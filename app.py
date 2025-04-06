# TODO:
# [x] add capture keyframe fn
# [x] add group video tab
# [x] add query 2nd textbox
# [x] add google translate (google translate py is bad)
# [x] add clip batch 2
#   [x] map-keyframe
#   [x] media-info
# [x] resize keyframe for thumbnail
# [x] fix potential bug because batch 2 contain discontinuous video
# [ ] use Selenium for Translation from web
# [x] add select for group-by-video
# [x] use `fps` from map-keyframe instead(important)
# [ ] run transnetv2 to extract more frames for better sumarization.
# [ ] add export to all for group-by-video view
# [x] limit search area by choosing which Lxx video to search

import gc
import io
import json
import os

import requests

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
from time import strftime
from typing import List


import gradio as gr
import pandas as pd
import torch
from tqdm import tqdm

from modules.retriever import FaissRetriever as Retriever, FaissRetriever
from modules.tag_retrieval import TagRetrieval
from modules.settings import (
    CLIP_MODEL_NAMES,
    DEFAULT_DB_DIR,
    DEFAULT_CLIP_FEATURE_DIR,
    DEFAULT_CLIP_FEATURE_V2_DIR,
    DEFAULT_CLIP_MODEL_CHOICE,
    DEFAULT_EMBEDDED_HTML,
    DEFAULT_KEYFRAMES_DIR,
    DEFAULT_KEYFRAMES_V2_DIR,
    DEFAULT_MAP_KEYFRAMES_DIR,
    DEFAULT_MAP_KEYFRAMES_V2_DIR,
    DEFAULT_MEDIA_INFO_DIR,
    DEFAULT_THUMBNAILS_DIR,
    DEFAULT_VIDEO_NAME,
    DEFAULT_YOUTUBE_ID,
    DEFAULT_DB_V1_DIR,
    DEFAULT_DB_V2_DIR,
    DEFAULT_FAISS_INDEX_V1_DIR,
    PROJECT_DIR,
)
from util.helper import (
    build_submit_kis_data,
    build_submit_qa_data,
    create_thumbnail,
    format_query_result,
    get_fps_from_video_name,
    get_keyframes_from_video_name,
    get_video_name_from_preview_video_html,
    get_video_name_from_youtube_id,
    get_youtube_id,
    parse_youtube_debug_info,
    read_txt,
    sanitize_model_name,
    get_faiss_id,
)

LIST_AVAILABLE_VIDEO_PACKS = [f"L{i+1:02d}" for i in range(30)]

with open("util/tag_list", "rb") as fp:
    TAGS_LIST = pickle.load(fp)


def test():
    print("Selected!")


def check_or_change_retriever(
    clip_model_name: str,
    db_dir: str,
    top_k: int = 100,
):
    global retriever
    # do nothing if the same model is already loaded.
    if isinstance(retriever, Retriever):
        if retriever.clip_model.model_name == clip_model_name:
            print("[DEBUG]: Using the same model.")
            pass
        else:
            print("[DEBUG]: Changing model.")
            retriever.set_model(clip_model_name)
            retriever.set_db_dir(db_dir)

        if retriever.db_dir != db_dir:
            print("  [DEBUG]: Changing db_dir.")
            retriever.set_db_dir(db_dir)

    # init retriever
    else:
        print("[DEBUG]: Initializing clip model.")
        retriever = Retriever(
            clip_model_name=clip_model_name,
            top_k=top_k,
            db_dir=db_dir,
            save_cache=False,
            cache_dir=None,
            device=None,
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if gr.NO_RELOAD:
    # from modules.vi_translator import Translator
    from googletrans import Translator

    retriever = None
    # preload model
    translator = Translator()
    check_or_change_retriever(
        DEFAULT_CLIP_MODEL_CHOICE,
        DEFAULT_DB_V1_DIR,
    )

    tag_retriever = TagRetrieval(db_dir=DEFAULT_DB_DIR, top_k=100)


def query_text(
    translated_textbox_str: str,
    k_value: int,
    db_dir: str,
    blacklist=None,
):
    global retriever
    retrieved_results = retriever.retrieve_text(
        translated_textbox_str,
        top_k=k_value,
        blacklist=blacklist,
    )
    # retrieved_results = retriever.retrieve_faiss(
    #     translated_textbox_str,
    #     top_k=k_value,
    #     blacklist=blacklist,
    # )
    retrieved_results = format_query_result(retrieved_results, db_dir)
    result_gallery = gr.Gallery(visible=True, value=retrieved_results)
    gr.Info("Query Text successfully!", duration=5)
    return result_gallery


def query_2scenes_text(
    translated_textbox_str_1: str,
    translated_textbox_str_2: str,
    k_value: int,
    next_keyframe_duration: float,
    first_k_limit: int,
    db_dir: str,
    blacklist=None,
):
    global retriever
    retrieved_results = retriever.retrieve_2_text(
        translated_textbox_str_1,
        translated_textbox_str_2,
        next_keyframe_duration=next_keyframe_duration,
        top_k=k_value,
        first_k_limit=first_k_limit,
        blacklist=blacklist,
    )

    retrieved_results = format_query_result(retrieved_results, db_dir)
    result_gallery = gr.Gallery(visible=True, value=retrieved_results)
    gr.Info("Query Text successfully!", duration=5)
    return result_gallery


def query_text_chooser(
    translated_textbox_str_1: str,
    translated_textbox_str_2: str,
    k_value: int,
    next_keyframe_duration: float,
    first_k_limit: int,
    clip_model_names: List[str],
    list_checkbox,
    db_dir: str,
):
    global retriever

    gr.Info("Querying...", duration=5)
    # TODO: add multiple model
    # right now we only get the first choice:
    if not isinstance(clip_model_names, list):
        clip_model_names = [clip_model_names]
    for clip_model_name in clip_model_names:
        check_or_change_retriever(
            clip_model_name,
            db_dir=db_dir,
        )
        break

    blacklist = [
        video_pack
        for video_pack in LIST_AVAILABLE_VIDEO_PACKS
        if video_pack not in list_checkbox
    ]

    if translated_textbox_str_1 and translated_textbox_str_2:
        return query_2scenes_text(
            translated_textbox_str_1,
            translated_textbox_str_2,
            k_value=k_value,
            next_keyframe_duration=next_keyframe_duration,
            first_k_limit=first_k_limit,
            db_dir=db_dir,
            blacklist=blacklist,
        )
    if translated_textbox_str_1:
        return query_text(
            translated_textbox_str_1,
            k_value,
            db_dir,
            blacklist=blacklist,
        )
    if translated_textbox_str_2:
        return query_text(
            translated_textbox_str_2,
            k_value,
            db_dir,
            blacklist=blacklist,
        )
    return


def query_text_next_search(
    next_search_textbox_str: str,
    k_value: int,
    db_dir: str,
    last_results: List[str],
):
    global retriever

    if not isinstance(retriever, FaissRetriever):
        gr.Warning("Not using Faiss Retriever, this won't work")
        return gr.skip()

    last_results_keyframes = []
    for _, label in last_results:
        keyframe_path = "_".join(label.split("_")[:2])
        last_results_keyframes.append(keyframe_path)

    retrieved_results = retriever.retrieve_faiss_next_search(
        next_search_textbox_str,
        top_k=k_value,
        faiss_idx=get_faiss_id(last_results_keyframes, retriever.idx2keyframe),
    )
    # retrieved_results = retriever.retrieve_faiss(
    #     translated_textbox_str,
    #     top_k=k_value,
    #     blacklist=blacklist,
    # )
    retrieved_results = format_query_result(retrieved_results, db_dir)
    # result_gallery = gr.Gallery(visible=True, value=retrieved_results)
    # gr.Info("Query Text successfully!", duration=5)
    # return result_gallery, keyframe_names
    # print(retrieved_results)
    return retrieved_results


def query_image(
    image,
    k_value: int,
    db_dir: str,
    blacklist,
):
    global retriever

    image = retriever.preprocess_image(image).unsqueeze(0)
    retrieved_results = retriever.retrieve_image(
        image,
        top_k=k_value,
        blacklist=blacklist,
    )

    retrieved_results = format_query_result(retrieved_results, db_dir)

    result_gallery = gr.Gallery(visible=True, value=retrieved_results)
    gr.Info("Query Image successfully!", duration=5)
    return result_gallery


def query_2scenes_image(
    image1,
    image2,
    k_value: int,
    db_dir: str,
    next_keyframe_duration: float = 15,
    first_k_limit: int = 2000,
    blacklist=None,
):
    global retriever

    image1 = retriever.preprocess_image(image1).unsqueeze(0)
    image2 = retriever.preprocess_image(image2).unsqueeze(0)

    retrieved_results = retriever.retrieve_2_images(
        image1,
        image2,
        next_keyframe_duration=next_keyframe_duration,
        top_k=k_value,
        first_k_limit=first_k_limit,
        blacklist=blacklist,
    )
    retrieved_results = format_query_result(retrieved_results, db_dir)
    result_gallery = gr.Gallery(visible=True, value=retrieved_results)
    gr.Info("Query Images successfully!", duration=5)
    return result_gallery


def query_images_chooser(
    image1,
    image2,
    k_value: int,
    first_k_limit: int,
    next_keyframe_duration: float,
    list_checkbox,
    clip_model_names: List[str],
    db_dir: str,
):
    global retriever
    gr.Info("Querying...", duration=5)
    # TODO: add multiple model
    # right now we only get the first choice:
    if not isinstance(clip_model_names, list):
        clip_model_names = [clip_model_names]
    for clip_model_name in clip_model_names:
        check_or_change_retriever(
            clip_model_name,
            db_dir=db_dir,
        )
        break

    blacklist = [
        video_pack
        for video_pack in LIST_AVAILABLE_VIDEO_PACKS
        if video_pack not in list_checkbox
    ]

    if image1 and image2:
        return query_2scenes_image(
            image1,
            image2,
            k_value=k_value,
            next_keyframe_duration=next_keyframe_duration,
            first_k_limit=first_k_limit,
            db_dir=db_dir,
            blacklist=blacklist,
        )

    image = image1 if image1 else image2 if image2 else None
    if image:
        return query_image(
            image,
            k_value,
            db_dir=db_dir,
            blacklist=blacklist,
        )
    return


def query_tags(
    tags_string: str,
    k_value: int,
    database_version: str,
):
    global tag_retriever

    if database_version != "v1":
        gr.Warning("Must choose database version 1 (v1) to continue", duration=5)
        return

    retrieved_results = tag_retriever(tags_string, k_value)
    retrieved_results = format_query_result(
        retrieved_results,
        db_dir=DEFAULT_DB_V1_DIR,
    )
    result_gallery = gr.Gallery(visible=True, value=retrieved_results)
    gr.Info("Query Tags successfully!", duration=5)
    return result_gallery


def upload_query_textbox_value(source_query_textbox):
    dest_query_textbox = gr.Textbox(value=source_query_textbox)
    return (
        dest_query_textbox,
        dest_query_textbox,
        dest_query_textbox,
        dest_query_textbox,
    )


def reset_upload_status():
    return gr.File(value=None)


def clear_gallery():
    return gr.Gallery(selected_index=None, value=None)


def clear_image_query_gallery():
    return gr.Gallery(selected_index=None, value=None)


def export_result_to_csv(
    gallery: gr.Gallery,
    answer_textbox: str,
    db_dir: str,
):
    results = []

    for *_, img_caption in gallery:
        # f"{video_keyframe}_{score:.3f}_{time_in_second}_{h:02d}:{m:02d}:{s:02d}",
        # L01_V001/001_score_0_00:00:00
        video_keyframe = img_caption.split(".jpg")[0]
        video_name, keyframe = video_keyframe.split("/")
        time_in_second = img_caption.split("_")[3]
        time_in_second = float(time_in_second)
        fps = get_fps_from_video_name(video_name, db_dir)
        frame_idx = int(time_in_second * fps)
        results.append((video_name, frame_idx, answer_textbox))

    df = pd.DataFrame(
        results,
        columns=["video", "frame_idx", "answer"],
    )

    if answer_textbox == "":
        df = df[["video", "frame_idx"]]

    time_str = strftime("%Y_%m_%d_%H_%M_%S")

    df.to_csv(f"output/{time_str}.csv", index=False, header=False)
    # df = df.style.set_properties(**{"font-size": "11pt"})
    gr.Info("Exported successfully!", duration=5)
    return df, df.to_csv(index=False)


def add_selected_image(
    evt: gr.SelectData,
    view_mode: str,
    result_gallery: list,
    selected_gallery: list,
    db_dir: str,
):
    idx = evt.index
    img_info = evt.value

    img_path = img_info["image"]["path"]
    img_caption = img_info["caption"]
    timestamp = img_caption.split("_")[3]
    video_name = img_caption.split("/")[0]
    fps = get_fps_from_video_name(video_name, db_dir)
    youtube_embed_id = get_youtube_id(video_name)
    video_preview = gr.HTML(
        DEFAULT_EMBEDDED_HTML.format(
            video_name=video_name,
            fps=fps,
            youtube_id=youtube_embed_id,
            start_time=int(float(timestamp)),
        )
    )

    if view_mode == "Select":
        if selected_gallery:
            if (img_path, img_caption) not in selected_gallery:
                selected_gallery.append((img_path, img_caption))
        else:
            selected_gallery = [(img_path, img_caption)]

        return (
            gr.Gallery(selected_index=None),
            selected_gallery,
            video_preview,
        )
    if view_mode == "Remove":
        result_gallery.pop(idx)  # ("temp_path", "caption")
        new_result = result_gallery
        return new_result, gr.skip(), gr.skip()
    else:
        return gr.Gallery(), gr.skip(), video_preview


def preview_or_remove_selected_image(
    evt: gr.SelectData,
    view_mode: str,
    gallery_samples: gr.Gallery,
    db_dir: str,
):
    idx = evt.index
    img_info = evt.value
    # img_path = img_info["image"]["path"]
    img_caption = img_info["caption"]
    timestamp = img_caption.split("_")[3]
    video_name = img_caption.split("/")[0]
    fps = get_fps_from_video_name(video_name, db_dir)
    youtube_embed_id = get_youtube_id(video_name)
    video_preview = gr.HTML(
        DEFAULT_EMBEDDED_HTML.format(
            video_name=video_name,
            fps=fps,
            youtube_id=youtube_embed_id,
            start_time=int(float(timestamp)),
        )
    )
    if view_mode == "Preview":
        return gr.skip(), gr.skip(), gr.skip()
    if view_mode == "View Video Keyframes":
        return gr.Gallery(selected_index=None), video_preview, gr.skip()
    else:
        gallery_samples.pop(idx)  # ("temp_path", "caption")
        new_result = gallery_samples
        video_preview = gr.HTML(
            DEFAULT_EMBEDDED_HTML.format(
                video_name=DEFAULT_VIDEO_NAME,
                fps=0,
                youtube_id=DEFAULT_YOUTUBE_ID,
                start_time=0,
            )
        )
        return (
            gr.Gallery(value=new_result, selected_index=None),
            gr.Gallery(),
            gr.Gallery(),
        )


def change_result_mode(result_mode):
    if result_mode == "Preview":
        return gr.Gallery(allow_preview=True)
    if result_mode == "Remove":
        return gr.Gallery(allow_preview=False)
    if result_mode == "Select":
        return gr.Gallery(allow_preview=False)
    else:
        return gr.Gallery(allow_preview=False)


def video_select_fn(
    filepath: str,
    db_dir: str,
):
    if not filepath:
        video_preview = gr.HTML(
            DEFAULT_EMBEDDED_HTML.format(
                video_name=DEFAULT_VIDEO_NAME,
                fps=0,
                youtube_id=DEFAULT_YOUTUBE_ID,
                start_time=0,
            )
        )
        return video_preview, None

    video_name = os.path.basename(filepath).replace(".json", "")
    video_id = get_youtube_id(video_name=video_name)
    fps = get_fps_from_video_name(video_name, db_dir)
    keyframe_paths_labels_pair = get_keyframes_from_video_name(video_name, db_dir)
    video_preview = gr.HTML(
        DEFAULT_EMBEDDED_HTML.format(
            video_name=video_name,
            fps=fps,
            youtube_id=video_id,
            start_time=0,
        )
    )
    return video_preview, keyframe_paths_labels_pair


def _get_keyframes_from_current_video_fn(
    html_doc: str,
    db_dir: str,
):
    video_name = get_video_name_from_preview_video_html(html_doc)
    keyframe_paths_labels_pair = get_keyframes_from_video_name(video_name, db_dir)
    return keyframe_paths_labels_pair


def _capture_frame_from_debug_info_fn(
    debug_info: str,
    output_textbox_str: str,
    db_dir: str,
):
    youtube_id, time = parse_youtube_debug_info(debug_info)
    video_name = get_video_name_from_youtube_id(youtube_id)

    fps = get_fps_from_video_name(video_name, db_dir)

    frame_idx = int(time * fps + 0.5)
    result = f"{video_name},{frame_idx}\n"
    result = output_textbox_str + result
    return result


def _capture_json_from_debug_info_fn(
    debug_info: str,
    qa_answer: str,
    db_dir: str,
):
    youtube_id, time = parse_youtube_debug_info(debug_info)
    video_name = get_video_name_from_youtube_id(youtube_id)

    time_in_ms = int(time * 1000)
    if qa_answer != "":
        result = f"{video_name},{time_in_ms},{qa_answer}\n"
    else:
        result = f"{video_name},{time_in_ms}\n"
    return result


def _on_capture_info_json_textbox_change(captured_json_answer: str):
    answers = captured_json_answer.strip().split(",")
    submit_data = "Cannot read captured_json_answer"
    if len(answers) == 3:
        submit_data = str(json.dumps(build_submit_qa_data(*answers), indent=4))
    if len(answers) == 2:
        submit_data = str(json.dumps(build_submit_kis_data(*answers), indent=4))
    return submit_data


def _on_login_btn_click(
    username: str,
    password: str,
):
    login_url = "https://eventretrieval.one/api/v2/login"
    login_data = {
        "username": username,
        "password": password,
    }
    response = requests.post(login_url, json=login_data)
    if response.ok:
        response = json.loads(response.content)
        session_id = response["sessionId"]
        gr.Info(f"Login successful with sessionId: {session_id}")
        return session_id
    gr.Warning(f"Login unsucessful!")
    return f"LOGIN UNSUCCESSFUL!!!!! - response: {response.reason}"


def _on_get_evaluation_id_btn(session_id: str):
    get_evaluation_id_url = "https://eventretrieval.one/api/v2/client/evaluation/list"
    evaluation_data = {"session": session_id}
    response = requests.get(get_evaluation_id_url, params=evaluation_data)
    if response.ok:
        response = json.loads(response.content)
        evaluation_id = response[0]["id"]
        evaluation_name = response[0]["name"]
        return evaluation_id, evaluation_name
    return f"Failed to get evaluation ID... {response.reason}", "NONE"


def _on_submit_btn(captured_json_answer: str, session_id: str, evaluation_id: str):
    answers = captured_json_answer.strip().split(",")
    submit_url = (
        f"https://eventretrieval.one/api/v2/submit/{evaluation_id}?session={session_id}"
    )
    if len(answers) == 3:
        submit_data = build_submit_qa_data(*answers)
    if len(answers) == 2:
        submit_data = build_submit_kis_data(*answers)

    response = requests.post(submit_url, json=submit_data)
    if response.ok:
        gr.Info("Submit successful!")
        return str(response.content)
    return f"Failed to get evaluation ID... {response.reason}"


def _parse_debug_info_detail(debug_info, db_dir: str):
    youtube_id, t = parse_youtube_debug_info(debug_info)
    video_name = get_video_name_from_youtube_id(youtube_id)
    fps = get_fps_from_video_name(video_name, db_dir)
    frame_index = int(t * fps + 0.5)
    timestamp = int(t)
    h = timestamp // 3600
    m = timestamp // 60
    s = timestamp % 60
    ms = t - timestamp
    time_str = f"{h:02d} : {m:02d} : {s:02d} ({ms:.3f})"
    result = (
        f"{'video_name':<15}: {video_name}\n"
        f"{'time':<15}: {t}\n"
        f"{'fps':<15}: {fps}\n"
        f"{'timestamp':<15}: {time_str}\n"
        f"{'frame_idx':<15}: {frame_index}\n"
        f"{'id':<15}: {youtube_id}"
    )
    return result


def _on_show_keyframe_in_video(caption: str, db_dir: str):
    _, _, time_in_second, video_name = caption.split("-")
    fps = get_fps_from_video_name(video_name, db_dir)
    youtube_embed_id = get_youtube_id(video_name)
    video_preview = gr.HTML(
        DEFAULT_EMBEDDED_HTML.format(
            video_name=video_name,
            fps=fps,
            youtube_id=youtube_embed_id,
            start_time=int(float(time_in_second)),
        )
    )
    return video_preview


def _on_capture_this_keyframe_btn(
    caption: str,
    output_textbox_str: str,
    db_dir: str,
):
    _, _, time_in_second, video_name = caption.split("-")
    time_in_second = float(time_in_second)
    fps = get_fps_from_video_name(video_name, db_dir)
    frame_idx = int(time_in_second * fps + 0.5)

    result = f"{video_name},{frame_idx}\n"
    result = output_textbox_str + result
    return result


def _on_keyframe_gallery_select(evt: gr.SelectData):
    return evt.value["caption"]


def _on_create_thumbnail_confirm(progress=gr.Progress(), thumbnails_dir: str = None):
    gr.Warning("Not working at the moment.")
    return
    # if not thumbnails_dir:
    #     pass
    # import joblib

    # n_jobs = 4
    # verbose = 1

    # fwalker = os.walk(DEFAULT_KEYFRAMES_DIR)
    # next(fwalker)
    # inputs = []
    # for r, _, fs in fwalker:
    #     inputs.extend([os.path.join(r, f) for f in fs if ".jpg" in f])

    # progress(0, "Starting...")
    # jobs = [
    #     joblib.delayed(create_thumbnail)(i, thumbnails_dir)
    #     for i in progress.tqdm(inputs)
    # ]
    # joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)


def on_select_all_checkboxes():
    return LIST_AVAILABLE_VIDEO_PACKS


def on_deselect_all_checkboxes():
    return []


def on_database_version_change(version):
    if version == "v2":
        return DEFAULT_DB_V2_DIR
    return DEFAULT_DB_V1_DIR


####################################################################################################
####################################################################################################
####################################################################################################
with gr.Blocks(
    title="Demo",
    css='#translated_textbox textarea { border: 3px solid #ffc342; } #debug_info_parsed_textbox textarea { font-family: "IBM Plex Mono" }',
    #    css=".contain { display: flex !important; flex-direction: column !important; }"
    #         "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
    #         "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
    #         "#col { height: 100vh !important; }",
    theme=gr.themes.Default(
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_none,
    ),
    fill_width=True,
    fill_height=True,
) as demo:
    #############################

    database_version = gr.State("v1")
    db_dir = gr.State(DEFAULT_DB_V1_DIR)

    with gr.Tab("Query options"):
        with gr.Row(variant="panel", equal_height=True):
            upload_file = gr.File(scale=5)
            uploaded_textbox = gr.Textbox(
                label="Uploaded Text",
                # info="Initial text",
                lines=5,
                value="The quick brown fox jumped over the lazy dogs.",
                interactive=True,
                scale=8,
            )

            translate_btn = gr.Button("Translate", scale=1)

        with gr.Row(equal_height=True):
            query_textbox1 = gr.Textbox(
                label="Describe first frame",
                lines=3,
                value="English text here!",
                interactive=True,
                elem_id="translated_textbox",
            )
            # TODO: add logic for this next frame search
            query_textbox2 = gr.Textbox(
                label="Describe next frame (optional)",
                lines=3,
                value="",
                interactive=True,
                elem_id="translated_textbox",
            )
        with gr.Row(equal_height=True):
            database_selection = gr.Radio(
                ["v1", "v2"],
                label="Choose Database version",
                value="v1",
                interactive=True,
            )
            clip_options = gr.Radio(
                CLIP_MODEL_NAMES,
                value=DEFAULT_CLIP_MODEL_CHOICE,
                label="CLIP Model",
                interactive=True,
                info="Select CLIP Models that you want to run",
            )
            k_value = gr.Number(value=100, minimum=0, maximum=2000)

            with gr.Column():
                text_next_frame_duration = gr.Number(
                    value=15, minimum=3, maximum=100, label="Next Frame Duration Limit"
                )
                text_1st_scene_k_limit = gr.Number(
                    value=1000,
                    minimum=100,
                    maximum=200000,
                    label="Top Score From First Scene For Next Search",
                )

        with gr.Row(equal_height=True):

            list_checkbox = gr.CheckboxGroup(
                LIST_AVAILABLE_VIDEO_PACKS,
                value=LIST_AVAILABLE_VIDEO_PACKS,
                label="Choose video to query",
            )
            with gr.Row(equal_height=True):
                select_all_checkbox = gr.Button(value="Select All")
                deselect_all_checkbox = gr.Button(value="Deselect All")

        query_text_btn = gr.Button("Query", variant="primary")

    #############################
    with gr.Tab("Result"):
        with gr.Row(equal_height=True):
            query_textbox3 = gr.Textbox(
                label="First frame",
                lines=2,
                # value="English text here!",
                interactive=False,
                # elem_id="translated_textbox",
            )
            query_textbox4 = gr.Textbox(
                label="Next frame",
                lines=2,
                # value="",
                interactive=False,
                # elem_id="translated_textbox",
            )
        with gr.Row(equal_height=True):
            query_txtbox_for_next_search = gr.Textbox(
                label="Next search",
                lines=2,
                # value="",
                interactive=True,
            )
            with gr.Column():
                top_k_next_search = gr.Number(
                    value=100,
                    minimum=0,
                    maximum=2000,
                    interactive=True,
                )
                btn_query_next_search = gr.Button("Next Search")

        result_mode = gr.Radio(
            ["Preview", "Select", "Remove"],
            label="Select or Preview Image",
            value="Select",
            interactive=True,
        )
        with gr.Tab("Normal"):
            # TODO: add function
            export_all_keyframes_to_selected_btn = gr.Button(
                "Export to Selected",
                variant="primary",
            )
            result_gallery = gr.Gallery(
                columns=10,
                rows=10,
                show_fullscreen_button=True,
                visible=True,
                # height="max-content",
                interactive=False,
                allow_preview=False,
            )
        with gr.Tab("Group-by-video"):
            # TODO: add function
            export_all_grouped_keyframes_to_selected_btn = gr.Button(
                "Export to Selected",
                variant="primary",
            )

            @gr.render(inputs=[result_gallery])
            def group_result_gallery_by_video(result_gallery):
                if result_gallery:
                    if len(result_gallery) == 0:
                        gr.Markdown("## No results")
                    else:
                        d = {}
                        for keyframe_path, caption in result_gallery:
                            video_name = caption.split("/")[0]
                            if video_name not in d:
                                d[video_name] = [(keyframe_path, caption)]
                            else:
                                d[video_name].append((keyframe_path, caption))
                        list_galleries = []
                        for video_name in d:
                            gr.Markdown(f"## {video_name}")
                            with gr.Accordion():
                                gallery = gr.Gallery(
                                    d[video_name],
                                    columns=10,
                                    rows=3,
                                    show_fullscreen_button=True,
                                    visible=True,
                                    # height="max-content",
                                    interactive=False,
                                    allow_preview=False,
                                    object_fit="contain",
                                )
                            list_galleries.append(gallery)
                        for gallery in list_galleries:
                            gallery.select(
                                add_selected_image,
                                inputs=[
                                    result_mode,
                                    gallery,
                                    selected_gallery,
                                    db_dir,
                                ],
                                outputs=[
                                    gallery,
                                    selected_gallery,
                                    video_preview,
                                ],
                            )

                else:
                    gr.Markdown("## Result is None")

            # result_grouped_gallery = gr.Gallery(
            #     columns=10,
            #     rows=10,
            #     show_fullscreen_button=True,
            #     visible=True,
            #     # height="max-content",
            #     interactive=False,
            #     allow_preview=False,
            # )
    #############################
    with gr.Tab("Query Images"):
        with gr.Row(equal_height=True):
            query_textbox5 = gr.Textbox(
                label="First frame",
                lines=2,
                # value="English text here!",
                interactive=False,
                # elem_id="translated_textbox",
            )
            query_textbox6 = gr.Textbox(
                label="Next frame",
                lines=2,
                # value="",
                interactive=False,
                # elem_id="translated_textbox",
            )
        query_image_clear_btn = gr.Button("Clear", variant="stop")
        with gr.Row(equal_height=True):
            query_image_placeholder_1 = gr.Image(type="pil", label="First Scene")
            query_image_placeholder_2 = gr.Image(type="pil", label="Second Scene")
        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Row(equal_height=True):
                    query_image_database_selection = gr.Radio(
                        ["v1", "v2"],
                        label="Database version selection",
                        value="v1",
                        interactive=True,
                    )
                    query_image_clip_options = gr.Radio(
                        CLIP_MODEL_NAMES,
                        value=DEFAULT_CLIP_MODEL_CHOICE,
                        label="CLIP Model",
                        interactive=True,
                        info="Select CLIP Models that you want to run",
                    )
                    query_image_k_value = gr.Number(
                        value=100, minimum=0, maximum=2000, label="Top_k"
                    )
                    with gr.Column():
                        image_next_frame_duration = gr.Number(
                            value=15,
                            minimum=3,
                            maximum=100,
                            label="Next Frame Duration Limit",
                        )
                        image_first_scene_limit = gr.Number(
                            value=1000,
                            minimum=100,
                            maximum=200000,
                            label="Top Score From First Scene For Next Search",
                        )

                with gr.Row(equal_height=True):
                    list_checkbox_query_image = gr.CheckboxGroup(
                        LIST_AVAILABLE_VIDEO_PACKS,
                        value=LIST_AVAILABLE_VIDEO_PACKS,
                        label="Choose video to query",
                        interactive=True,
                    )
                    # for i in range(24):
                    #     checkbox = gr.Checkbox(
                    #         label=f"L{i+1:02d}", interactive=True, value=True
                    #     )
                    #     list_checkbox.append(checkbox)
                    with gr.Row(equal_height=True):
                        select_all_checkbox_query_image = gr.Button(value="Select All")
                        deselect_all_checkbox_query_image = gr.Button(
                            value="Deselect All"
                        )

                query_image_query_btn = gr.Button("Query", variant="primary")
                result_image_mode = gr.Radio(
                    ["Preview", "Select", "Remove"],
                    label="Select or Preview Image",
                    value="Select",
                    interactive=True,
                )

        # with gr.Row():
        with gr.Tab("Normal"):
            query_image_result_gallery = gr.Gallery(
                columns=10,
                show_fullscreen_button=True,
                visible=True,
                height="max-content",
                interactive=False,
                allow_preview=False,
            )
        with gr.Tab("Group-by-video"):

            @gr.render(inputs=[query_image_result_gallery])
            def group_result_gallery_by_video(query_image_result_gallery):
                if query_image_result_gallery:
                    if len(query_image_result_gallery) == 0:
                        gr.Markdown("## No results")
                    else:
                        d = {}
                        for keyframe_path, caption in query_image_result_gallery:
                            video_name = caption.split("/")[0]
                            if video_name not in d:
                                d[video_name] = [(keyframe_path, caption)]
                            else:
                                d[video_name].append((keyframe_path, caption))
                        list_galleries_images = []
                        for video_name in d:
                            gr.Markdown(f"## {video_name}")
                            with gr.Accordion():
                                gallery = gr.Gallery(
                                    d[video_name],
                                    columns=10,
                                    rows=1,
                                    show_fullscreen_button=True,
                                    visible=True,
                                    interactive=False,
                                    allow_preview=False,
                                    object_fit="contain",
                                )
                            list_galleries_images.append(gallery)
                        for gallery in list_galleries_images:
                            gallery.select(
                                add_selected_image,
                                inputs=[
                                    result_image_mode,
                                    gallery,
                                    selected_gallery,
                                    db_dir,
                                ],
                                outputs=[
                                    gallery,
                                    selected_gallery,
                                    video_preview,
                                ],
                            )

                else:
                    gr.Markdown("## Result is None")

    #############################
    with gr.Tab("Query Tags"):

        with gr.Row(equal_height=True):
            upload_file2 = gr.File(scale=2)

            with gr.Column(scale=7):
                uploaded_textbox2 = gr.Textbox(
                    label="Uploaded Text",
                    # info="Initial text",
                    lines=3,
                    value="The quick brown fox jumped over the lazy dogs.",
                    interactive=True,
                )

                recommended_tags = gr.Textbox(
                    label="Recommended Tags",
                    # info="Initial text",
                    lines=3,
                    value="",
                    interactive=True,
                )

            recommended_tag_btn = gr.Button("Tag Recommendation", scale=1)

        with gr.Row(equal_height=True):
            select_tags = gr.Dropdown(
                choices=TAGS_LIST,
                multiselect=True,
                label="Selected Tags",
                show_label=True,
                interactive=True,
                scale=9,
            )

            compile_tag_btn = gr.Button("Compiled Tag Query", scale=1)

        with gr.Row(equal_height=True):
            compiled_tags = gr.Textbox(
                label="Compiled Tag",
                # info="Initial text",
                lines=2,
                value="",
                interactive=True,
                scale=8,
            )

            query_tag_k_value = gr.Number(
                value=100, minimum=0, maximum=200000, label="Top_k", scale=1
            )

            query_tag_btn = gr.Button("Query Tag", scale=1, variant="primary")

        with gr.Row(equal_height=True):
            result_tag_mode = gr.Radio(
                ["Preview", "Select", "Remove"],
                label="Select or Preview Tag",
                value="Select",
                interactive=True,
            )

        with gr.Tab("Normal"):
            query_tag_result_gallery = gr.Gallery(
                columns=10,
                show_fullscreen_button=True,
                visible=True,
                height="max-content",
                interactive=False,
                allow_preview=False,
            )
        with gr.Tab("Group-by-video"):

            @gr.render(inputs=[query_tag_result_gallery])
            def group_result_gallery_by_video(query_tag_result_gallery):
                if query_tag_result_gallery:
                    if len(query_tag_result_gallery) == 0:
                        gr.Markdown("## No results")
                    else:
                        d = {}
                        for keyframe_path, caption in query_tag_result_gallery:
                            video_name = caption.split("/")[0]
                            if video_name not in d:
                                d[video_name] = [(keyframe_path, caption)]
                            else:
                                d[video_name].append((keyframe_path, caption))
                        list_galleries_images = []
                        for video_name in d:
                            gr.Markdown(f"## {video_name}")
                            with gr.Accordion():
                                gallery = gr.Gallery(
                                    d[video_name],
                                    columns=10,
                                    rows=1,
                                    show_fullscreen_button=True,
                                    visible=True,
                                    interactive=False,
                                    allow_preview=False,
                                )
                            list_galleries_images.append(gallery)
                        for gallery in list_galleries_images:
                            gallery.select(
                                add_selected_image,
                                inputs=[
                                    result_image_mode,
                                    gallery,
                                    selected_gallery,
                                    db_dir,
                                ],
                                outputs=[
                                    gallery,
                                    selected_gallery,
                                    video_preview,
                                ],
                            )

                else:
                    gr.Markdown("## Result is None")

    #############################

    with gr.Tab("Selected Images"):
        with gr.Row(equal_height=True):
            query_textbox7 = gr.Textbox(
                label="First frame",
                lines=2,
                # value="English text here!",
                interactive=False,
                # elem_id="translated_textbox",
            )
            query_textbox8 = gr.Textbox(
                label="Next frame",
                lines=2,
                # value="",
                interactive=False,
                # elem_id="translated_textbox",
            )

        clear_selected_gallery_btn = gr.Button("Clear", variant="stop")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                answer_textbox = gr.Textbox(
                    label="Answer (Optional)",
                    # info="Initial text",
                    lines=5,
                    max_lines=5,
                    value="",
                    interactive=True,
                )
            with gr.Column(scale=2):
                with gr.Tab("DataFrame"):
                    export_to_textbox_btn = gr.Button("Export to TextBox")
                    csv_result_dataframe = gr.Dataframe(
                        label="Result",
                        interactive=True,
                        max_height=300,
                    )
                with gr.Tab("TextBox"):
                    export_to_dataframe_btn = gr.Button("Export to DataFrame")
                    csv_result_textbox = gr.Textbox(
                        label="Result",
                        interactive=True,
                        lines=5,
                        max_lines=15,
                    )
        with gr.Row(equal_height=True):
            export_selected_to_result_btn = gr.Button("Import")
        result_selected_image_mode = gr.Radio(
            ["Preview", "Remove", "View Video Keyframes"],
            label="Preview or Remove Image",
            value="Preview",
            interactive=True,
        )
        with gr.Row(equal_height=True):
            selected_gallery = gr.Gallery(
                columns=5,
                show_fullscreen_button=True,
                visible=True,
                height="max-content",
                interactive=False,
                allow_preview=True,
            )

    #############################
    with gr.Tab("Preview"):
        with gr.Row(equal_height=True):
            query_textbox9 = gr.Textbox(
                label="First frame",
                lines=2,
                # value="English text here!",
                interactive=False,
                # elem_id="translated_textbox",
            )
            query_textbox10 = gr.Textbox(
                label="Next frame",
                lines=2,
                # value="",
                interactive=False,
                # elem_id="translated_textbox",
            )
        with gr.Row(equal_height=True):
            with gr.Row(equal_height=True):

                with gr.Column(scale=1):
                    video_explorer = gr.FileExplorer(
                        f"*.json",
                        label="Select Video",
                        root_dir=DEFAULT_MEDIA_INFO_DIR,
                        height=600,
                        file_count="single",
                        ignore_glob="*.jpg",
                    )

                with gr.Column(scale=5):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            with gr.Row(equal_height=True):
                                get_current_video_keyframes_btn = gr.Button("Get current video keyframes")  # fmt: skip
                                show_keyframe_in_video_btn = gr.Button("Show keyframe in video")  # fmt: skip
                                capture_current_selected_keyframe_btn = gr.Button(
                                    "Capture this keyframe",
                                    variant="primary",
                                )
                                selected_keyframe_caption_state = gr.State()
                            keyframe_gallery = gr.Gallery(
                                columns=4,
                                # rows=3,
                                show_fullscreen_button=True,
                                visible=True,
                                # scale=2,
                                interactive=False,
                                allow_preview=True,
                                # preview=True
                                object_fit="contain",
                            )
                        video_preview = gr.HTML(
                            DEFAULT_EMBEDDED_HTML.format(
                                video_name="NO VIDEO",
                                fps=0,
                                youtube_id=DEFAULT_YOUTUBE_ID,
                                start_time=0,
                            ),
                        )
                    with gr.Row(variant="panel", equal_height=True):
                        with gr.Column():
                            debug_info_textbox = gr.Textbox(
                                label="Debug Info here",
                                lines=1,
                                max_lines=1,
                                interactive=True,
                            )
                            with gr.Row(equal_height=True):
                                debug_info_parsed_textbox = gr.Textbox(
                                    label="Parsed Info",
                                    lines=5,
                                    max_lines=5,
                                    interactive=True,
                                    elem_id="debug_info_parsed_textbox",
                                )
                                answer_json_textbox = gr.Textbox(
                                    label="Answer (QA)",
                                    lines=5,
                                    max_lines=5,
                                )

                            capture_from_debug_info_btn = gr.Button(
                                "Capture",
                                variant="primary",
                            )
                        with gr.Column():
                            with gr.Tab("Json"):
                                evaluation_display_label = gr.Label(
                                    "NONE", label="Current Evaluation Name"
                                )
                                with gr.Row(equal_height=True):
                                    session_id_textbox = gr.Textbox(
                                        label="Session ID",
                                        lines=1,
                                        max_lines=1,
                                        show_copy_button=True,
                                        show_label=True,
                                        interactive=True,
                                    )
                                    evalution_id_textbox = gr.Textbox(
                                        label="Evaluation ID",
                                        lines=1,
                                        max_lines=1,
                                        show_copy_button=True,
                                        show_label=True,
                                        interactive=True,
                                    )
                                    get_evaluation_id_btn = gr.Button(
                                        "Get Evaluation ID",
                                        variant="stop",
                                    )
                                capture_info_json_textbox = gr.Textbox(
                                    label="Captured answer",
                                    lines=1,
                                    max_lines=1,
                                    interactive=True,
                                    show_copy_button=True,
                                )
                                output_json_data_textbox = gr.Textbox(
                                    label="Submit Data Json",
                                    lines=5,
                                    interactive=False,
                                    show_copy_button=True,
                                    show_label=True,
                                )
                                submit_json_btn = gr.Button("SUBMIT", variant="stop")
                                submit_json_output_label = gr.Label(
                                    "", label="Submit output"
                                )

                            with gr.Tab("Frame Index"):
                                capture_info_textbox = gr.Textbox(
                                    label="Captured info",
                                    lines=5,
                                    max_lines=5,
                                    interactive=True,
                                    show_copy_button=True,
                                )

    #############################
    with gr.Tab("Misc"):
        with gr.Row(equal_height=True):
            username_textbox = gr.Textbox(
                label="Username",
                lines=1,
                max_lines=1,
                show_copy_button=True,
                show_label=True,
            )
            password_textbox = gr.Textbox(
                label="Password",
                type="password",
                lines=1,
                max_lines=1,
                show_copy_button=True,
                show_label=True,
            )
            login_btn = gr.Button("Login", variant="primary")

        with gr.Row(equal_height=True):
            create_thumbnail_setting_text = gr.Label(
                "Thumbnails", container=False, scale=1
            )
            with gr.Row(equal_height=True):
                create_thumbnail_btn = gr.Button("Generate Thumbnails")

                create_thumbnail_confirm_btn = gr.Button(
                    "Confirm",
                    variant="stop",
                    visible=False,
                )
                create_thumbnail_cancel_btn = gr.Button("Cancel", visible=False)
    #############################

    upload_file.upload(
        read_txt,
        inputs=upload_file,
        outputs=uploaded_textbox,
    )
    uploaded_textbox.change(
        reset_upload_status,
        inputs=None,
        outputs=upload_file,
    )
    translate_btn.click(
        lambda x: translator.translate(x, dest="en", src="vi").text,
        inputs=uploaded_textbox,
        outputs=query_textbox1,
    )
    query_image_query_btn.click(
        query_images_chooser,
        inputs=[
            query_image_placeholder_1,
            query_image_placeholder_2,
            query_image_k_value,
            image_first_scene_limit,
            image_next_frame_duration,
            list_checkbox_query_image,
            query_image_clip_options,
            db_dir,
        ],
        outputs=[query_image_result_gallery],
    )
    query_text_btn.click(
        query_text_chooser,
        inputs=[
            query_textbox1,
            query_textbox2,
            k_value,
            text_next_frame_duration,
            text_1st_scene_k_limit,
            clip_options,
            list_checkbox,
            db_dir,
        ],
        outputs=[result_gallery],
    )
    query_textbox1.change(
        upload_query_textbox_value,
        inputs=[query_textbox1],
        outputs=[
            query_textbox3,
            query_textbox5,
            query_textbox7,
            query_textbox9,
        ],
    )
    query_textbox2.change(
        upload_query_textbox_value,
        inputs=[query_textbox2],
        outputs=[
            query_textbox4,
            query_textbox6,
            query_textbox8,
            query_textbox10,
        ],
    )
    result_gallery.select(
        add_selected_image,
        inputs=[result_mode, result_gallery, selected_gallery, db_dir],
        outputs=[result_gallery, selected_gallery, video_preview],
    )
    query_image_result_gallery.select(
        add_selected_image,
        inputs=[
            result_image_mode,
            query_image_result_gallery,
            selected_gallery,
            db_dir,
        ],
        outputs=[query_image_result_gallery, selected_gallery, video_preview],
    )
    query_tag_result_gallery.select(
        add_selected_image,
        inputs=[
            result_tag_mode,
            query_tag_result_gallery,
            selected_gallery,
            db_dir,
        ],
        outputs=[query_tag_result_gallery, selected_gallery, video_preview],
    )
    selected_gallery.select(
        preview_or_remove_selected_image,
        inputs=[result_selected_image_mode, selected_gallery, db_dir],
        outputs=[selected_gallery, video_preview, keyframe_gallery],
    )
    clear_selected_gallery_btn.click(
        clear_gallery,
        inputs=None,
        outputs=selected_gallery,
    )
    query_image_clear_btn.click(
        clear_image_query_gallery,
        inputs=None,
        outputs=query_image_result_gallery,
    )
    result_mode.change(
        change_result_mode,
        inputs=result_mode,
        outputs=result_gallery,
    )
    result_image_mode.change(
        change_result_mode,
        inputs=result_image_mode,
        outputs=query_image_result_gallery,
    )
    result_tag_mode.change(
        change_result_mode,
        inputs=result_tag_mode,
        outputs=query_tag_result_gallery,
    )
    result_selected_image_mode.change(
        change_result_mode,
        inputs=result_selected_image_mode,
        outputs=selected_gallery,
    )
    export_selected_to_result_btn.click(
        export_result_to_csv,
        inputs=[selected_gallery, answer_textbox, db_dir],
        outputs=[csv_result_dataframe, csv_result_textbox],
    )
    export_to_textbox_btn.click(
        lambda inp: inp.to_csv(index=False),
        inputs=[csv_result_dataframe],
        outputs=[csv_result_textbox],
    )
    export_to_dataframe_btn.click(
        lambda inp: pd.read_csv(io.StringIO(inp), sep=","),
        inputs=[csv_result_textbox],
        outputs=[csv_result_dataframe],
    )
    get_current_video_keyframes_btn.click(
        _get_keyframes_from_current_video_fn,
        inputs=[video_preview, db_dir],
        outputs=[keyframe_gallery],
    )
    video_explorer.change(
        video_select_fn,
        inputs=[video_explorer, db_dir],
        outputs=[video_preview, keyframe_gallery],
    )
    keyframe_gallery.select(
        _on_keyframe_gallery_select,
        inputs=None,
        outputs=[selected_keyframe_caption_state],
    )
    show_keyframe_in_video_btn.click(
        _on_show_keyframe_in_video,
        inputs=[selected_keyframe_caption_state, db_dir],
        outputs=[video_preview],
    )
    capture_from_debug_info_btn.click(
        _capture_frame_from_debug_info_fn,
        inputs=[debug_info_textbox, capture_info_textbox, db_dir],
        outputs=[capture_info_textbox],
    )

    capture_from_debug_info_btn.click(
        _capture_json_from_debug_info_fn,
        inputs=[debug_info_textbox, answer_json_textbox, db_dir],
        outputs=[capture_info_json_textbox],
    )

    debug_info_textbox.change(
        _parse_debug_info_detail,
        inputs=[debug_info_textbox, db_dir],
        outputs=[debug_info_parsed_textbox],
    )
    capture_current_selected_keyframe_btn.click(
        _on_capture_this_keyframe_btn,
        inputs=[
            selected_keyframe_caption_state,
            capture_info_textbox,
            db_dir,
        ],
        outputs=[capture_info_textbox],
    )
    create_thumbnail_btn.click(
        lambda: [
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        ],
        inputs=None,
        outputs=[
            create_thumbnail_btn,
            create_thumbnail_confirm_btn,
            create_thumbnail_cancel_btn,
        ],
    )
    create_thumbnail_cancel_btn.click(
        lambda: [
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ],
        inputs=None,
        outputs=[
            create_thumbnail_btn,
            create_thumbnail_confirm_btn,
            create_thumbnail_cancel_btn,
        ],
    )
    create_thumbnail_confirm_btn.click(_on_create_thumbnail_confirm)
    export_all_keyframes_to_selected_btn.click(
        lambda x: x,
        inputs=[result_gallery],
        outputs=[selected_gallery],
    )
    select_all_checkbox.click(
        on_select_all_checkboxes,
        inputs=None,
        outputs=[list_checkbox],
    )
    deselect_all_checkbox.click(
        on_deselect_all_checkboxes,
        None,
        outputs=[list_checkbox],
    )
    select_all_checkbox_query_image.click(
        on_select_all_checkboxes,
        inputs=None,
        outputs=[list_checkbox_query_image],
    )
    deselect_all_checkbox_query_image.click(
        on_deselect_all_checkboxes,
        None,
        outputs=[list_checkbox_query_image],
    )
    database_selection.change(
        lambda x: (x, x),
        inputs=[database_selection],
        outputs=[database_version, query_image_database_selection],
    )
    query_image_database_selection.change(
        lambda x: (x, x),
        inputs=[query_image_database_selection],
        outputs=[database_version, database_selection],
    )
    database_version.change(
        on_database_version_change,
        inputs=[database_version],
        outputs=[db_dir],
    )

    btn_query_next_search.click(
        query_text_next_search,
        inputs=[
            query_txtbox_for_next_search,
            top_k_next_search,
            db_dir,
            result_gallery,
        ],
        outputs=[result_gallery],
    )

    recommended_tag_btn.click(
        # todo: use gemini to generate tags
        TagRetrieval.recommend_tag,
        inputs=[uploaded_textbox2],
        outputs=[recommended_tags],
    )
    compile_tag_btn.click(
        lambda x: " ".join(x),
        inputs=[select_tags],
        outputs=[compiled_tags],
    )
    query_tag_btn.click(
        query_tags,
        inputs=[compiled_tags, query_tag_k_value, database_version],
        # output is the gallary
        outputs=[query_tag_result_gallery],
    )
    login_btn.click(
        _on_login_btn_click,
        inputs=[username_textbox, password_textbox],
        outputs=[session_id_textbox],
    )
    capture_info_json_textbox.change(
        _on_capture_info_json_textbox_change,
        inputs=[capture_info_json_textbox],
        outputs=[output_json_data_textbox],
    )
    get_evaluation_id_btn.click(
        _on_get_evaluation_id_btn,
        inputs=[session_id_textbox],
        outputs=[evalution_id_textbox, evaluation_display_label],
    )
    submit_json_btn.click(
        _on_submit_btn,
        inputs=[capture_info_json_textbox, session_id_textbox, evalution_id_textbox],
        outputs=[submit_json_output_label],
    )

allowed_paths = [
    DEFAULT_KEYFRAMES_DIR,
    DEFAULT_KEYFRAMES_V2_DIR,
    DEFAULT_CLIP_FEATURE_DIR,
    DEFAULT_CLIP_FEATURE_V2_DIR,
    DEFAULT_MAP_KEYFRAMES_DIR,
    DEFAULT_MAP_KEYFRAMES_V2_DIR,
    os.path.realpath(DEFAULT_KEYFRAMES_DIR),
    os.path.realpath(DEFAULT_KEYFRAMES_V2_DIR),
    os.path.realpath(DEFAULT_CLIP_FEATURE_DIR),
    os.path.realpath(DEFAULT_CLIP_FEATURE_V2_DIR),
    os.path.realpath(DEFAULT_MAP_KEYFRAMES_DIR),
    os.path.realpath(DEFAULT_MAP_KEYFRAMES_V2_DIR),
]

# demo.launch(server_name="0.0.0.0",server_port=7777, share=True, allowed_paths=allowed_paths)
demo.launch(server_name="0.0.0.0", server_port=7777, allowed_paths=allowed_paths)
