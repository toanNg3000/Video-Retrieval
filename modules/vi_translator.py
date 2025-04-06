from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from modules.settings import DEVICE


class Translator:
    def __init__(self, device=None):
        self.model_name = "vinai/vinai-translate-vi2en-v2"
        self.src_lang = "vi_VN"
        self.device = device if device else DEVICE
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.src_lang,
        )

    def __call__(self, vi_texts: str, device=None) -> str:
        return self.translate_vi2en(vi_texts, device)

    def translate_vi2en(self, vi_texts: str, device=None) -> str:
        device = device if device else self.device

        input_ids = self.tokenizer(
            vi_texts,
            padding=True,
            return_tensors="pt",
        ).to(device)

        output_ids = self.model.generate(
            **input_ids,
            decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        en_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return en_texts[0]


# The input may consist of multiple text sequences, with the number of text sequences in the input ranging from 1 up to 8, 16, 32, or even higher, depending on the GPU memory.
# vi_texts = ["Cô cho biết: trước giờ tôi không đến phòng tập công cộng, mà tập cùng giáo viên Yoga riêng hoặc tự tập ở nhà.",
#             "Khi tập thể dục trong không gian riêng tư, tôi thoải mái dễ chịu hơn.",
#             "cô cho biết trước giờ tôi không đến phòng tập công cộng mà tập cùng giáo viên yoga riêng hoặc tự tập ở nhà khi tập thể dục trong không gian riêng tư tôi thoải mái dễ chịu hơn"]
# print(translate_vi2en(vi_texts))
