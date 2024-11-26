from abc import ABC, abstractmethod
from deep_translator import GoogleTranslator
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

#GPU HERE
import torch


class Translator(ABC):

    def __init__(self,
                 source_lang: str,
                 dest_lang: str):
        self.source_lang = source_lang
        self.dest_lang = dest_lang

    @abstractmethod
    def translate_text(self, input):
        pass


class FaceBookTranslatorProvider(Translator):

    def __init__(self,
                 source_lang: str,
                 dest_lang: str) -> None:
        super().__init__(source_lang, dest_lang)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.translator_tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-" + source_lang + "-" + dest_lang)
        self.translator_model = FSMTForConditionalGeneration.from_pretrained(
            "facebook/wmt19-" + source_lang + "-" + dest_lang).to(self.device)
        self.translator_model.eval()

    def translate_text(self, input):
        input_ids = self.translator_tokenizer.encode(input, return_tensors="pt").to(self.device)
        outputs = self.translator_model.generate(input_ids)
        decoded = self.translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded


class GoogleTranslationProvider(Translator):

    def __init__(self,
                 source_lang: str,
                 dest_lang: str) -> None:
        super().__init__(source_lang, dest_lang)

    def translate_text(self, input):
        translator = GoogleTranslator(source=self.source_lang, target=self.dest_lang)
        translated_text = translator.translate(input)
        return translated_text
