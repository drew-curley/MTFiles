import docx
import fasttext
import gc
import os
import torch
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, TRANSFORMERS_CACHE
from nltk.tokenize import sent_tokenize
from zipfile import BadZipFile
import nltk
import json
from TranslatorInterface import TranslatorInterface


nltk.download('punkt')
nltk.download('punkt_tab')

class TxtTranslator(TranslatorInterface):

    def __init__(self, pretrained_lang_model="./lid218e.bin"):
        self.ext_in = "txt"
        self.ext_out = "txt"
        self.pretrained_lang_model = pretrained_lang_model

        with open("./constants/model_checkpoints.json", 'r') as json_file:
            self.checkpoints = json.load(json_file)
            # "NLLB": "facebook/nllb-200-3.3B",
            # "MADLAD": "google/madlad400-3b-mt",
            # "Llama-3.1-405B": "meta-llama/Meta-Llama-3.1-405B"

        with open("./constants/languages.json", 'r') as json_file:
            self.languages = json.load(json_file)
                # "Ethiopian": "amh_Ethi",
                # "Arabic": "arb_Arab",
                # "Assamese": "asm_Beng",
                # "Bangal": "ben_Beng",
                # "BPortugese": "por_Latn",
                # "Burmese": "mya_Mymr",
                # "Cebuano": "ceb_Latn",
                # "Chinese": "zsm_Latn",
                # "French": "fra_Latn",
                # "Gujarati": "guj_Gujr",
                # "Hausa": "hau_Latn",
                # "Hindi": "hin_Deva",
                # "Illocano": "ilo_Latn",
                # "Indonesian": "ind_Latn",
                # "Kannada": "kan_Knda",
                # "Khmer": "khm_Khmr",
                # "Laotian": "lao_Laoo",
                # "LASpanish": "spa_Latn",
                # "Malayalam": "mal_Mlym",
                # "Nepali": "npi_Deva",
                # "Oriya": "ory_Orya",
                # "PlatMalagasy": "plt_Latn",
                # "EPunjabi": "pan_Guru",
                # "Russian": "rus_Cyrl",
                # "Swahili": "swh_Latn",
                # "Tagalog": "tgl_Latn",
                # "Tamil": "tam_Taml",
                # "Telugu": "tel_Telu",
                # "Thai": "tha_Thai",
                # "TokPisin": "tpi_Latn",
                # "Urdu": "urd_Arab",

        self._prepare_language_detection_model()


    def _prepare_language_detection_model(self):
        if not os.path.isfile(self.pretrained_lang_model):
            # Download the model if it doesn't exist
            os.system(f"wget https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin")


    def load_model(self, model_name):
        model_dir = f"{TRANSFORMERS_CACHE}/{model_name}"

        if not os.path.exists(model_dir):
            print(f"{model_name} not found")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        else:
            print(f"{model_name} found")
            tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")
            model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}")

        return model, tokenizer


    def unload_model(self, model, tokenizer):
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


    def get_languages(self, file):
        print("TODO")
        # TODO: finish

    def translate(self, text, source_language, target_language):
        # TODO: account for this being a .txt file
        """Translate a given text from src_lang to tgt_lang using the specified model."""
        # TODO: allow the translate method to specify what model to use. Possibly make an Enum class from 
        #  the languages.json file. 
        model, tokenizer = self.load_model(self.checkpoints["NLLB-distilled"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        translation_pipeline = pipeline('translation',
                                        model=model,
                                        tokenizer=tokenizer,
                                        src_lang=source_language,
                                        tgt_lang=target_language,
                                        max_length=400,
                                        device=device)

        translated_text = translation_pipeline(text)[0]['translation_text']

        self.unload_model(model, tokenizer)
        return translated_text


# Example usage:
# translator = TxtTranslator()
# translated_text = translator.translate("Hello, world!", "eng_Latn", "fra_Latn")
# print(translated_text)
