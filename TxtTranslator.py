import docx
import fasttext
import gc
import os
import torch
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, TRANSFORMERS_CACHE
from nltk.tokenize import sent_tokenize, word_tokenize
from zipfile import BadZipFile
import nltk
import json
from TranslatorInterface import TranslatorInterface

nltk.download('punkt')
nltk.download('punkt_tab')

class TxtTranslator(TranslatorInterface):

    def __init__(self, pretrained_lang_model="./lid218e.bin"):
        self.chunk_word_count=200
        self.ext_in = "txt"
        self.ext_out = "txt"
        self.pretrained_lang_model = pretrained_lang_model

        with open("./constants/model_checkpoints.json", 'r') as json_file:
            self.checkpoints = json.load(json_file)

        with open("./constants/languages.json", 'r') as json_file:
            self.languages = json.load(json_file)

        self._prepare_language_detection_model()

    def _get_languages(self, file):
        print("TODO")
        return True
        # TODO: finish

    def translate(self, filePath, source_language, target_language, model_name):

        # TODO: add src lang check here.

        if target_language not in self.languages:
            print(f"Cannot translate. {target_language} is not a supported target language.")
            return
        
        if model_name not in self.checkpoints:
            print(f"Cannot translate. {model_name} is not a supported model.")
            return 

        model, tokenizer = self._load_model(self.checkpoints[model_name])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        translation_pipeline = pipeline('translation',
                                        model=model,
                                        tokenizer=tokenizer,
                                        src_lang=source_language,
                                        tgt_lang=target_language,
                                        max_length=400,
                                        device=device)

        translated_chunks = []

        with open(filePath, 'r', encoding='utf-8') as file:
            text = file.read()
            words = word_tokenize(text)
            
            current_chunk = []
            current_word_count = 0

            for word in words:
                current_chunk.append(word)
                current_word_count += 1

                if current_word_count >= self.chunk_word_count:
                    chunk_text = ' '.join(current_chunk)
                    sentences = sent_tokenize(chunk_text)
                    for sentence in sentences:
                        translated_sentence = translation_pipeline(sentence)[0]['translation_text']
                        translated_chunks.append(translated_sentence)
                    current_chunk = []
                    current_word_count = 0
            
            # Handle any remaining words in the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                sentences = sent_tokenize(chunk_text)
                for sentence in sentences:
                    translated_sentence = translation_pipeline(sentence)[0]['translation_text']
                    translated_chunks.append(translated_sentence)

        self._unload_model(model, tokenizer)

        # Frees the pipeline
        del translation_pipeline
        gc.collect()
        torch.cuda.empty_cache()   

        output_file_path = filePath.with_name(f"{filePath.stem}_translated.{self.ext_out}")
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            for translated_chunk in translated_chunks:
                out_f.write(translated_chunk + "\n")

        print(f"Translation saved to {output_file_path}")
        return output_file_path



# Example usage:
# translator = TxtTranslator()
# file_path = Path("/mnt/c/Users/hilld/Documents/Github/MTFiles/Input/test.txt")
# translated_text = translator.translate(file_path, "eng_Latn", "fra_Latn", "NLLB-distilled")
