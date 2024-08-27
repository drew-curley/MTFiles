import docx
import fasttext
import gc
import os
import torch
from collections import Counter
from pathlib import Path
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from zipfile import BadZipFile
import nltk
import json
from TranslatorInterface import TranslatorInterface, SupportedFileType


nltk.download('punkt')
nltk.download('punkt_tab')

class DocxTranslator(TranslatorInterface):

    def __init__(self, pretrained_lang_model="./lid218e.bin", ):
        self.ext_in = SupportedFileType.DOCX.value
        self.ext_out = SupportedFileType.DOCX.value
        self.pretrained_lang_model = pretrained_lang_model

        with open("./constants/model_checkpoints.json", 'r') as json_file:
            self.checkpoints = json.load(json_file)

        with open("./constants/languages.json", 'r') as json_file:
            self.languages = json.load(json_file)

        self._prepare_language_detection_model()


    def _translate_paragraph(self, paragraph, translation_pipeline):
        """Translate the content of a paragraph and replace its text."""
        original_text = paragraph.text
        if original_text.strip():  # Only translate if the paragraph is not empty
            print(f"translting {original_text}")
            translated_text = translation_pipeline(original_text)[0]['translation_text']
            self._replace_text_in_runs(paragraph, translated_text)


    def _replace_text_in_runs(self, paragraph, translated_text):
        """Replace text in each run while preserving the original formatting."""
        original_text = "".join(run.text for run in paragraph.runs)
        # Ensure we correctly replace text while preserving formatting
        if len(original_text) == len(translated_text):
            current_char_index = 0
            for run in paragraph.runs:
                run_length = len(run.text)
                run.text = translated_text[current_char_index:current_char_index + run_length]
                current_char_index += run_length
        else:
            # If lengths don't match, replace text by matching run lengths
            current_char_index = 0
            for run in paragraph.runs:
                run_length = len(run.text)
                run.text = translated_text[current_char_index:current_char_index + run_length]
                current_char_index += run_length

            # Handle any leftover text by adding it as a new run
            if current_char_index < len(translated_text):
                remaining_text = translated_text[current_char_index:]
                paragraph.add_run(remaining_text)


    def _translate_table(self, table, translation_pipeline):
        """Translate all the cells in a table."""
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    self._translate_paragraph(paragraph, translation_pipeline)


    def _get_languages(self, file):
        file = file.resolve()
        fasttext_model = fasttext.load_model(self.pretrained_lang_model)
        
        try :
            document = docx.Document(file)
        except BadZipFile:
            print(f"BadZipFile Error on opening {file}")

        languageCounter = Counter()

        self._get_languages_in_paragraphs(document.paragraphs, fasttext_model, languageCounter)
        self._get_languages_in_tables(document.tables, fasttext_model, languageCounter)

        del fasttext_model
        gc.collect()
        torch.cuda.empty_cache()

        return languageCounter


    def _get_languages_in_paragraphs(self, paragraphs, model, counter):
        sentences = [sentence for para in paragraphs for sentence in sent_tokenize(para.text)]

        for sentence in sentences:
            predictions = model.predict(sentence, k=1)
            output_lang = predictions[0][0].replace('__label__', '')
            counter.update([output_lang])


    def _get_languages_in_tables(self, tables, model, counter):
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    self._get_languages_in_paragraphs(cell.paragraphs, model, counter)


    def translate(self, filePath, source_language, target_language, model_name):
        file = filePath.resolve()
        languages_in_file = self._get_languages(file)
        top_language_in_file = languages_in_file.most_common(1)[0][0]
        file_is_src_lang = (top_language_in_file == source_language)

        if not file_is_src_lang:
            print(f"Cannot translate. File is in {top_language_in_file} expected {source_language}")
            return

        if target_language not in self.languages:
            print(f"Cannot translate. {target_language} is not a supported target language.")
            return

        try:
            document = docx.Document(file)
        except BadZipFile:
            print(f"BadZipFile Error on opening {file}")
            return
        
        if model_name not in self.checkpoints:
            print(f"Cannot translate. {model_name} is not a supported model.")
            return 

        print(f"Loading model: {self.checkpoints[model_name]}")
        model, tokenizer = self._load_model(self.checkpoints[model_name])

        file_name = self.languages[target_language]
        output_dir_for_model = Path(f'./Translated/{model_name}')
        output_dir_for_model.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_for_model / f"{file.stem}_{file_name}.{self.ext_out}"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        translation_pipeline = pipeline('translation',
                                        model=model,
                                        tokenizer=tokenizer,
                                        src_lang=source_language,
                                        tgt_lang=target_language,
                                        max_length=400,
                                        device=device)

        # Iterate over paragraphs and tables to translate the content
        for paragraph in document.paragraphs:
            self._translate_paragraph(paragraph, translation_pipeline)

        for table in document.tables:
            self._translate_table(table, translation_pipeline)

        # Save the translated document
        document.save(output_path)

        # Frees loaded model and tokenizer
        self._unload_model(model, tokenizer)   

        # Frees the pipeline
        del translation_pipeline
        gc.collect()
        torch.cuda.empty_cache()   

        return output_path   

# Example usage:
# translator = DocxTranslator()
# file = Path("./Input/Spiritual Terms Eval with defs and refs for IT 1.docx")
# translator.translate(file, "eng_Latn", "spa_Latn", "NLLB-distilled")
