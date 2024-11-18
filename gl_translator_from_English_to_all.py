import os
import shutil
from pathlib import Path
from tqdm import tqdm
import docx
from nltk.tokenize import sent_tokenize
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from zipfile import BadZipFile

# Install dependencies (comment out in production)
# !pip install -U pip transformers sentencepiece python-docx nltk

# Setup and initialization
nltk.download('punkt')

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model checkpoints
nllb_checkpoint = "facebook/nllb-200-3.3B"
translation_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_checkpoint).to(device)
translation_tokenizer = AutoTokenizer.from_pretrained(nllb_checkpoint)
translation_pipeline = pipeline("translation", model=translation_model, tokenizer=translation_tokenizer, device=0 if device.type == "cuda" else -1)

# Directories
input_folder = Path("/home/drew/Documents/GitHub/MTFiles/Input")
output_folder = Path("/home/drew/Documents/GitHub/MTFiles/Translated")
ext_in, ext_out = "docx", "docx"

# List of target languages
target_languages = {
    "amh_Ethi": "Ethiopian",
    "arb_Arab": "Arabic",
    "asm_Beng": "Assamese",
    "ben_Beng": "Bangal",
    "por_Latn": "BPortugese",
    "mya_Mymr": "Burmese",
    "ceb_Latn": "Cebuano",
    "zsm_Latn": "Chinese",
    "fra_Latn": "French",
    "guj_Gujr": "Gujarati",
    "hau_Latn": "Hausa",
    "hin_Deva": "Hindi",
    "ilo_Latn": "Illocano",
    "ind_Latn": "Indonesian",
    "kan_Knda": "Kannada",
    "khm_Khmr": "Khmer",
    "lao_Laoo": "Laotian",
    "spa_Latn": "LASpanish",
    "mal_Mlym": "Malayalam",
    "npi_Deva": "Nepali",
    "ory_Orya": "Oriya",
    "plt_Latn": "PlatMalagasy",
    "pan_Guru": "EPunjabi",
    "rus_Cyrl": "Russian",
    "swh_Latn": "Swahili",
    "tgl_Latn": "Tagalog",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "tha_Thai": "Thai",
    "tpi_Latn": "TokPisin",
    "urd_Arab": "Urdu",
    "vie_Latn": "Vietnamese"
}

def translate_docx(file: Path, target_lang: str):
    """Translate DOCX content to the target language and save with the language name in the filename."""
    # Get the target language name
    target_lang_name = target_languages.get(target_lang, target_lang).replace(" ", "_")
    output_file = output_folder / f"{file.stem}_{target_lang_name}.{ext_out}"
    
    document = docx.Document(file)
    for para in document.paragraphs:
        sentences = sent_tokenize(para.text)
        translations = [
            translation_pipeline(
                sentence, 
                src_lang="eng_Latn", 
                tgt_lang=target_lang
            )[0]['translation_text'] 
            for sentence in sentences
        ]
        para.text = " ".join(translations)
    
    document.save(output_file)
    print(f"Saved translated file: {output_file}")

def manual_target_language_selection():
    """Prompt the user to manually select one or more target languages."""
    print("Available target languages:")
    for i, (lang_code, lang_name) in enumerate(target_languages.items(), 1):
        print(f"{i}: {lang_name} ({lang_code})")

    print("\nEnter numbers separated by commas (e.g., 1,3,5) or 'all' to translate into all languages.")
    choice = input("Select target languages: ").strip()

    if choice.lower() == "all":
        return list(target_languages.keys())  # All target languages
    else:
        selected_indices = [int(num.strip()) for num in choice.split(",") if num.strip().isdigit()]
        valid_indices = [index for index in selected_indices if 1 <= index <= len(target_languages)]

        if valid_indices:
            return [list(target_languages.keys())[index - 1] for index in valid_indices]
        else:
            print("Invalid selection. No languages selected.")
            return []

def process_files(files):
    """Process and translate a list of files."""
    for i, file in enumerate(files, 1):
        print(f"\nProcessing {file.name} ({i}/{len(files)})")
        target_languages_list = manual_target_language_selection()

        if target_languages_list:
            for target_language in target_languages_list:
                print(f"Translating {file.name} to {target_languages[target_language]}.")
                translate_docx(file, target_lang=target_language)
                print(f"Translation complete: {file.name} -> {target_languages[target_language]}")
        else:
            print(f"Skipping {file.name}.")


# Execution
if __name__ == "__main__":
    input_files = list(input_folder.rglob(f"*.{ext_in}"))
    print(f"Found {len(input_files)} {ext_in} files in {input_folder.resolve()}")
    process_files(input_files)
