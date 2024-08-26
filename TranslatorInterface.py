from abc import ABC, abstractmethod
from pathlib import Path
import gc
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, TRANSFORMERS_CACHE

class TranslatorInterface(ABC):
    @abstractmethod
    def translate(self, filePath: Path, source_language: str, target_language: str, model_name: str) -> Path | None:
        """Translate the given text from the source language to the target language. Returns path to translated file
            or None if file translation was unsuccessful. 
        """
        pass

    def _prepare_language_detection_model(self):
        if not os.path.isfile(self.pretrained_lang_model):
            # Download the model if it doesn't exist
            os.system(f"wget https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin")

    def _load_model(self, model_name):
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


    def _unload_model(self, model, tokenizer):
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()