from abc import ABC, abstractmethod
from typing import TextIO

class TranslatorInterface(ABC):
    @abstractmethod
    def translate(self, text: TextIO, source_language: str, target_language: str) -> str:
        """Translate the given text from the source language to the target language."""
        pass