from TranslatorInterface import SupportedFileType
from DocxTranslator import DocxTranslator
from TxtTranslator import TxtTranslator


class TranslatorFactory:
    def get_translator(self, fileType):
        fileType = SupportedFileType.from_extension(fileType)
        if fileType == SupportedFileType.DOCX:
            return DocxTranslator()
        elif fileType == SupportedFileType.TEXT:
            return TxtTranslator()
        else:
            print("translator not found")
