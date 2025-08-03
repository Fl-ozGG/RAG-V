import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

nltk.download('wordnet')
nltk.download('stopwords')


# Interfaz común para todos los pasos
class TextProcessor:
    def process(self, text: str) -> str:
        raise NotImplementedError("Debe implementarse en las subclases")

class TextNormalizer(TextProcessor):
    def process(self, text: str) -> str:
        text = text.lower()
        text = ' '.join(text.split())  # elimina espacios redundantes
        return text

class TextCollector:
    def collect(self, texts: list[str]) -> list[str]:
        # Elimina duplicados y vacíos
        return list({t.strip() for t in texts if t.strip()})
    
class StopwordRemover(TextProcessor):
    def __init__(self, language='spanish'):
        self.stopwords = set(stopwords.words(language))

    def process(self, text: str) -> str:
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords]
        return ' '.join(tokens)
    
class Lemmatizer(TextProcessor):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def process(self, text: str) -> str:
        tokens = text.split()
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized)
    
class Tokenizer(TextProcessor):
    def process(self, text: str) -> list[str]:
        return text.split()
    
class SpellCorrector(TextProcessor):
    def __init__(self, language='es'):
        self.spell = SpellChecker(language=language)

    def process(self, text: str) -> str:
        tokens = text.split()
        corrected = [self.spell.correction(token) or token for token in tokens]
        return ' '.join(corrected)

class LanguageDetector:
    def detect_language(self, text: str) -> str:
        return detect(text)
    
class KeywordFilter(TextProcessor):
    def __init__(self, keywords: list[str]):
        self.keywords = set(keywords)

    def process(self, text: str) -> str:
        return text if any(k in text for k in self.keywords) else ""
    
class TextPipeline:
    def __init__(self, steps: list[TextProcessor]):
        self.steps = steps

    def process(self, text: str) -> str:
        for step in self.steps:
            text = step.process(text)
            if text == "":  # Si fue filtrado
                break
        return text    