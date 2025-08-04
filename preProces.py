import logging
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

nltk.download('wordnet')
nltk.download('stopwords')

# Configuración básica del logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Interfaz común para todos los pasos
class TextProcessor:
    def process(self, text: str) -> str:
        logging.debug("Base TextProcessor process() called - debe implementarse en subclase")
        raise NotImplementedError("Debe implementarse en las subclases")

class TextNormalizer(TextProcessor):
    def process(self, text: str) -> str:
        logging.debug(f"Normalizing text: {text[:30]}...")
        text = text.lower()
        text = ' '.join(text.split())  # elimina espacios redundantes
        logging.debug(f"Normalized text: {text[:30]}...")
        return text

class TextCollector:
    def collect(self, texts: list[str]) -> list[str]:
        logging.debug(f"Collecting texts, original count: {len(texts)}")
        collected = list({t.strip() for t in texts if t.strip()})
        logging.debug(f"Collected texts count (sin duplicados ni vacíos): {len(collected)}")
        return collected
    
class StopwordRemover(TextProcessor):
    def __init__(self, language='spanish'):
        logging.debug(f"Loading stopwords for language: {language}")
        self.stopwords = set(stopwords.words(language))

    def process(self, text: str) -> str:
        tokens = text.split()
        filtered_tokens = [t for t in tokens if t not in self.stopwords]
        logging.debug(f"Removed stopwords: {len(tokens) - len(filtered_tokens)} words removed")
        return ' '.join(filtered_tokens)
    
class Lemmatizer(TextProcessor):
    def __init__(self):
        logging.debug("Initializing Lemmatizer")
        self.lemmatizer = WordNetLemmatizer()

    def process(self, text: str) -> str:
        tokens = text.split()
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        logging.debug(f"Lemmatized text sample: {' '.join(lemmatized[:5])}...")
        return ' '.join(lemmatized)
    
class Tokenizer(TextProcessor):
    def process(self, text: str) -> list[str]:
        tokens = text.split()
        logging.debug(f"Tokenized into {len(tokens)} tokens")
        return tokens
    
class NoiseRemover(TextProcessor):
    def process(self, text: str) -> str:
        import re
        logging.debug(f"Removing noise from text: {text[:30]}...")
        text = re.sub(r'\d+', '', text)  # elimina números
        text = re.sub(r'[^\w\s]', '', text)  # elimina puntuación
        logging.debug(f"Noise removed text sample: {text[:30]}...")
        return text

class SpellCorrector(TextProcessor):
    def __init__(self, language='es'):
        logging.debug(f"Initializing SpellCorrector with language: {language}")
        self.spell = SpellChecker(language=language)

    def process(self, text: str) -> str:
        tokens = text.split()
        corrected = []
        corrections_count = 0
        for token in tokens:
            correction = self.spell.correction(token) or token
            if correction != token:
                corrections_count += 1
            corrected.append(correction)
        logging.debug(f"SpellCorrector: corrected {corrections_count} tokens")
        return ' '.join(corrected)

class LanguageDetector:
    def detect_language(self, text: str) -> str:
        language = detect(text)
        logging.debug(f"Detected language: {language} for text sample: {text[:30]}...")
        return language
    
class KeywordFilter(TextProcessor):
    def __init__(self, keywords: list[str]):
        self.keywords = set(keywords)
        logging.debug(f"KeywordFilter initialized with keywords: {self.keywords}")

    def process(self, text: str) -> str:
        match = any(k in text for k in self.keywords)
        if match:
            logging.debug(f"Text passed keyword filter")
            return text
        else:
            logging.debug(f"Text filtered out by keyword filter")
            return ""
    
class TextPipeline:
    def __init__(self, steps: list[TextProcessor]):
        self.steps = steps
        logging.debug(f"TextPipeline initialized with steps: {[step.__class__.__name__ for step in steps]}")

    def process(self, text: str) -> str:
        logging.debug(f"Starting pipeline process for text sample: {text[:30]}...")
        for step in self.steps:
            text = step.process(text)
            logging.debug(f"After {step.__class__.__name__}, text sample: {text[:30]}...")
            if text == "":  # Si fue filtrado
                logging.debug(f"Text filtered out, stopping pipeline")
                break
        return text
