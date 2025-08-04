import logging
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Configuración básica del logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TextVectorizer:
    def fit(self, texts: list[str]):
        logging.debug("Base fit() called - should be implemented in subclass")
        raise NotImplementedError()

    def transform(self, texts: list[str]):
        logging.debug("Base transform() called - should be implemented in subclass")
        raise NotImplementedError()

    def fit_transform(self, texts: list[str]):
        logging.debug("Calling fit_transform()")
        self.fit(texts)
        return self.transform(texts)
    
class BoWVectorizer(TextVectorizer):
    def __init__(self):
        logging.debug("Initializing BoWVectorizer")
        self.vectorizer = CountVectorizer()

    def fit(self, texts: list[str]):
        logging.debug(f"Fitting BoWVectorizer on {len(texts)} texts")
        self.vectorizer.fit(texts)
        logging.debug(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def transform(self, texts: list[str]):
        logging.debug(f"Transforming {len(texts)} texts using BoWVectorizer")
        transformed = self.vectorizer.transform(texts).toarray()
        logging.debug(f"Transformed shape: {transformed.shape}")
        return transformed

    def fit_transform(self, texts: list[str]):
        logging.debug(f"Fit and transform {len(texts)} texts using BoWVectorizer")
        transformed = self.vectorizer.fit_transform(texts).toarray()
        logging.debug(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logging.debug(f"Transformed shape: {transformed.shape}")
        return transformed

class TFIDFVectorizer(TextVectorizer):
    def __init__(self):
        logging.debug("Initializing TFIDFVectorizer")
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts: list[str]):
        logging.debug(f"Fitting TFIDFVectorizer on {len(texts)} texts")
        self.vectorizer.fit(texts)
        logging.debug(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def transform(self, texts: list[str]):
        logging.debug(f"Transforming {len(texts)} texts using TFIDFVectorizer")
        transformed = self.vectorizer.transform(texts).toarray()
        logging.debug(f"Transformed shape: {transformed.shape}")
        return transformed

    def fit_transform(self, texts: list[str]):
        logging.debug(f"Fit and transform {len(texts)} texts using TFIDFVectorizer")
        transformed = self.vectorizer.fit_transform(texts).toarray()
        logging.debug(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logging.debug(f"Transformed shape: {transformed.shape}")
        return transformed

class Word2VecVectorizer(TextVectorizer):
    def __init__(self, size=100, window=5, min_count=1):
        logging.debug(f"Initializing Word2VecVectorizer with size={size}, window={window}, min_count={min_count}")
        self.model = None
        self.size = size
        self.window = window
        self.min_count = min_count

    def fit(self, texts: list[str]):
        logging.debug(f"Training Word2Vec model on {len(texts)} texts")
        tokenized = [t.split() for t in texts]
        self.model = Word2Vec(sentences=tokenized, vector_size=self.size,
                              window=self.window, min_count=self.min_count)
        logging.debug(f"Word2Vec vocabulary size: {len(self.model.wv)}")

    def transform(self, texts: list[str]):
        logging.debug(f"Transforming {len(texts)} texts using Word2VecVectorizer")
        tokenized = [t.split() for t in texts]
        vectors = []
        for i, tokens in enumerate(tokenized):
            word_vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
            if word_vectors:
                mean_vector = np.mean(word_vectors, axis=0)
                logging.debug(f"Text {i}: computed mean vector")
                vectors.append(mean_vector)
            else:
                logging.warning(f"Text {i}: no known words found, using zero vector")
                vectors.append(np.zeros(self.size))
        transformed = np.array(vectors)
        logging.debug(f"Transformed shape: {transformed.shape}")
        return transformed

class TransformerVectorizer(TextVectorizer):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        logging.debug(f"Loading Transformer model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def transform(self, texts: list[str]):
        logging.debug(f"Transforming {len(texts)} texts using TransformerVectorizer")
        embeddings = []
        for i, text in enumerate(texts):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_embedding)
            logging.debug(f"Text {i}: obtained CLS embedding of shape {cls_embedding.shape}")
        return embeddings

class VectorizationPipeline:
    def __init__(self, vectorizer: TextVectorizer):
        logging.debug(f"Initializing VectorizationPipeline with {vectorizer.__class__.__name__}")
        self.vectorizer = vectorizer

    def fit(self, texts: list[str]):
        logging.debug(f"Pipeline fit called with {len(texts)} texts")
        self.vectorizer.fit(texts)
        logging.debug("Pipeline fit completed")

    def transform(self, texts: list[str]):
        logging.debug(f"Pipeline transform called with {len(texts)} texts")
        transformed = self.vectorizer.transform(texts)
        logging.debug("Pipeline transform completed")
        return transformed

    def fit_transform(self, texts: list[str]):
        logging.debug(f"Pipeline fit_transform called with {len(texts)} texts")
        transformed = self.vectorizer.fit_transform(texts)
        logging.debug("Pipeline fit_transform completed")
        return transformed
