from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np


class TextVectorizer:
    def fit(self, texts: list[str]):
        raise NotImplementedError()

    def transform(self, texts: list[str]):
        raise NotImplementedError()

    def fit_transform(self, texts: list[str]):
        self.fit(texts)
        return self.transform(texts)
    
    

class BoWVectorizer(TextVectorizer):
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts).toarray()

class TFIDFVectorizer(TextVectorizer):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts).toarray()

class Word2VecVectorizer(TextVectorizer):
    def __init__(self, size=100, window=5, min_count=1):
        self.model = None
        self.size = size
        self.window = window
        self.min_count = min_count

    def fit(self, texts: list[str]):
        tokenized = [t.split() for t in texts]
        self.model = Word2Vec(sentences=tokenized, vector_size=self.size,
                              window=self.window, min_count=self.min_count)

    def transform(self, texts: list[str]):
        tokenized = [t.split() for t in texts]
        vectors = []
        for tokens in tokenized:
            word_vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
            if word_vectors:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                vectors.append(np.zeros(self.size))
        return np.array(vectors)

class TransformerVectorizer(TextVectorizer):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def transform(self, texts: list[str]):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
        return embeddings

class VectorizationPipeline:
    def __init__(self, vectorizer: TextVectorizer):
        self.vectorizer = vectorizer

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)
