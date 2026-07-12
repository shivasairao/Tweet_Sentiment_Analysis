import numpy as np
from gensim.models import Word2Vec

class MeanEmbeddingVectorizer:
    def __init__(self, vector_size=200, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.w2v_model = None

    def fit(self, tokenized_docs):
        """ Trains word representations across context window spans. """
        self.w2v_model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=15
        )
        return self

    def transform(self, tokenized_docs):
        """ Maps token lists into vector aggregates. """
        if self.w2v_model is None:
            raise ValueError("Word2Vec vector space must be fitted prior to serialization pipelines.")
        
        features = []
        for tokens in tokenized_docs:
            valid_vectors = [self.w2v_model.wv[word] for word in tokens if word in self.w2v_model.wv]
            if len(valid_vectors) > 0:
                features.append(np.mean(valid_vectors, axis=0))
            else:
                features.append(np.zeros(self.vector_size))
        return np.array(features)

    def save(self, filepath):
        self.w2v_model.save(filepath)

    @classmethod
    def load(cls, filepath):
        instance = cls()
        instance.w2v_model = Word2Vec.load(filepath)
        instance.vector_size = instance.w2v_model.vector_size
        return instance