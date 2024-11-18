from abc import ABC, abstractmethod
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
import torch

class EmbeddingService(ABC):
    @abstractmethod
    def get_embeddings(self, text):
        pass


class FastTextEmbeddingService(EmbeddingService):
    def __init__(self, model_path, binary=False):
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=binary)

    def get_embeddings(self, text):
        if isinstance(text, list):
            return torch.stack([torch.tensor(self.model[word.lower()]) for word in text if word in self.model], dim=0)
        if text in self.model:
            return torch.tensor(self.model[text.lower()])
        print(text + ' is not in model')


class GloveEmbeddingService(EmbeddingService):
    def __init__(self, model_path):
        self.model = self.load_glove_model(model_path)

    def load_glove_model(self, glove_file):
        model = {}
        with open(glove_file, 'r', encoding='utf8') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = torch.tensor([float(val) for val in split_line[1:]])
                model[word] = embedding
        return model

    def get_embeddings(self, text):
        if isinstance(text, list):
            return torch.tensor([self.model.get(word.lower()) for word in text if word.lower() in self.model])
        return self.model.get(text.lower()) if text.lower() in self.model else None


class SentenceTransformerEmbeddingService(EmbeddingService):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, text):
        return torch.tensor(self.model.encode(text))

class BertEmbeddingService(EmbeddingService):
    def __init__(self, model_name='google/bert_uncased_L-24_H-1024_A-16'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Extrahiert die [CLS] Token Repräsentationen
        cls_embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return torch.tensor(cls_embeddings)
