from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, text):
        return self.model.encode([text])[0]

    def predict(self, text1, text2):
        text_vec1 = self.encode(text1)
        text_vec2 = self.encode(text2)

        similarity_score = 1 - distance.cosine(text_vec1, text_vec2)

        return similarity_score
