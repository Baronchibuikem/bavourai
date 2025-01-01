from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer



class EmbeddingGenerator:
    def __init__(self, model = None):
        self.model = model
        # if model is None:
        #     self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # else:
        self.embedding_model = OllamaEmbeddings(model=model)
        

    def generate_embedding(self, prompt: dict):
        return self.embedding_model.embed_query(prompt)

    def encode(self, text):
        # For sentence-transformers/all-MiniLM-L6-v2
        return self.embedding_model.encode(text, convert_to_tensor=True)
