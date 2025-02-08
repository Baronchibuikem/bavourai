from langchain_ollama import OllamaEmbeddings


class EmbeddingGenerator:
    def __init__(self, model: str):
        self.model = model
        self.embedding_model = OllamaEmbeddings(model=model)
        

    def generate_embedding(self, prompt: dict):
        return self.embedding_model.embed_query(prompt)

