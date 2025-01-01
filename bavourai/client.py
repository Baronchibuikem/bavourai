import os
from dotenv import load_dotenv
from bavourai.embeddings.embeddings import EmbeddingGenerator
from bavourai.databases.vector_stores import VectorDatabase
from langchain_core.documents import Document
from langchain_community.llms import Ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class BavourClient:
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
    ):
        os.environ['LANGCHAIN_API_KEY'] = api_key
        self.embedding_generator = EmbeddingGenerator(model=model)
        self.vector_db = VectorDatabase(
            embedding_model=self.embedding_generator.embedding_model,
        )
        self.llm_model = Ollama(model=model)
   
    def add_prompt(self, prompt):
        """
        Adds a prompt or a list of prompts to the database.

        Args:
            prompt (dict or list): A single prompt or a list of prompts with {"role": "", "content": ""}.
            user_id (str): The ID of the user associated with the prompts.
        """
        if isinstance(prompt, dict):
            # Single prompt
            prompts = [prompt]
        elif isinstance(prompt, list):
            # List of prompts
            prompts = prompt
        else:
            raise ValueError("Prompt must be a dictionary or a list of dictionaries with 'role' and 'content' keys.")

        documents = []
        # Generate embeddings and add to the database
        for index, prompt in enumerate(prompts):
            embedding = self.embedding_generator.generate_embedding(prompt["content"])
            doc = Document(
                page_content=prompt["content"],
                metadata={**prompt.get('metadata', {})},
                id=prompt["id"] if prompt["id"] else index + 1,
                embedding=embedding,

            )
            documents.append(doc)
        self.vector_db.add_data(documents=documents)
        
        return "Prompt(s) added successfully."

    def search(self, query: str, user_id: str, top_k: int = 2):
        results = self.vector_db.get(query=query, top_k=top_k, user_id=user_id)
        res = [{"text": r.page_content} for r in results]
        return self.rank_with_llama(query=query, results=res)

    def delete_document(self, document_id):
        self.vector_db.delete(id=document_id)
        return "Prompt removed successfully."
    

    def rank_with_llama(self, query: str, results: list):
        """
        Ranks the retrieved results based on their relevance to the query using Ollama embeddings.
        """
        # Generate embeddings for the query
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Ensure query embedding is a 2D array
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Generate a list of documents and their embeddings for comparison
        document_embeddings = []
        documents = []
        for r in results:
            doc_text = r["text"]
            documents.append(doc_text)

            # Get document embedding
            doc_embedding = self.embedding_generator.generate_embedding(doc_text)

            # Check if the embedding is valid
            if doc_embedding is not None and len(doc_embedding) > 0:
                # Ensure document embedding is a 2D array
                doc_embedding = np.array(doc_embedding).reshape(1, -1)
                document_embeddings.append(doc_embedding)
            else:
                print(f"Warning: Empty embedding for document: {doc_text}")

        if not document_embeddings:
            print("Error: Could find any related query result.")
            return []  # Return an empty list if no valid embeddings are found

        # Step 1: Stack document embeddings into a 2D array
        document_embeddings = np.vstack(document_embeddings)

        # Step 2: Compute cosine similarity between query and document embeddings
        cosine_similarities = cosine_similarity(query_embedding, document_embeddings)[0]

        # Step 3: Sort documents based on similarity score
        ranked_results = [r for _, r in sorted(zip(cosine_similarities, results), key=lambda x: x[0], reverse=True)]

        return ranked_results
    


