import os
from typing import Literal, Union
from dotenv import load_dotenv
from bavourai.embeddings.embeddings import EmbeddingGenerator
from bavourai.databases.vector_stores import VectorDatabase
from langchain_core.documents import Document
from langchain_community.llms import Ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class BavourClient:
    def __init__(self, model: Literal['llama3.2']):
        self.embedding_generator = EmbeddingGenerator(model=model)
        self.vector_db = VectorDatabase(
            embedding_model=self.embedding_generator.embedding_model,
        )
        self.llm_model = Ollama(model=model)
   
    def add_data(self, prompt: Union[list, dict], batch_size=100):
        """
        Adds a prompt or a list of prompts to the database in batches.

        Args:
            prompt (dict or list[dict]): A single prompt or a list of prompts with {"role": "", "content": ""}.
            user_id (str): The ID of the user associated with the prompts.
            batch_size (int): The size of each batch to process at once.
        """
        if isinstance(prompt, dict):
            # Single prompt
            prompts = [prompt]
        elif isinstance(prompt, list):
            # List of prompts
            prompts = prompt
        else:
            raise ValueError("Prompt must be a dictionary or a list of dictionaries with 'role' and 'content' keys.")

        # Process prompts in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch = prompts[batch_start:batch_end]

            documents = []
            # Generate embeddings and add to the database for each prompt in the batch
            for index, prompt in enumerate(batch):
                embedding = self.embedding_generator.generate_embedding(prompt["content"])
                doc = Document(
                    page_content=prompt["content"],
                    metadata={**prompt.get('metadata', {})},
                    id=batch_start + index + 1,  # Ensure unique IDs across batches
                    embedding=embedding,
                )

                documents.append(doc)
            
            # Add the batch to the vector DB
            self.vector_db.add_data(documents=documents)
            print(f"Processed batch {batch_start // batch_size + 1} of {len(prompts) // batch_size + 1}")


    def search(self, query: str, total_expected_result: int):
        results = self.vector_db.get(query=query, top_k=total_expected_result)  # Use self.get instead of self.vector_db.get
        if not results:
            return []

        res = [{"text": r.page_content} for r in results]
        return self.rank_with_llama(query=query, results=res)

    def delete_document(self, document_id):
        self.vector_db.delete(id=document_id)
        return "Prompt removed successfully."
    

    def rank_with_llama(self, query: str, results: list):
        """
        Ranks the retrieved results based on their relevance to the query using Ollama embeddings.
        """
        query_embedding = self.embedding_generator.generate_embedding(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)  # Ensure query embedding is a 2D array

        document_embeddings = []
        documents = []
        for r in results:
            doc_text = r["text"]
            documents.append(doc_text)

            doc_embedding = self.embedding_generator.generate_embedding(doc_text)
            if doc_embedding and len(doc_embedding) > 0:
                document_embeddings.append(np.array(doc_embedding).reshape(1, -1))
            else:
                print(f"Warning: Empty embedding for document: {doc_text}")

        if not document_embeddings:
            print("Error: Could not find any related query result.")
            return []

        document_embeddings = np.vstack(document_embeddings)  # Stack embeddings into 2D array
        cosine_similarities = cosine_similarity(query_embedding, document_embeddings)[0]

        ranked_results = [r for _, r in sorted(zip(cosine_similarities, results), key=lambda x: x[0], reverse=True)]
        return ranked_results