from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_chroma import Chroma
from loguru import logger
from langchain_core.documents import Document
from typing import List

import re
import logging

logger = logging.getLogger(__name__)


def extract_rating_condition(query: str):
    """
    Extracts rating condition (e.g., "rating above 4") dynamically from the query.
    Returns a comparison operator and the numeric rating threshold.
    """
    # Define natural language mappings to operators
    operator_mappings = {
        "above": ">",
        "greater than": ">",
        "bigger than": ">",
        "below": "<",
        "less than": "<",
        "smaller than": "<",
        "equal to": "==",
        "equal": "==",
    }

    # Check for explicit operators (>, <, >=, <=, =)
    match = re.search(r"rating\s*(>=|>|<=|<|=)?\s*(\d+)", query, re.IGNORECASE)
    if match:
        operator = match.group(1) or "=="  # Default to equality if no operator is found
        rating = int(match.group(2))
        return operator, rating

    # Check for natural language phrases
    for phrase, operator in operator_mappings.items():
        match = re.search(rf"{phrase}\s*(\d+)", query, re.IGNORECASE)
        print({'match': match})
        if match:
            rating = int(match.group(1))
            print({"rating": rating, "operator": operator})
            return operator, rating

    return None, None  # No rating condition found




class VectorDatabase:
    def __init__(self, embedding_model: OllamaEmbeddings ):
        self.db = Chroma(
            collection_name="bavour_ai_data",
            persist_directory="./bavour_db",
            embedding_function=embedding_model,
        )

    def add_data(self, documents: List[Document]):
        self.uuids = [doc.id for doc in documents]
        result = self.db.add_documents(documents=documents, ids=self.uuids)
        logger.debug(f"Added {len(result)} documents to the database.")

    def update_data(
        self,
        updated_document: Document,
    ):
        self.db.update_document(
            document_id=self.uuids[updated_document["id"]], document=updated_document
        )
        logger.debug(f"Updated document with id {updated_document['id']}")

    def get(self, query: str, top_k: int):
        # Retrieve results with similarity scores
        results_with_scores = self.db.similarity_search_with_score(
            query=query,
            k=top_k,
        )
        print({'results_with_scores': results_with_scores})

        # Extract rating condition from query
        operator, rating_threshold = extract_rating_condition(query)

        # Apply dynamic filtering based on extracted rating condition
        filtered_results = []
        for doc, score in results_with_scores:
            doc_rating = int(doc.metadata.get("rating", 0))  # Convert rating to int

            if operator and rating_threshold is not None:
                if (
                    (operator == ">" and doc_rating > rating_threshold) or
                    (operator == ">=" and doc_rating >= rating_threshold) or
                    (operator == "<" and doc_rating < rating_threshold) or
                    (operator == "<=" and doc_rating <= rating_threshold) or
                    (operator == "=" and doc_rating == rating_threshold) or
                    (operator == "==" and doc_rating == rating_threshold)
                ):
                    filtered_results.append((doc, score))
            else:
                # If no rating filter found in query, return all results
                filtered_results.append((doc, score))
        return [doc for doc, score in filtered_results]


    def delete(self, id):
        self.db.delete(ids=id)
        return "Prompt(s) removed successfully."

    def get_all_documents(self):
        """
        Retrieves all documents stored in the Chroma vector database.
        """
        # Retrieve documents and metadata
        documents = (
            self.db.get()
        )  # Depending on Chroma's API, use the method to retrieve documents
        return documents

