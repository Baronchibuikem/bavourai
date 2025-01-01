from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from typing import List
from langchain_chroma import Chroma
from loguru import logger
from langchain_core.documents import Document
import spacy

# Load a spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_entities(query: str):
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    return entities


class VectorDatabase:
    def __init__(self, embedding_model: Embeddings = None):
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

    def get(self, query: str, top_k: int = 4, user_id=None):
        keyword = extract_entities(query)
        keyword_list = str(max(keyword))
        # Retrieve initial results without filtering on category substring
        results = self.db.similarity_search(
            query=query,
            k=top_k,
            # filter={"user_id": user_id}
        )
        print({"results": results})
        # Filter results manually if a category substring filter is provided
        if keyword_list:
            filtered_results = [
                result
                for result in results
                if keyword_list.lower() in result.metadata.get("category", "").lower()
                or keyword_list.lower() in result.page_content.lower()
            ]
        else:
            filtered_results = results

        # Return the top_k results after filtering
        return filtered_results[:top_k]

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
