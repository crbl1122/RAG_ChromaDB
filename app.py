import os
import streamlit as st
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings, IDs, Metadatas
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
import torch
from typing import List, Tuple, Dict, Any, Optional

# Embedding model
device: str = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model: SentenceTransformer = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

class CustomEmbeddingFunction(EmbeddingFunction):
    """
    A custom embedding function that uses a SentenceTransformer model
    to generate embeddings for a list of documents.
    """
    def __init__(self, model: SentenceTransformer):
        """
        Initializes the CustomEmbeddingFunction.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to use for embeddings.
        """
        self.model: SentenceTransformer = model

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generates embeddings for a list of documents.

        Args:
            input (Documents): A list of documents (strings) to embed.
                               In chromadb types, this is `chromadb.api.types.Documents`
                               which is an alias for `Sequence[Document]`, where
                               `Document` is an alias for `str`.

        Returns:
            Embeddings: A list of embeddings, where each embedding is a list of floats.
                        In chromadb types, this is `chromadb.api.types.Embeddings`
                        which is an alias for `Sequence[Embedding]`, where
                        `Embedding` is an alias for `Sequence[float]`.
        """
        return self.model.encode(input, convert_to_numpy=True).tolist()

embedding_fn: CustomEmbeddingFunction = CustomEmbeddingFunction(embedding_model)

# Initialize ChromaDB client and collection
chroma_client: chromadb.PersistentClient = chromadb.PersistentClient(path="chromadb_store")
collection: chromadb.Collection = chroma_client.get_or_create_collection(
    name="txt_documents",
    embedding_function=embedding_fn
)

splitter: SentenceSplitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def load_and_chunk_documents() -> None:
    """
    Loads documents from the 'data' folder, splits them into chunks (nodes),
    and adds them to the ChromaDB collection if they don't already exist.
    """
    # Load docs from 'data' folder
    documents: List[Document] = SimpleDirectoryReader("data").load_data()
    nodes: List[BaseNode] = splitter.get_nodes_from_documents(documents)

    for idx, node in enumerate(nodes):
        chunk_text: str = node.get_text()
        doc_id: str = f"{node.ref_doc_id}_chunk_{idx}"
        existing: chromadb.GetResult = collection.get(ids=[doc_id])
        if not existing['ids']: # Check if 'ids' list in the GetResult is empty
            collection.add(
                documents=[chunk_text],
                metadatas=[{"source": node.ref_doc_id}], # type: ignore
                ids=[doc_id]
            )

def generate_response(query: str, client: OpenAI) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Generates a response to a query using a RAG (Retrieval Augmented Generation) approach.
    It retrieves relevant documents from ChromaDB, constructs a prompt with this context,
    and then uses an OpenAI model to generate an answer.

    Args:
        query (str): The user's question.
        client (OpenAI): An initialized OpenAI client.

    Returns:
        Tuple[str, List[str], List[Dict[str, Any]]]:
            - The generated answer string.
            - A list of retrieved document texts (context).
            - A list of metadatas corresponding to the retrieved documents.
    """
    results: chromadb.QueryResult = collection.query(query_texts=[query], n_results=5)

    retrieved_documents: Optional[List[str]] = results["documents"][0] if results["documents"] else [] # pyright: ignore [reportOptionalSubscript]
    metadatas: Optional[List[Optional[Dict[str, Any]]]] = results["metadatas"][0] if results["metadatas"] else [] # pyright: ignore [reportOptionalSubscript]

    # Ensure documents and metadatas are not None and are lists
    # The query result for documents is List[List[str]] and for metadatas is List[List[Optional[Metadata]]]
    # We are interested in the first list of results (index 0)
    context_docs: List[str] = retrieved_documents if retrieved_documents is not None else []
    context_metas: List[Dict[str, Any]] = [m if m is not None else {} for m in (metadatas if metadatas is not None else [])]


    context: str = "\n\n".join(context_docs)

    prompt: str = f"""You are a helpful assistant. Answer the question using only the context below.

    Context:
    {context}

    Question: {query}
    Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    answer: str = response.choices[0].message.content.strip() if response.choices[0].message.content else "No answer generated."
    return answer, context_docs, context_metas

def main() -> None:
    """
    Main function to run the Streamlit application.
    It sets up the UI, handles document indexing, and processes user queries
    to display answers and retrieved context.
    """
    st.title("📚 Deutsche Telekom RAG App with: ChromaDB + LlamaIndex + BGE embeddings + GPT-4o-mini")
    st.markdown("Place `.txt` files in the `data/` folder. Documents are chunked and indexed for retrieval.")

    if collection.count() == 0:
        with st.spinner("Indexing documents..."):
            load_and_chunk_documents()

    query: Optional[str] = st.text_input("Enter your question:", "What is the Christmas campaign about?")

    if query:
        client: OpenAI = OpenAI()
        answer: str
        docs: List[str]
        metas: List[Dict[str, Any]]
        answer, docs, metas = generate_response(query, client)

        st.subheader("Answer:")
        st.write(answer)

        with st.expander("Show Retrieved Context (Source Nodes)"):
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                source: Any = meta.get("source", "Unknown source")
                st.write(f"**Source Node {i+1} — Source: {source}**")
                st.info(doc.strip())
                st.markdown("---")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set your OPENAI_API_KEY environment variable.")
    main()