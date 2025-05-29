import os
import streamlit as st
import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import torch

# Embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, convert_to_numpy=True).tolist()

embedding_fn = CustomEmbeddingFunction(embedding_model)

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="chromadb_store")
collection = chroma_client.get_or_create_collection(
    name="txt_documents",
    embedding_function=embedding_fn
)

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def load_and_chunk_documents():
    # Load docs from 'data' folder
    documents = SimpleDirectoryReader("data").load_data()
    nodes = splitter.get_nodes_from_documents(documents)

    for idx, node in enumerate(nodes):
        chunk_text = node.get_text()
        doc_id = f"{node.ref_doc_id}_chunk_{idx}"
        existing = collection.get(ids=[doc_id])
        if not existing['ids']:
            collection.add(
                documents=[chunk_text],
                metadatas=[{"source": node.ref_doc_id}],
                ids=[doc_id]
            )

def generate_response(query, client):
    results = collection.query(query_texts=[query], n_results=5)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)

    prompt = f"""You are a helpful assistant. Answer the question using only the context below.

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
    answer = response.choices[0].message.content.strip()
    return answer, documents, metadatas

def main():
    st.title("📚 Deutsche Telekom RAG App with ChromaDB + llama_index + BGE embeddings + gpt-4o-mini")
    st.markdown("Place `.txt` files in the `data/` folder. Documents are chunked and indexed for retrieval.")

    if collection.count() == 0:
        with st.spinner("Indexing documents..."):
            load_and_chunk_documents()

    query = st.text_input("Enter your question:", "What is the Christmas campaign about?")

    if query:
        client = OpenAI()
        answer, docs, metas = generate_response(query, client)

        st.subheader("Answer:")
        st.write(answer)

        with st.expander("Show Retrieved Context (Source Nodes)"):
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                source = meta.get("source", "Unknown source")
                st.write(f"**Source Node {i+1} — Source: {source}**")
                st.info(doc.strip())
                st.markdown("---")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set your OPENAI_API_KEY environment variable.")
    main()
