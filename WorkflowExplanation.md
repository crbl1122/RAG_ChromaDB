This document outlines the step-by-step workflow and techniques employed in the Python RAG (Retrieval Augmented Generation) application. The application allows users to ask questions about documents stored in a specified directory, and it generates answers based on the content of these documents using a language model.

### Core Components
The application is built using several key libraries and services:
* **Streamlit**: For creating the interactive web user interface.
* **ChromaDB**: As a vector database to store document embeddings and enable semantic search.
* **Sentence Transformers (`BAAI/bge-small-en-v1.5`)**: To generate dense vector embeddings for text chunks .
* **LlamaIndex (`SimpleDirectoryReader`, `SentenceSplitter`)**: For loading and chunking documents .
* **OpenAI API (`gpt-4o-mini`)**: As the large language model (LLM) for generating answers based on retrieved context.

### Step-by-Step Workflow

**1. Setup and Initialization** 
   * **Import Libraries**: Essential Python libraries are imported, including `os`, `streamlit`, `chromadb`, `sentence_transformers`, `openai`, and components from `llama_index.core` .
   * **Embedding Model Configuration**:
        * The device for computation (CUDA if available, otherwise CPU) is determined .
        * The `BAAI/bge-small-en-v1.5` model from Sentence Transformers is loaded to convert text into numerical embeddings .
   * **Custom Embedding Function**:
        * A `CustomEmbeddingFunction` class is defined, inheriting from ChromaDB's `EmbeddingFunction` . This class wraps the Sentence Transformer model to make it compatible with ChromaDB's API .
        * An instance of this custom function is created .
   * **ChromaDB Initialization**:
        * A persistent ChromaDB client is initialized, storing its data in a directory named `chromadb_store` .
        * A ChromaDB collection named `txt_documents` is either retrieved or created . This collection is configured to use the `CustomEmbeddingFunction` for generating embeddings automatically when documents are added .
   * **Text Splitter**:
        * A `SentenceSplitter` from LlamaIndex is initialized . This tool is configured to divide documents into chunks of 512 characters with an overlap of 50 characters between chunks .

**2. Document Ingestion and Indexing (`load_and_chunk_documents` function)**
   * **Load Documents**: Text documents (`.txt` files) are loaded from a local directory named `data` using LlamaIndex's `SimpleDirectoryReader` .
   * **Chunk Documents**: The loaded documents are then processed by the `SentenceSplitter` to break them down into smaller text segments called "nodes" .
   * **Store Chunks in ChromaDB**:
        * Each node (chunk) is iterated over .
        * A unique ID is generated for each chunk (e.g., `original_doc_id_chunk_0`) .
        * The system checks if a document with this ID already exists in the ChromaDB collection to prevent duplicates .
        * If the chunk is new, its text content and associated metadata (specifically, the ID of the original document it came from, `node.ref_doc_id` ) are added to the `txt_documents` collection in ChromaDB . ChromaDB automatically uses the `CustomEmbeddingFunction` to convert the chunk's text into an embedding before storage.

**3. Query Processing and Response Generation (`generate_response` function)**
   * **Retrieve Relevant Chunks**:
        * When a user submits a query, this function is called.
        * The ChromaDB collection is queried using the user's query text. The query text is embedded using the same `bge-small-en-v1.5` model, and ChromaDB searches for the top 5 most semantically similar document chunks from the stored collection.
        * The text content and metadata of these retrieved chunks are extracted.
   * **Construct Context**: The text content of the retrieved chunks is concatenated together, separated by double newlines, to form a single "context" string.
   * **Generate Answer with LLM**:
        * A prompt is constructed for the OpenAI `gpt-4o-mini` model . This prompt includes explicit instructions for the LLM to answer the user's question *using only the provided context* . The context string and the original user query are embedded within this prompt .
        * The OpenAI API is called using the `client.chat.completions.create` method. The request includes:
            * The target model (`gpt-4o-mini`).
            * A messages list containing a system message (defining the AI's role) and the user's prompt (with context and question).
            * A `temperature` setting of 0.7.
            * A `max_tokens` limit of 500 for the generated answer .
        * The LLM's response content (the answer) is extracted and any leading/trailing whitespace is removed.
   * **Return Results**: The function returns the generated answer, the list of retrieved document texts, and their corresponding metadata.

**4. User Interface (`main` function)**
   * **Application Title and Description**: A title and a brief description of the RAG application are set for the Streamlit UI.
   * **Initial Document Indexing**:
        * The application checks if the ChromaDB collection contains any documents.
        * If the collection is empty (i.e., `collection.count() == 0`), it means documents haven't been indexed yet. In this case, the `load_and_chunk_documents` function is called to perform the ingestion and indexing process. A spinner is displayed in the UI during this operation.
   * **User Input**: A text input field is provided where users can type their questions. A default example question, "What is the Christmas campaign about?", is pre-filled.
   * **Process Query and Display Results**:
        * When a user enters a query and presses Enter :
            * An OpenAI API client is initialized . (Note: The `OPENAI_API_KEY` environment variable must be set .)
            * The `generate_response` function is called with the user's query and the OpenAI client .
            * The generated answer is displayed under a "Answer:" subheader .
            * An expandable section titled "Show Retrieved Context (Source Nodes)" is provided . Inside this section, each retrieved document chunk (source node) that was used to generate the answer is displayed, along with its original source document ID .

**5. Application Entry Point (`if __name__ == "__main__":`)**
   * **API Key Check**: Before starting the Streamlit application, the code checks if the `OPENAI_API_KEY` environment variable is set . If not, it raises a `ValueError` to alert the user .
   * **Run Main Application**: If the API key is present, the `main()` function is called, launching the Streamlit UI.

### Techniques Employed
* **Retrieval Augmented Generation (RAG)**: This is the core methodology. Instead of relying solely on the LLM's pre-trained knowledge, the system first retrieves relevant information from a custom knowledge base (the documents in the `data` folder) and then provides this information to the LLM as context to generate a more accurate and grounded answer.
* **Vector Embeddings**: Text is converted into numerical vectors (embeddings) using the `BAAI/bge-small-en-v1.5` model . These embeddings capture the semantic meaning of the text, allowing for similarity searches.
* **Vector Database**: ChromaDB is used to store these embeddings and perform efficient similarity searches (i.e., finding text chunks whose embeddings are closest to the query embedding) .
* **Text Chunking**: Documents are broken into smaller, manageable chunks using `SentenceSplitter` . This is crucial because LLMs have limited context windows, and retrieval is often more effective on smaller, focused pieces of text.
* **In-Context Learning with LLM**: The `gpt-4o-mini` model is prompted to use the retrieved chunks as the sole basis for its answer, reducing hallucinations and improving relevance to the provided documents .
