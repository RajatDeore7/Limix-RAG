import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()


# --------------------------------------------------------------------
# SAFE JSON LOADER
# --------------------------------------------------------------------
def safe_load_json(path: str, default_data=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: {path} is empty. Using default data.")
                return default_data if default_data else {}
            return json.loads(content)
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using default data.")
        return default_data if default_data else {}
    except json.JSONDecodeError as e:
        print(f"Error decoding {path}: {e}. Using default data.")
        return default_data if default_data else {}


# --------------------------------------------------------------------
# LOAD CODE DATA AND METADATA
# --------------------------------------------------------------------
code_data = safe_load_json(
    "./processed_code.json",
    default_data={"anchor": [], "solana": [], "cookbook": [], "snippets_from_pdf": []},
)
all_code_chunks = (
    code_data.get("anchor", [])
    + code_data.get("solana", [])
    + code_data.get("cookbook", [])
    + code_data.get("snippets_from_pdf", [])
)

metadata = safe_load_json(
    "metadata.json", default_data={"code": [], "vulnerabilities": []}
)

# --------------------------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

# Get embedding dimension dynamically
sample_vector = np.array(embedding_model.embed_query("test"), dtype=np.float32)
embedding_dim = sample_vector.shape[0]


# --------------------------------------------------------------------
# FAISS INDEX LOADING WITH FALLBACK
# --------------------------------------------------------------------
def load_faiss_index(path: str, dim: int):
    if os.path.exists(path):
        index = faiss.read_index(path)
        if index.d != dim:
            print(
                f"Embedding dimension mismatch: {dim} != FAISS index dim {index.d}. Rebuilding index."
            )
            return faiss.IndexFlatL2(dim)
        return index
    else:
        print(f"Warning: FAISS index {path} not found. Using empty index.")
        return faiss.IndexFlatL2(dim)


code_index = load_faiss_index("solana_code_faiss.index", dim=embedding_dim)
vulnerabilities_index = load_faiss_index("vulnerabilities.index", dim=embedding_dim)


# --------------------------------------------------------------------
# GROQ MODEL
# --------------------------------------------------------------------
groq_api_key = "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"
if not groq_api_key:
    raise RuntimeError("Missing GROQ_API_KEY (or GROQ_KEY) environment variable.")

groq_llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    temperature=0.2,
)


# --------------------------------------------------------------------
# QUERY CODE
# --------------------------------------------------------------------
def query_code(prompt, top_k=3):
    prompt_vector = np.array(
        embedding_model.embed_query(prompt), dtype=np.float32
    ).reshape(1, -1)
    distances, indices = code_index.search(prompt_vector, top_k)
    return [all_code_chunks[i] for i in indices[0]]


# --------------------------------------------------------------------
# CHAT WITH MODEL
# --------------------------------------------------------------------
def chat_with_model(user_query: str) -> str:
    retrieved_code = query_code(user_query, top_k=3)
    formatted_code = "\n\n".join(retrieved_code)

    prompt = f"""
    Use the retrieved Solana code snippets below to provide the best answer to the user's question
    or suggest appropriate completions for the partial user code.

    # Retrieved Code snippets
    {formatted_code}

    # User Question or partial user code
    {user_query}
    """.strip()

    messages = [
        SystemMessage(
            content="You are an expert in Solana smart contract development."
        ),
        HumanMessage(content=prompt),
    ]

    chunks = []
    for chunk in groq_llm.stream(messages):
        if chunk.content:
            chunks.append(chunk.content)

    return "".join(chunks).strip()


# --------------------------------------------------------------------
# RETRIEVE SIMILAR CODE
# --------------------------------------------------------------------
def retrieve_similar_code(input_code, top_k=3):
    prompt_vector = np.array(
        embedding_model.embed_query(input_code), dtype=np.float32
    ).reshape(1, -1)
    distances, indices = vulnerabilities_index.search(prompt_vector, top_k)

    retrieved_snippets = []
    retrieved_vulnerabilities = []

    for idx in indices[0]:
        if idx < len(metadata.get("code", [])):
            retrieved_snippets.append(metadata["code"][idx])
        if idx < len(metadata.get("vulnerabilities", [])):
            retrieved_vulnerabilities.extend(metadata["vulnerabilities"][idx])

    return retrieved_snippets, list(set(retrieved_vulnerabilities))


# --------------------------------------------------------------------
# DETECT VULNERABILITIES
# --------------------------------------------------------------------
def detect_vulnerabilities(input_code: str) -> str:
    retrieved_snippets, retrieved_vulnerabilities = retrieve_similar_code(input_code)

    prompt = f"""
    The user has provided the following Solana code snippet:
    
    ```rust
    {input_code}
    ```
    Based on the following past vulnerabilities, analyze and detect security flaws in the given code:
    **Past Vulnerabilities Found in Similar Code:**
    {retrieved_vulnerabilities}
    **Provide a structured analysis of potential security risks and mitigation strategies.**
    """.strip()

    messages = [
        SystemMessage(
            content="You are an expert in Solana smart contract security analysis."
        ),
        HumanMessage(content=prompt),
    ]

    chunks = []
    for chunk in groq_llm.stream(messages):
        if chunk.content:
            chunks.append(chunk.content)

    return "".join(chunks).strip()
