from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import requests
import numpy as np
import pickle
import os
import json
import time

# Load environment variables (Render supports this via its dashboard)
load_dotenv()

app = FastAPI()

# Enable CORS for any frontend that might access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Proxy setup
PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/"
PROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not PROXY_TOKEN:
    raise EnvironmentError("Missing AIPROXY_TOKEN")

# Pydantic model
class UserQuery(BaseModel):
    question: str
    image: Optional[str] = None

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")  # Flattened for Render
os.makedirs(DATA_DIR, exist_ok=True)

POSTS_JSON = os.path.join(DATA_DIR, "DiscourseData.json")
INDEX_FILE = os.path.join(DATA_DIR, "post_index.pkl")
EMBED_FILE = os.path.join(DATA_DIR, "post_embeddings.pkl")

# Load metadata
def load_post_snippets(limit=30):
    try:
        with open(POSTS_JSON, "r", encoding="utf-8") as f:
            posts = json.load(f)
        return [{"url": p["url"], "text": p["text"][:500]} for p in posts[:limit]]
    except Exception as e:
        print(f"Error loading {POSTS_JSON}: {e}")
        return []

# Load or create metadata
if os.path.exists(INDEX_FILE):
    try:
        with open(INDEX_FILE, "rb") as f:
            metadata = pickle.load(f)["metadata"]
    except Exception as e:
        print("Index error:", e)
        metadata = load_post_snippets()
else:
    metadata = load_post_snippets()
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"metadata": metadata}, f)

# Get embeddings
def fetch_embeddings(texts: List[str], batch=5, retry=3) -> List[List[float]]:
    vectors = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        for attempt in range(retry):
            try:
                res = requests.post(
                    f"{PROXY_URL}v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {PROXY_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={"model": "text-embedding-3-small", "input": chunk},
                    timeout=10
                )
                res.raise_for_status()
                vectors += [d["embedding"] for d in res.json()["data"]]
                break
            except Exception as e:
                print(f"Embedding error on batch {i//batch+1}, try {attempt+1}: {e}")
                if attempt == retry - 1:
                    vectors += [[0] * 1536 for _ in chunk]
                time.sleep(2 ** attempt)
    return vectors

# Load or generate post embeddings
if os.path.exists(EMBED_FILE):
    try:
        with open(EMBED_FILE, "rb") as f:
            post_embeddings = pickle.load(f)
    except Exception:
        post_embeddings = None
else:
    post_embeddings = fetch_embeddings([p["text"] for p in metadata])
    with open(EMBED_FILE, "wb") as f:
        pickle.dump(post_embeddings, f)

if not post_embeddings:
    post_embeddings = fetch_embeddings([p["text"] for p in metadata])

# Chat completion
def generate_response(question: str, context: str, image: Optional[str] = None) -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI teaching assistant for a university-level data science course. "
                    "Respond to students with clear, fact-based answers using the context provided."
                )
            },
            {
                "role": "user",
                "content": f"Student asked: {question}\nRelevant context:\n{context}"
            }
        ]
        if image:
            messages.append({
                "role": "user",
                "content": f"Also consider this image (base64 or URL): {image}"
            })
        res = requests.post(
            f"{PROXY_URL}v1/chat/completions",
            headers={
                "Authorization": f"Bearer {PROXY_TOKEN}",
                "Content-Type": "application/json"
            },
            json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": 200},
            timeout=15
        )
        res.raise_for_status()
        return res.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    except Exception as err:
        print("Chat completion error:", err)
        return "I'm unable to answer right now. Please refer to your course material."

# Main API
@app.api_route("/api/", methods=["POST", "OPTIONS"])
async def handle_query(request: Request):
    if request.method == "OPTIONS":
        return {}

    try:
        data = await request.json()
        user_input = UserQuery(**data)
        question = user_input.question
        image = user_input.image

        query_vector = fetch_embeddings([question])[0]
        if not query_vector or all(x == 0 for x in query_vector):
            raise ValueError("Invalid embedding generated.")

        # Cosine similarity
        sims = [
            np.dot(query_vector, emb) / (np.linalg.norm(query_vector) * np.linalg.norm(emb))
            if np.linalg.norm(emb) > 0 else 0
            for emb in post_embeddings
        ]
        top_indices = np.argsort(sims)[-5:][::-1]

        context = ""
        links = []
        for idx in top_indices:
            if sims[idx] > 0.1:
                snippet = metadata[idx]
                short_text = snippet["text"].split(".")[0]
                if len(short_text) > 100:
                    short_text = short_text[:97] + "..."
                context += f"- {snippet['text'][:500]}\n"
                links.append({"url": snippet["url"], "text": short_text})

        response = generate_response(question, context, image)
        return {"answer": response, "links": links}

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(err)}")

