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

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/"
PROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not PROXY_TOKEN:
    raise EnvironmentError("Missing AIPROXY_TOKEN")

# File paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

COURSE_JSON = os.path.join(DATA_DIR, "CoursecontentData.json")
DISCOURSE_JSON = os.path.join(DATA_DIR, "DiscourseData.json")
INDEX_FILE = os.path.join(DATA_DIR, "post_index.pkl")
EMBED_FILE = os.path.join(DATA_DIR, "post_embeddings.pkl")

# Model
class UserQuery(BaseModel):
    question: str
    image: Optional[str] = None

# Load posts and merge
def load_all_snippets(limit=50):
    items = []
    try:
        if os.path.exists(COURSE_JSON):
            with open(COURSE_JSON, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        items.append({
                            "text": data.get("text", "")[:1000],
                            "url": "Course Content"
                        })
                    except:
                        continue

        if os.path.exists(DISCOURSE_JSON):
            with open(DISCOURSE_JSON, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        items.append({
                            "text": data.get("text", "")[:1000],
                            "url": data.get("url", "Discourse")
                        })
                    except:
                        continue
    except Exception as e:
        print(f"Error loading files: {e}")
    return items[:limit]

# Load or create index
if os.path.exists(INDEX_FILE):
    try:
        with open(INDEX_FILE, "rb") as f:
            metadata = pickle.load(f)["metadata"]
    except Exception as e:
        print("Index error:", e)
        metadata = load_all_snippets()
else:
    metadata = load_all_snippets()
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"metadata": metadata}, f)

# Embedding generator
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
                batch_vecs = [item["embedding"] for item in res.json()["data"]]
                vectors.extend(batch_vecs)
                break
            except Exception as err:
                print(f"Embedding error (batch {i//batch + 1}, try {attempt + 1}): {err}")
                if attempt == retry - 1:
                    vectors.extend([[0] * 1536 for _ in chunk])
                time.sleep(2 ** attempt)
    return vectors

# Load or create embeddings
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

# Completion logic
def generate_response(question: str, context: str, image: Optional[str] = None) -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI teaching assistant for a university-level data science course. "
                    "Base your answers only on the course materials and discussion posts provided in the context."
                )
            },
            {"role": "user", "content": f"Student asked: {question}\nRelevant context:\n{context}"}
        ]
        if image:
            messages.append({"role": "user", "content": f"Attached image: {image}"})
        
        res = requests.post(
            f"{PROXY_URL}v1/chat/completions",
            headers={
                "Authorization": f"Bearer {PROXY_TOKEN}",
                "Content-Type": "application/json"
            },
            json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": 400},
            timeout=15
        )
        res.raise_for_status()
        return res.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    except Exception as err:
        print("Chat completion error:", err)
        return "I'm unable to answer right now. Please check again later."

# Main API route
@app.api_route("/api/", methods=["POST", "OPTIONS"])
async def handle_query(request: Request):
    if request.method == "OPTIONS":
        return {}

    try:
        raw = await request.body()
        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, str):
                payload = json.loads(payload)
            user_input = UserQuery(**payload)
        except json.JSONDecodeError:
            user_input = UserQuery(**await request.json())
        
        question = user_input.question
        image = user_input.image

        query_vec = fetch_embeddings([question])[0]
        if not query_vec or all(val == 0 for val in query_vec):
            raise ValueError("Invalid embedding generated")

        sims = [
            np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
            if np.linalg.norm(emb) > 0 else 0
            for emb in post_embeddings
        ]
        top_matches = np.argsort(sims)[-5:][::-1]

        context = ""
        references = []
        for idx in top_matches:
            if sims[idx] > 0.1 and idx < len(metadata):
                post = metadata[idx]
                preview = post["text"].split(".")[0]
                if len(preview) > 100:
                    preview = preview[:97] + "..."
                references.append({"url": post["url"], "text": preview})
                context += f"- {post['text'][:500]}\n"

        response = generate_response(question, context, image)
        return {"answer": response, "links": references}

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(err)}")
