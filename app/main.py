from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import pickle
import os
import json
from typing import List
import time
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AIPROXY
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
print("DEBUG AIPROXY_TOKEN:", AIPROXY_TOKEN)
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set")

class Query(BaseModel):
    question: str
    image: str | None = None

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
os.makedirs(data_dir, exist_ok=True)
index_path = os.path.join(data_dir, "post_index_combined.pkl")

# Combine discourse and course content
def load_combined_metadata():
    combined = []

    # Load DiscourseData (JSONL format)
    discourse_path = os.path.join(data_dir, "discourse_posts.json")
    if os.path.exists(discourse_path):
        with open(discourse_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                    combined.append({
                        "url": p.get("url", ""),
                        "text": p.get("content", "")[:500]
                    })
                except json.JSONDecodeError:
                    continue
    else:
        print("Discourse data not found!")

    # Load CoursecontentData (JSONL format)
    course_path = os.path.join(data_dir, "course_content.json")
    if os.path.exists(course_path):
        with open(course_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                    combined.append({
                        "url": p.get("url", ""),
                        "text": p.get("content", "")[:500]
                    })
                except json.JSONDecodeError:
                    continue
    else:
        print("Course content data not found!")

    return combined[:100]  # Trim for performance

# Load or regenerate index
if os.path.exists(index_path):
    try:
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
        metadata = index_data["metadata"]
    except Exception as e:
        print(f"Error loading index: {e}")
        metadata = load_combined_metadata()
else:
    print("Generating new metadata...")
    metadata = load_combined_metadata()
    try:
        with open(index_path, "wb") as f:
            pickle.dump({"metadata": metadata}, f)
        print("Metadata saved.")
    except Exception as e:
        print(f"Error saving metadata: {e}")

# Embeddings
def get_embeddings(texts: List[str], batch_size: int = 5, retries: int = 3) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{AIPROXY_URL}v1/embeddings",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {AIPROXY_TOKEN}"
                    },
                    json={
                        "model": "text-embedding-3-small",
                        "input": batch
                    },
                    timeout=10
                )
                response.raise_for_status()
                batch_embeddings = [item["embedding"] for item in response.json()["data"]]
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                print(f"Embedding error batch {i//batch_size + 1}, attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    embeddings.extend([[0] * 1536 for _ in batch])
                time.sleep(2 ** attempt)
    return embeddings

# Chat Completion
def get_chat_completion(question: str, context: str, image: str | None = None) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a teaching assistant for a data science course. Provide a concise, accurate answer."},
            {"role": "user", "content": f"Question: {question}\nContext: {context}"}
        ]
        if image:
            messages.append({"role": "user", "content": f"Attached image: {image}"})
        response = requests.post(
            f"{AIPROXY_URL}v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AIPROXY_TOKEN}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "max_tokens": 200
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer available")
    except Exception as e:
        print(f"Chat completion error: {e}")
        return "Sorry, I couldn't generate an answer. Please refer to the linked resources."

# Load or compute embeddings
embedding_cache_path = os.path.join(data_dir, "post_embeddings_combined.pkl")
if os.path.exists(embedding_cache_path):
    try:
        with open(embedding_cache_path, "rb") as f:
            post_embeddings = pickle.load(f)
    except Exception as e:
        print(f"Embedding load error: {e}")
        post_embeddings = None
else:
    print("Generating new embeddings...")
    post_texts = [post["text"] for post in metadata]
    post_embeddings = get_embeddings(post_texts)
    try:
        with open(embedding_cache_path, "wb") as f:
            pickle.dump(post_embeddings, f)
    except Exception as e:
        print(f"Embedding save error: {e}")

@app.api_route("/api/", methods=["POST", "OPTIONS"])
async def answer_question(request: Request):
    if request.method == "OPTIONS":
        return {}

    try:
        body = await request.body()
        try:
            data = json.loads(body.decode("utf-8"))
            if isinstance(data, str):
                data = json.loads(data)
            query = Query(**data)
        except json.JSONDecodeError:
            query = Query(**await request.json())

        question = query.question
        question_embedding = get_embeddings([question])[0]
        if not question_embedding or all(v == 0 for v in question_embedding):
            raise Exception("Embedding failed")

        similarities = [
            np.dot(question_embedding, post_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(post_emb))
            if np.linalg.norm(post_emb) > 0 else 0
            for post_emb in post_embeddings
        ]
        top_indices = np.argsort(similarities)[-5:][::-1]

        links = []
        context = ""
        for idx in top_indices:
            if similarities[idx] > 0.1 and idx < len(metadata):
                post = metadata[idx]
                text = post["text"].split(".")[0]
                if len(text) > 100:
                    text = text[:97] + "..."
                links.append({
                    "url": post["url"],
                    "text": text
                })
                context += f"- {post['text'][:500]}\n"

        answer = get_chat_completion(question, context, query.image)

        return {
            "answer": answer,
            "links": links
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
