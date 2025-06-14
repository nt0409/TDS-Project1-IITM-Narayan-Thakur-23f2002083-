import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
import json

# Your session cookie (_t value only, not the full cookie header)
DISCOURSE_COOKIE = "<ADD_YOUR_COOKIE_HERE>"

session = requests.Session()
session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

def get_topic_ids(category_slug="courses/tds-kb", category_id=34):
    topics = []
    for page in tqdm(range(0, 20), desc="Fetching topics"):
        url = f"{BASE_URL}/c/{category_slug}/{category_id}.json?page={page}"
        r = session.get(url)
        if r.status_code != 200:
            print(f"Failed to fetch page {page}: {r.status_code}")
            break
        data = r.json()
        new_topics = data.get("topic_list", {}).get("topics", [])
        if not new_topics:
            break
        topics.extend(new_topics)
    return topics

def get_posts_in_topic(topic_id):
    r = session.get(f"{BASE_URL}/t/{topic_id}.json")
    if r.status_code != 200:
        print(f"Failed to fetch topic {topic_id}: {r.status_code}")
        return []
    data = r.json()
    return [
        {
            "username": post["username"],
            "created_at": post["created_at"],
            "content": BeautifulSoup(post["cooked"], "html.parser").get_text(),
            "post_url": f"{BASE_URL}/t/{topic_id}/{post['post_number']}"
        }
        for post in data.get("post_stream", {}).get("posts", [])
    ]

def main():
    all_posts = []
    topics = get_topic_ids()

    for topic in tqdm(topics, desc="Processing topics"):
        created_at = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))
        if created_at >= datetime(2025, 1, 1, tzinfo=timezone.utc):
            posts = get_posts_in_topic(topic["id"])
            all_posts.extend(posts)

    # Save the scraped posts into a JSON file
    with open("DiscourseData.json", "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)

    print(f"Scraped {len(all_posts)} posts.")

if __name__ == "__main__":
    main()
