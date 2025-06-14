# 🧠 TDS Virtual Teaching Assistant

This project is a Promptfoo-powered evaluation system for a Virtual Teaching Assistant tailored to the **Tools for Data Science (TDS)** course. It uses LLM APIs and image-based inputs (e.g., `.webp` screenshots from forums or content) to help answer student queries and evaluate different prompt strategies.

---

## 🚀 Features

- 🤖 Prompt evaluation using [Promptfoo](https://promptfoo.dev)
- 📸 Image + question based test cases
- 🧪 Compare prompt variations for quality
- 📊 CLI & web-based results viewer
- ☁️ LLM API provider support (OpenAI, Proxy)

---

## 📂 Project Structure

TDS-Project1-IITM-Narayan-Thakur-23f2002083-/
├── ScrappingScript/
│   ├── CourseContent_Scrapping.py
│   ├── Discourse_Scrapping.py
│   └── app/
│       ├── __pycache__/
│       ├── __init__.py
│       └── main.py
├── data/
├── .gitignore
├── LICENSE
├── project-tds-virtual-ta-promptfoo.yaml
├── project-tds-virtual-ta-q1.webp
├── requirements.txt


Install Promptfoo
npm install -g promptfoo


Run Evaluation
npx promptfoo eval -c project-tds-virtual-ta-promptfoo.yaml


Sample Test Case Format (JSON)
json
Copy
Edit
{
  "question": "What is the deadline for the TDS Week 5 quiz?",
  "image": "images/project-tds-virtual-ta-q1.webp"
}

 License
This project is licensed under the MIT License.

Author
Narayan Thakur
IIT Madras BS Degree, Batch 23f2002083
