# 🧠 TDS Virtual Teaching Assistant

This project is a Promptfoo-powered evaluation system for a Virtual Teaching Assistant tailored to the **Tools for Data Science (TDS)** course offered by IIT Madras. It leverages LLM APIs and image-based inputs (such as `.webp` screenshots of forum discussions or course content) to assist students by answering course-related questions. It also allows for systematic prompt evaluations to improve response quality.

---

## 🚀 Features

- 🤖 Prompt evaluation using [Promptfoo](https://promptfoo.dev)
- 📸 Image + question-based test cases for realistic input simulation
- 🧪 Side-by-side comparison of different prompt strategies
- 📊 Web & CLI-based viewer for evaluation results
- ☁️ API provider support (OpenAI, AIPipe, AIPROXY, and other proxy-compatible providers)

---

## 📡 API Endpoint

**POST** `/api/`

---

## 📥 Request Format

```json
{
  "question": "What is the deadline for the TDS Week 5 quiz?",
  "image": "images/project-tds-virtual-ta-q1.webp"
}
Author
Narayan Thakur
Student at IIT Madras
BSc in Data Science and Applications
Student ID: 23f2002083
GitHub: nt0409
