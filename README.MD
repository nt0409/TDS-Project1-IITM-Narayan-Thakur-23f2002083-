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

## Endpoint

**POST** `/api/`

---

## Request Format

```json
{
  "question": "What is the deadline for the TDS Week 5 quiz?",
  "image": "images/project-tds-virtual-ta-q1.webp"
}
