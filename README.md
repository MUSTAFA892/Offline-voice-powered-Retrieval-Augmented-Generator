# 🎤🧠 Offline Voice RAG Assistant

An **offline AI-powered voice assistant** that supports **local speech-to-text (Whisper)** and **local LLMs (Ollama)** to provide contextual answers using **RAG (Retrieval-Augmented Generation)**. No internet required after setup!

Built with:
- 🗣️ [faster-whisper](https://github.com/guillaumekln/faster-whisper) for offline speech recognition
- 🤖 [Ollama](https://ollama.com/) for running local LLMs like `llama3.2:1b`
- 📄 LangChain for prompt chaining and retrieval
- 🐍 Python, NumPy, SoundDevice

---

## 🚀 Features

- Fully **offline** voice assistant
- Uses **Whisper** STT to transcribe microphone input
- Uses **local LLMs** (via Ollama) to answer questions
- Includes a **RAG pipeline** to augment answers with context from CSV or other documents
- Dual-mode input: **voice or typed**
- Personalized assistant with your name and knowledge injected

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/offline-voice-rag-assistant.git
cd offline-voice-rag-assistant
````

### 2. Install Python Dependencies

Make sure you're using Python 3.9 or later.

```bash
pip install -r requirements.txt
```

**Or manually install:**

```bash
pip install numpy sounddevice faster-whisper langchain langchain-core langchain-community ollama
```

---

## 🤖 Model Setup

### 3. Download Whisper Model (Offline STT)

Before running the app, manually **download the Whisper model** to avoid online fetch:

```python
from faster_whisper import WhisperModel
WhisperModel("base.en", compute_type="int8", local_files_only=False)
```

> ⚠️ This step will **download the model** and cache it locally (\~50MB). You only need to do this **once**.
> Once downloaded, set `local_files_only=True` to ensure **completely offline mode** during real use.

---

### 4. Install Ollama and LLM Model

Make sure [Ollama](https://ollama.com/download) is installed on your system.

```bash
ollama pull llama3.2:1b
```

> You can use any local model supported by Ollama.
> The default used here is `llama3.2:1b`, which is lightweight and fast.

---

## 🧠 Folder Structure

```bash
offline-voice-rag-assistant/
│
├── main.py                # Main assistant script
├── vector.py              # Your custom retriever logic for RAG
├── requirements.txt
└── README.md
```

---

## 🎙️ Running the Assistant

Simply run:

```bash
python main.py
```

### Modes of Input:

* Speak into the mic when prompted 🎤
* Or press **Enter** to type your question ⌨️
* Type **`q`** anytime to quit

---

## 🧠 Customization

### Personalize Your Assistant

```python
YOUR_NAME = "Mustafa"
EXTRA_KNOWLEDGE = f"""
You are talking to {YOUR_NAME}, the creator of this offline RAG assistant.
"""
```

### Change Whisper Model Size

Options:

* `"tiny.en"` – Fastest, least accurate
* `"base.en"` – Default, balanced
* `"small.en"` – Better quality, slightly slower

---

## 📁 Document RAG Integration

In `vector.py`, implement your custom retriever (CSV, PDF, etc.).

Example (simplified):

```python
from langchain_community.vectorstores import FAISS
# Setup embedding model, load CSV, etc.
```

---

## ✅ Example Output

```bash
🎙️ Listening for 5 seconds...
🧠 Question: What is the rating of this product?
🤖 Answer: Based on the reviews, the average rating is 4.5 stars.
```

---

## 🧪 Troubleshooting

* 🔇 **No audio recorded**: Check microphone permissions or change device index.
* 🐌 **Slow response**: Try a smaller Whisper or LLM model.
* 📥 **Model not found**: Ensure Whisper and Ollama models are downloaded.

---

## ❤️ Credits

* Whisper by [OpenAI](https://github.com/openai/whisper)
* faster-whisper by [@guillaumekln](https://github.com/guillaumekln/faster-whisper)
* Ollama by [Ollama Team](https://ollama.com/)
* LangChain for chaining LLMs

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🌐 Author

**Mustafa Tinwala**
AI + Voice + Offline Tech Enthusiast
[LinkedIn](https://www.linkedin.com/in/mustafatinwala) | [GitHub](https://github.com/Mustafa892)
