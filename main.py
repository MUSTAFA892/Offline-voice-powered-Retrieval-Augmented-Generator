import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# === Personalization ===
YOUR_NAME = "Mustafa"
EXTRA_KNOWLEDGE = f"""
You are talking to {YOUR_NAME}, the creator of this offline RAG assistant.
The system runs entirely without internet using local LLMs and speech-to-text.
"""

# === Load Ollama LLM ===
model = OllamaLLM(model="llama3.2:1b")

# === Prompt Templates ===
template_with_reviews = """
You are an expert in answering questions about the CSV data provided.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}

Additional context:
{extra_knowledge}
"""

template_general = """
You are a smart assistant who answers general and specific questions clearly.

Here is the question to answer: {question}

Additional context:
{extra_knowledge}
"""

prompt_with_reviews = ChatPromptTemplate.from_template(template_with_reviews)
prompt_general = ChatPromptTemplate.from_template(template_general)

# === Whisper STT Setup ===
model_size = "base.en"
whisper_model = WhisperModel(model_size, compute_type="int8", local_files_only=True)

samplerate = 16000
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio error:", status)
    q.put(indata.copy())

def record_audio(seconds=5):
    print(f"🎙️ Listening for {seconds} seconds...")
    frames = []
    for _ in range(int(samplerate / 1024 * seconds)):
        frames.append(q.get())
    audio = np.concatenate(frames, axis=0)
    return audio

def get_voice_input():
    while True:
        audio_data = record_audio(seconds=5)
        audio_data = np.squeeze(audio_data).astype(np.float32) / 32768.0
        segments, _ = whisper_model.transcribe(audio_data, language="en", beam_size=5)

        for segment in segments:
            question = segment.text.strip()
            break

        if not question:
            print("❌ Could not understand. Try again.")
            continue

        print(f"\n🗣️ You said: \"{question}\"")
        confirm = input("✅ Is this correct? (y/n): ").strip().lower()
        if confirm == "y":
            return question
        else:
            print("🔁 Let's try again...\n")

def get_text_input():
    return input("⌨️ Type your question: ").strip()

# === Main Loop ===
print("\n🎤 Voice RAG Assistant (Offline LLM + Whisper)")
print("Press [v] for voice input or [t] to type. [q] to quit.\n")

try:
    with sd.InputStream(samplerate=samplerate, blocksize=1024, dtype='int16',
                        channels=1, callback=audio_callback):
        while True:
            mode = input("👉 Choose input mode [v/t/q]: ").strip().lower()

            if mode == "q":
                break
            elif mode == "v":
                question = get_voice_input()
            elif mode == "t":
                question = get_text_input()
                if not question:
                    continue
            else:
                print("❓ Invalid option. Please enter 'v' or 't'.")
                continue

            if not question:
                continue

            print(f"\n🧠 Question: {question}")
            reviews = retriever.invoke(question)

            # Choose prompt based on RAG or not
            if reviews:
                prompt = prompt_with_reviews
                chain = prompt | model
                inputs = {
                    "reviews": reviews,
                    "question": question,
                    "extra_knowledge": EXTRA_KNOWLEDGE
                }
            else:
                prompt = prompt_general
                chain = prompt | model
                inputs = {
                    "question": question,
                    "extra_knowledge": EXTRA_KNOWLEDGE
                }

            response = chain.invoke(inputs)

            print(f"\n🤖 Answer: {response}\n")
            print("-" * 40)

except KeyboardInterrupt:
    print("\n🛑 Exiting. Goodbye!")
