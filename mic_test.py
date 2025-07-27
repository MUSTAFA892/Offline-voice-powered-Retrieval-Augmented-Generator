import sounddevice as sd
import numpy as np
import queue
import time

q = queue.Queue()
samplerate = 16000
duration = 5  # seconds

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Error:", status)
    q.put(indata.copy())

def test_microphone():
    print("🎤 Starting microphone test...")
    print(f"🎙️ Please speak for {duration} seconds.")
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=audio_callback):
            frames = []
            for _ in range(int(samplerate / 1024 * duration)):
                frames.append(q.get())
            audio = np.concatenate(frames, axis=0)
            print("✅ Mic test complete! Data captured successfully.")
            print("🎧 (Optional) You can play it back using any audio library if needed.)")
    except Exception as e:
        print("❌ Mic test failed:", e)

if __name__ == "__main__":
    test_microphone()
