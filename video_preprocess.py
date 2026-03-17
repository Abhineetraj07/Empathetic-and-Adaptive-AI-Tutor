# preprocess_video.py

import os
import json
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline

VIDEO_PATH = "videos/Introduction to Agriculture _ Crop Production and Management _ Don t Memorise.mp4"
OUTPUT_JSON = "video_metadata.json"

def extract_topic_and_segments(video_path):
    print("\n🧩 Extracting topic and transcript segments...")
    if not os.path.exists(video_path):
        print("❌ Video not found:", video_path)
        return None

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile("temp_audio.wav", logger=None)

    print("🔊 Transcribing audio with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe("temp_audio.wav")

    transcript = result["text"]
    segments = result.get("segments", [])
    print(f"✅ Transcription done — {len(segments)} segments")

    print("🧠 Summarizing transcript to get topic...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcript, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
    print("📘 Detected Topic:", summary)

    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
        print("🧹 Deleted temp_audio.wav")

    metadata = {
        "video_path": video_path,
        "topic": summary,
        "segments": [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip()
            }
            for seg in segments
        ]
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved metadata to {OUTPUT_JSON}")
    return metadata

if __name__ == "__main__":
    extract_topic_and_segments(VIDEO_PATH)