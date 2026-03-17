import os
import sys
import json
from deepface import DeepFace
import cv2
import time
import threading
import subprocess
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
from data_logger import save_emotions_to_csv, clear_emotion_log
from question_generator import generate_questions_from_csv, launch_quiz_ui
from adaptive_engine import evaluate_and_adapt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
EMOTION_CSV_PATH = "emotion_data.csv"


def extract_transcript_with_timestamps(video_path):
    """
    Extract transcript WITH timestamps from video audio using Whisper.
    Returns: (topic_summary, transcript_segments)
    
    transcript_segments format:
    [{"start": 0.0, "end": 2.5, "text": "Hello everyone..."}, ...]
    """
    print("\n Extracting transcript with timestamps...")
    
    # Extract audio
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()

    # Transcribe with Whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    # Get segments with timestamps
    transcript_segments = []
    for segment in result["segments"]:
        transcript_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })
    
    full_transcript = result["text"]
    print(f" Transcription complete: {len(transcript_segments)} segments")

    # Generate topic summary
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    truncated = full_transcript[:1024] if len(full_transcript) > 1024 else full_transcript
    summary = summarizer(truncated, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
    print(f"📚 Topic: {summary}")
    
    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    return summary, transcript_segments


def get_phrase_at_timestamp(timestamp_str, transcript_segments):
    """
    Given timestamp (HH:MM:SS), find the phrase being spoken.
    """
    parts = timestamp_str.split(":")
    total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
    # Find matching segment
    for segment in transcript_segments:
        if segment["start"] <= total_seconds <= segment["end"]:
            return segment["text"]
    
    # Find closest segment
    if transcript_segments:
        closest = min(
            transcript_segments,
            key=lambda s: min(abs(s["start"] - total_seconds), abs(s["end"] - total_seconds))
        )
        return closest["text"]
    
    return ""


def background_emotion_detection(stop_flag, emotion_log, topic_name, transcript_segments):
    """
    Run webcam emotion detection in background.
    Logs negative emotions WITH the phrase being spoken.
    """
    webcam = cv2.VideoCapture(0)
    start_time = time.time()
    print("\n Emotion monitoring started...")

    while not stop_flag["stop"]:
        ret, frame = webcam.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            dominant_emotion = result[0]["dominant_emotion"]
            emotion_scores = result[0].get("emotion", {})

            # Detect confusion: high mix of fear + surprise + sad (no single dominant)
            fear_score = emotion_scores.get("fear", 0)
            surprise_score = emotion_scores.get("surprise", 0)
            sad_score = emotion_scores.get("sad", 0)
            angry_score = emotion_scores.get("angry", 0)
            top_score = max(emotion_scores.values()) if emotion_scores else 100

            # Confused = no strong dominant emotion + elevated negative signals
            is_confused = (top_score < 45 and (fear_score + surprise_score + sad_score) > 50)

            # Detect disengagement: face not detected or neutral for too long
            face_detected = result[0].get("face_confidence", 1.0) > 0.5 if result else True

        except Exception:
            dominant_emotion = "unknown"
            is_confused = False
            face_detected = False
            emotion_scores = {}

        # Determine what to log
        log_emotion = None
        if not face_detected:
            log_emotion = "disengaged"
        elif is_confused:
            log_emotion = "confused"
        elif dominant_emotion in ["sad", "angry", "fear"]:
            log_emotion = dominant_emotion

        elapsed = time.time() - start_time
        if log_emotion and elapsed >= 3:
            timestamp = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            # Get phrase at this timestamp
            phrase = get_phrase_at_timestamp(timestamp, transcript_segments)

            emotion_log.append({
                "timestamp": timestamp,
                "emotion": log_emotion,
                "topic": topic_name,
                "phrase": phrase
            })

            # Show what was detected
            phrase_preview = phrase[:40] + "..." if len(phrase) > 40 else phrase
            print(f"[{timestamp}] {log_emotion.upper()} | \"{phrase_preview}\"")

        time.sleep(1)

    webcam.release()
    print("\n Emotion monitoring stopped.")


def detect_emotions_during_video(video_path):
    """
    Play video while monitoring emotions.
    Returns (emotion_log, negative_detected, topic_name, video_description, cancelled)
    cancelled=True if user closed the video early.
    """
    # Extract transcript with timestamps FIRST
    topic_name, transcript_segments = extract_transcript_with_timestamps(video_path)

    # Build video description from transcript for quiz generation
    video_description = " ".join([seg["text"] for seg in transcript_segments])

    emotion_log = []

    # Start emotion detection thread
    stop_flag = {"stop": False}
    monitor_thread = threading.Thread(
        target=background_emotion_detection,
        args=(stop_flag, emotion_log, topic_name, transcript_segments),
    )
    monitor_thread.start()

    # Try to play video with different players
    print(f"\n Playing: {video_path}\n")
    video_process = None

    # Get video duration first
    clip = VideoFileClip(video_path)
    video_duration = clip.duration
    clip.close()

    # Try different video players - prioritize ffplay for reliable video+audio
    import platform
    system = platform.system()

    players = [
        # ffplay (FFmpeg) - reliable, shows window with audio
        ["ffplay", "-autoexit", "-window_title", "AI Tutor - Lecture", video_path],
        # VLC
        ["vlc", "--play-and-exit", video_path],
        # mpv
        ["mpv", video_path],
    ]

    if system == "Windows":
        players.append(["start", "", video_path])

    play_start = time.time()
    for player_cmd in players:
        try:
            print(f" Trying player: {player_cmd[0]}...")
            video_process = subprocess.Popen(player_cmd)
            video_process.wait()
            break

        except FileNotFoundError:
            print(f"    {player_cmd[0]} not found, trying next...")
            continue
        except Exception as e:
            print(f"    Error with {player_cmd[0]}: {e}")
            continue

    if video_process is None:
        print(" No video player found! Video will not play.")
        print("   Install ffplay (FFmpeg) or VLC, or the video will be skipped.")
        print(f"   Waiting {video_duration:.0f} seconds for emotion detection...")
        time.sleep(video_duration)

    # Check if video was closed early
    elapsed_play = time.time() - play_start
    cancelled = elapsed_play < (video_duration * 0.8)  # closed before 80% of video

    # Stop monitoring
    stop_flag["stop"] = True
    monitor_thread.join()

    # Terminate video process if still running
    if video_process:
        try:
            video_process.terminate()
        except:
            pass

    if cancelled:
        return emotion_log, False, topic_name, video_description, True

    negative_detected = any(e["emotion"] in ["sad", "angry", "fear", "confused", "disengaged"] for e in emotion_log)

    return emotion_log, negative_detected, topic_name, video_description, False


# =========================================
#  STARTUP MENU (runs in subprocess)
# =========================================
MENU_SCRIPT = '''
import tkinter as tk
import json
import sys
import os

BG_DARK = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT_BLUE = "#2979FF"
ACCENT_GREEN = "#4CAF50"
GOLD = "#ffd700"
WHITE = "#ffffff"
GRAY = "#aaaaaa"

def run_menu(data_path, result_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    curriculum = data["curriculum"]
    completed_ids = set(data.get("completed_ids", []))

    selected_id = [None]

    root = tk.Tk()
    root.title("AI Tutor - Select a Lecture")
    root.geometry("850x650")
    root.configure(bg=BG_DARK)

    # Header
    header = tk.Frame(root, bg="#0d0d1a", pady=15)
    header.pack(fill="x")
    tk.Label(header, text="Empathetic & Adaptive AI Tutor",
             font=("Arial", 22, "bold"), bg="#0d0d1a", fg=GOLD).pack()
    tk.Label(header, text="Select a lecture to begin",
             font=("Arial", 13), bg="#0d0d1a", fg=GRAY).pack(pady=(5, 0))

    # Progress
    completed_count = len([c for c in curriculum if c["id"] in completed_ids])
    tk.Label(root, text=f"Progress: {completed_count}/{len(curriculum)} lectures completed",
             font=("Arial", 12, "bold"), bg=BG_DARK, fg=ACCENT_GREEN).pack(pady=(10, 5))

    # Scrollable list
    canvas = tk.Canvas(root, bg=BG_DARK, highlightthickness=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg=BG_DARK)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True, padx=(30, 0), pady=10)
    scrollbar.pack(side="right", fill="y", pady=10)

    def select_video(vid_id):
        selected_id[0] = vid_id
        root.destroy()

    for entry in sorted(curriculum, key=lambda x: x["order"]):
        is_done = entry["id"] in completed_ids
        exists = os.path.exists(entry["video_path"])

        card = tk.Frame(scroll_frame, bg=BG_CARD, padx=20, pady=15, relief="ridge", bd=2)
        card.pack(fill="x", pady=6, padx=10)

        # Top row: order number + topic + status
        top = tk.Frame(card, bg=BG_CARD)
        top.pack(fill="x")

        status = "  (completed)" if is_done else ""
        status_color = ACCENT_GREEN if is_done else WHITE
        tk.Label(top, text=f"{entry['order']}. {entry['topic_name']}{status}",
                 font=("Arial", 15, "bold"), bg=BG_CARD, fg=status_color,
                 anchor="w").pack(side="left")

        # Description
        desc = entry.get("description", "")
        if len(desc) > 120:
            desc = desc[:120] + "..."
        tk.Label(card, text=desc, font=("Arial", 11), bg=BG_CARD, fg=GRAY,
                 wraplength=700, justify="left", anchor="w").pack(fill="x", pady=(5, 8))

        # Button (tk.Label to fix macOS visibility)
        if exists:
            btn_text = "Re-watch" if is_done else "Start Lecture"
            btn_color = ACCENT_BLUE if not is_done else "#555555"
            btn = tk.Label(card, text=btn_text, font=("Arial", 12, "bold"),
                           bg=btn_color, fg=WHITE, width=16, cursor="hand2",
                           padx=10, pady=6, relief="raised")
            btn.pack(anchor="e")
            btn.bind("<Button-1>", lambda e, vid=entry["id"]: select_video(vid))
        else:
            tk.Label(card, text="Video file not found", font=("Arial", 11, "italic"),
                     bg=BG_CARD, fg="#f44336").pack(anchor="e")

    # Quit button (tk.Label to fix macOS visibility)
    quit_btn = tk.Label(root, text="Quit Tutor", font=("Arial", 12, "bold"),
                        bg="#555555", fg=WHITE, width=14, cursor="hand2",
                        padx=10, pady=8, relief="raised")
    quit_btn.pack(pady=15)
    quit_btn.bind("<Button-1>", lambda e: root.destroy())

    root.mainloop()

    with open(result_path, "w") as f:
        f.write(str(selected_id[0]) if selected_id[0] is not None else "")

run_menu(sys.argv[1], sys.argv[2])
'''


def show_startup_menu(curriculum, completed_ids):
    """Launch the startup menu in a subprocess. Returns selected curriculum entry or None."""
    import tempfile

    data_file = os.path.join(tempfile.gettempdir(), "tutor_menu_data.json")
    result_file = os.path.join(tempfile.gettempdir(), "tutor_menu_result.txt")

    with open(data_file, "w") as f:
        json.dump({"curriculum": curriculum, "completed_ids": list(completed_ids)}, f)

    # Write menu script to temp file and run it
    script_file = os.path.join(tempfile.gettempdir(), "tutor_menu_script.py")
    with open(script_file, "w") as f:
        f.write(MENU_SCRIPT)

    subprocess.run([sys.executable, script_file, data_file, result_file],
                   cwd=os.path.dirname(os.path.abspath(__file__)))

    # Read result
    selected_id = None
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            content = f.read().strip()
            if content:
                try:
                    selected_id = int(content)
                except ValueError:
                    selected_id = None
        os.remove(result_file)

    # Cleanup
    for f in [data_file, script_file]:
        if os.path.exists(f):
            os.remove(f)

    if selected_id is None:
        return None

    # Find the matching curriculum entry
    for entry in curriculum:
        if entry["id"] == selected_id:
            return entry
    return None


def load_curriculum(path="curriculum.json"):
    """Load curriculum from JSON file."""
    if not os.path.exists(path):
        print(f" curriculum.json not found at {path}")
        return []
    with open(path, "r") as f:
        return json.load(f)




def run_full_pipeline(entry, emotion_csv_path=EMOTION_CSV_PATH):
    """Run the full pipeline for a single video: emotion detection → quiz → adaptation."""
    video_path = entry["video_path"]
    curriculum_description = entry.get("description", "")

    print("\n" + "=" * 60)
    print(f"   Lecture: {entry['topic_name']}")
    print("=" * 60)
    print(f" Video: {os.path.basename(video_path)}")

    if not os.path.exists(video_path):
        print(f"\n Video not found: {video_path}")
        return False

    # 1. Play video + detect emotions
    emotion_log, negative_detected, topic_name, video_description, cancelled = detect_emotions_during_video(video_path)

    if cancelled:
        print("\n Video closed early — returning to menu.")
        return False

    # Use curriculum topic name if available (more reliable than Whisper summary)
    if entry.get("topic_name"):
        topic_name = entry["topic_name"]

    # 2. Save emotions to CSV
    save_emotions_to_csv(emotion_log, filename=emotion_csv_path)

    # 3. Emotion summary
    print("\n" + "=" * 60)
    print("    Emotion Detection Summary")
    print("=" * 60)

    if negative_detected:
        unique_phrases = list(set([e["phrase"] for e in emotion_log if e.get("phrase")]))
        print(f"\n Negative emotions detected at {len(emotion_log)} points")
        print(f" Unique confusion phrases: {len(unique_phrases)}")
        print("\nSample confusion points:")
        for phrase in unique_phrases[:3]:
            print(f'   - "{phrase[:60]}..."')
    else:
        print("\n No significant confusion detected")

    # 4. Generate quiz (using both video transcript + curriculum description)
    print("\n" + "=" * 60)
    print("    Generating Targeted Quiz")
    print("=" * 60)

    # Combine video transcript with curriculum description for richer context
    combined_description = video_description
    if curriculum_description:
        combined_description = f"CURRICULUM TOPIC: {curriculum_description}\n\nVIDEO TRANSCRIPT: {video_description}"

    questions = generate_questions_from_csv(
        csv_path=emotion_csv_path,
        num_qs=10,
        fallback_topic=topic_name,
        video_description=combined_description
    )

    if not questions:
        print(" Failed to generate questions")
        return

    print(f" {len(questions)} questions ready - targeting confusion points!")

    # 5. Launch Quiz
    print("\n Launching Quiz...")
    score = launch_quiz_ui(questions)

    # 6. Results and Adaptation
    print("\n" + "=" * 60)
    print("    Results & Recommendations")
    print("=" * 60)
    print(f"\n Final Score: {score}/10")

    action = evaluate_and_adapt(score, topic_name, combined_description)

    if action == "reteach":
        print("\n Review session complete!")
        print("   The student has reviewed the concepts they struggled with.")
    else:
        print("\n Lecture complete!")
        print("   The student understood the material well.")

    return True


# =========================================
#  MAIN ENTRY POINT
# =========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   Empathetic & Adaptive AI Tutor")
    print("=" * 60)

    # Load curriculum
    curriculum = load_curriculum()
    if not curriculum:
        print(" No curriculum found. Create a curriculum.json file.")
        exit()

    print(f" Loaded {len(curriculum)} lectures from curriculum.json")

    completed_ids = set()

    # Main loop: menu → pipeline → back to menu
    while True:
        selected = show_startup_menu(curriculum, completed_ids)

        if selected is None:
            print("\n Tutor closed. Goodbye!")
            break

        print(f"\n Selected: {selected['topic_name']}")
        finished = run_full_pipeline(selected)
        if finished:
            completed_ids.add(selected["id"])
            print(f"\n Completed {len(completed_ids)}/{len(curriculum)} lectures this session.")