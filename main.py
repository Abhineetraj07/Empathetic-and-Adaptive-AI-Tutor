import os
from emotion_detector import detect_emotions_during_video, video_path
from data_logger import save_emotions_to_csv
from question_generator import generate_questions
from adaptive_engine import evaluate_and_adapt

def main():
    print("\n🎓 Empathetic & Adaptive AI Tutor\n")

    # Step 0: Use same video path from emotion_detector.py
    print(f"Using lecture video from emotion_detector.py:\n   {video_path}")

    if not os.path.exists(video_path):
        print(f"\n Video file not found at: {video_path}")
        print("Please make sure the file exists or update 'video_path' inside emotion_detector.py")
        return

    # Step 1: Detect emotions during video
    emotion_log, negative_detected = detect_emotions_during_video(video_path)
    save_emotions_to_csv(emotion_log)

    # Step 2: Generate quiz based on detected topic
    topic_name = emotion_log[0]["topic"] if emotion_log else "Current Lecture"

    if negative_detected:
        print("\n Negative emotions detected — generating topic-specific quiz.")
    else:
        print("\n No confusion detected — verifying understanding with quiz.")

    questions = generate_questions(topic_name)
    for i, q in enumerate(questions, start=1):
        print(f"Q{i}: {q}")

    # Step 3: Get learner’s score and adapt
    try:
        score = int(input("\nEnter learner quiz score (out of 10): "))
    except ValueError:
        print(" Invalid score entered. Defaulting to 0.")
        score = 0

    action = evaluate_and_adapt(score, topic_name)

    if action == "reteach":
        print(" Replay same lecture with gamified reinforcement.\n")
    else:
        print(" Proceeding to next lecture.\n")

if __name__ == "__main__":
    main()