import pandas as pd
import os
from datetime import datetime

def save_emotions_to_csv(emotion_log, filename="emotion_data.csv"):
    """
    Save emotion log entries to a CSV file.
    
    Expected format for each entry in emotion_log:
    {
        "timestamp": "00:01:23",      # Time in video when emotion detected
        "emotion": "sad",              # Detected emotion (sad, angry, fear)
        "phrase": "The plant needs..." # What was being said at that moment
    }
    
    CSV Output columns: timestamp, emotion, phrase
    """

    if not emotion_log:
        print(" No emotion data to save.")
        return

    # Normalize structure and ensure all fields exist
    formatted_data = []
    for entry in emotion_log:
        formatted_data.append({
            "timestamp": entry.get("timestamp", datetime.now().strftime("%H:%M:%S")),
            "emotion": entry.get("emotion", "unknown"),
            "phrase": entry.get("phrase", "")  # Now this will be populated!
        })

    df = pd.DataFrame(formatted_data, columns=["timestamp", "emotion", "phrase"])

    # Write or append to CSV
    write_header = not os.path.isfile(filename)
    df.to_csv(filename, mode="a", index=False, header=write_header)
    
    print(f" Emotion data saved to {filename}")
    print(f"   → {len(formatted_data)} entries with phrases logged")
    
    # Show sample of what was saved
    if formatted_data:
        sample = formatted_data[0]
        print(f"   → Sample: [{sample['timestamp']}] {sample['emotion']} - \"{sample['phrase'][:50]}...\"")


def load_emotions_from_csv(filename="emotion_data.csv"):
    """
    Load emotion data from CSV for analysis or question generation.
    
    Returns:
        pandas.DataFrame or None if file doesn't exist
    """
    if not os.path.isfile(filename):
        print(f" No emotion data file found: {filename}")
        return None
    
    df = pd.read_csv(filename)
    print(f" Loaded {len(df)} emotion records from {filename}")
    return df


def get_confusion_phrases(filename="emotion_data.csv"):
    """
    Extract unique phrases where negative emotions were detected.
    Useful for generating targeted quiz questions.
    
    Returns:
        list: Unique phrases where confusion/negative emotions occurred
    """
    df = load_emotions_from_csv(filename)
    if df is None or df.empty:
        return []
    
    # Filter for non-empty phrases
    phrases = df[df["phrase"].notna() & (df["phrase"] != "")]["phrase"].unique().tolist()
    return phrases


def clear_emotion_log(filename="emotion_data.csv"):
    """
    Clear/delete the emotion log file.
    Useful for starting fresh with a new video.
    """
    if os.path.isfile(filename):
        os.remove(filename)
        print(f" Emotion log cleared: {filename}")
    else:
        print(f" No file to clear: {filename}")