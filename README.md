# Empathetic & Adaptive AI Tutor

An AI-powered educational tutoring system that detects student emotions in real-time and adapts the learning experience accordingly. The system uses facial emotion recognition to identify confusion, disengagement, and negative emotions during video lectures, then generates personalized quizzes and gamified review sessions to reinforce learning.

## Features

- **Real-time Emotion Detection** — Uses DeepFace to monitor facial expressions during video lectures, detecting emotions like sadness, anger, fear, confusion, and disengagement
- **Adaptive Quiz Generation** — LLM-generated quizzes tailored to the video content and student's emotional responses (confusion points get extra focus)
- **Gamified Review System** — Four engaging game modes based on quiz performance:
  - **Boss Battle** — Mixed challenges (MCQ, short answer, fill-in-the-blank, unscramble) in an RPG-style battle
  - **Memory Match** — Card-matching game pairing concepts with definitions
  - **Word Scramble** — Unscramble key terms with hints and explanations
  - **Who Wants to Be a Millionaire** — Progressive MCQ game with lifelines (50:50, Ask the Audience)
- **Concept Review Cards** — Teaching phase before games that covers all important concepts from the curriculum
- **Curriculum Management** — JSON-based curriculum system with a startup menu for lecture selection
- **Score-Based Adaptation** — Game difficulty and type are selected based on quiz performance
- **Detailed Explanations** — Wrong answers trigger teaching moments with detailed explanations

## Architecture

```
emotion_detector.py    — Main pipeline: video playback, emotion detection, curriculum menu
question_generator.py  — LLM-powered quiz generation and UI
gamified_review.py     — All four game modes + concept review cards
adaptive_engine.py     — Score evaluation and game routing
data_logger.py         — Emotion and quiz data logging to CSV
video_preprocess.py    — Video transcription using Whisper
curriculum.json        — Curriculum metadata (topics, descriptions, video paths)
```

## Tech Stack

- **Python** with **Tkinter** for all UIs
- **DeepFace** for facial emotion recognition
- **OpenAI Whisper** for video transcription
- **LLM APIs** (OpenAI / Groq / Gemini) for content generation
- **OpenCV** for webcam capture
- **FFplay** for video playback

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Abhineetraj07/Empathetic-and-Adaptive-AI-Tutor.git
   cd Empathetic-and-Adaptive-AI-Tutor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export TUTOR_API_KEY="your-api-key-here"
   export API_PROVIDER="openai"  # or "groq" or "gemini"
   ```

4. Add video files to a `videos/` directory and update `curriculum.json` with the correct paths.

5. Run the tutor:
   ```bash
   python emotion_detector.py
   ```

## How It Works

1. **Startup Menu** — Select a lecture from the curriculum
2. **Video Playback** — Watch the lecture while the system monitors your emotions via webcam
3. **Quiz Phase** — Answer LLM-generated questions based on video content and detected confusion points
4. **Adaptation** — Based on your quiz score:
   - **0-3/10** — Boss Battle (most support needed)
   - **4/10** — Word Scramble or Memory Match
   - **5/10** — Who Wants to Be a Millionaire
   - **6+/10** — Move to next lecture
5. **Concept Review** — Before the game, review flashcards covering key concepts
6. **Gamified Review** — Play the selected game to reinforce learning
7. **Return to Menu** — Select the next lecture or re-watch

## Research Graphs

The `research_graphs/` directory contains visualizations generated from emotion and quiz data:
- Emotion distribution over time
- Emotion frequency analysis
- Quiz performance vs emotional state correlation
