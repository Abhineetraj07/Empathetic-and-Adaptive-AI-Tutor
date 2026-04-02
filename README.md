<h1 align="center">🧠 Empathetic & Adaptive AI Tutor</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/DeepFace-Emotion_Detection-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Whisper-Transcription-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Quiz_Generation-10B981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tkinter-UI-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  An AI-powered tutoring system that <b>watches how you feel</b> while you learn — detecting confusion, disengagement, and frustration in real-time via webcam — and <b>adapts the learning experience</b> with personalized quizzes and gamified review sessions.
</p>

---

## ✨ What Makes This Different

Most learning platforms deliver the same content regardless of how you're responding to it. This system is different:

- 👁️ **Sees your emotions** — DeepFace monitors your facial expressions during video playback
- 🧩 **Targets your confusion** — LLM generates quizzes focused on timestamps where you looked confused
- 🎮 **Adapts the game** — your quiz score determines which of 4 game modes you play
- 📊 **Tracks everything** — emotion + quiz data logged for research and improvement

---

## 🎮 Four Game Modes

| Score | Game | Description |
|-------|------|-------------|
| 0–3 / 10 | **Boss Battle** | RPG-style mixed challenge (MCQ, fill-in-blank, unscramble, short answer) |
| 4 / 10 | **Memory Match** or **Word Scramble** | Card-pair matching or key term unscrambling with hints |
| 5 / 10 | **Who Wants to Be a Millionaire** | Progressive MCQ with 50:50 and Ask-the-Audience lifelines |
| 6+ / 10 | ✅ Next Lecture | You're ready — move on! |

---

## 🔄 How It Works — Full Flow

```
1. STARTUP MENU
   └── Select lecture from curriculum (curriculum.json)

2. VIDEO PLAYBACK
   └── Watch lecture video (FFplay)
   └── Webcam ON → DeepFace detects emotions per frame
   └── Flagged timestamps: sadness, anger, fear, confusion, disengagement

3. QUIZ PHASE
   └── Whisper transcribes lecture audio
   └── LLM generates 10 questions → confusion timestamps get extra focus
   └── Student answers questions in Tkinter UI

4. ADAPTATION ENGINE
   └── Score evaluated → game mode selected
   └── Concept Review Cards shown (flashcard teaching phase)

5. GAMIFIED REVIEW
   └── One of 4 games played
   └── Wrong answers → detailed teaching moment shown

6. RETURN TO MENU
   └── Next lecture or re-watch current
```

---

## 🏗️ Architecture

```
emotion_detector.py     — Main pipeline: video playback + webcam emotion loop
question_generator.py   — Whisper transcription + LLM quiz generation + UI
gamified_review.py      — All 4 game modes + concept review cards
adaptive_engine.py      — Score → game mode routing logic
data_logger.py          — CSV logging of emotions and quiz performance
video_preprocess.py     — Video audio extraction + Whisper transcription
curriculum.json         — Lecture metadata (topics, descriptions, video paths)
Generate_research_graphs.py — Visualize emotion/performance data
research_graphs/        — Output charts and analysis
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-Emotion_AI-FF6F00?style=flat-square)
![Whisper](https://img.shields.io/badge/OpenAI_Whisper-Transcription-412991?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-Webcam-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-UI-1565C0?style=flat-square)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI%2FGroq%2FGemini-412991?style=flat-square&logo=openai&logoColor=white)

**Supported LLM providers:** OpenAI · Groq · Gemini (swappable via env var)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Abhineetraj07/Empathetic-and-Adaptive-AI-Tutor.git
cd Empathetic-and-Adaptive-AI-Tutor
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export TUTOR_API_KEY="your-api-key-here"
export API_PROVIDER="openai"   # or "groq" or "gemini"
```

### 3. Add Your Lectures

Place `.mp4` video files in a `videos/` folder and update `curriculum.json`:

```json
{
  "lectures": [
    {
      "id": 1,
      "title": "Introduction to Neural Networks",
      "description": "Basics of perceptrons and backpropagation",
      "video_path": "videos/neural_networks_intro.mp4"
    }
  ]
}
```

### 4. Run

```bash
python emotion_detector.py
```

---

## 📊 Research & Analytics

The `research_graphs/` directory contains auto-generated visualizations:
- Emotion distribution over time during lectures
- Emotion frequency analysis per student session
- Quiz performance vs emotional state correlation
- Confusion heatmaps by lecture timestamp

Run `python Generate_research_graphs.py` after a session to generate graphs.

---

## 🔬 Research Potential

This system generates rich datasets correlating **facial emotion signals** with **learning performance** — useful for:
- EdTech product research
- Human-Computer Interaction studies
- Affective computing benchmarks
- Adaptive learning algorithm research

---

## 👨‍💻 Author

**Abhineet Raj** · CS @ SRM Institute of Science & Technology
🌐 [Portfolio](https://aabhineet07-portfolio.netlify.app/) · 🐙 [GitHub](https://github.com/Abhineetraj07)

---

## 📄 License

MIT License
