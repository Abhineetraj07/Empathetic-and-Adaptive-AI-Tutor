"""
AI-Driven Gamified Review System

The AI analyzes what the student got wrong and automatically picks the best game:
- Boss Battle: Many wrong answers, needs comprehensive review
- Memory Match: Confused similar terms, needs association-building
- Word Scramble: Vocabulary/terminology gaps
- Millionaire: Close to passing, targeted reinforcement
"""

import tkinter as tk
from tkinter import messagebox
import json
import os
import re
import random
import time
import math
import pandas as pd

from question_generator import call_llm

# =========================================
#  CONSTANTS
# =========================================
BG_DARK = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT_RED = "#e94560"
ACCENT_GREEN = "#4CAF50"
ACCENT_BLUE = "#2196F3"
ACCENT_ORANGE = "#FF9800"
ACCENT_PURPLE = "#9C27B0"
GOLD = "#ffd700"
WHITE = "#ffffff"
GRAY = "#aaaaaa"
FONT_TITLE = ("Arial", 22, "bold")
FONT_LARGE = ("Arial", 18, "bold")
FONT_MEDIUM = ("Arial", 14)
FONT_SMALL = ("Arial", 12)


# =========================================
#  DATA EXTRACTION
# =========================================
def get_wrong_topics(quiz_csv="quiz_results.csv"):
    """Extract questions the student got wrong from the most recent session."""
    if not os.path.exists(quiz_csv):
        return []
    df = pd.read_csv(quiz_csv)
    if df.empty:
        return []
    latest_timestamp = df["timestamp"].max()
    session = df[df["timestamp"] == latest_timestamp]
    wrong = session[session["is_correct"] == False]

    # Map letter to column for actual option text
    letter_to_col = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}

    topics = []
    for _, row in wrong.iterrows():
        student_letter = str(row["student_answer"]).strip().upper()
        correct_letter = str(row["correct_answer"]).strip().upper()

        # Get actual text of the answers, not just the letter
        student_text = row.get(letter_to_col.get(student_letter, ""), student_letter)
        correct_text = row.get(letter_to_col.get(correct_letter, ""), correct_letter)

        # Also grab all 4 options for potential MCQ use
        all_options = [
            str(row.get("option_a", "A")),
            str(row.get("option_b", "B")),
            str(row.get("option_c", "C")),
            str(row.get("option_d", "D")),
        ]

        topics.append({
            "question": row["question_text"],
            "student_answer": student_text,
            "correct_answer": correct_text,
            "correct_letter": correct_letter,
            "all_options": all_options,
        })
    return topics


def get_session_score(quiz_csv="quiz_results.csv"):
    """Get the numeric score from the most recent quiz session."""
    if not os.path.exists(quiz_csv):
        return 0
    df = pd.read_csv(quiz_csv)
    if df.empty:
        return 0
    latest_timestamp = df["timestamp"].max()
    session = df[df["timestamp"] == latest_timestamp]
    return int(session["is_correct"].sum())


def get_confusion_phrases(emotion_csv="emotion_data.csv"):
    """Get unique confusion phrases from emotion data."""
    if not os.path.exists(emotion_csv):
        return []
    df = pd.read_csv(emotion_csv)
    if "phrase" not in df.columns:
        return []
    phrases = df["phrase"].dropna().unique().tolist()
    seen = set()
    clean = []
    for p in phrases:
        p = p.strip()
        if p and p not in seen and len(p) > 10:
            seen.add(p)
            clean.append(p)
    return clean


# =========================================
#  LLM GAME ORCHESTRATOR
# =========================================
def parse_llm_json(response):
    """Robustly extract JSON object from LLM response."""
    clean = response.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _parse_llm_array(response):
    """Extract JSON array from LLM response."""
    clean = response.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\[.*\]', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _build_boss_battle_challenges(wrong_topics, topic_name, video_description="", confusion_phrases=None, curriculum_description=""):
    """Generate boss battle challenges with GUARANTEED mix of all 4 types.
    Makes 4 separate LLM calls — one per challenge type, 2 items each.
    Then interleaves them so the student gets variety every round."""

    wrong_summary = "\n".join(
        [f"- {t['question']} (correct: {t['correct_answer']})" for t in wrong_topics]
    )

    # Build context blocks
    curriculum_context = ""
    if curriculum_description:
        curriculum_context = f"\nCURRICULUM METADATA (what this lecture covers — use as primary source):\n{curriculum_description}\n"

    # Strip out curriculum part if embedded in video_description to avoid duplication
    clean_transcript = video_description
    if clean_transcript and "VIDEO TRANSCRIPT:" in clean_transcript:
        clean_transcript = clean_transcript.split("VIDEO TRANSCRIPT:", 1)[-1].strip()
    elif clean_transcript and "CURRICULUM TOPIC:" in clean_transcript:
        clean_transcript = clean_transcript.split("\n\n", 1)[-1].strip()

    video_context = ""
    if clean_transcript:
        truncated_desc = clean_transcript[:1000] if len(clean_transcript) > 1000 else clean_transcript
        video_context = f"\nVIDEO TRANSCRIPT (what was actually said):\n{truncated_desc}\n"

    phrase_context = ""
    if confusion_phrases:
        phrase_context = "\nPHRASES WHERE STUDENT SHOWED CONFUSION:\n" + "\n".join(
            [f"- \"{p}\"" for p in confusion_phrases[:6]]
        ) + "\n"

    base_context = f"""Topic: "{topic_name}"
{curriculum_context}{video_context}{phrase_context}
Questions the student got WRONG in the quiz (do NOT repeat these):
{wrong_summary}

RULES:
- Use CURRICULUM METADATA as primary source for topics to cover.
- Cover the FULL topic, not just wrong answers. The student should learn ALL key concepts.
- Every question must be UNIQUE — no two should test the same concept or be worded similarly.
- Do NOT repeat or rephrase the quiz questions listed above."""

    challenges = []

    # --- 1. MCQs (2 items) ---
    print("   Generating MCQ challenges...")
    mcq_prompt = f"""{base_context}

Generate exactly 2 multiple-choice questions. Each must have 4 options that are factually accurate, clearly distinct, and not overlapping. No "All of the above" or "None of the above".

Return ONLY a JSON array:
[{{"type": "mcq", "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "B", "explanation": "2-3 sentence teaching explanation of WHY this is correct and what the concept means"}}]"""

    mcq_items = _parse_llm_array(call_llm(mcq_prompt))
    for item in mcq_items[:2]:
        item["type"] = "mcq"
        challenges.append(item)

    # --- 2. Type Answer (2 items) ---
    print("   Generating Type Answer challenges...")
    type_prompt = f"""{base_context}

Generate exactly 2 questions where the student must TYPE the answer (one word or short phrase). The answer should be a key term from the video content.

Return ONLY a JSON array:
[{{"type": "type_answer", "question": "What process involves breaking down food into simpler substances?", "answer": "digestion", "hint": "starts with D"}}]"""

    type_items = _parse_llm_array(call_llm(type_prompt))
    for item in type_items[:2]:
        item["type"] = "type_answer"
        challenges.append(item)

    # --- 3. Fill Blank (2 items) ---
    print("   Generating Fill Blank challenges...")
    blank_prompt = f"""{base_context}

Generate exactly 2 fill-in-the-blank sentences. Each has ONE blank (marked ___) where a key term goes. The answer should be a single word.

Return ONLY a JSON array:
[{{"type": "fill_blank", "sentence": "The process of ___ involves exchange of gases in lungs.", "answer": "respiration", "hint": "starts with R"}}]"""

    blank_items = _parse_llm_array(call_llm(blank_prompt))
    for item in blank_items[:2]:
        item["type"] = "fill_blank"
        challenges.append(item)

    # --- 4. Unscramble (2 items) ---
    print("   Generating Unscramble challenges...")
    scramble_prompt = f"""{base_context}

Generate exactly 2 word unscramble challenges. Pick key terms with 5+ letters from the video content. The clue should be an indirect question (NOT the definition). Scramble the letters of the word.

Return ONLY a JSON array:
[{{"type": "unscramble", "clue": "What happens to food after you eat it?", "scrambled_word": "GIDNOIEST", "answer": "digestion", "hint": "9 letters"}}]"""

    scramble_items = _parse_llm_array(call_llm(scramble_prompt))
    for item in scramble_items[:2]:
        item["type"] = "unscramble"
        challenges.append(item)

    # --- Interleave: take one from each type in rotation ---
    by_type = {"mcq": [], "type_answer": [], "fill_blank": [], "unscramble": []}
    for c in challenges:
        t = c.get("type", "mcq")
        if t in by_type:
            by_type[t].append(c)

    interleaved = []
    for round_num in range(2):  # 2 rounds of 4 types = 8 challenges
        for t in ["mcq", "type_answer", "fill_blank", "unscramble"]:
            if round_num < len(by_type[t]):
                interleaved.append(by_type[t][round_num])

    # Shuffle within each pair to keep it unpredictable
    if len(interleaved) >= 8:
        first_half = interleaved[:4]
        second_half = interleaved[4:]
        random.shuffle(first_half)
        random.shuffle(second_half)
        interleaved = first_half + second_half

    return interleaved if interleaved else challenges



def _generate_single_game(game_type, wrong_topics, topic_name, video_description="", confusion_phrases=None, curriculum_description=""):
    """Generate content for a single game type, grounded in curriculum metadata + video content + confusion data."""

    wrong_summary = "\n".join(
        [f"- {t['question']} (correct: {t['correct_answer']})" for t in wrong_topics]
    )

    # Build shared context blocks
    curriculum_context = ""
    if curriculum_description:
        curriculum_context = f"\nCURRICULUM METADATA (what this lecture covers — use this as primary source for topics):\n{curriculum_description}\n"

    # Strip out curriculum part if it was embedded in video_description to avoid duplication
    clean_transcript = video_description
    if clean_transcript and "VIDEO TRANSCRIPT:" in clean_transcript:
        clean_transcript = clean_transcript.split("VIDEO TRANSCRIPT:", 1)[-1].strip()
    elif clean_transcript and "CURRICULUM TOPIC:" in clean_transcript:
        clean_transcript = clean_transcript.split("\n\n", 1)[-1].strip()

    video_context = ""
    if clean_transcript:
        truncated_desc = clean_transcript[:1000] if len(clean_transcript) > 1000 else clean_transcript
        video_context = f"\nVIDEO TRANSCRIPT (what was actually said in the lecture):\n{truncated_desc}\n"

    phrase_context = ""
    if confusion_phrases:
        phrase_context = "\nPHRASES WHERE STUDENT SHOWED CONFUSION:\n" + "\n".join(
            [f"- \"{p}\"" for p in confusion_phrases[:6]]
        ) + "\n"

    base_context = f"""Topic: "{topic_name}"
{curriculum_context}{video_context}{phrase_context}
Questions the student got WRONG in the quiz (do NOT repeat these — generate completely different questions):
{wrong_summary}

RULES:
- Use the CURRICULUM METADATA as the primary source for what topics to cover.
- Use the VIDEO TRANSCRIPT for specific facts, terms, and examples mentioned in the lecture.
- Cover the FULL topic — not just what the student got wrong. The student should learn ALL key concepts.
- Every question must be UNIQUE — no two questions should test the same concept or be worded similarly.
- Do NOT repeat or rephrase the quiz questions listed above. Generate fresh, different questions."""

    if game_type == "boss_battle":
        challenges = _build_boss_battle_challenges(
            wrong_topics, topic_name,
            video_description=video_description,
            confusion_phrases=confusion_phrases,
            curriculum_description=curriculum_description
        )
        # Generate a thematic boss name
        boss_names = [
            f"The {topic_name} Guardian", f"Professor {topic_name.split()[0]}",
            f"The Knowledge Keeper", f"The Quiz Dragon"
        ]
        return {
            "game_type": "boss_battle",
            "reason": "comprehensive review for low score",
            "boss_name": random.choice(boss_names),
            "boss_taunt": "Let's see if you can defeat me this time!",
            "challenges": challenges
        }

    elif game_type == "memory_match":
        prompt = f"""{base_context}

Generate exactly 8 term-definition pairs for a memory match card game. Terms should be key concepts from the curriculum. Definitions should be concise (under 10 words).

Return ONLY a JSON array:
[{{"term": "Kharif Crops", "definition": "Crops grown in rainy season (June-Sept)"}}]"""

        pairs = _parse_llm_array(call_llm(prompt))
        return {"game_type": "memory_match", "reason": "build term associations", "pairs": pairs}

    elif game_type == "word_scramble":
        prompt = f"""{base_context}

Generate exactly 8 word scramble items. Pick key terms with 5+ letters from the curriculum. The clue should be an indirect question that makes the student think, NOT a definition. Scramble the letters.

Return ONLY a JSON array:
[{{"word": "digestion", "clue": "What happens to food after you eat it?", "hint": "starts with D", "category": "Biology", "meaning": "The process of breaking down food into nutrients your body can absorb and use for energy."}}]"""

        words = _parse_llm_array(call_llm(prompt))
        return {"game_type": "word_scramble", "reason": "vocabulary reinforcement", "words": words}

    elif game_type == "millionaire":
        prompt = f"""{base_context}

Generate exactly 8 MCQs with increasing difficulty (1=easy to 8=hard). Questions should be grounded in the curriculum. Options must be factually accurate, clearly distinct, no "All/None of the above".

Return ONLY a JSON array:
[{{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "B", "explanation": "2-3 sentence teaching explanation of WHY this is correct and what the concept means", "difficulty": 1}}]"""

        questions = _parse_llm_array(call_llm(prompt))
        return {"game_type": "millionaire", "reason": "targeted reinforcement", "questions": questions}

    return {"game_type": game_type, "reason": "fallback"}


def _generate_concept_cards(wrong_topics, topic_name, video_description="", confusion_phrases=None, curriculum_description=""):
    """Generate teaching concept cards covering the full topic + what the student got wrong."""
    wrong_summary = "\n".join(
        [f"- Q: {t['question']}\n  Student answered: {t['student_answer']}, Correct: {t['correct_answer']}"
         for t in wrong_topics]
    )

    curriculum_context = ""
    if curriculum_description:
        curriculum_context = f"\nCURRICULUM METADATA (full topic coverage — use this to decide what to teach):\n{curriculum_description}\n"

    video_context = ""
    if video_description:
        truncated = video_description[:800] if len(video_description) > 800 else video_description
        video_context = f"\nVIDEO TRANSCRIPT:\n{truncated}\n"

    phrase_context = ""
    if confusion_phrases:
        phrase_context = "\nPHRASES WHERE STUDENT WAS CONFUSED:\n" + "\n".join(
            [f"- \"{p}\"" for p in confusion_phrases[:5]]
        ) + "\n"

    prompt = f"""You are a patient, friendly teacher. A student just scored poorly on a quiz about "{topic_name}".
{curriculum_context}{video_context}{phrase_context}
QUESTIONS THE STUDENT GOT WRONG:
{wrong_summary}

Read the CURRICULUM METADATA description carefully. It tells you exactly what important concepts this lecture covers. Create a teaching card for each important concept mentioned in the description.

For concepts the student got wrong, explain why the correct answer is right.
For all other important concepts, teach them clearly.

Generate one card for EVERY important concept in the curriculum description. Do not limit the number — cover all topics until everything important is included.

Rules:
- Use simple, friendly language (imagine explaining to a curious 13-year-old)
- Each explanation should be 2-4 sentences
- Include a concrete example or analogy for each concept
- Do NOT just restate quiz questions — actually teach the concept

Return ONLY a JSON array:
[{{"title": "What is Respiration?", "explanation": "Respiration is the process where your body uses oxygen to break down glucose (sugar from food) and release energy. Think of it like burning fuel in a car engine — your body 'burns' glucose with oxygen to power everything you do.", "example": "When you run fast and start breathing heavily, that's your body demanding more oxygen for respiration to produce extra energy for your muscles."}}]"""

    cards = _parse_llm_array(call_llm(prompt))
    if cards:
        print(f"   Generated {len(cards)} concept review cards")
    return cards


def generate_fallback_content(wrong_topics, topic_name):
    """Fallback: generate boss_battle challenges directly from wrong answers."""
    challenges = []
    types = ["type_answer", "fill_blank", "mcq", "type_answer"]
    for i, t in enumerate(wrong_topics):
        # Extract clean answer text (strip "A) " prefix if present)
        correct_text = re.sub(r'^[A-D]\)\s*', '', str(t['correct_answer']))
        c_type = types[i % len(types)]

        if c_type == "type_answer":
            challenges.append({
                "type": "type_answer",
                "question": t["question"],
                "answer": correct_text,
                "hint": f"Starts with '{correct_text[0]}'" if correct_text else ""
            })
        elif c_type == "fill_blank":
            challenges.append({
                "type": "fill_blank",
                "sentence": t["question"].replace("?", " is ___."),
                "answer": correct_text,
                "hint": f"Starts with '{correct_text[0]}'" if correct_text else ""
            })
        else:
            # Use actual options from the quiz if available
            options = t.get("all_options", [])
            if options and len(options) == 4:
                challenges.append({
                    "type": "mcq",
                    "question": t["question"],
                    "options": options,
                    "correct": t.get("correct_letter", "A"),
                    "explanation": f"The correct answer is {correct_text}"
                })
            else:
                # If no options, fall back to type_answer
                challenges.append({
                    "type": "type_answer",
                    "question": t["question"],
                    "answer": correct_text,
                    "hint": f"Starts with '{correct_text[0]}'" if correct_text else ""
                })
    fallback_game = {
        "game_type": "boss_battle",
        "reason": "Fallback mode - comprehensive review",
        "boss_name": f"The {topic_name} Guardian",
        "boss_taunt": "Let's see what you've really learned!",
        "challenges": challenges
    }
    return {"games": [fallback_game]}


# =========================================
#  BASE GAME CLASS
# =========================================
class BaseGame:
    """Shared functionality for all game types."""

    def __init__(self, root, game_data, topic_name):
        self.root = root
        self.game_data = game_data
        self.topic_name = topic_name
        self.total_points = 0
        self.streak = 0

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_header(self, title, subtitle="", color=ACCENT_RED):
        header = tk.Frame(self.root, bg=color, pady=12)
        header.pack(fill="x")
        tk.Label(header, text=title, font=FONT_TITLE, bg=color, fg=WHITE).pack()
        if subtitle:
            tk.Label(header, text=subtitle, font=("Arial", 11, "italic"),
                     bg=color, fg="#ffcccc").pack()

        points_frame = tk.Frame(self.root, bg=BG_DARK, pady=6)
        points_frame.pack(fill="x")
        self.points_label = tk.Label(
            points_frame,
            text=f"Points: {self.total_points}  |  Streak: {self.streak}",
            font=("Arial", 14, "bold"), bg=BG_DARK, fg=GOLD
        )
        self.points_label.pack()

    def update_points(self, earned, correct=True):
        if correct:
            self.streak += 1
            bonus = min(self.streak - 1, 5) * 5
            self.total_points += earned + bonus
        else:
            self.streak = 0
            self.total_points += earned
        if hasattr(self, 'points_label'):
            streak_text = f" STREAK x{self.streak}!" if self.streak >= 3 else ""
            self.points_label.config(
                text=f"Points: {self.total_points}  |  Streak: {self.streak}{streak_text}"
            )

    def show_completion(self, title, stats):
        self.clear_screen()
        self.show_header(title, color=ACCENT_GREEN)

        frame = tk.Frame(self.root, bg=BG_CARD, padx=40, pady=30, relief="ridge", bd=2)
        frame.pack(expand=True, padx=60, pady=30)

        tk.Label(frame, text=f"Total Points: {self.total_points}",
                 font=("Arial", 28, "bold"), bg=BG_CARD, fg=GOLD).pack(pady=15)

        for label, value in stats.items():
            tk.Label(frame, text=f"{label}: {value}",
                     font=FONT_MEDIUM, bg=BG_CARD, fg=WHITE).pack(pady=3)

        done_btn = tk.Label(
            self.root, text="Done", font=("Arial", 14, "bold"),
            bg=ACCENT_GREEN, fg=WHITE, width=16, padx=10, pady=12,
            cursor="hand2", relief="raised"
        )
        done_btn.pack(pady=20)
        done_btn.bind("<Button-1>", lambda e: self.root.destroy())


# =========================================
#  CONCEPT REVIEW (Teaching Phase)
# =========================================
class ConceptReviewUI:
    """Shows concept explanation cards BEFORE the game starts. Teaches, not tests."""

    def __init__(self, root, game_data, topic_name):
        self.root = root
        self.cards = game_data.get("cards", [])
        self.topic_name = topic_name
        self.card_index = 0
        self.total_points = 0  # Compatibility with game runner
        self.show_card()

    def show_card(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.configure(bg=BG_DARK)

        if self.card_index >= len(self.cards):
            self.show_done()
            return

        card = self.cards[self.card_index]

        # Header
        header = tk.Frame(self.root, bg="#1B5E20", pady=12)
        header.pack(fill="x")
        tk.Label(header, text="Concept Review", font=FONT_TITLE,
                 bg="#1B5E20", fg=WHITE).pack()
        tk.Label(header, text="Learn these concepts before the game!",
                 font=("Arial", 11, "italic"), bg="#1B5E20", fg="#c8e6c9").pack()

        # Progress
        tk.Label(self.root,
                 text=f"Card {self.card_index + 1} of {len(self.cards)}",
                 font=("Arial", 12), bg=BG_DARK, fg=GRAY).pack(pady=(10, 5))

        # Card frame
        card_frame = tk.Frame(self.root, bg=BG_CARD, padx=30, pady=25, relief="ridge", bd=2)
        card_frame.pack(fill="both", expand=True, padx=40, pady=10)

        # Title
        title = card.get("title", f"Concept {self.card_index + 1}")
        tk.Label(card_frame, text=title, font=("Arial", 18, "bold"),
                 bg=BG_CARD, fg=GOLD, wraplength=700).pack(pady=(5, 15))

        # Explanation
        explanation = card.get("explanation", "")
        tk.Label(card_frame, text=explanation, font=("Arial", 14),
                 bg=BG_CARD, fg=WHITE, wraplength=700, justify="left").pack(fill="x", pady=5)

        # Example (if available)
        example = card.get("example", "")
        if example:
            ex_frame = tk.Frame(card_frame, bg="#0f3460", padx=15, pady=10, relief="ridge", bd=1)
            ex_frame.pack(fill="x", pady=(15, 5))
            tk.Label(ex_frame, text="Example:", font=("Arial", 12, "bold"),
                     bg="#0f3460", fg=ACCENT_ORANGE).pack(anchor="w")
            tk.Label(ex_frame, text=example, font=("Arial", 13),
                     bg="#0f3460", fg=WHITE, wraplength=660, justify="left").pack(anchor="w")

        # Navigation button
        is_last = self.card_index >= len(self.cards) - 1
        btn_text = "Start Game!" if is_last else "Next Concept"
        btn_color = ACCENT_GREEN if is_last else ACCENT_BLUE

        nav_btn = tk.Label(self.root, text=btn_text, font=("Arial", 14, "bold"),
                           bg=btn_color, fg=WHITE, width=18, padx=10, pady=12,
                           cursor="hand2", relief="raised")
        nav_btn.pack(pady=15)
        nav_btn.bind("<Button-1>", lambda e: self.next_card())

    def next_card(self):
        self.card_index += 1
        self.show_card()

    def show_done(self):
        self.root.destroy()


# =========================================
#  BOSS BATTLE GAME (Mixed Mini-Challenges)
# =========================================
class BossBattleGame(BaseGame):

    def __init__(self, root, game_data, topic_name):
        super().__init__(root, game_data, topic_name)
        self.challenges = game_data.get("challenges", [])
        # Backward compat: old "questions" format → treat as mcq challenges
        if not self.challenges and game_data.get("questions"):
            self.challenges = [
                {**q, "type": "mcq"} for q in game_data["questions"]
            ]
        self.boss_name = game_data.get("boss_name", "The Knowledge Boss")
        self.boss_taunt = game_data.get("boss_taunt", "Prepare yourself!")
        self.c_index = 0
        self.boss_hp = 100
        self.player_hp = 100
        self.boss_max_hp = 100
        self.player_max_hp = 100
        self.damage_per_hit = max(12, 100 // max(len(self.challenges), 1))
        self.boss_damage = 20
        self.correct_count = 0
        self.show_intro()

    def show_intro(self):
        self.clear_screen()
        self.root.configure(bg="#0d0d1a")

        tk.Label(self.root, text="BOSS BATTLE",
                 font=("Arial", 32, "bold"), bg="#0d0d1a", fg=ACCENT_RED).pack(pady=(50, 10))
        tk.Label(self.root, text=f"VS  {self.boss_name}",
                 font=("Arial", 24, "bold"), bg="#0d0d1a", fg=GOLD).pack(pady=10)
        tk.Label(self.root, text=f'"{self.boss_taunt}"',
                 font=("Arial", 14, "italic"), bg="#0d0d1a", fg=GRAY,
                 wraplength=600).pack(pady=20)

        challenge_types = set(c.get("type", "mcq") for c in self.challenges)
        type_labels = {"mcq": "MCQ", "type_answer": "Type Answer",
                       "fill_blank": "Fill Blank", "unscramble": "Unscramble"}
        types_str = ", ".join(type_labels.get(t, t) for t in challenge_types)

        tk.Label(self.root, text=f"Mixed challenges ahead: {types_str}\n"
                 f"Answer correctly to deal damage!\nWrong answers let the boss attack!",
                 font=FONT_MEDIUM, bg="#0d0d1a", fg=WHITE, justify="center").pack(pady=20)

        fight_btn = tk.Label(
            self.root, text="FIGHT!", font=("Arial", 18, "bold"),
            bg=ACCENT_RED, fg=WHITE, width=14, padx=10, pady=14,
            cursor="hand2", relief="raised"
        )
        fight_btn.pack(pady=30)
        fight_btn.bind("<Button-1>", lambda e: self.show_round())

    # ---- Health bars (reused every round) ----
    def draw_health_bars(self):
        hp_frame = tk.Frame(self.root, bg=BG_DARK, pady=8)
        hp_frame.pack(fill="x", padx=40)

        # Boss HP
        tk.Label(hp_frame, text=self.boss_name, font=("Arial", 12, "bold"),
                 bg=BG_DARK, fg=ACCENT_RED).pack(anchor="w")
        self.boss_canvas = tk.Canvas(hp_frame, width=780, height=25, bg="#333", highlightthickness=0)
        self.boss_canvas.pack(fill="x", pady=(2, 5))
        bw = int(780 * self.boss_hp / self.boss_max_hp)
        self.boss_bar = self.boss_canvas.create_rectangle(0, 0, bw, 25, fill="#f44336")
        self.boss_hp_label = tk.Label(hp_frame, text=f"HP: {self.boss_hp}/{self.boss_max_hp}",
                                       font=("Arial", 10), bg=BG_DARK, fg=ACCENT_RED)
        self.boss_hp_label.pack(anchor="e")

        # Player HP
        tk.Label(hp_frame, text="You", font=("Arial", 12, "bold"),
                 bg=BG_DARK, fg=ACCENT_GREEN).pack(anchor="w", pady=(5, 0))
        self.player_canvas = tk.Canvas(hp_frame, width=780, height=25, bg="#333", highlightthickness=0)
        self.player_canvas.pack(fill="x", pady=2)
        pw = int(780 * self.player_hp / self.player_max_hp)
        self.player_bar = self.player_canvas.create_rectangle(0, 0, pw, 25, fill=ACCENT_GREEN)
        self.player_hp_label = tk.Label(hp_frame, text=f"HP: {self.player_hp}/{self.player_max_hp}",
                                         font=("Arial", 10), bg=BG_DARK, fg=ACCENT_GREEN)
        self.player_hp_label.pack(anchor="e")

    def update_hp_bars(self):
        bw = int(780 * self.boss_hp / self.boss_max_hp)
        self.boss_canvas.coords(self.boss_bar, 0, 0, bw, 25)
        self.boss_hp_label.config(text=f"HP: {self.boss_hp}/{self.boss_max_hp}")
        pw = int(780 * self.player_hp / self.player_max_hp)
        self.player_canvas.coords(self.player_bar, 0, 0, pw, 25)
        self.player_hp_label.config(text=f"HP: {self.player_hp}/{self.player_max_hp}")

    # ---- Main round dispatcher ----
    def show_round(self):
        self.clear_screen()
        self.show_header("Boss Battle", f"VS {self.boss_name}", ACCENT_RED)

        if self.c_index >= len(self.challenges):
            self.victory()
            return

        self.draw_health_bars()

        challenge = self.challenges[self.c_index]
        c_type = challenge.get("type", "mcq")

        # Challenge type label
        type_labels = {"mcq": "MCQ Attack", "type_answer": "Type Your Answer",
                       "fill_blank": "Fill the Blank", "unscramble": "Unscramble Attack"}
        tk.Label(self.root, text=f"Round {self.c_index + 1}  —  {type_labels.get(c_type, c_type)}",
                 font=("Arial", 12, "bold"), bg=BG_DARK, fg=ACCENT_ORANGE).pack(pady=(8, 0))

        if c_type == "mcq":
            self._render_mcq(challenge)
        elif c_type == "type_answer":
            self._render_type_answer(challenge)
        elif c_type == "fill_blank":
            self._render_fill_blank(challenge)
        elif c_type == "unscramble":
            self._render_unscramble(challenge)
        else:
            self._render_type_answer(challenge)  # fallback

    # ---- MCQ challenge ----
    def _render_mcq(self, challenge):
        q_frame = tk.Frame(self.root, bg=BG_CARD, padx=25, pady=15, relief="ridge", bd=2)
        q_frame.pack(fill="x", padx=40, pady=10)

        tk.Label(q_frame, text=challenge.get("question", ""),
                 font=("Arial", 14, "bold"), bg=BG_CARD, fg=WHITE,
                 wraplength=700, justify="left").pack(anchor="w", pady=8)

        self.battle_feedback = tk.Label(q_frame, text="", font=("Arial", 12, "bold"),
                                         bg=BG_CARD, fg=WHITE)
        self.battle_feedback.pack(pady=5)

        opts_frame = tk.Frame(self.root, bg=BG_DARK)
        opts_frame.pack(pady=8)
        self.battle_buttons = []
        colors = ["#2979FF", "#00BFA5", "#66BB6A", "#AB47BC"]
        options = challenge.get("options", [])

        for i, opt in enumerate(options):
            row, col = divmod(i, 2)
            btn = tk.Label(
                opts_frame, text=opt, font=("Arial", 12, "bold"),
                bg=colors[i % 4], fg=WHITE, width=38, height=2,
                cursor="hand2", wraplength=350,
                padx=10, pady=10, relief="raised", bd=2
            )
            btn.grid(row=row, column=col, padx=6, pady=5)
            btn.bind("<Button-1>", lambda e, letter=chr(65 + i): self._check_mcq(challenge, letter))
            self.battle_buttons.append(btn)

    def _check_mcq(self, challenge, answer):
        correct = challenge.get("correct", "A").strip().upper()
        for btn in self.battle_buttons:
            btn.unbind("<Button-1>")
            btn.config(cursor="", relief="flat", bg="#333333")
        is_correct = (answer == correct)
        explanation = challenge.get("explanation", f"Correct was {correct}")
        self._resolve_round(is_correct, explanation)

    # ---- Type Answer challenge ----
    def _render_type_answer(self, challenge):
        q_frame = tk.Frame(self.root, bg=BG_CARD, padx=25, pady=15, relief="ridge", bd=2)
        q_frame.pack(fill="x", padx=40, pady=10)

        tk.Label(q_frame, text=challenge.get("question", ""),
                 font=("Arial", 14, "bold"), bg=BG_CARD, fg=WHITE,
                 wraplength=700, justify="left").pack(anchor="w", pady=8)

        hint = challenge.get("hint", "")
        if hint:
            tk.Label(q_frame, text=f"Hint: {hint}",
                     font=("Arial", 11, "italic"), bg=BG_CARD, fg=GOLD).pack(anchor="w")

        self.battle_entry = tk.Entry(q_frame, font=("Arial", 18), width=25,
                                      bg="#0f3460", fg=WHITE, insertbackground=WHITE, justify="center")
        self.battle_entry.pack(pady=12)
        self.battle_entry.focus()

        self.battle_feedback = tk.Label(q_frame, text="", font=("Arial", 12, "bold"),
                                         bg=BG_CARD, fg=WHITE)
        self.battle_feedback.pack(pady=5)

        self.battle_submit = tk.Label(
            self.root, text="ATTACK!", font=("Arial", 14, "bold"),
            bg=ACCENT_RED, fg=WHITE, width=16, padx=10, pady=12,
            cursor="hand2", relief="raised"
        )
        self.battle_submit.pack(pady=8)
        self.battle_submit.bind("<Button-1>", lambda e: self._check_typed(challenge))
        self.battle_entry.bind("<Return>", lambda e: self._check_typed(challenge))

    def _check_typed(self, challenge):
        user_input = self.battle_entry.get().strip().lower()
        if not user_input:
            return
        answer = challenge.get("answer", "").strip().lower()
        is_correct = (user_input == answer or answer in user_input or user_input in answer)
        self.battle_entry.config(state="disabled")
        self.battle_submit.unbind("<Button-1>")
        self.battle_submit.config(cursor="", relief="flat", bg="#333333")
        explanation = f"The answer was: {challenge.get('answer', '')}"
        self._resolve_round(is_correct, explanation)

    # ---- Fill Blank challenge ----
    def _render_fill_blank(self, challenge):
        q_frame = tk.Frame(self.root, bg=BG_CARD, padx=25, pady=15, relief="ridge", bd=2)
        q_frame.pack(fill="x", padx=40, pady=10)

        tk.Label(q_frame, text=challenge.get("sentence", ""),
                 font=("Arial", 14, "bold"), bg=BG_CARD, fg=WHITE,
                 wraplength=700, justify="left").pack(anchor="w", pady=8)

        hint = challenge.get("hint", "")
        if hint:
            tk.Label(q_frame, text=f"Hint: {hint}",
                     font=("Arial", 11, "italic"), bg=BG_CARD, fg=GOLD).pack(anchor="w")

        self.battle_entry = tk.Entry(q_frame, font=("Arial", 18), width=25,
                                      bg="#0f3460", fg=WHITE, insertbackground=WHITE, justify="center")
        self.battle_entry.pack(pady=12)
        self.battle_entry.focus()

        self.battle_feedback = tk.Label(q_frame, text="", font=("Arial", 12, "bold"),
                                         bg=BG_CARD, fg=WHITE)
        self.battle_feedback.pack(pady=5)

        self.battle_submit = tk.Label(
            self.root, text="ATTACK!", font=("Arial", 14, "bold"),
            bg=ACCENT_RED, fg=WHITE, width=16, padx=10, pady=12,
            cursor="hand2", relief="raised"
        )
        self.battle_submit.pack(pady=8)
        self.battle_submit.bind("<Button-1>", lambda e: self._check_typed(challenge))
        self.battle_entry.bind("<Return>", lambda e: self._check_typed(challenge))

    # ---- Unscramble challenge ----
    def _render_unscramble(self, challenge):
        q_frame = tk.Frame(self.root, bg=BG_CARD, padx=25, pady=15, relief="ridge", bd=2)
        q_frame.pack(fill="x", padx=40, pady=10)

        clue = challenge.get("clue", "")
        if clue:
            tk.Label(q_frame, text=f"Clue: {clue}",
                     font=("Arial", 13), bg=BG_CARD, fg=ACCENT_BLUE,
                     wraplength=700, justify="left").pack(anchor="w", pady=5)

        # Scrambled letter tiles
        scrambled = challenge.get("scrambled_word", "")
        if not scrambled:
            word = challenge.get("answer", "")
            letters = list(word.upper())
            random.shuffle(letters)
            scrambled = "".join(letters)

        tiles_frame = tk.Frame(q_frame, bg=BG_CARD)
        tiles_frame.pack(pady=15)
        for letter in scrambled.upper():
            tk.Label(tiles_frame, text=letter, font=("Arial", 22, "bold"),
                     bg=ACCENT_ORANGE, fg=WHITE, width=3, height=1,
                     relief="raised", bd=3).pack(side="left", padx=3)

        hint = challenge.get("hint", "")
        if hint:
            tk.Label(q_frame, text=f"Hint: {hint}",
                     font=("Arial", 11, "italic"), bg=BG_CARD, fg=GOLD).pack(pady=3)

        self.battle_entry = tk.Entry(q_frame, font=("Arial", 18), width=25,
                                      bg="#0f3460", fg=WHITE, insertbackground=WHITE, justify="center")
        self.battle_entry.pack(pady=10)
        self.battle_entry.focus()

        self.battle_feedback = tk.Label(q_frame, text="", font=("Arial", 12, "bold"),
                                         bg=BG_CARD, fg=WHITE)
        self.battle_feedback.pack(pady=5)

        self.battle_submit = tk.Label(
            self.root, text="ATTACK!", font=("Arial", 14, "bold"),
            bg=ACCENT_RED, fg=WHITE, width=16, padx=10, pady=12,
            cursor="hand2", relief="raised"
        )
        self.battle_submit.pack(pady=8)
        self.battle_submit.bind("<Button-1>", lambda e: self._check_typed(challenge))
        self.battle_entry.bind("<Return>", lambda e: self._check_typed(challenge))

    # ---- Resolve round (shared by all types) ----
    def _resolve_round(self, is_correct, explanation=""):
        if is_correct:
            self.correct_count += 1
            self.boss_hp = max(0, self.boss_hp - self.damage_per_hit)
            self.update_points(20, True)
            self.battle_feedback.config(
                text=f"CRITICAL HIT! -{self.damage_per_hit} HP to boss!", fg=ACCENT_GREEN)
        else:
            self.player_hp = max(0, self.player_hp - self.boss_damage)
            self.update_points(0, False)
            self.battle_feedback.config(
                text=f"Boss attacks! -{self.boss_damage} HP!", fg="#f44336")

            # Show teaching explanation prominently
            if explanation:
                explain_frame = tk.Frame(self.root, bg="#1B3A4B", padx=15, pady=10,
                                         relief="ridge", bd=1)
                explain_frame.pack(fill="x", padx=40, pady=5)
                tk.Label(explain_frame, text="Learn:", font=("Arial", 11, "bold"),
                         bg="#1B3A4B", fg=GOLD).pack(anchor="w")
                tk.Label(explain_frame, text=explanation, font=("Arial", 12),
                         bg="#1B3A4B", fg=WHITE, wraplength=700,
                         justify="left").pack(anchor="w")

        self.update_hp_bars()
        self.c_index += 1

        # Give more time to read explanation on wrong answers
        delay = 1800 if is_correct else 4000

        if self.player_hp <= 0:
            self.root.after(delay, self.defeat)
        elif self.boss_hp <= 0:
            self.root.after(delay, self.victory)
        else:
            self.root.after(delay, self.show_round)

    def victory(self):
        self.update_points(50, True)
        self.show_completion("VICTORY!", {
            "Boss Defeated": self.boss_name,
            "Challenges Correct": f"{self.correct_count}/{len(self.challenges)}",
            "Boss Kill Bonus": "+50 pts",
            "HP Remaining": f"{self.player_hp}/{self.player_max_hp}"
        })

    def defeat(self):
        self.show_completion("DEFEATED!", {
            "Boss": self.boss_name,
            "Challenges Correct": f"{self.correct_count}/{len(self.challenges)}",
            "You Reached": f"Round {self.c_index}/{len(self.challenges)}",
            "Tip": "Review the concepts and try again!"
        })


# =========================================
#  MEMORY MATCH GAME
# =========================================
class MemoryMatchGame(BaseGame):

    def __init__(self, root, game_data, topic_name):
        super().__init__(root, game_data, topic_name)
        self.pairs = game_data.get("pairs", [])[:8]  # Max 8 pairs = 16 cards
        self.matches_found = 0
        self.attempts = 0
        self.first_card = None
        self.can_click = True
        self.start_time = time.time()
        self.setup_cards()
        self.show_board()

    def setup_cards(self):
        """Create card data from pairs."""
        self.cards = []
        for i, pair in enumerate(self.pairs):
            self.cards.append({"id": i, "type": "term", "text": pair["term"], "pair_id": i})
            self.cards.append({"id": i, "type": "def", "text": pair["definition"], "pair_id": i})
        random.shuffle(self.cards)

        n = len(self.cards)
        self.cols = 4
        self.rows = math.ceil(n / self.cols)

    def show_board(self):
        self.clear_screen()
        self.show_header("Memory Match", "Flip cards to match terms with definitions!", ACCENT_BLUE)

        # Stats
        stats_frame = tk.Frame(self.root, bg=BG_DARK)
        stats_frame.pack(fill="x", padx=40)
        self.stats_label = tk.Label(
            stats_frame,
            text=f"Matches: {self.matches_found}/{len(self.pairs)}  |  Attempts: {self.attempts}",
            font=FONT_MEDIUM, bg=BG_DARK, fg=WHITE
        )
        self.stats_label.pack()

        # Card grid
        grid_frame = tk.Frame(self.root, bg=BG_DARK, padx=20, pady=10)
        grid_frame.pack(expand=True)

        self.card_buttons = []
        for idx, card in enumerate(self.cards):
            row, col = divmod(idx, self.cols)

            btn = tk.Label(
                grid_frame, text="?", font=("Arial", 12, "bold"),
                bg=ACCENT_PURPLE, fg=WHITE, width=22, height=4,
                cursor="hand2", wraplength=170,
                relief="raised", bd=3
            )
            btn.grid(row=row, column=col, padx=6, pady=6)
            btn.bind("<Button-1>", lambda e, i=idx: self.flip_card(i))
            self.card_buttons.append(btn)

            # If already matched, show it
            if card.get("matched"):
                btn.config(text=card["text"], bg="#1B5E20", relief="sunken")
                btn.unbind("<Button-1>")
                btn.config(cursor="")

    def flip_card(self, idx):
        if not self.can_click:
            return

        card = self.cards[idx]
        btn = self.card_buttons[idx]

        if card.get("matched") or card.get("flipped"):
            return

        # Show card
        card["flipped"] = True
        label = "Term" if card["type"] == "term" else "Def"
        btn.config(text=f"[{label}]\n{card['text']}", bg="#0D47A1")

        if self.first_card is None:
            self.first_card = idx
        else:
            self.can_click = False
            self.attempts += 1
            second_idx = idx
            first_card = self.cards[self.first_card]

            # Check match
            if (first_card["pair_id"] == card["pair_id"] and
                    first_card["type"] != card["type"]):
                # Match!
                first_card["matched"] = True
                card["matched"] = True
                self.matches_found += 1
                self.update_points(15, True)

                self.card_buttons[self.first_card].config(bg="#1B5E20", relief="sunken")
                self.card_buttons[self.first_card].unbind("<Button-1>")
                self.card_buttons[self.first_card].config(cursor="")
                btn.config(bg="#1B5E20", relief="sunken")
                btn.unbind("<Button-1>")
                btn.config(cursor="")

                self.stats_label.config(
                    text=f"Matches: {self.matches_found}/{len(self.pairs)}  |  Attempts: {self.attempts}"
                )

                self.first_card = None
                self.can_click = True

                if self.matches_found == len(self.pairs):
                    self.root.after(500, self.game_complete)
            else:
                # No match - flip back after delay
                first_idx = self.first_card
                self.first_card = None
                self.update_points(0, False)
                self.stats_label.config(
                    text=f"Matches: {self.matches_found}/{len(self.pairs)}  |  Attempts: {self.attempts}"
                )
                self.root.after(1200, lambda: self.flip_back(first_idx, second_idx))

    def flip_back(self, idx1, idx2):
        self.cards[idx1]["flipped"] = False
        self.cards[idx2]["flipped"] = False
        self.card_buttons[idx1].config(text="?", bg=ACCENT_PURPLE)
        self.card_buttons[idx2].config(text="?", bg=ACCENT_PURPLE)
        self.can_click = True

    def game_complete(self):
        elapsed = time.time() - self.start_time
        speed_bonus = max(0, int(120 - elapsed))
        self.total_points += speed_bonus
        self.show_completion("All Matched!", {
            "Pairs Found": f"{self.matches_found}/{len(self.pairs)}",
            "Attempts": self.attempts,
            "Time": f"{elapsed:.1f}s",
            "Speed Bonus": f"+{speed_bonus} pts"
        })


# =========================================
#  WORD SCRAMBLE GAME
# =========================================
class WordScrambleGame(BaseGame):

    def __init__(self, root, game_data, topic_name):
        super().__init__(root, game_data, topic_name)
        self.words = game_data.get("words", [])
        self.w_index = 0
        self.correct_count = 0
        self.hints_used = 0
        self.show_word()

    def scramble(self, word):
        """Scramble a word, ensuring it differs from the original."""
        letters = list(word.upper())
        for _ in range(20):
            random.shuffle(letters)
            if "".join(letters) != word.upper():
                return "".join(letters)
        # If still same (short word), just reverse
        return word.upper()[::-1]

    def show_word(self):
        self.clear_screen()
        self.show_header("Word Scramble", "Unscramble the letters!", ACCENT_ORANGE)

        if self.w_index >= len(self.words):
            self.game_complete()
            return

        item = self.words[self.w_index]
        word = item.get("word", "").strip()
        self.current_answer = word.lower()
        scrambled = self.scramble(word)
        category = item.get("category", "")

        # Progress
        tk.Label(self.root, text=f"Word {self.w_index + 1} of {len(self.words)}",
                 font=("Arial", 13), bg=BG_DARK, fg=GRAY).pack(pady=8)

        if category:
            tk.Label(self.root, text=f"Category: {category}",
                     font=("Arial", 12, "italic"), bg=BG_DARK, fg=ACCENT_BLUE).pack()

        # Clue/question at the top so student knows the context
        clue = item.get("clue", "")
        if clue:
            clue_frame = tk.Frame(self.root, bg="#0f3460", padx=20, pady=12, relief="ridge", bd=1)
            clue_frame.pack(fill="x", padx=60, pady=(10, 0))
            tk.Label(clue_frame, text="Clue:", font=("Arial", 12, "bold"),
                     bg="#0f3460", fg=GOLD).pack(anchor="w")
            tk.Label(clue_frame, text=clue, font=("Arial", 14),
                     bg="#0f3460", fg=WHITE, wraplength=700, justify="left").pack(anchor="w")

        # Scrambled letters as tiles
        card_frame = tk.Frame(self.root, bg=BG_CARD, padx=30, pady=25, relief="ridge", bd=2)
        card_frame.pack(fill="x", padx=60, pady=15)

        tiles_frame = tk.Frame(card_frame, bg=BG_CARD)
        tiles_frame.pack(pady=20)

        for letter in scrambled:
            tile = tk.Label(
                tiles_frame, text=letter, font=("Arial", 24, "bold"),
                bg=ACCENT_ORANGE, fg=WHITE, width=3, height=1,
                relief="raised", bd=3
            )
            tile.pack(side="left", padx=4)

        # Hint area
        self.hint_label = tk.Label(card_frame, text="", font=("Arial", 12, "italic"),
                                    bg=BG_CARD, fg=GOLD)
        self.hint_label.pack(pady=5)

        # Input
        input_frame = tk.Frame(card_frame, bg=BG_CARD)
        input_frame.pack(pady=10)

        self.word_entry = tk.Entry(
            input_frame, font=("Arial", 20), width=20,
            bg="#0f3460", fg=WHITE, insertbackground=WHITE,
            justify="center"
        )
        self.word_entry.pack()
        self.word_entry.focus()

        # Feedback
        self.word_feedback = tk.Label(card_frame, text="", font=("Arial", 14, "bold"),
                                      bg=BG_CARD, fg=WHITE)
        self.word_feedback.pack(pady=5)

        # Buttons
        btn_frame = tk.Frame(self.root, bg=BG_DARK)
        btn_frame.pack(pady=10)

        self.hint_btn = tk.Label(
            btn_frame, text="Hint", font=("Arial", 12, "bold"),
            bg=GOLD, fg="#333", width=10, padx=8, pady=6,
            cursor="hand2", relief="raised"
        )
        self.hint_btn.pack(side="left", padx=10)
        self.hint_btn.bind("<Button-1>", lambda e, it=item: self.show_hint(it))

        self.submit_btn = tk.Label(
            btn_frame, text="Check", font=("Arial", 14, "bold"),
            bg=ACCENT_ORANGE, fg=WHITE, width=14, padx=10, pady=12,
            cursor="hand2", relief="raised"
        )
        self.submit_btn.pack(side="left", padx=10)
        self.submit_btn.bind("<Button-1>", lambda e: self.check_word())

        self.word_entry.bind("<Return>", lambda e: self.check_word())

    def show_hint(self, item):
        hint = item.get("hint", "No hint available")
        self.hint_label.config(text=f"Hint: {hint}")
        self.hints_used += 1
        self.hint_btn.unbind("<Button-1>")
        self.hint_btn.config(cursor="", relief="flat", bg="#333333")

    def check_word(self):
        user_input = self.word_entry.get().strip().lower()
        if not user_input:
            return

        if user_input == self.current_answer:
            self.correct_count += 1
            self.update_points(15, True)
            self.word_feedback.config(text="Correct! +15 pts", fg=ACCENT_GREEN)
        else:
            self.update_points(0, False)
            self.word_feedback.config(
                text=f"The word was: {self.current_answer.upper()}", fg="#f44336"
            )
            # Show teaching context for the word
            item = self.words[self.w_index]
            meaning = item.get("meaning", item.get("clue", ""))
            if meaning:
                explain_frame = tk.Frame(self.root, bg="#1B3A4B", padx=15, pady=8,
                                         relief="ridge", bd=1)
                explain_frame.pack(fill="x", padx=60, pady=5)
                tk.Label(explain_frame, text=f"Learn: {self.current_answer.upper()} — {meaning}",
                         font=("Arial", 12), bg="#1B3A4B", fg=WHITE,
                         wraplength=660, justify="left").pack(anchor="w")

        self.word_entry.config(state="disabled")
        self.submit_btn.unbind("<Button-1>")
        self.submit_btn.config(text="Next >>", bg=ACCENT_BLUE)
        self.submit_btn.bind("<Button-1>", lambda e: self.next_word())

    def next_word(self):
        self.w_index += 1
        self.show_word()

    def game_complete(self):
        self.show_completion("Scramble Complete!", {
            "Words Solved": f"{self.correct_count}/{len(self.words)}",
            "Hints Used": self.hints_used,
            "Accuracy": f"{(self.correct_count / max(len(self.words), 1)) * 100:.0f}%"
        })


# =========================================
#  MILLIONAIRE GAME
# =========================================
class MillionaireGame(BaseGame):

    PRIZE_LADDER = [
        (100, "100"), (200, "200"), (500, "500"),
        (1000, "1,000"), (2000, "2,000"), (5000, "5,000"),
        (10000, "10,000"), (50000, "50,000")
    ]

    def __init__(self, root, game_data, topic_name):
        super().__init__(root, game_data, topic_name)
        self.questions = game_data.get("questions", [])
        # Sort by difficulty if available
        self.questions.sort(key=lambda q: q.get("difficulty", 0))
        self.q_index = 0
        self.lifelines = {"fifty_fifty": True, "phone": True, "audience": True}
        self.eliminated_options = []
        self.show_question()

    def show_question(self):
        self.clear_screen()

        if self.q_index >= len(self.questions):
            self.win_game()
            return

        # Top bar with lifelines
        top_frame = tk.Frame(self.root, bg="#00004d", pady=10)
        top_frame.pack(fill="x")

        tk.Label(top_frame, text="WHO WANTS TO BE A MILLIONAIRE?",
                 font=("Arial", 16, "bold"), bg="#00004d", fg=GOLD).pack()

        # Current prize
        prize_idx = min(self.q_index, len(self.PRIZE_LADDER) - 1)
        current_prize = self.PRIZE_LADDER[prize_idx][1]
        tk.Label(top_frame, text=f"Playing for: ${current_prize}",
                 font=("Arial", 13, "bold"), bg="#00004d", fg=WHITE).pack()

        # Main content area
        content_frame = tk.Frame(self.root, bg="#000033")
        content_frame.pack(fill="both", expand=True)

        # Left: Question area
        left_frame = tk.Frame(content_frame, bg="#000033")
        left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)

        # Lifelines
        ll_frame = tk.Frame(left_frame, bg="#000033")
        ll_frame.pack(pady=10)

        self.ll_buttons = {}
        lifeline_config = [
            ("fifty_fifty", "50:50", "#FF5722"),
            ("phone", "Phone", "#2196F3"),
            ("audience", "Audience", "#4CAF50"),
        ]

        for key, text, color in lifeline_config:
            bg = color if self.lifelines[key] else "#555"
            btn = tk.Label(
                ll_frame, text=text, font=("Arial", 11, "bold"),
                bg=bg, fg=WHITE, width=10,
                cursor="hand2" if self.lifelines[key] else "",
                padx=8, pady=6, relief="raised", bd=2
            )
            btn.pack(side="left", padx=8)
            if self.lifelines[key]:
                btn.bind("<Button-1>", lambda e, k=key: self.use_lifeline(k))
            self.ll_buttons[key] = btn

        # Question
        q = self.questions[self.q_index]
        q_frame = tk.Frame(left_frame, bg="#001a4d", padx=20, pady=20, relief="ridge", bd=2)
        q_frame.pack(fill="x", pady=10)

        tk.Label(q_frame, text=q["question"], font=("Arial", 15, "bold"),
                 bg="#001a4d", fg=WHITE, wraplength=550, justify="center").pack(pady=10)

        # Feedback
        self.ml_feedback = tk.Label(q_frame, text="", font=("Arial", 13, "bold"),
                                     bg="#001a4d", fg=WHITE)
        self.ml_feedback.pack(pady=5)

        # Audience chart area (hidden by default)
        self.audience_frame = tk.Frame(left_frame, bg="#000033")
        self.audience_frame.pack(fill="x", pady=5)

        # Options 2x2
        opts_frame = tk.Frame(left_frame, bg="#000033")
        opts_frame.pack(pady=10)

        self.ml_buttons = []
        self.eliminated_options = []
        colors = ["#2979FF", "#FFB300", "#66BB6A", "#EF5350"]
        options = q.get("options", [])

        for i, opt in enumerate(options):
            row, col = divmod(i, 2)
            btn = tk.Label(
                opts_frame, text=opt, font=("Arial", 13, "bold"),
                bg=colors[i], fg=WHITE, width=30, height=2,
                cursor="hand2", wraplength=280,
                padx=10, pady=10, relief="raised", bd=2
            )
            btn.grid(row=row, column=col, padx=6, pady=5)
            btn.bind("<Button-1>", lambda e, letter=chr(65 + i): self.check_ml_answer(letter))
            self.ml_buttons.append(btn)

        # Right: Prize ladder
        right_frame = tk.Frame(content_frame, bg="#000022", padx=15, pady=10, relief="ridge", bd=1)
        right_frame.pack(side="right", fill="y", padx=(0, 10), pady=10)

        tk.Label(right_frame, text="PRIZE LADDER",
                 font=("Arial", 11, "bold"), bg="#000022", fg=GOLD).pack(pady=(0, 8))

        for i in range(len(self.PRIZE_LADDER) - 1, -1, -1):
            pts, display = self.PRIZE_LADDER[i]
            is_current = (i == self.q_index)
            bg = GOLD if is_current else "#000022"
            fg = "#000" if is_current else (WHITE if i <= self.q_index else "#555")
            font = ("Arial", 12, "bold") if is_current else ("Arial", 11)

            tk.Label(right_frame, text=f"  ${display}  ",
                     font=font, bg=bg, fg=fg, width=12).pack(pady=1)

    def use_lifeline(self, key):
        if not self.lifelines[key]:
            return
        self.lifelines[key] = False
        self.ll_buttons[key].unbind("<Button-1>")
        self.ll_buttons[key].config(bg="#555", cursor="", relief="flat")

        q = self.questions[self.q_index]
        correct = q.get("correct", "A").strip().upper()

        if key == "fifty_fifty":
            # Remove 2 wrong options
            wrong_indices = [i for i in range(len(self.ml_buttons))
                             if chr(65 + i) != correct]
            to_remove = random.sample(wrong_indices, min(2, len(wrong_indices)))
            for idx in to_remove:
                self.ml_buttons[idx].config(bg="#333", text="---")
                self.ml_buttons[idx].unbind("<Button-1>")
                self.eliminated_options.append(idx)

        elif key == "phone":
            explanation = q.get("explanation", "I think it might be " + correct)
            self.ml_feedback.config(
                text=f'Friend says: "{explanation}"', fg=GOLD
            )

        elif key == "audience":
            # Show audience vote bar chart
            for w in self.audience_frame.winfo_children():
                w.destroy()

            canvas = tk.Canvas(self.audience_frame, width=400, height=100,
                               bg="#001133", highlightthickness=0)
            canvas.pack()

            correct_idx = ord(correct) - 65
            votes = [random.randint(5, 20) for _ in range(4)]
            votes[correct_idx] = random.randint(45, 70)
            total = sum(votes)
            pcts = [int(v / total * 100) for v in votes]

            labels = ["A", "B", "C", "D"]
            bar_colors = ["#1565C0", "#FF8F00", "#2E7D32", "#C62828"]

            for i in range(4):
                x = 40 + i * 90
                bar_h = pcts[i] * 0.8
                canvas.create_rectangle(x, 90 - bar_h, x + 60, 90, fill=bar_colors[i])
                canvas.create_text(x + 30, 95, text=f"{labels[i]}: {pcts[i]}%",
                                   fill=WHITE, font=("Arial", 9, "bold"))

    def check_ml_answer(self, answer):
        q = self.questions[self.q_index]
        correct = q.get("correct", "A").strip().upper()

        for btn in self.ml_buttons:
            btn.unbind("<Button-1>")
            btn.config(cursor="", relief="flat", bg="#333333")
        for btn in self.ll_buttons.values():
            btn.unbind("<Button-1>")
            btn.config(bg="#555", cursor="", relief="flat")

        if answer == correct:
            prize_idx = min(self.q_index, len(self.PRIZE_LADDER) - 1)
            prize_pts = self.PRIZE_LADDER[prize_idx][0] // 10  # Scale down
            self.update_points(prize_pts, True)
            self.ml_feedback.config(text=f"Correct! +{prize_pts} pts!", fg=ACCENT_GREEN)
            self.q_index += 1
            self.root.after(1800, self.show_question)
        else:
            self.ml_feedback.config(text="Wrong!", fg="#f44336")
            self.update_points(0, False)

            # Show teaching explanation prominently
            explanation = q.get("explanation", f"The correct answer was {correct}")
            explain_frame = tk.Frame(self.root, bg="#1B3A4B", padx=15, pady=10,
                                     relief="ridge", bd=1)
            explain_frame.pack(fill="x", padx=40, pady=5)
            tk.Label(explain_frame, text="Learn:", font=("Arial", 11, "bold"),
                     bg="#1B3A4B", fg=GOLD).pack(anchor="w")
            tk.Label(explain_frame, text=explanation, font=("Arial", 12),
                     bg="#1B3A4B", fg=WHITE, wraplength=700,
                     justify="left").pack(anchor="w")

            self.root.after(4000, self.game_over)

    def game_over(self):
        self.show_completion("Game Over!", {
            "Reached Level": f"{self.q_index + 1}/{len(self.questions)}",
            "Questions Correct": self.q_index,
            "Lifelines Remaining": sum(1 for v in self.lifelines.values() if v),
        })

    def win_game(self):
        self.update_points(100, True)  # Completion bonus
        self.show_completion("MILLIONAIRE!", {
            "All Questions Correct": f"{len(self.questions)}/{len(self.questions)}",
            "Completion Bonus": "+100 pts",
            "Lifelines Remaining": sum(1 for v in self.lifelines.values() if v),
        })


# =========================================
#  SUBPROCESS ENTRY POINT
# =========================================
def _run_review_subprocess(data_path, score_file):
    """Entry point for gamified review subprocess. Plays 2 games in sequence."""
    with open(data_path, "r") as f:
        data = json.load(f)

    topic = data.get("topic_name", "")
    games_list = data.get("games", [])

    # Backward compat: if old single-game format
    if not games_list and "game_type" in data:
        games_list = [data]

    game_classes = {
        "concept_review": ConceptReviewUI,
        "boss_battle": BossBattleGame,
        "memory_match": MemoryMatchGame,
        "word_scramble": WordScrambleGame,
        "millionaire": MillionaireGame,
    }

    total_points = 0

    for i, game_data in enumerate(games_list):
        game_type = game_data.get("game_type", "boss_battle")
        game_cls = game_classes.get(game_type, BossBattleGame)

        root = tk.Tk()
        root.title(f"AI Tutor - Game {i + 1} of {len(games_list)}: {game_type.replace('_', ' ').title()}")
        root.geometry("900x700")
        root.configure(bg=BG_DARK)

        game = game_cls(root, game_data, topic)
        root.mainloop()

        total_points += game.total_points

    with open(score_file, "w") as f:
        f.write(str(total_points))


# =========================================
#  LAUNCHER
# =========================================
def launch_gamified_review(topic_name="", quiz_csv="quiz_results.csv", emotion_csv="emotion_data.csv", video_description=""):
    """Score-based game selection + AI-generated content, launched in subprocess."""
    import subprocess
    import sys
    import tempfile

    print("\n" + "=" * 60)
    print("    AI Game Orchestrator")
    print("=" * 60)

    # 1. Extract data
    wrong_topics = get_wrong_topics(quiz_csv)
    confusion_phrases = get_confusion_phrases(emotion_csv)
    score = get_session_score(quiz_csv)

    if not wrong_topics:
        print(" No wrong answers found - skipping review.")
        return 0

    # 1b. Load curriculum metadata for this topic
    curriculum_description = ""
    curriculum_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curriculum.json")
    if os.path.exists(curriculum_path):
        with open(curriculum_path, "r") as f:
            curriculum = json.load(f)
        for entry in curriculum:
            if entry.get("topic_name", "").lower() in topic_name.lower() or topic_name.lower() in entry.get("topic_name", "").lower():
                curriculum_description = entry.get("description", "")
                print(f" Curriculum metadata loaded: {entry['topic_name']}")
                break

    print(f" Score: {score}/10 | Wrong answers: {len(wrong_topics)}")

    # 2. Score-based game selection (no LLM call for picking)
    if score <= 3:
        game_types = ["boss_battle"]
        print(" Score 0-3: Launching BOSS BATTLE (comprehensive mixed-challenge review)")
    elif score <= 4:
        game_types = [random.choice(["word_scramble", "memory_match"])]
        print(f" Score 4: Launching {game_types[0].replace('_', ' ').upper()} (lighter review)")
    else:
        game_types = ["millionaire"]
        print(" Score 5: Launching MILLIONAIRE (targeted reinforcement with lifelines)")

    # 3. Generate concept review cards (TEACH phase)
    print("\n Generating concept review cards (teaching phase)...\n")
    concept_cards = _generate_concept_cards(
        wrong_topics, topic_name, video_description, confusion_phrases,
        curriculum_description=curriculum_description
    )

    games = []
    if concept_cards:
        games.append({"game_type": "concept_review", "cards": concept_cards})

    # 4. Generate content for each selected game (TEST phase)
    print(" Generating game content with AI...\n")
    for game_type in game_types:
        print(f" Generating {game_type.replace('_', ' ').title()} content...")
        game_data = _generate_single_game(
            game_type, wrong_topics, topic_name,
            video_description=video_description,
            confusion_phrases=confusion_phrases,
            curriculum_description=curriculum_description
        )
        games.append(game_data)

    game_data = {"games": games, "topic_name": topic_name}

    # 3. Save and launch subprocess
    data_file = os.path.join(tempfile.gettempdir(), "tutor_review_data.json")
    score_file = os.path.join(tempfile.gettempdir(), "tutor_review_score.txt")

    with open(data_file, "w") as f:
        json.dump(game_data, f)

    games = game_data.get("games", [])
    game_names = " → ".join(g.get("game_type", "?").replace("_", " ").title() for g in games)
    print(f"\n Launching: {game_names}")

    subprocess.run(
        [sys.executable, "-c",
         f"from gamified_review import _run_review_subprocess; "
         f"_run_review_subprocess({data_file!r}, {score_file!r})"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    # 4. Read score back
    points = 0
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            try:
                points = int(f.read().strip())
            except ValueError:
                points = 0
        os.remove(score_file)

    if os.path.exists(data_file):
        os.remove(data_file)

    return points


# =========================================
#  STANDALONE TEST
# =========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AI Gamified Review - Test Mode")
    print("=" * 60)

    points = launch_gamified_review(
        topic_name="Agriculture and Crop Production",
        quiz_csv="quiz_results.csv",
        emotion_csv="emotion_data.csv",
        video_description=""
    )

    print(f"\n Review complete! Total points earned: {points}")
