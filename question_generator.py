import spacy
import tkinter as tk
from tkinter import messagebox
import json
import re
import pandas as pd
import os
import time

# =========================================
#  CONFIGURATION - CHOOSE YOUR API
# =========================================
API_PROVIDER = os.environ.get("API_PROVIDER", "openai")
API_KEY = os.environ.get("TUTOR_API_KEY", "")

# =========================================
#  API SETUP
# =========================================
nlp = spacy.load("en_core_web_sm")

def setup_api():
    """Initialize the chosen API client."""
    global client, model_name
    
    if API_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=API_KEY)
        model_name = "llama-3.1-8b-instant"
        print(f" Using Groq API with {model_name}")
        
    elif API_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        model_name = "gpt-3.5-turbo"
        print(f" Using OpenAI API with {model_name}")
        
    elif API_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        client = genai
        model_name = "gemini-2.0-flash"
        print(f" Using Gemini API with {model_name}")
    
    else:
        raise ValueError(f"Unknown API provider: {API_PROVIDER}")

setup_api()


def call_llm(prompt, max_retries=3):
    """Universal LLM caller with retry logic."""
    for attempt in range(max_retries):
        try:
            if API_PROVIDER == "groq":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            elif API_PROVIDER == "openai":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            elif API_PROVIDER == "gemini":
                model = client.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                wait_time = 30 * (attempt + 1)
                print(f"⚠️ Rate limit hit. Waiting {wait_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f" API Error: {e}")
                raise e
    
    raise Exception(f"Failed after {max_retries} retries")


# =========================================
#  CSV KEYWORD EXTRACTION
# =========================================
def load_confusion_phrases_from_csv(csv_path="emotion_data.csv"):
    """Load phrases from CSV where negative emotions were detected."""
    if not os.path.isfile(csv_path):
        print(f" CSV file not found: {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    
    if "phrase" not in df.columns:
        print(" CSV doesn't have 'phrase' column")
        return []
    
    phrases = df[df["phrase"].notna() & (df["phrase"] != "")]["phrase"].unique().tolist()
    print(f" Loaded {len(phrases)} unique confusion phrases from CSV")
    
    return phrases


def extract_keywords_from_phrases(phrases):
    """Extract key concepts from confusion phrases using spaCy."""
    all_keywords = []
    
    for phrase in phrases:
        doc = nlp(phrase)
        noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text) > 2]
        all_keywords.extend(noun_chunks)
        entities = [ent.text for ent in doc.ents]
        all_keywords.extend(entities)
        important_tokens = [
            token.text for token in doc 
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2
        ]
        all_keywords.extend(important_tokens)
    
    unique_keywords = list(dict.fromkeys(all_keywords))
    stopwords = {"it", "this", "that", "the", "a", "an", "is", "are", "was", "were"}
    filtered_keywords = [kw for kw in unique_keywords if kw.lower() not in stopwords]
    
    print(f" Extracted {len(filtered_keywords)} keywords")
    return filtered_keywords


# =========================================
#  QUESTION GENERATOR
# =========================================
def generate_questions_from_csv(csv_path="emotion_data.csv", num_qs=10, fallback_topic=None, video_description=None):
    """Generate MCQ questions based on confusion phrases from CSV + video description."""
    phrases = load_confusion_phrases_from_csv(csv_path)

    if not phrases:
        if fallback_topic:
            print(f" No phrases in CSV, using fallback topic")
            return generate_questions_from_topic(fallback_topic, num_qs)
        return []

    keywords = extract_keywords_from_phrases(phrases)

    if not keywords:
        if fallback_topic:
            return generate_questions_from_topic(fallback_topic, num_qs)
        return []

    return generate_questions_from_keywords(keywords, phrases, num_qs, video_description)


def generate_questions_from_keywords(keywords, original_phrases, num_qs=10, video_description=None):
    """Generate MCQs targeting extracted keywords + video content."""
    limited_keywords = keywords[:10]
    limited_phrases = original_phrases[:5]

    keywords_str = ", ".join(limited_keywords)
    phrases_context = "\n".join([f"- {p[:100]}" for p in limited_phrases])

    # Build video context (truncate to keep prompt manageable)
    video_context = ""
    if video_description:
        truncated_desc = video_description[:800]
        video_context = f"""
VIDEO CONTENT (what the lecture taught):
{truncated_desc}

"""

    prompt = f"""Generate {num_qs} multiple-choice questions for 8th standard students.
{video_context}CONFUSION POINTS (where the student struggled - prioritize these):
{phrases_context}

KEY CONCEPTS: {keywords_str}

INSTRUCTIONS:
- Generate questions that cover BOTH the video content AND the confusion points
- About 60% of questions should target the confusion points (where the student struggled)
- About 40% of questions should cover other important concepts from the video content
- This ensures the quiz tests overall understanding, not just the confusing parts
- Every question MUST be unique — no two questions should test the same concept or be worded similarly
- Each question should cover a DIFFERENT aspect of the topic

FORMAT (follow EXACTLY):
Q: <question>
A) <option>
B) <option>
C) <option>
D) <option>
---

Generate exactly {num_qs} questions:"""

    try:
        response_text = call_llm(prompt)
        print(f"\n Raw API Response:\n{response_text[:500]}...\n")  # Debug
        
        raw_questions = response_text.split("---")
        
        questions = []
        for q in raw_questions:
            q = q.strip()
            if q and "Q:" in q and "A)" in q:
                q_clean = re.sub(r'CORRECT:.*', '', q, flags=re.IGNORECASE).strip()
                questions.append(q_clean)
        
        print(f" Parsed {len(questions)} questions")
        
        # Debug: Print first question structure
        if questions:
            print(f"\n Sample Question Structure:\n{questions[0]}\n")
        
        return questions[:num_qs]
        
    except Exception as e:
        print(f" Error: {e}")
        return []


def generate_questions_from_topic(topic_text, num_qs=10):
    """Fallback: Generate questions from topic text."""
    prompt = f"""Topic: {topic_text}

Generate exactly {num_qs} multiple-choice questions for 8th standard students.

FORMAT:
Q: <question>
A) <option>
B) <option>
C) <option>
D) <option>
---

Generate {num_qs} questions:"""

    try:
        response_text = call_llm(prompt)
        raw_questions = response_text.split("---")
        questions = [q.strip() for q in raw_questions if q.strip() and "Q:" in q]
        return questions[:num_qs]
    except Exception as e:
        print(f" Error: {e}")
        return []


# =========================================
#  AI-BASED ANSWER CHECKING 
# =========================================
def check_answers_with_llm(questions, user_answers):
    """Evaluate answers using AI and return correct answers."""
    evaluation_data = ""
    for i, (q, a) in enumerate(zip(questions, user_answers), start=1):
        evaluation_data += f"Question {i}:\n{q.strip()}\nUser's Selected Option: {a}\n\n"

    prompt = f"""You are an academic grader. Evaluate these answers.

{evaluation_data}

For each question:
1. Determine if the user selected the CORRECT answer
2. Provide the correct answer option (A, B, C, or D)

Respond ONLY with a JSON object in this format:
{{
    "results": [true, false, true, ...],
    "correct_answers": ["A", "C", "B", ...]
}}

Your response (JSON only):"""

    try:
        response_text = call_llm(prompt)
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON response
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        
        if match:
            data = json.loads(match.group())
            results = data.get("results", [False] * len(questions))
            correct_answers = data.get("correct_answers", [""] * len(questions))
            return sum(results[:len(questions)]), results[:len(questions)], correct_answers[:len(questions)]
        
        return 0, [False] * len(questions), [""] * len(questions)
        
    except Exception as e:
        print(f" Grading Error: {e}")
        return 0, [False] * len(questions), [""] * len(questions)


def save_quiz_results_to_csv(questions, user_answers, correct_answers, results, score, filename="quiz_results.csv"):
    """
    Save quiz results to CSV file.
    
    Columns: timestamp, question_num, question_text, option_a, option_b, option_c, option_d,
             student_answer, correct_answer, is_correct, session_score
    """
    from datetime import datetime
    
    quiz_data = []
    session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for i, (q, student_ans, correct_ans, is_correct) in enumerate(zip(questions, user_answers, correct_answers, results), 1):
        # Parse question to extract question text and options
        q_text, options = parse_question(q)
        
        quiz_data.append({
            "timestamp": session_timestamp,
            "question_num": i,
            "question_text": q_text,
            "option_a": options[0] if len(options) > 0 else "",
            "option_b": options[1] if len(options) > 1 else "",
            "option_c": options[2] if len(options) > 2 else "",
            "option_d": options[3] if len(options) > 3 else "",
            "student_answer": student_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct,
            "session_score": f"{score}/{len(questions)}"
        })
    
    df = pd.DataFrame(quiz_data)
    
    # Append to existing CSV or create new
    write_header = not os.path.isfile(filename)
    df.to_csv(filename, mode="a", index=False, header=write_header)
    
    print(f" Quiz results saved to {filename}")
    print(f"   → {len(quiz_data)} questions logged")
    
    return df


# =========================================
#  IMPROVED QUESTION PARSER
# =========================================
def parse_question(q_data):
    """
    Parse a question string into question text and options.
    Handles various formats from different LLMs.
    """
    lines = [l.strip() for l in q_data.split("\n") if l.strip()]
    
    question_text = ""
    options = ["Option A", "Option B", "Option C", "Option D"]  # Defaults
    
    for line in lines:
        # Handle question line
        if line.startswith("Q:") or line.startswith("Q."):
            question_text = line[2:].strip()
        elif line.startswith("**Q") or line.lower().startswith("question"):
            # Handle markdown or "Question 1:" format
            match = re.search(r'[:\.](.+)', line)
            if match:
                question_text = match.group(1).strip()
        
        # Handle options - multiple formats
        elif line.startswith("A)") or line.startswith("A.") or line.startswith("a)"):
            options[0] = line
        elif line.startswith("B)") or line.startswith("B.") or line.startswith("b)"):
            options[1] = line
        elif line.startswith("C)") or line.startswith("C.") or line.startswith("c)"):
            options[2] = line
        elif line.startswith("D)") or line.startswith("D.") or line.startswith("d)"):
            options[3] = line
        # Handle "(A)" format
        elif line.startswith("(A)") or line.startswith("(a)"):
            options[0] = "A) " + line[3:].strip()
        elif line.startswith("(B)") or line.startswith("(b)"):
            options[1] = "B) " + line[3:].strip()
        elif line.startswith("(C)") or line.startswith("(c)"):
            options[2] = "C) " + line[3:].strip()
        elif line.startswith("(D)") or line.startswith("(d)"):
            options[3] = "D) " + line[3:].strip()
    
    # If no question found, use first line
    if not question_text and lines:
        question_text = lines[0]
    
    return question_text, options


# =========================================
#  TKINTER QUIZ UI (FIXED)
# =========================================
class QuizApp:
    def __init__(self, root, questions, score_callback=None):
        self.root = root
        self.questions = questions
        self.current_q = 0
        self.answers = []
        self.score = 0
        self.score_callback = score_callback

        # Debug: Print questions
        print(f"\n QuizApp received {len(questions)} questions")
        if questions:
            print(f"First question preview: {questions[0][:100]}...")

        # Window setup
        self.root.title("🎓 AI Tutor - Targeted Quiz")
        self.root.geometry("850x650")
        self.root.configure(bg="#f0f4f8")
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 425
        y = (self.root.winfo_screenheight() // 2) - 325
        self.root.geometry(f'+{x}+{y}')

        # Header
        header = tk.Frame(root, bg="#2196F3", pady=15)
        header.pack(fill="x")
        
        tk.Label(
            header, 
            text="Targeted Knowledge Check", 
            font=("Arial", 20, "bold"),
            bg="#2196F3", fg="white"
        ).pack()
        
        tk.Label(
            header,
            text="Questions based on your confusion points",
            font=("Arial", 11, "italic"),
            bg="#2196F3", fg="#E3F2FD"
        ).pack()

        # Progress
        self.progress = tk.Label(
            root, text="", 
            font=("Arial", 13, "bold"), 
            bg="#f0f4f8", fg="#333"
        )
        self.progress.pack(pady=15)

        # Question frame
        q_frame = tk.Frame(root, bg="white", padx=30, pady=25, relief="ridge", bd=2)
        q_frame.pack(fill="both", expand=True, padx=40, pady=15)

        # Question label - FIXED: Larger font, explicit fg color
        self.question_label = tk.Label(
            q_frame, 
            text="Loading question...", 
            font=("Arial", 15, "bold"),
            wraplength=750, 
            justify="left", 
            bg="white",
            fg="black",  # Explicit text color
            anchor="w"
        )
        self.question_label.pack(pady=20, anchor="w", fill="x")

        # Options frame
        opts_frame = tk.Frame(q_frame, bg="white")
        opts_frame.pack(fill="x", pady=15)

        self.var = tk.StringVar(value="")
        self.option_buttons = []
        self.option_colors = ["#555555", "#555555", "#555555", "#555555"]

        for i, opt in enumerate(["A", "B", "C", "D"]):
            btn = tk.Label(
                opts_frame,
                text=f"Option {opt}",
                font=("Arial", 13, "bold"),
                anchor="w",
                justify="left",
                bg=self.option_colors[i],
                fg="white",
                wraplength=700,
                padx=15,
                pady=12,
                relief="raised",
                bd=2,
                cursor="hand2"
            )
            btn.pack(fill="x", pady=5, padx=15)
            btn.bind("<Button-1>", lambda e, v=opt: self._select_option(v))
            self.option_buttons.append(btn)

        # Button frame
        btn_frame = tk.Frame(root, bg="#f0f4f8", pady=20)
        btn_frame.pack(fill="x")

        self.next_button = tk.Label(
            btn_frame,
            text="Next  →",
            font=("Arial", 13, "bold"),
            bg="#2196F3",
            fg="white",
            padx=30,
            pady=12,
            relief="raised",
            bd=2,
            cursor="hand2"
        )
        self.next_button.pack()
        self.next_button.bind("<Button-1>", lambda e: self.next_question())

        # Load first question
        self.load_question()

    def _select_option(self, value):
        """Handle option button click — highlight selected, dim others."""
        self.var.set(value)
        letters = ["A", "B", "C", "D"]
        for i, btn in enumerate(self.option_buttons):
            if letters[i] == value:
                btn.config(bg="#FFD600", fg="black", relief="sunken", bd=3)
            else:
                btn.config(bg=self.option_colors[i], fg="white", relief="raised", bd=2)

    def load_question(self):
        """Load and display the current question."""
        if self.current_q < len(self.questions):
            q_data = self.questions[self.current_q]

            # Debug output
            print(f"\n Loading Q{self.current_q + 1}:")
            print(f"Raw data: {q_data[:200]}...")

            question_text, options = parse_question(q_data)

            print(f"Parsed question: {question_text}")
            print(f"Parsed options: {options}")

            # Update progress
            self.progress.config(
                text=f"Question {self.current_q + 1} of {len(self.questions)}"
            )

            # Update question text
            if question_text:
                self.question_label.config(text=question_text)
            else:
                self.question_label.config(text="[Question text not found]")

            # Update options
            for i, opt_text in enumerate(options):
                if opt_text and opt_text.strip():
                    self.option_buttons[i].config(
                        text=opt_text,
                        bg=self.option_colors[i], fg="white", relief="raised", bd=2
                    )
                else:
                    self.option_buttons[i].config(
                        text=f"Option {['A','B','C','D'][i]}",
                        bg=self.option_colors[i], fg="white", relief="raised", bd=2
                    )

            # Reset selection
            self.var.set("")

            # Update button for last question
            if self.current_q == len(self.questions) - 1:
                self.next_button.config(text="Submit ✓")

            # Force UI update
            self.root.update_idletasks()

    def next_question(self):
        """Handle next button click."""
        if not self.var.get():
            messagebox.showwarning("Select Answer", "Please select an option before continuing.")
            return
        
        self.answers.append(self.var.get())
        self.current_q += 1
        
        if self.current_q < len(self.questions):
            self.load_question()
        else:
            self.finish_quiz()

    def finish_quiz(self):
        """Complete quiz and show results."""
        self.question_label.config(text=" AI is evaluating your answers...")
        self.root.update()
        
        # Get score, results, AND correct answers
        self.score, results, correct_answers = check_answers_with_llm(self.questions, self.answers)
        
        # Save to CSV
        save_quiz_results_to_csv(
            self.questions, 
            self.answers, 
            correct_answers, 
            results, 
            self.score
        )
        
        result_msg = f"Score: {self.score}/{len(self.questions)}\n\n"
        for i, (correct, student_ans, correct_ans) in enumerate(zip(results, self.answers, correct_answers), 1):
            if correct:
                result_msg += f"Q{i}:  Correct! (You selected: {student_ans})\n"
            else:
                result_msg += f"Q{i}:  Wrong (You: {student_ans}, Correct: {correct_ans})\n"
        
        if self.score >= 6:
            result_msg += "\n Great job! Proceeding to next lecture."
        else:
            result_msg += "\n Let's review with interactive activities!"
        
        messagebox.showinfo("Quiz Results", result_msg)
        
        if self.score_callback:
            self.score_callback(self.score)
        
        self.root.destroy()


# =========================================
#  LAUNCHER FUNCTION
# =========================================
def _run_quiz_subprocess(questions_json_path, score_file_path):
    """Entry point for quiz subprocess — runs Tkinter in a clean process."""
    import json
    with open(questions_json_path, "r") as f:
        questions = json.load(f)

    score_container = {"score": 0}

    def on_complete(score):
        score_container["score"] = score

    root = tk.Tk()
    app = QuizApp(root, questions, score_callback=on_complete)
    root.mainloop()

    with open(score_file_path, "w") as f:
        f.write(str(score_container["score"]))


def launch_quiz_ui(questions):
    """Launch Tkinter quiz in a separate process to avoid SDL/Tkinter conflict on macOS."""
    import subprocess, sys, tempfile

    if not questions:
        print(" No questions to display")
        return 0

    print(f"\n Launching Quiz with {len(questions)} questions...")

    # Write questions to a temp file so the subprocess can read them
    questions_file = os.path.join(tempfile.gettempdir(), "tutor_quiz_questions.json")
    score_file = os.path.join(tempfile.gettempdir(), "tutor_quiz_score.txt")

    with open(questions_file, "w") as f:
        json.dump(questions, f)

    # Launch quiz in a clean Python subprocess (no SDL contamination)
    result = subprocess.run(
        [sys.executable, "-c",
         f"from question_generator import _run_quiz_subprocess; "
         f"_run_quiz_subprocess({questions_file!r}, {score_file!r})"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    # Read score back
    score = 0
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            try:
                score = int(f.read().strip())
            except ValueError:
                score = 0
        os.remove(score_file)

    if os.path.exists(questions_file):
        os.remove(questions_file)

    return score


# =========================================
#  MAIN TEST
# =========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   Question Generator Test")
    print("="*60)
    
    # Generate questions from CSV
    csv_path = "emotion_data.csv"
    
    if os.path.exists(csv_path):
        print(f"\n Using CSV: {csv_path}")
        questions = generate_questions_from_csv(csv_path=csv_path, num_qs=10)
    else:
        print(f"\n CSV not found, using fallback topic")
        questions = generate_questions_from_topic("Agriculture, crops, farming", num_qs=10)
    
    if questions:
        print(f"\n Generated {len(questions)} questions")
        score = launch_quiz_ui(questions)
        print(f"\n Final Score: {score}/{len(questions)}")
    else:
        print(" Failed to generate questions")