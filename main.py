"""
Nova Chat - Multi-LLM Intelligent Routing Chatbot
Main application entry point using Flask + SocketIO + Groq + Semantic Routing
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional



from flask import Flask, render_template, request, redirect, url_for, session

from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv

from authlib.integrations.flask_client import OAuth


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

from collections import defaultdict, deque

# ────────────────────────────────────────────────
# 1. Load environment variables FIRST
# ────────────────────────────────────────────────
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY is required!")
    sys.exit(1)

# ────────────────────────────────────────────────
# 2. Flask app setup
# ────────────────────────────────────────────────
app = Flask(__name__)


oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)




app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///nova.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'super-secret-key-change-me-please')

app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
}

# ────────────────────────────────────────────────
# 3. Initialize extensions
# ────────────────────────────────────────────────
db = SQLAlchemy(app)
migrate = Migrate(app, db)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ────────────────────────────────────────────────
# 4. LLM Models & Semantic Routing
# ────────────────────────────────────────────────
MODELS = {
    "model_xl": "openai/gpt-oss-safeguard-20b",           # placeholder
    "model_l": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "model_m": "meta-llama/llama-guard-4-12b",
    "model_s": "groq/compound-mini",
    "model_quiz": "groq/compound-mini",                   # fast model for quizzes
}

ROUTES = {
    "model_xl": "Safety analysis, policy sensitive, high risk content",
    "model_l": "Complex reasoning, system design, deep explanations",
    "model_m": "Cybersecurity, vulnerabilities, exploits, moderation",
    "model_s": "Casual chat, greetings, short or simple questions",
}

router_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

ROUTE_EMBEDDINGS = {
    k: router_model.encode(v, normalize_embeddings=True)
    for k, v in ROUTES.items()
}

def route_prompt(prompt: str, threshold: float = 0.45) -> str:
    prompt_emb = router_model.encode(prompt, normalize_embeddings=True)
    scores = {k: cosine_similarity([prompt_emb], [v])[0][0] for k, v in ROUTE_EMBEDDINGS.items()}
    best_model, best_score = max(scores.items(), key=lambda x: x[1])
    return best_model if best_score >= threshold else "model_s"

# ────────────────────────────────────────────────
# 5. LLM Helpers
# ────────────────────────────────────────────────
def call_llm(model_key: str, messages: list[dict]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODELS[model_key],
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM call failed: {e}")
        return f"Error: {str(e)}"

# ────────────────────────────────────────────────
# 6. Global state
# ────────────────────────────────────────────────
chat_histories = defaultdict(lambda: deque(maxlen=12))   # ~6 turns
quiz_states: Dict[str, Dict[str, Any]] = {}              # sid → quiz state

# ────────────────────────────────────────────────
# 7. Quiz helpers — MCQ ONLY
# ────────────────────────────────────────────────
def is_quiz_trigger(message: str) -> bool:
    msg_lower = message.lower().strip()
    return any(word in msg_lower for word in ["quiz me", "start quiz", "give me a quiz", "quiz on", "quiz about"])

def extract_quiz_topic(message: str) -> Optional[str]:
    msg = message.strip()
    patterns = [
        r"(?i)quiz\s*me\s*(?:on|about)\s+(.+?)(?:\s*$|\?)",
        r"(?i)start\s+quiz\s*(?:on|about)\s+(.+?)(?:\s*$|\?)",
        r"(?i)give\s+me\s+a\s+quiz\s*(?:on|about)\s+(.+?)(?:\s*$|\?)",
        r"(?i)quiz\s+(?:on|about)\s+(.+?)(?:\s*$|\?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg)
        if match:
            topic = match.group(1).strip().rstrip(".")
            return topic if topic else None
    return None

def build_quiz_prompt(topic: str) -> str:
    return (
        "You are an expert quiz generator. Create an engaging educational quiz.\n"
        "**Only generate multiple-choice questions (MCQ). No short-answer or open-ended questions.**\n\n"
        f"Topic: {topic}\n\n"
        "Generate 6–10 multiple-choice questions.\n"
        "Return **only valid JSON** with this exact structure:\n"
        "{\n"
        '  "topic": "string",\n'
        '  "questions": [\n'
        '    {\n'
        '      "type": "mcq",\n'
        '      "question": "string",\n'
        '      "options": ["A) option one", "B) option two", "C) option three", "D) option four"],\n'
        '      "answer": "A) option one",   // must match one of the options exactly\n'
        '      "explanation": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Every question MUST have exactly 4 options.\n"
        "- Options must start with A), B), C), D)\n"
        "- The 'answer' field must be the full correct option string (e.g. 'C) Paris is the capital of France')\n"
        "- Make questions clear, accurate and educational\n"
        "- Return ONLY the JSON — no extra text, no markdown, no explanation outside JSON"
    )

def parse_quiz_response(raw: str) -> Optional[dict]:
    try:
        # Clean common markdown/code fences
        cleaned = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
        cleaned = re.sub(r'^`+|`+$', '', cleaned.strip())
        data = json.loads(cleaned)
        if "questions" in data and isinstance(data["questions"], list):
            # Basic validation
            for q in data["questions"]:
                if not all(k in q for k in ["question", "options", "answer", "explanation"]):
                    return None
                if len(q["options"]) != 4:
                    return None
                if q["answer"] not in q["options"]:
                    return None
            return data
    except Exception:
        pass
    return None

def format_question(q: dict, index: int, total: int) -> str:
    lines = [f"**Question {index}/{total}**", q["question"]]
    for opt in q.get("options", []):
        lines.append(opt)
    return "\n".join(lines)

def normalize_answer(user_input: str) -> str:
    """Convert user answer like 'a', 'A', 'A)', 'a)' → full option text if possible"""
    ans = user_input.strip().lower()
    if ans in ('a', 'b', 'c', 'd'):
        return ans.upper() + ')'
    if len(ans) == 1 and ans in 'abcd':
        return ans.upper() + ')'
    # fallback: return as is
    return user_input.strip()


# ────────────────────────────────────────────────
# 8. Flask Routes
# ────────────────────────────────────────────────

@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/auth/google')
def google_login():
    return google.authorize_redirect(url_for('google_callback', _external=True))


@app.route('/auth/google/callback')
def google_callback():
    token = google.authorize_access_token()
    user = token['userinfo']

    session['user'] = {
        "email": user['email'],
        "name": user['name'],
        "picture": user['picture']
    }

    return redirect('/')

@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return render_template('index.html')


@app.route('/quiz')
def quiz():
    if 'user' not in session:
        return redirect('/login')
    return render_template('quiz.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')



# ────────────────────────────────────────────────
# 9. SocketIO Handlers
# ────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    sid = request.sid
    print(f"[CONNECT] {sid}")
    emit('chat_message', {
        'role': 'system',
        'content': 'Connected! You can start chatting or request a quiz (all questions will be MCQ).'
    })

@socketio.on('message')
def on_message(data):
    sid = request.sid
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return

    print(f"[MSG] {sid} → {user_msg[:80]}{'...' if len(user_msg)>80 else ''}")

    # Echo user message
    emit('chat_message', {
        'role': 'user',
        'content': user_msg,
        'timestamp': datetime.utcnow().isoformat()
    })

    # ──────────────────────────────
    # Handle quiz answer (MCQ only)
    # ──────────────────────────────
    if sid in quiz_states and quiz_states[sid].get('awaiting_answer'):
        state = quiz_states[sid]
        q = state['questions'][state['current']]

        user_answer = normalize_answer(user_msg)
        correct_answer = q['answer']

        # Try to match by letter prefix if user gave 'A' or 'a'
        correct_letter = correct_answer[0].upper() if correct_answer and len(correct_answer) > 1 else None
        user_letter = user_answer[0].upper() if user_answer and len(user_answer) > 1 else None

        is_correct = (
            user_answer == correct_answer or
            (user_letter and correct_letter and user_letter == correct_letter)
        )

        state['score'] += 1 if is_correct else 0

        feedback = " **Correct!**" if is_correct else f" **Incorrect** — Correct answer: {correct_answer}"
        feedback += f"\n\n**Explanation:** {q.get('explanation', 'No explanation provided.')}"
        emit('chat_message', {'role': 'assistant', 'content': feedback})

        state['current'] += 1

        if state['current'] >= len(state['questions']):
            summary = (
                f"**Quiz finished!**\n"
                f"Topic: {state['topic']}\n"
                f"Score: **{state['score']} / {len(state['questions'])}**"
            )
            emit('chat_message', {'role': 'assistant', 'content': summary})
            del quiz_states[sid]
        else:
            next_q = format_question(state['questions'][state['current']], state['current']+1, len(state['questions']))
            emit('chat_message', {'role': 'assistant', 'content': next_q})
            state['awaiting_answer'] = True

        return

    # ──────────────────────────────
    # New quiz request
    # ──────────────────────────────
    if is_quiz_trigger(user_msg):
        topic = extract_quiz_topic(user_msg)

        if not topic:
            emit('chat_message', {
                'role': 'assistant',
                'content': "Sure! What topic would you like the MCQ quiz on?\n(Just reply with the subject)"
            })
            quiz_states[sid] = {"awaiting_topic": True}
            return

        emit('chat_message', {'role': 'assistant', 'content': f"Preparing MCQ quiz on **{topic}** …"})
        emit('typing', {'status': True})

        try:
            prompt = build_quiz_prompt(topic)
            raw_response = call_llm("model_quiz", [
                {"role": "system", "content": "You are a strict JSON-only quiz generator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ])

            quiz_data = parse_quiz_response(raw_response)
            if not quiz_data or not quiz_data.get('questions'):
                raise ValueError("Could not generate valid MCQ quiz")

            questions = quiz_data['questions']
            if len(questions) < 5:
                raise ValueError("Too few questions generated")

            quiz_states[sid] = {
                "topic": topic,
                "questions": questions,
                "current": 0,
                "score": 0,
                "awaiting_answer": True
            }

            intro = f"**MCQ Quiz on {topic}**\n{len(questions)} questions. Choose A, B, C or D."
            emit('chat_message', {'role': 'assistant', 'content': intro})

            first_q = format_question(questions[0], 1, len(questions))
            emit('chat_message', {'role': 'assistant', 'content': first_q})

        except Exception as e:
            error_msg = f"Sorry, I couldn't create the MCQ quiz right now.\n({str(e)})"
            emit('chat_message', {'role': 'assistant', 'content': error_msg})
            quiz_states.pop(sid, None)

        finally:
            emit('typing', {'status': False})
        return

    # ──────────────────────────────
    # Normal chat
    # ──────────────────────────────
    model_key = route_prompt(user_msg)

    emit('typing', {'status': True})

    try:
        messages = [
            {"role": "system", "content": "You are a helpful, concise assistant."},
        ]
        for msg in list(chat_histories[sid])[-8:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": user_msg})

        answer = call_llm(model_key, messages)

        chat_histories[sid].append({"role": "user", "content": user_msg})
        chat_histories[sid].append({"role": "assistant", "content": answer})

        emit('chat_message', {
            'role': 'assistant',
            'content': answer,
            'model': model_key
        })

    except Exception as e:
        emit('chat_message', {'role': 'assistant', 'content': f"Error: {str(e)}"})

    finally:
        emit('typing', {'status': False})

@socketio.on('disconnect')
def on_disconnect():
    print(f"[DISCONNECT] {request.sid}")

# ────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────
if __name__ == '__main__':
    print("═" * 60)
    print(" Nova Chat   |   http://127.0.0.1:5000")
    print("═" * 60)
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        allow_unsafe_werkzeug=True
    )