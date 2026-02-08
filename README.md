# Nova Chat — Multi-LLM Intelligent Learning Chatbot

Nova is a modern AI-powered EdTech platform that intelligently routes user questions across multiple large language models and reinforces learning through a built-in interactive quiz system.

Students can chat with AI for explanations and immediately test their understanding using topic-wise quizzes — combining AI assistance with active learning.


## Features

- Smart Model Routing — automatically chooses the best model based on prompt content  
- Supports multiple Groq-powered LLMs with specialized roles  
- Modern dark-themed real-time chat UI  
- WebSocket streaming responses (Socket.IO)  

### Authentication
- Google OAuth 2.0 signup/login (no passwords required)  
- Secure session-based user authentication  
- Protected chat and quiz pages  

### Learning Tools
- AI-powered topic-wise quizzes  
- MCQ style questions with instant feedback  
- Reinforces learning after chat sessions  


## Currently Supported Models & Routing Logic

| Key     | Model ID                                      | Purpose / Specialty                              |
|---------|-----------------------------------------------|--------------------------------------------------|
| model_xl| openai/gpt-oss-safeguard-20b                  | Safety analysis, policy-sensitive content        |
| model_l | meta-llama/llama-4-maverick-17b-128e-instruct | Complex reasoning, system design, deep answers   |
| model_m | meta-llama/llama-guard-4-12b                  | Cybersecurity, exploits, moderation              |
| model_s | groq/compound-mini                            | Casual chat, greetings, simple & fast questions  |

Routing uses sentence-transformers + cosine similarity (very lightweight)

## Tech Stack

- Backend: Flask + Flask-SocketIO
- LLM: Groq API
- Embedding/Routing: sentence-transformers + scikit-learn
- Frontend: Pure HTML/CSS + JavaScript + Socket.IO
- Environment: python-dotenv
- Authlib (OAuth for Google login)
- Backend: Flask + Flask-SocketIO
- Database: SQLAlchemy + Flask-Migrate
- Authentication: Google OAuth 2.0 (Authlib)


## Quick Start

1. Clone the repository
```bash
git clone https://github.com/yourusername/nova-chat.git
cd nova-chat
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create .env file and add your key
```bash

GROQ_API_KEY=your_groq_api_key  
GOOGLE_CLIENT_ID=your_google_client_id  
GOOGLE_CLIENT_SECRET=your_google_client_secret  
SECRET_KEY=your_flask_secret_key
```

4. Run the app
```bash
python main.py
```

5. Open in browser → http://127.0.0.1:5000


## Project Status (Jan 2026)

- Basic routing + chat working
- Nice dark UI with hover controls
- Secure Google OAuth authentication implemented  
- Special quiz learning section added  
- Chat system fully functional  
- Chat history persistence & analytics planned
- Missing: History storing in DataBase

