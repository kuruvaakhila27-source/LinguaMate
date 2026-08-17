import streamlit as st
import ollama
import json
import os
import io
import re
from datetime import date, datetime
from difflib import SequenceMatcher

import speech_recognition as sr
from pydub import AudioSegment
from deep_translator import GoogleTranslator
from streamlit_mic_recorder import mic_recorder


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="LinguaMate",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL = "llama3.2:3b"
DATA_FILE = "linguamate_data.json"


# ============================================================
# FFmpeg SETUP
# ============================================================

# Winget installation path
WINGET_FFMPEG = (
    r"C:\Users\AKHILA\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.1-essentials_build\bin"
)

if os.path.exists(os.path.join(WINGET_FFMPEG, "ffmpeg.exe")):
    AudioSegment.converter = os.path.join(
        WINGET_FFMPEG,
        "ffmpeg.exe"
    )

    AudioSegment.ffmpeg = os.path.join(
        WINGET_FFMPEG,
        "ffmpeg.exe"
    )

    AudioSegment.ffprobe = os.path.join(
        WINGET_FFMPEG,
        "ffprobe.exe"
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f8fafc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Hero */

    .hero {
        padding: 2.2rem;
        border-radius: 25px;
        background:
            linear-gradient(
                135deg,
                #eef2ff 0%,
                #f5f3ff 45%,
                #ecfeff 100%
            );
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(15,23,42,0.05);
    }

    .hero h1 {
        font-size: 3.1rem;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: #475569;
        font-size: 1.08rem;
        margin-top: 0.5rem;
    }

    /* Cards */

    .card {
        background: white;
        padding: 1.3rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 20px rgba(15,23,42,0.04);
        margin-bottom: 1rem;
    }

    .feature-card {
        background: white;
        padding: 1.25rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        min-height: 155px;
        box-shadow: 0 5px 20px rgba(15,23,42,0.04);
        transition: transform .2s;
    }

    .feature-card:hover {
        transform: translateY(-3px);
    }

    .feature-icon {
        font-size: 2rem;
    }

    .feature-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .5rem;
    }

    .muted {
        color: #64748b;
        font-size: .9rem;
    }

    /* Section title */

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: .8rem;
    }

    /* Metric cards */

    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 5px 18px rgba(15,23,42,0.04);
    }

    .metric-number {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .metric-label {
        color: #64748b;
        font-size: .9rem;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #64748b;
        padding: 1.5rem;
        font-size: .9rem;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA
# ============================================================

DEFAULT_DATA = {
    "xp": 0,
    "saved_words": [],
    "history": [],
    "voice_history": [],
    "pronunciation_history": [],
    "scores": [],
    "completed_challenges": [],
    "analysis": None
}


def load_data():

    if not os.path.exists(DATA_FILE):
        return DEFAULT_DATA.copy()

    try:

        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        for key, value in DEFAULT_DATA.items():

            if key not in data:
                data[key] = value

        return data

    except Exception:

        return DEFAULT_DATA.copy()


saved_data = load_data()

for key, value in saved_data.items():

    if key not in st.session_state:
        st.session_state[key] = value


def save_data():

    data = {
        "xp": st.session_state.xp,
        "saved_words": st.session_state.saved_words,
        "history": st.session_state.history,
        "voice_history": st.session_state.voice_history,
        "pronunciation_history": st.session_state.pronunciation_history,
        "scores": st.session_state.scores,
        "completed_challenges": st.session_state.completed_challenges,
        "analysis": st.session_state.analysis
    }

    try:

        with open(
            DATA_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception:
        pass


# ============================================================
# SESSION DEFAULTS
# ============================================================

if "current_messages" not in st.session_state:
    st.session_state.current_messages = []

if "conversation_settings" not in st.session_state:
    st.session_state.conversation_settings = {}

if "voice_history" not in st.session_state:
    st.session_state.voice_history = []

if "pronunciation_history" not in st.session_state:
    st.session_state.pronunciation_history = []


# ============================================================
# HELPERS
# ============================================================

def ask_ollama(
    prompt,
    system="You are LinguaMate, a helpful language learning assistant."
):

    try:

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"ERROR: {e}"


def extract_json(text):

    try:

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            return json.loads(
                text[start:end + 1]
            )

    except Exception:
        pass

    return None


def level_info(xp):

    levels = [
        (0, "Beginner", 100),
        (100, "Explorer", 250),
        (250, "Communicator", 450),
        (450, "Confident Speaker", 700),
        (700, "Fluent", 1000)
    ]

    current = levels[0]

    for level in levels:

        if xp >= level[0]:
            current = level

    return current


def add_xp(amount):

    st.session_state.xp += amount
    save_data()


def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return " ".join(
        text.split()
    )


def speech_match_score(target, spoken):

    target = normalize_text(target)
    spoken = normalize_text(spoken)

    if not target or not spoken:
        return 0

    return round(
        SequenceMatcher(
            None,
            target,
            spoken
        ).ratio() * 100
    )


def convert_audio_to_wav(audio_bytes):

    audio_stream = io.BytesIO(audio_bytes)

    audio_segment = AudioSegment.from_file(
        audio_stream
    )

    audio_segment = (
        audio_segment
        .set_channels(1)
        .set_frame_rate(16000)
        .set_sample_width(2)
    )

    wav_buffer = io.BytesIO()

    audio_segment.export(
        wav_buffer,
        format="wav"
    )

    wav_buffer.seek(0)

    return wav_buffer


def recognize_audio(audio_bytes, language_code):

    recognizer = sr.Recognizer()

    wav_buffer = convert_audio_to_wav(
        audio_bytes
    )

    with sr.AudioFile(
        wav_buffer
    ) as source:

        recorded_audio = recognizer.record(
            source
        )

    spoken_text = recognizer.recognize_google(
        recorded_audio,
        language=language_code
    )

    return spoken_text


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:10px;
    ">
        <div style="font-size:2.4rem;">🌍</div>
        <h2 style="margin:0;">LinguaMate</h2>
        <p style="color:#64748b;">
        Learn Languages. Speak Confidently.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "💬 Conversation",
        "🧠 Analyzer",
        "📚 Vocabulary",
        "🎯 Daily Challenge",
        "📊 Progress",
        "🌎 Translation",
        "🎤 Voice & Pronunciation",
        "💾 History"
    ]
)

st.sidebar.divider()

current_level = level_info(
    st.session_state.xp
)

st.sidebar.metric(
    "⭐ XP",
    st.session_state.xp
)

st.sidebar.write(
    f"🏆 **{current_level[1]}**"
)

level_progress = min(
    st.session_state.xp / current_level[2],
    1
)

st.sidebar.progress(
    level_progress
)

st.sidebar.caption(
    f"{st.session_state.xp} / {current_level[2]} XP"
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">
            <h1>🌍 LinguaMate</h1>
            <p>
                Learn Languages. Speak Confidently.
            </p>
            <p>
                Practice conversations, improve grammar,
                build vocabulary and develop speaking confidence
                with your personal AI language partner.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    avg_score = 0

    if st.session_state.scores:

        avg_score = round(
            sum(st.session_state.scores)
            / len(st.session_state.scores)
        )

    metrics = [
        (
            "💬",
            len(st.session_state.history),
            "Conversations"
        ),
        (
            "📚",
            len(st.session_state.saved_words),
            "Saved Words"
        ),
        (
            "⭐",
            st.session_state.xp,
            "XP"
        ),
        (
            "📈",
            f"{avg_score}%",
            "Average Score"
        )
    ]

    cols = st.columns(4)

    for col, metric in zip(cols, metrics):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.7rem;">
                        {metric[0]}
                    </div>
                    <div class="metric-number">
                        {metric[1]}
                    </div>
                    <div class="metric-label">
                        {metric[2]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">✨ Everything you need to improve</div>',
        unsafe_allow_html=True
    )

    features = [
        (
            "💬",
            "Conversation Practice",
            "Practice realistic conversations with AI."
        ),
        (
            "🎭",
            "Roleplay",
            "Practice interviews, travel, college and more."
        ),
        (
            "🧠",
            "Communication Analyzer",
            "Get grammar, vocabulary and fluency feedback."
        ),
        (
            "📚",
            "Vocabulary Builder",
            "Save useful words and grow your vocabulary."
        ),
        (
            "🎯",
            "Daily Challenge",
            "Complete a small speaking or writing challenge."
        ),
        (
            "🌎",
            "Translation",
            "Translate sentences and learn their usage."
        ),
        (
            "🎤",
            "Voice Practice",
            "Speak naturally and convert speech to text."
        ),
        (
            "📊",
            "Progress Tracking",
            "Track XP, scores and learning progress."
        )
    ]

    cols = st.columns(4)

    for i, feature in enumerate(features):

        with cols[i % 4]:

            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">
                        {feature[0]}
                    </div>
                    <div class="feature-title">
                        {feature[1]}
                    </div>
                    <p class="muted">
                        {feature[2]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# CONVERSATION
# ============================================================

elif page == "💬 Conversation":

    st.title("💬 Conversation Practice")

    st.caption(
        "Talk with LinguaMate and practice real-world communication."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        language = st.selectbox(
            "🌐 Language",
            [
                "English",
                "Hindi",
                "Telugu",
                "Spanish",
                "French",
                "Japanese",
                "Korean"
            ]
        )

    with c2:

        difficulty = st.selectbox(
            "🧩 Difficulty",
            [
                "Beginner",
                "Intermediate",
                "Advanced"
            ]
        )

    with c3:

        situation = st.selectbox(
            "🎯 Situation",
            [
                "Daily Conversation",
                "College",
                "Job Interview",
                "Travel",
                "Restaurant",
                "Shopping",
                "Making Friends",
                "Presentation",
                "Airport"
            ]
        )

    mode = st.selectbox(
        "🎭 Communication Mode",
        [
            "Normal Conversation",
            "Roleplay Mode",
            "Professional Communication"
        ]
    )

    if mode == "Roleplay Mode":

        role_map = {
            "Daily Conversation": "Friend",
            "College": "Professor",
            "Job Interview": "Interviewer",
            "Travel": "Travel Guide",
            "Restaurant": "Waiter",
            "Shopping": "Shop Assistant",
            "Making Friends": "New Friend",
            "Presentation": "Audience Member",
            "Airport": "Immigration Officer"
        }

        st.info(
            f"🎭 AI Role: **{role_map[situation]}**"
        )

    if mode == "Professional Communication":

        st.info(
            "🧑‍💼 Professional mode focuses on workplace communication."
        )

    if st.button(
        "🚀 Start New Conversation",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.current_messages = []

        st.session_state.conversation_settings = {
            "language": language,
            "difficulty": difficulty,
            "situation": situation,
            "mode": mode
        }

        system = f"""
You are LinguaMate.

Language: {language}
Situation: {situation}
Difficulty: {difficulty}
Mode: {mode}

Act as a natural language conversation partner.

Rules:
- Use the selected language.
- Match the learner's difficulty.
- Keep answers reasonably short.
- Ask one useful follow-up question.
- Correct mistakes naturally when useful.
"""

        with st.spinner(
            "🤖 Starting your conversation..."
        ):

            reply = ask_ollama(
                "Start the conversation.",
                system
            )

        st.session_state.current_messages = [
            {
                "role": "assistant",
                "content": reply
            }
        ]

        st.rerun()

    if st.session_state.current_messages:

        st.divider()

        for message in st.session_state.current_messages:

            if message["role"] == "user":

                st.chat_message(
                    "user",
                    avatar="👤"
                ).write(
                    message["content"]
                )

            else:

                st.chat_message(
                    "assistant",
                    avatar="🌍"
                ).write(
                    message["content"]
                )

        user_text = st.chat_input(
            "Type your message..."
        )

        if user_text:

            st.session_state.current_messages.append(
                {
                    "role": "user",
                    "content": user_text
                }
            )

            settings = st.session_state.conversation_settings

            messages = [
                {
                    "role": "system",
                    "content": f"""
You are LinguaMate.

Language: {settings.get("language", "English")}
Situation: {settings.get("situation", "Daily Conversation")}
Difficulty: {settings.get("difficulty", "Beginner")}
Mode: {settings.get("mode", "Normal Conversation")}

Continue naturally.
Keep your response concise.
Ask one follow-up question.
"""
                }
            ]

            messages.extend(
                st.session_state.current_messages
            )

            try:

                response = ollama.chat(
                    model=MODEL,
                    messages=messages
                )

                reply = response["message"]["content"]

            except Exception as e:

                reply = f"ERROR: {e}"

            st.session_state.current_messages.append(
                {
                    "role": "assistant",
                    "content": reply
                }
            )

            st.rerun()

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "💾 Save Conversation",
                use_container_width=True
            ):

                user_messages = [
                    x["content"]
                    for x in st.session_state.current_messages
                    if x["role"] == "user"
                ]

                if user_messages:

                    settings = st.session_state.conversation_settings

                    st.session_state.history.append(
                        {
                            "date": datetime.now().strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                            "language": settings.get(
                                "language",
                                "English"
                            ),
                            "situation": settings.get(
                                "situation",
                                "Conversation"
                            ),
                            "difficulty": settings.get(
                                "difficulty",
                                "Beginner"
                            ),
                            "mode": settings.get(
                                "mode",
                                "Normal Conversation"
                            ),
                            "messages": list(
                                st.session_state.current_messages
                            )
                        }
                    )

                    add_xp(10)

                    st.success(
                        "✅ Conversation saved to History! +10 XP"
                    )

                else:

                    st.warning(
                        "Send at least one message first."
                    )

        with col2:

            if st.button(
                "🗑️ Clear Current Chat",
                use_container_width=True
            ):

                st.session_state.current_messages = []

                st.rerun()


# ============================================================
# ANALYZER
# ============================================================

elif page == "🧠 Analyzer":

    st.title("🧠 Communication Analyzer")

    if not st.session_state.current_messages:

        st.info(
            "💬 Start a conversation and send a few messages first."
        )

    else:

        user_messages = [
            x["content"]
            for x in st.session_state.current_messages
            if x["role"] == "user"
        ]

        if not user_messages:

            st.info(
                "Send some messages before analyzing."
            )

        else:

            text = "\n".join(
                user_messages
            )

            with st.expander(
                "📝 Your messages"
            ):

                st.write(text)

            if st.button(
                "🧠 Analyze My Communication",
                use_container_width=True,
                type="primary"
            ):

                prompt = f"""
Analyze this language learner's messages:

{text}

Return ONLY valid JSON:

{{
    "grammar": 80,
    "vocabulary_score": 75,
    "fluency": 78,
    "sentence_formation": 82,
    "overall": 79,
    "mistakes": [
        {{
            "original": "I am studying AI from two years.",
            "better": "I have been studying AI for two years.",
            "why": "Use 'for' with a duration."
        }}
    ],
    "vocabulary": [
        {{
            "word": "confident",
            "meaning": "feeling sure about yourself",
            "example": "She spoke confidently.",
            "difficulty": "Intermediate"
        }}
    ],
    "tips": [
        "Practice longer sentences."
    ]
}}
"""

                with st.spinner(
                    "🔎 Analyzing your communication..."
                ):

                    result = ask_ollama(
                        prompt,
                        "You are an expert language teacher."
                    )

                data = extract_json(result)

                if data:

                    st.session_state.analysis = data

                    score = int(
                        data.get(
                            "overall",
                            0
                        )
                    )

                    st.session_state.scores.append(
                        score
                    )

                    add_xp(20)

                    st.success(
                        "✅ Analysis complete! +20 XP"
                    )

                else:

                    st.error(
                        "Could not read the analysis. Please try again."
                    )

            data = st.session_state.analysis

            if data:

                st.divider()

                cols = st.columns(5)

                score_items = [
                    ("Grammar", "grammar"),
                    ("Vocabulary", "vocabulary_score"),
                    ("Fluency", "fluency"),
                    ("Sentence", "sentence_formation"),
                    ("Overall", "overall")
                ]

                for col, item in zip(
                    cols,
                    score_items
                ):

                    with col:

                        st.metric(
                            item[0],
                            f"{data.get(item[1], 0)}/100"
                        )

                st.divider()

                st.subheader(
                    "✍️ Grammar Corrections"
                )

                mistakes = data.get(
                    "mistakes",
                    []
                )

                if mistakes:

                    for mistake in mistakes:

                        st.error(
                            "❌ " +
                            mistake.get(
                                "original",
                                ""
                            )
                        )

                        st.success(
                            "✅ " +
                            mistake.get(
                                "better",
                                ""
                            )
                        )

                        st.info(
                            "💡 " +
                            mistake.get(
                                "why",
                                ""
                            )
                        )

                else:

                    st.success(
                        "🎉 No major mistakes found."
                    )

                st.subheader(
                    "📚 Useful Vocabulary"
                )

                for i, word in enumerate(
                    data.get(
                        "vocabulary",
                        []
                    )
                ):

                    name = word.get(
                        "word",
                        ""
                    )

                    with st.container():

                        st.markdown(
                            f"### 📖 {name}"
                        )

                        st.write(
                            "**Meaning:**",
                            word.get(
                                "meaning",
                                ""
                            )
                        )

                        st.write(
                            "**Example:**",
                            word.get(
                                "example",
                                ""
                            )
                        )

                        if st.button(
                            f"⭐ Save {name}",
                            key=f"save_word_{i}"
                        ):

                            if name not in st.session_state.saved_words:

                                st.session_state.saved_words.append(
                                    name
                                )

                                save_data()

                                st.success(
                                    f"Saved '{name}'!"
                                )

                st.subheader(
                    "💡 Improvement Tips"
                )

                for tip in data.get(
                    "tips",
                    []
                ):

                    st.write(
                        "• " + tip
                    )


# ============================================================
# VOCABULARY
# ============================================================

elif page == "📚 Vocabulary":

    st.title("📚 Vocabulary Builder")

    st.caption(
        "Your saved words in one place."
    )

    if not st.session_state.saved_words:

        st.info(
            "📖 No saved words yet. Analyze a conversation and save useful words."
        )

    else:

        st.write(
            f"**{len(st.session_state.saved_words)} saved words**"
        )

        for i, word in enumerate(
            st.session_state.saved_words
        ):

            c1, c2 = st.columns(
                [6, 1]
            )

            with c1:

                st.markdown(
                    f"### 📖 {word}"
                )

            with c2:

                if st.button(
                    "Remove",
                    key=f"remove_word_{i}"
                ):

                    st.session_state.saved_words.pop(i)

                    save_data()

                    st.rerun()


# ============================================================
# DAILY CHALLENGE
# ============================================================

elif page == "🎯 Daily Challenge":

    st.title("🎯 Daily Challenge")

    challenges = [
        "Introduce yourself in five sentences.",
        "Describe your college.",
        "Talk about your favorite hobby.",
        "Explain your career goals.",
        "Describe a memorable trip.",
        "Explain why learning languages is useful.",
        "Describe your daily routine.",
        "Talk about a person who inspires you.",
        "Describe your dream job.",
        "Explain your favorite movie."
    ]

    challenge = challenges[
        date.today().day % len(challenges)
    ]

    st.markdown(
        f"""
        <div class="card">
            <h2>🎯 Today's Challenge</h2>
            <p>{challenge}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    answer = st.text_area(
        "✍️ Your answer",
        height=180,
        placeholder="Write your answer here..."
    )

    if st.button(
        "🏆 Submit Challenge",
        use_container_width=True,
        type="primary"
    ):

        if not answer.strip():

            st.warning(
                "Write your answer first."
            )

        else:

            prompt = f"""
Challenge:
{challenge}

Student answer:
{answer}

Evaluate the answer.

Return ONLY JSON:

{{
    "score": 86,
    "feedback": "Good response.",
    "better_version": "Improved natural version."
}}
"""

            with st.spinner(
                "🤖 Checking your answer..."
            ):

                result = ask_ollama(
                    prompt,
                    "You are a friendly language teacher."
                )

            data = extract_json(result)

            if data:

                score = int(
                    data.get(
                        "score",
                        0
                    )
                )

                st.success(
                    f"🏆 Score: {score}/100"
                )

                st.write(
                    data.get(
                        "feedback",
                        ""
                    )
                )

                st.subheader(
                    "✨ Better Version"
                )

                st.info(
                    data.get(
                        "better_version",
                        ""
                    )
                )

                today = date.today().isoformat()

                if today not in st.session_state.completed_challenges:

                    st.session_state.completed_challenges.append(
                        today
                    )

                    add_xp(20)

                    st.success(
                        "⭐ +20 XP earned!"
                    )


# ============================================================
# PROGRESS
# ============================================================

elif page == "📊 Progress":

    st.title("📊 Your Progress")

    average = 0

    if st.session_state.scores:

        average = round(
            sum(st.session_state.scores)
            / len(st.session_state.scores)
        )

    cols = st.columns(4)

    progress_data = [
        (
            "💬",
            len(st.session_state.history),
            "Conversations"
        ),
        (
            "📚",
            len(st.session_state.saved_words),
            "Words"
        ),
        (
            "⭐",
            st.session_state.xp,
            "XP"
        ),
        (
            "📈",
            f"{average}%",
            "Average Score"
        )
    ]

    for col, item in zip(
        cols,
        progress_data
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">
                        {item[0]}
                    </div>
                    <div class="metric-number">
                        {item[1]}
                    </div>
                    <div class="metric-label">
                        {item[2]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    current = level_info(
        st.session_state.xp
    )

    st.subheader(
        f"🏆 Current Level: {current[1]}"
    )

    st.progress(
        min(
            st.session_state.xp / current[2],
            1
        )
    )

    st.caption(
        f"{st.session_state.xp} / {current[2]} XP"
    )

    st.divider()

    st.subheader(
        "📈 Communication Skills"
    )

    if st.session_state.analysis:

        data = st.session_state.analysis

        skills = {
            "Grammar": data.get(
                "grammar",
                0
            ),
            "Vocabulary": data.get(
                "vocabulary_score",
                0
            ),
            "Fluency": data.get(
                "fluency",
                0
            ),
            "Sentence Formation": data.get(
                "sentence_formation",
                0
            ),
            "Overall": data.get(
                "overall",
                0
            )
        }

        for name, score in skills.items():

            st.write(
                f"**{name} — {score}%**"
            )

            st.progress(
                min(
                    score / 100,
                    1
                )
            )

    else:

        st.info(
            "Complete a communication analysis to see your skill scores."
        )


# ============================================================
# TRANSLATION
# ============================================================

elif page == "🌎 Translation":

    st.title("🌎 Translation + Learning")

    languages = {
        "English": "en",
        "Telugu": "te",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "Japanese": "ja",
        "Korean": "ko"
    }

    c1, c2 = st.columns(2)

    with c1:

        source = st.selectbox(
            "From",
            list(languages.keys())
        )

    with c2:

        target = st.selectbox(
            "To",
            list(languages.keys()),
            index=1
        )

    text = st.text_area(
        "📝 Enter sentence",
        placeholder="Where is the railway station?",
        height=140
    )

    if st.button(
        "🌎 Translate",
        use_container_width=True,
        type="primary"
    ):

        if not text.strip():

            st.warning(
                "Enter a sentence."
            )

        elif source == target:

            st.warning(
                "Choose different languages."
            )

        else:

            try:

                translated = GoogleTranslator(
                    source=languages[source],
                    target=languages[target]
                ).translate(
                    text
                )

                st.success(
                    "✅ Translation"
                )

                st.markdown(
                    f"""
                    <div class="card">
                        <h2>{translated}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.divider()

                st.subheader(
                    "📚 Learning Mode"
                )

                with st.spinner(
                    "📖 Preparing learning explanation..."
                ):

                    learning = ask_ollama(
                        f"""
Original sentence:
{text}

Translated sentence:
{translated}

Explain:
1. Simple meaning
2. Useful words
3. One example sentence
4. Where this sentence can be used
5. Simple pronunciation guidance if possible

Keep it concise.
""",
                        "You are a language teacher."
                    )

                st.write(
                    learning
                )

            except Exception as e:

                st.error(
                    f"Translation error: {e}"
                )


# ============================================================
# VOICE + PRONUNCIATION
# ============================================================

elif page == "🎤 Voice & Pronunciation":

    st.title("🎤 Voice & Pronunciation")

    st.caption(
        "Speak naturally and let LinguaMate understand your voice."
    )

    language = st.selectbox(
        "🌐 Speech Language",
        [
            "English",
            "Hindi",
            "Telugu"
        ]
    )

    speech_codes = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Telugu": "te-IN"
    }

    situation = st.selectbox(
        "🎯 Situation",
        [
            "Daily Conversation",
            "Job Interview",
            "Travel",
            "College",
            "Restaurant",
            "Making Friends"
        ]
    )

    st.divider()

    # --------------------------------------------------------
    # VOICE CONVERSATION
    # --------------------------------------------------------

    st.subheader(
        "🎙️ Voice Conversation"
    )

    st.info(
        "Click Start Recording, speak clearly, then stop recording."
    )

    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="voice_recording_main"
    )

    if audio:

        st.success(
            "🎙️ Recording captured!"
        )

        st.audio(
            audio["bytes"],
            format="audio/wav"
        )

        if st.button(
            "📝 Convert Voice to Text",
            use_container_width=True,
            type="primary"
        ):

            try:

                with st.spinner(
                    "🎧 Converting your voice..."
                ):

                    spoken_text = recognize_audio(
                        audio["bytes"],
                        speech_codes[language]
                    )

                st.success(
                    "✅ Speech converted successfully!"
                )

                st.subheader(
                    "📝 You said"
                )

                st.markdown(
                    f"""
                    <div class="card">
                        <h3>🗣️ {spoken_text}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.divider()

                st.subheader(
                    "🧠 LinguaMate Response"
                )

                with st.spinner(
                    "🤖 Thinking..."
                ):

                    response = ask_ollama(
                        f"""
The learner said:

"{spoken_text}"

Language:
{language}

Situation:
{situation}

Reply naturally as a language practice partner.

Keep your response short.
Ask one follow-up question.
""",
                        "You are LinguaMate, a friendly speaking practice partner."
                    )

                st.chat_message(
                    "assistant",
                    avatar="🌍"
                ).write(
                    response
                )

                # AUTO SAVE VOICE HISTORY

                st.session_state.voice_history.append(
                    {
                        "date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "language": language,
                        "situation": situation,
                        "user": spoken_text,
                        "assistant": response
                    }
                )

                add_xp(5)

                st.success(
                    "💾 Voice conversation automatically saved to History! +5 XP"
                )

            except sr.UnknownValueError:

                st.error(
                    "❌ I couldn't understand the speech. "
                    "Please speak clearly and try again."
                )

            except sr.RequestError:

                st.error(
                    "❌ Speech recognition service is unavailable."
                )

            except Exception as e:

                st.error(
                    f"❌ Audio processing error: {e}"
                )

    st.divider()

    # --------------------------------------------------------
    # PRONUNCIATION
    # --------------------------------------------------------

    st.subheader(
        "🔊 Pronunciation Practice"
    )

    st.caption(
        "This gives a speech-match score based on recognized words."
    )

    target = st.text_input(
        "🎯 Enter a sentence to practice",
        placeholder="I am learning English every day."
    )

    if target.strip():

        st.info(
            f"🎯 Target sentence: **{target}**"
        )

        pronunciation_audio = mic_recorder(
            start_prompt="🎤 Practice Sentence",
            stop_prompt="⏹️ Stop Practice",
            just_once=True,
            use_container_width=True,
            key="pronunciation_recording"
        )

        if pronunciation_audio:

            st.audio(
                pronunciation_audio["bytes"],
                format="audio/wav"
            )

            if st.button(
                "🔊 Check My Practice",
                use_container_width=True,
                type="primary"
            ):

                try:

                    with st.spinner(
                        "🎧 Checking pronunciation..."
                    ):

                        spoken = recognize_audio(
                            pronunciation_audio["bytes"],
                            "en-US"
                        )

                    score = speech_match_score(
                        target,
                        spoken
                    )

                    st.divider()

                    st.subheader(
                        "📊 Practice Result"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Speech Match",
                        f"{score}%"
                    )

                    c2.metric(
                        "Target Words",
                        len(target.split())
                    )

                    c3.metric(
                        "Recognized Words",
                        len(spoken.split())
                    )

                    st.write(
                        "🎯 **Target:**",
                        target
                    )

                    st.write(
                        "🗣️ **Recognized:**",
                        spoken
                    )

                    if score >= 90:

                        st.success(
                            "🔥 Excellent! Very close match."
                        )

                    elif score >= 75:

                        st.success(
                            "👏 Good! Keep practicing."
                        )

                    elif score >= 50:

                        st.warning(
                            "🙂 Good attempt. Speak slowly and clearly."
                        )

                    else:

                        st.warning(
                            "💪 Keep practicing. Break the sentence into smaller parts."
                        )

                    feedback = ask_ollama(
                        f"""
Target:
{target}

Recognized:
{spoken}

Speech-match score:
{score}%

Give:
1. One short pronunciation practice tip
2. Words that may need attention
3. One practice instruction

Do not claim phoneme-level acoustic analysis.
""",
                        "You are a pronunciation practice coach."
                    )

                    st.info(
                        feedback
                    )

                    # SAVE PRONUNCIATION HISTORY

                    st.session_state.pronunciation_history.append(
                        {
                            "date": datetime.now().strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                            "target": target,
                            "recognized": spoken,
                            "score": score
                        }
                    )

                    add_xp(5)

                    st.success(
                        "💾 Pronunciation result saved to History! +5 XP"
                    )

                except sr.UnknownValueError:

                    st.error(
                        "❌ Could not understand your pronunciation."
                    )

                except sr.RequestError:

                    st.error(
                        "❌ Speech recognition service unavailable."
                    )

                except Exception as e:

                    st.error(
                        f"❌ Pronunciation processing error: {e}"
                    )


# ============================================================
# HISTORY
# ============================================================

elif page == "💾 History":

    st.title("💾 Learning History")

    st.caption(
        "Your conversations, voice practice and pronunciation results."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "💬 Conversations",
            "🎤 Voice History",
            "🔊 Pronunciation"
        ]
    )

    # --------------------------------------------------------
    # NORMAL CONVERSATION HISTORY
    # --------------------------------------------------------

    with tab1:

        history = st.session_state.history

        if not history:

            st.info(
                "No saved conversations yet."
            )

        else:

            st.write(
                f"**{len(history)} saved conversations**"
            )

            for i, item in enumerate(
                reversed(history)
            ):

                title = (
                    f"{item.get('date', '')} • "
                    f"{item.get('language', 'English')} • "
                    f"{item.get('situation', 'Conversation')}"
                )

                with st.expander(
                    title
                ):

                    st.write(
                        f"**Difficulty:** "
                        f"{item.get('difficulty', '')}"
                    )

                    st.write(
                        f"**Mode:** "
                        f"{item.get('mode', 'Normal Conversation')}"
                    )

                    for message in item.get(
                        "messages",
                        []
                    ):

                        if message["role"] == "user":

                            st.chat_message(
                                "user",
                                avatar="👤"
                            ).write(
                                message["content"]
                            )

                        else:

                            st.chat_message(
                                "assistant",
                                avatar="🌍"
                            ).write(
                                message["content"]
                            )

        st.divider()

        if st.button(
            "🗑️ Clear Conversation History",
            use_container_width=True
        ):

            st.session_state.history = []

            save_data()

            st.success(
                "Conversation history cleared."
            )

            st.rerun()

    # --------------------------------------------------------
    # VOICE HISTORY
    # --------------------------------------------------------

    with tab2:

        voice_history = st.session_state.voice_history

        if not voice_history:

            st.info(
                "No voice conversations yet."
            )

        else:

            st.write(
                f"**{len(voice_history)} voice conversations**"
            )

            for item in reversed(
                voice_history
            ):

                with st.expander(
                    f"🎤 {item.get('date', '')} • "
                    f"{item.get('language', 'English')} • "
                    f"{item.get('situation', '')}"
                ):

                    st.write(
                        "🗣️ **You said:**"
                    )

                    st.write(
                        item.get(
                            "user",
                            ""
                        )
                    )

                    st.write(
                        "🌍 **LinguaMate:**"
                    )

                    st.write(
                        item.get(
                            "assistant",
                            ""
                        )
                    )

        if voice_history:

            if st.button(
                "🗑️ Clear Voice History",
                use_container_width=True
            ):

                st.session_state.voice_history = []

                save_data()

                st.success(
                    "Voice history cleared."
                )

                st.rerun()

    # --------------------------------------------------------
    # PRONUNCIATION HISTORY
    # --------------------------------------------------------

    with tab3:

        pronunciation_history = (
            st.session_state.pronunciation_history
        )

        if not pronunciation_history:

            st.info(
                "No pronunciation practice yet."
            )

        else:

            st.write(
                f"**{len(pronunciation_history)} pronunciation attempts**"
            )

            for item in reversed(
                pronunciation_history
            ):

                with st.expander(
                    f"🔊 {item.get('date', '')} • "
                    f"Score: {item.get('score', 0)}%"
                ):

                    st.write(
                        "🎯 **Target:**",
                        item.get(
                            "target",
                            ""
                        )
                    )

                    st.write(
                        "🗣️ **Recognized:**",
                        item.get(
                            "recognized",
                            ""
                        )
                    )

                    st.metric(
                        "Speech Match",
                        f"{item.get('score', 0)}%"
                    )

        if pronunciation_history:

            if st.button(
                "🗑️ Clear Pronunciation History",
                use_container_width=True
            ):

                st.session_state.pronunciation_history = []

                save_data()

                st.success(
                    "Pronunciation history cleared."
                )

                st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🌍 <b>LinguaMate</b> • Learn Languages. Speak Confidently.
        <br>
        AI-powered language learning assistant
    </div>
    """,
    unsafe_allow_html=True
)