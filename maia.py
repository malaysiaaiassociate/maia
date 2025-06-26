import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import pytz
import requests
from typing import Optional, List, Dict

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Constants
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = 'Ⓜ️'
MODEL_NAME = "gemini-2.5-flash"
MAX_HISTORY_LENGTH = 10
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID', '95594d69ad5634e0d')
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# System Prompt Template
SYSTEM_PROMPT = """
YYou are a helpful AI assistant named MAIA. Your AI model was developed and trained by maia.aio. Follow these guidelines:
1. Maintain context of the conversation history provided
2. Respond concisely in the user's language
3. For current info, use provided web results with sources
4. For weather queries, provide detailed forecasts without sources
5. Specify Malaysia Time (MYT, UTC+8) for time-sensitive info when asked
7. Maintain professional yet friendly tone
8. For news requests, use the provided news data

Current Malaysia Time: {{time}}
{{weatherContext}}
{{searchContext}}
{{newsContext}}

CONVERSATION HISTORY:
{{chatHistory}}

Current Message: {{userMessage}}
"""

# Initialize session state
def init_session_state():
    defaults = {
        'messages': [],
        '_history': [],
        'chat_id': f'{time.time()}',
        'chat_title': f'ChatSession-{time.time()}',
        'model': genai.GenerativeModel(MODEL_NAME),
        'last_active': time.time()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Malaysia time formatter
def get_malaysia_time() -> str:
    kl_timezone = pytz.timezone('Asia/Kuala_Lumpur')
    return datetime.now(kl_timezone).strftime('%A, %d %B %Y, %I:%M:%S %p (MYT)')

# Enhanced search function
def google_search(query: str, num_results: int = 3) -> Optional[List[Dict]]:
    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                'q': query,
                'key': GOOGLE_API_KEY,
                'cx': SEARCH_ENGINE_ID,
                'num': min(num_results, 5),
                'gl': 'MY'
            },
            timeout=5
        )
        response.raise_for_status()
        return response.json().get('items', [])[:num_results]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

# News fetching function
def fetch_news(query: str = '', country: str = 'my', limit: int = 3) -> List[Dict]:
    try:
        # Try GNews first if API key available
        if GNEWS_API_KEY:
            try:
                response = requests.get(
                    f"https://gnews.io/api/v4/top-headlines?token={GNEWS_API_KEY}",
                    params={
                        'q': query,
                        'country': country,
                        'max': limit
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    return response.json().get('articles', [])
            except:
                pass  # Fall through to Google search
        
        # Fallback to Google search
        results = google_search(f"{query} site:thestar.com.my OR site:malaymail.com", limit)
        return [{
            'title': r.get('title', ''),
            'source': 'Google Search',
            'url': r.get('link', ''),
            'description': r.get('snippet', '')
        } for r in results] if results else []
    except:
        return []

# Weather detection
def needs_weather_info(prompt: str) -> bool:
    weather_keywords = ['weather', 'forecast', 'temperature', 'cuaca', '天气', 'வானிலை']
    return any(keyword in prompt.lower() for keyword in weather_keywords)

# News detection
def needs_news_info(prompt: str) -> bool:
    news_keywords = ['news', 'headlines', 'berita', '最新消息', 'செய்திகள்']
    return any(keyword in prompt.lower() for keyword in news_keywords)

# Format news for display
def format_news(news: List[Dict]) -> str:
    if not news:
        return "No news found"
    return "📰 Latest News:\n" + "\n\n".join(
        f"{i+1}. {item['title']}\n   {item.get('description', '')}\n   Source: {item.get('source', 'Unknown')}"
        for i, item in enumerate(news)
    )

# Initialize the app
init_session_state()
os.makedirs('data/', exist_ok=True)

# Load past chats
try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar for past chats
with st.sidebar:
    st.write('# Past Chats')
    st.session_state.chat_id = st.selectbox(
        label='Pick a past chat',
        options=[st.session_state.chat_id] + list(past_chats.keys()),
        format_func=lambda x: past_chats.get(x, 'New Chat'),
        index=0
    )
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

    # Add clear chat button
    if st.button('Clear Current Chat'):
        st.session_state.messages = []
        st.session_state._history = []
        st.rerun()

st.title('Chat with MAIA')

# Load chat history
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state._history = joblib.load(
        f'data/{st.session_state.chat_id}-_messages'
    )
except:
    st.session_state.messages = []
    st.session_state._history = []

# Initialize chat model
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state._history
)

# Display messages
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

# Enhanced response generation
def generate_response(prompt: str) -> str:
    time_info = get_malaysia_time()
    weather_context = ''
    search_context = ''
    news_context = ''
    
    # Check for weather queries
    if needs_weather_info(prompt):
        location = 'Malaysia'  # Could extract location from prompt
        results = google_search(f"{location} weather forecast")
        if results:
            weather_context = "🌦️ Weather Data:\n" + "\n".join(
                f"- {r['title']}: {r['snippet']}" 
                for r in results[:2]
            )
    
    # Check for news queries
    if needs_news_info(prompt):
        query = prompt.replace('news', '').strip() or 'Malaysia'
        news = fetch_news(query)
        if news:
            news_context = format_news(news[:3])
            if prompt.lower().startswith('news'):
                return news_context  # Direct response for news queries
    
    # Prepare the prompt
    chat_history = "\n".join(
        f"{msg['role']}: {msg['content']}" 
        for msg in st.session_state.messages[-MAX_HISTORY_LENGTH:]
    )
    
    full_prompt = SYSTEM_PROMPT.format(
        time=time_info,
        weather_context=weather_context,
        search_context=search_context,
        news_context=news_context,
        chat_history=chat_history,
        user_message=prompt
    )
    
    # Generate response
    response = st.session_state.chat.send_message(full_prompt, stream=True)
    return "".join(chunk.text for chunk in response)

# User input handling
if prompt := st.chat_input('Your message here...'):
    # Save new chat
    if st.session_state.chat_id not in past_chats:
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    
    # Display user message
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })
    
    # Generate and display AI response
    with st.chat_message(MODEL_ROLE, avatar=AI_AVATAR_ICON):
        message_placeholder = st.empty()
        full_response = ''
        
        # For direct news queries, skip streaming
        if needs_news_info(prompt) and prompt.lower().startswith('news'):
            response = generate_response(prompt)
            message_placeholder.write(response)
        else:
            response = generate_response(prompt)
            for word in response.split(' '):
                full_response += word + ' '
                time.sleep(0.05)
                message_placeholder.write(full_response + '▌')
            message_placeholder.write(full_response)
    
    # Update history
    st.session_state.messages.append({
        'role': MODEL_ROLE,
        'content': full_response if 'full_response' in locals() else response,
        'avatar': AI_AVATAR_ICON
    })
    st.session_state._history = st.session_state.chat.history
    
    # Save conversation
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages'
    )
    joblib.dump(
        st.session_state._history,
        f'data/{st.session_state.chat_id}-_messages'
    )
