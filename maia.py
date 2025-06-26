import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Constants
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = 'Ⓜ️'
new_chat_id = f'{time.time()}'

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if '_history' not in st.session_state:
    st.session_state._history = []
if 'chat_id' not in st.session_state:
    st.session_state.chat_id = new_chat_id
if 'chat_title' not in st.session_state:
    st.session_state.chat_title = f'ChatSession-{new_chat_id}'

# Create data directory if not exists
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
        options=[new_chat_id] + list(past_chats.keys()),
        format_func=lambda x: past_chats.get(x, 'New Chat'),
        index=0,
    )
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

st.title('Chat with Gemini')

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

# Initialize model
st.session_state.model = genai.GenerativeModel('gemini-2.5-flash')
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

# User input
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

    # Get AI response
    response = st.session_state.chat.send_message(prompt, stream=True)
    
    # Display AI response
    with st.chat_message(MODEL_ROLE, avatar=AI_AVATAR_ICON):
        message_placeholder = st.empty()
        full_response = ''
        
        for chunk in response:
            for word in chunk.text.split(' '):
                full_response += word + ' '
                time.sleep(0.05)
                message_placeholder.write(full_response + '▌')
        
        message_placeholder.write(full_response)

    # Update history
    st.session_state.messages.append({
        'role': MODEL_ROLE,
        'content': full_response,
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
