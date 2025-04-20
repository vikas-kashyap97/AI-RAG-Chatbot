import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="wide", page_icon="🤖")

st.markdown("<h1 style='text-align: center;'>🤖 AI Chatbot Agents</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Design, query, and interact with intelligent agents powered by cutting-edge models.</p>", unsafe_allow_html=True)
st.markdown("---")

# Agent config
st.subheader("🧠 Define Your AI Agent")
system_prompt = st.text_area(
    label="System Prompt", 
    height=70, 
    placeholder="Type your system prompt here...",
    help="This sets the behavior/personality of the AI agent."
)
st.markdown("---")

st.subheader("⚙️ Select Model Configuration")
col1, col2 = st.columns(2)

with col1:
    provider = st.radio("Provider", ("Groq", "Google"), horizontal=True)

with col2:
    model_options = {
        "Groq": ["llama3-70b-8192", "llama-3.3-70b-versatile"],
        "Google": ["gemini-1.5-flash-latest"]
    }
    selected_model = st.selectbox("Model", model_options[provider])

allow_web_search = st.checkbox("Enable Web Search", help="Allow the agent to use web search for live information.")
st.markdown("---")

# User query
st.subheader("💬 Ask the Agent")
user_query = st.text_area(
    label="Your Question", 
    height=150, 
    placeholder="Ask anything...",
    help="This is your input to the AI agent."
)

API_URL = "https://ai-rag-chatbot.onrender.com/chat"

if st.button("🚀 Ask Agent!"):
    if user_query.strip():
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            response = requests.post(API_URL, json=payload)
            st.markdown("---")

            if response.status_code == 200:
                response_data = response.json()
                if "error" in response_data:
                    st.error(f"❌ Error: {response_data['error']}")
                else:
                    st.success("✅ Agent Response")
                    st.markdown(f"**Response:**\n\n{response_data['response']}")
            else:
                st.error(f"❌ Backend returned status code {response.status_code}")
        except Exception as e:
            st.error(f"⚠️ Request failed: {e}")
    else:
        st.warning("⚠️ Please enter a query before submitting.")
