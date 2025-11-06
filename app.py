import streamlit as st
from rag_pipeline import generate_answer

# Page setup
st.set_page_config(page_title="Marcus - Mental Health Companion", page_icon="💙", layout="centered")

st.title("💬 Marcus — Your Mental Health Companion")
st.write("Hi there 👋 I'm Marcus, here to listen and support you. You can share your thoughts below.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Marcus:** {msg['content']}")

# User input
query = st.chat_input("Type your message here...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate Marcus response
    with st.spinner("Marcus is thinking..."):
        answer = generate_answer(query)

    # Add bot message
    st.session_state.messages.append({"role": "bot", "content": answer})

    # Rerun UI
    st.rerun()
