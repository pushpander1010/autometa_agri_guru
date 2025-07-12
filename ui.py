import streamlit as st
import re
from langchain_community.vectorstores import FAISS
from backend import (
    get_llm,
    GROQ_MODELS,
    generate_crop_plan,
    RAGWithLLMFallback,
    embedding_model
)
from tools import get_state_coordinates

# === Static Data ===
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh"
]

# === Utility ===
def extract_image_urls(text):
    return re.findall(r'(https?://[^\s]+(?:\.png|\.jpg|\.jpeg|\.webp))', text)

# === Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "curr_loc" not in st.session_state:
    st.session_state.curr_loc = None

# === UI Header ===
st.title("🌾 AgriGuru: Your AI Farming Assistant")
st.markdown("1. Select your location and model.  \n2. Submit to get your crop plan.  \n3. Then ask follow-ups below.")

# === Location and Model Selection ===
with st.form("location_form"):
    state = st.selectbox("📍 Select State", INDIAN_STATES)
    village = st.text_input("🏡 Enter Village Name")
    selected_model = st.selectbox("🧠 Choose Groq Model", GROQ_MODELS)
    submit = st.form_submit_button("🚀 Generate Crop Plan")

# === On Submit ===
if submit:
    with st.spinner("Generating personalized crop plan..."):
        llm = get_llm(selected_model)

        # Construct location object
        location = get_state_coordinates.invoke(state)

        result = generate_crop_plan(llm, location)

        st.session_state.original_crop_plan = result["crop_plan"]
        st.session_state.crop_info = result["crop_info"]
        st.session_state.rag_docs = result["rag_docs"]
        st.session_state.curr_loc = result["curr_loc"]
        st.session_state.vectorstore = FAISS.from_documents(result["rag_docs"], embedding_model)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.session_state.smart_rag = RAGWithLLMFallback(llm, result["rag_docs"])

        st.success("✅ Crop plan generated!")

        with st.expander("📄 Crop Plan"):
            st.markdown(result["crop_plan"])

        with st.expander("🌱 Crop Info"):
            st.markdown(result["crop_info"])

        image_urls = extract_image_urls(result["crop_info"])
        if image_urls:
            st.markdown("### 🌾 Crop Images")
            for url in image_urls:
                st.image(url, width=250, caption="Crop")

# === Chatbot Interface ===
if "smart_rag" in st.session_state:
    st.divider()
    st.markdown("## 🤖 Chat with AgriGuru")

    for speaker, message in st.session_state.chat_history:
        with st.chat_message(speaker):
            st.markdown(message)

    user_q = st.chat_input("Ask anything about farming, feedback, or follow-up...")

    if user_q:
        st.chat_message("👨‍🌾").markdown(user_q)
        st.session_state.chat_history.append(("👨‍🌾", user_q))

        with st.spinner("💬 Thinking..."):
            response = st.session_state.smart_rag.invoke(user_q)

        st.chat_message("🤖 AgriGuru").markdown(response)
        st.session_state.chat_history.append(("🤖 AgriGuru", response))
        st.rerun()
