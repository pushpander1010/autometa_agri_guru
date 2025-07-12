import streamlit as st
import re
from langchain_community.vectorstores import FAISS
from backend import (
    get_llm,
    GROQ_MODELS,
    crop_graph,
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

# === Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "curr_loc" not in st.session_state:
    st.session_state.curr_loc = None

if "crop_plan_ready" not in st.session_state:
    st.session_state.crop_plan_ready = False

# === UI Header ===
st.title("🌾 AgriGuru: Your AI Farming Assistant")
st.markdown("1. Select your location and model.  \n2. Submit to get your crop plan.  \n3. Then ask follow-ups below.")

# === Location and Model Selection ===
with st.form("location_form"):
    state = st.selectbox("📍 Select State", INDIAN_STATES)
    village = st.text_input("🏡 Enter Village Name")
    selected_model = st.selectbox(
        "🧠 Choose Groq Model", 
        GROQ_MODELS, 
        index=GROQ_MODELS.index("meta-llama/llama-4-scout-17b-16e-instruct")
    )
    submit = st.form_submit_button("🚀 Generate Crop Plan")

# === On Submit ===
if submit:
    with st.spinner("Generating personalized crop plan..."):
        llm = get_llm(selected_model)
        location = get_state_coordinates.invoke(state)

        final_state = crop_graph.invoke({"location": location})

        st.session_state.original_crop_plan = final_state["crop_plan"]
        st.session_state.crop_info = final_state["crop_info"]
        st.session_state.rag_docs = final_state["rag_docs"]
        st.session_state.curr_loc = location
        st.session_state.vectorstore = FAISS.from_documents(final_state["rag_docs"], embedding_model)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.session_state.smart_rag = RAGWithLLMFallback(final_state["rag_docs"])
        st.session_state.crop_plan_ready = True

        st.success("✅ Crop plan generated!")

# === Display Crop Plan Section ===
if st.session_state.get("crop_plan_ready"):
    with st.expander("📄 Detailed Crop Plan", expanded=True):
        st.markdown(st.session_state.original_crop_plan)

    with st.expander("🌱 Expert Crop Info"):
        st.markdown(st.session_state.crop_info)

    # === Download Button ===
    st.download_button(
        label="📥 Download Crop Plan",
        data=st.session_state.original_crop_plan + "\n\n" + st.session_state.crop_info,
        file_name="AgriGuru_Crop_Plan.txt",
        mime="text/plain"
    )

    # === Chatbot Interface ===
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
