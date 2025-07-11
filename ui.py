import streamlit as st
from backend import (
    agri_agent, revise_crop_plan, extract_crop_name,
    get_farm_prices, get_soil_properties, get_seasonal_weather_data,
    RAGWithLLMFallback, embedding_model, llm
)
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "init_result" not in st.session_state:
    st.session_state.init_result = agri_agent.invoke({"query": "Which crops should I grow?"})
    st.session_state.original_crop_plan = st.session_state.init_result["crop_plan"].content
    st.session_state.rag_docs = st.session_state.init_result["rag_docs"]
    st.session_state.vectorstore = FAISS.from_documents(
        [Document(page_content=doc.page_content) for doc in st.session_state.rag_docs],
        embedding_model
    )
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    st.session_state.smart_rag = RAGWithLLMFallback(llm, st.session_state.retriever)

st.title("🌾 AgriGuru: Your AI Farming Assistant")
st.markdown("Ask farming-related questions or tell us what you want to grow!")

# Display past chat
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}**: {msg}")

# User input
user_q = st.chat_input("👨‍🌾 Type your question or feedback...")

if user_q:
    st.session_state.chat_history.append(("👨‍🌾", user_q))

    if any(word in user_q.lower() for word in ["i want", "can i grow", "instead", "prefer", "avoid", "only"]):
        st.markdown("🔁 Reworking crop plan based on your feedback...")
        st.session_state.chat_history = []  # Reset history on replan

        revised_plan = revise_crop_plan(st.session_state.original_crop_plan, user_q)
        crop_name = extract_crop_name(revised_plan)
        mandi_info = get_farm_prices.invoke(crop_name)
        soil_info = get_soil_properties.invoke(None)
        weather_info = get_seasonal_weather_data.invoke(None)

        prompt = f"""
Revised Plan:
{revised_plan}

Mandi Info:
{mandi_info}

Soil:
{soil_info}

Weather:
{weather_info}

Give crop info: diseases, fertilizers, steps, tips.
"""
        new_info = llm.invoke(prompt).content

        # Rebuild vectorstore
        new_docs = [
            Document(page_content=revised_plan, metadata={"type": "plan"}),
            Document(page_content=new_info, metadata={"type": "info"}),
            Document(page_content=soil_info, metadata={"type": "soil"}),
            Document(page_content=weather_info, metadata={"type": "weather"})
        ]
        st.session_state.rag_docs = new_docs
        st.session_state.vectorstore = FAISS.from_documents(new_docs, embedding_model)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.session_state.smart_rag = RAGWithLLMFallback(llm, st.session_state.retriever)
        st.session_state.original_crop_plan = revised_plan

        response = "✅ Crop plan revised. Ask questions about the new crop."
    else:
        response = st.session_state.smart_rag.invoke(user_q)

    st.session_state.chat_history.append(("🤖 AgriGuru", response))
    st.rerun()
