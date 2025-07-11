import streamlit as st
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from backend import (
    agri_agent,
    revise_crop_plan,
    get_curr_loc_tool,
    get_soil_location,
    get_farm_prices,
    get_soil_properties,
    get_seasonal_weather_data,
    RAGWithLLMFallback,
    embedding_model,
    llm
)

# === Initialization ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_docs" not in st.session_state:
    with st.spinner("🚜 Generating initial crop plan..."):
        result = agri_agent.invoke({"query": "Which crops should I grow?"})
        st.session_state.original_crop_plan = result["crop_plan"].content
        st.session_state.rag_docs = result["rag_docs"]
        st.session_state.vectorstore = FAISS.from_documents(
            [Document(page_content=doc.page_content) for doc in st.session_state.rag_docs],
            embedding_model
        )
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.session_state.smart_rag = RAGWithLLMFallback(llm, st.session_state.retriever)

# === UI Layout ===
st.title("🌾 AgriGuru: Your AI Farming Assistant")
st.markdown("Ask any farming-related question or tell us what you'd like to grow.")

# Display chat history
for speaker, message in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(message)

# User input
user_q = st.chat_input("Ask about crops, soil, weather, or give feedback like 'I want to grow wheat'...")

if user_q:
    st.chat_message("👨‍🌾").markdown(user_q)
    st.session_state.chat_history.append(("👨‍🌾", user_q))

    # === Feedback-based replanning ===
    if any(keyword in user_q.lower() for keyword in ["i want", "can i grow", "instead", "prefer", "avoid", "only"]):
        with st.spinner("🔁 Reworking crop plan based on your input..."):
            revised_plan = revise_crop_plan(st.session_state.original_crop_plan, user_q)
            loc = get_curr_loc_tool.invoke(None)
            soil_info_api = get_soil_properties.invoke(None)
            soil_info_llm = get_soil_location(loc.get("regionName", "Haryana"))
            seasonal_weather = get_seasonal_weather_data.invoke(None)

            prompt = f"""
Crop Plan:
{revised_plan}

Location:
{loc}

Soil Info:
{soil_info_api} {soil_info_llm}

Seasonal Weather:
{seasonal_weather}

Now generate:
- Common diseases & treatments
- Fertilizer types & usage
- Step-by-step growing instructions
- Tips for small/marginal farmers
- Government subsidies
- URLs for pictures of the crops
"""

 
            new_info = llm.invoke(prompt).content

            new_docs = [
                Document(page_content=revised_plan, metadata={"type": "plan"}),
                Document(page_content=new_info, metadata={"type": "info"}),
            ]

            # Update session state with new plan
            st.session_state.original_crop_plan = revised_plan
            st.session_state.rag_docs = new_docs
            st.session_state.vectorstore = FAISS.from_documents(new_docs, embedding_model)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever()
            st.session_state.smart_rag = RAGWithLLMFallback(llm, st.session_state.retriever)

            response = "✅ New crop plan ready. You can ask follow-up questions now."
    else:
        # === Standard query with RAG ===
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.smart_rag.invoke(user_q)

    st.session_state.chat_history.append(("🤖 AgriGuru", response))
    st.rerun()