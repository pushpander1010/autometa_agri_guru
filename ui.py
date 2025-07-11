import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from backend import agri_agent, revise_crop_plan, extract_crop_name, get_farm_prices, get_soil_properties, get_seasonal_weather_data, RAGWithLLMFallback, embedding_model, llm

st.set_page_config(page_title="AgriGuru 🧑‍🌾", page_icon="🌾")
st.title("🌾 AgriGuru - Your Smart Farming Assistant")

# Step 1: Run agent on load or trigger
if "init_result" not in st.session_state:
    with st.spinner("Preparing personalized crop plan..."):
        st.session_state.init_result = agri_agent.invoke({"query": "Which crops should I grow?"})
        st.session_state.original_crop_plan = st.session_state.init_result["crop_plan"].content
        st.session_state.curr_loc = st.session_state.init_result.get("curr_loc", {"regionName": "Haryana"})
        st.session_state.rag_docs = st.session_state.init_result["rag_docs"]
        vectorstore = FAISS.from_documents(
            [Document(page_content=doc.page_content) for doc in st.session_state.rag_docs],
            embedding_model
        )
        retriever = vectorstore.as_retriever()
        st.session_state.smart_rag = RAGWithLLMFallback(llm, retriever)
        st.success("Ready to assist you!")

# Step 2: Take user input
user_input = st.chat_input("Ask a question or give feedback like 'I want to grow wheat'")
if user_input:
    st.chat_message("user").markdown(user_input)

    if any(word in user_input.lower() for word in ["mandi price", "price", "market price"]):
        crop = extract_crop_name(st.session_state.original_crop_plan)
        state = st.session_state.curr_loc.get("regionName", "Haryana")
        mandi_info = get_farm_prices.invoke({"commodity": crop, "state": state})
        st.chat_message("assistant").markdown(f"📊 Mandi Info for **{crop}** in **{state}**:\n\n{mandi_info}")

    elif any(phrase in user_input.lower() for phrase in ["i want", "can i grow", "instead", "prefer", "avoid", "only"]):
        st.info("Revising crop plan...")
        revised = revise_crop_plan(st.session_state.original_crop_plan, user_input)
        st.session_state.original_crop_plan = revised  # update for next feedback

        crop_name = extract_crop_name(revised)
        mandi_info = get_farm_prices.invoke(crop_name)
        soil = get_soil_properties.invoke(None)
        weather = get_seasonal_weather_data.invoke(None)

        prompt = f"""
Revised Plan:\n{revised}

Mandi Info:\n{mandi_info}

Soil:\n{soil}

Seasonal Weather:\n{weather}

Give crop info: diseases, fertilizers, steps to grow, tips.
"""
        new_info = llm.invoke(prompt).content

        st.session_state.rag_docs = [
            Document(page_content=revised, metadata={"type": "plan"}),
            Document(page_content=new_info, metadata={"type": "info"}),
            Document(page_content=soil, metadata={"type": "soil"}),
            Document(page_content=weather, metadata={"type": "weather"})
        ]
        vectorstore = FAISS.from_documents(st.session_state.rag_docs, embedding_model)
        retriever = vectorstore.as_retriever()
        st.session_state.smart_rag = RAGWithLLMFallback(llm, retriever)

        st.success("✅ Plan revised! You can continue asking.")

    else:
        response = st.session_state.smart_rag.invoke(user_input)
        st.chat_message("assistant").markdown(f"🤖 {response}")
