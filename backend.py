from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tools import (
    get_state_coordinates,
    get_soil_properties,
    get_weather_and_soil_data,
    get_seasonal_weather_data,
    get_farm_prices
)
import os

# === Supported LLMs ===
GROQ_MODELS = [
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

# === Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Helper: Get LLM ===
def get_llm(model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    return ChatGroq(
        model=model_name,
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )

# === LangChain Tools ===
tool_list = [
    get_state_coordinates,
    get_soil_properties,
    get_weather_and_soil_data,
    get_seasonal_weather_data,
    get_farm_prices
]

# === Agent Initialization ===
def get_agent(llm):
    return initialize_agent(
        tools=tool_list,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )

# === Crop Plan Generator ===
def generate_crop_plan(llm, location: dict) -> dict:
    region = location.get("state", "Punjab")
    lat = location.get("latitude", 30.76)
    lon = location.get("longitude", 75.85)

    # 1. Weather and soil analysis
    weather_and_soil = get_weather_and_soil_data.invoke({"latitude": lat, "longitude": lon})
    crop_plan = llm.invoke(
        f"""Based on this weather and soil summary:\n\n{weather_and_soil}\n\n
        Suggest the best seasonal crops with water needs, temperature range, soil compatibility, etc."""
    ).content

    # 2. Soil and seasonal weather
    soil_info_api = get_soil_properties.invoke({"latitude": lat, "longitude": lon})
    soil_info_llm = llm.invoke(f"Give an expert soil summary for {region}").content
    seasonal_weather = get_seasonal_weather_data.invoke({"latitude": lat, "longitude": lon})

    # 3. Fetch mandi prices for popular crops
    mandi_data = []
    for crop in ["Wheat", "Rice", "Mustard"]:
        prices = get_farm_prices.invoke({"stateName": region, "commodity": crop})
        if "No mandi price" not in prices:
            mandi_data.append(prices)

    mandi_summary = "\n\n".join(mandi_data) if mandi_data else "No recent mandi price data found for your region."

    # 4. Crop-specific insights
    crop_info = llm.invoke(f"""
Crop Plan:
{crop_plan}

Location: {location}
Soil Info: {soil_info_api} {soil_info_llm}
Seasonal Weather: {seasonal_weather}
Mandi Prices: {mandi_summary}

Now generate:
- Common diseases & treatments
- Fertilizer types & usage
- Growing instructions
- Subsidies
- Image URLs
""").content

    # 5. RAG docs
    docs = [
        Document(page_content=crop_plan, metadata={"type": "plan"}),
        Document(page_content=crop_info, metadata={"type": "info"}),
        Document(page_content=weather_and_soil, metadata={"type": "summary"}),
        Document(page_content=mandi_summary, metadata={"type": "prices"})
    ]

    return {
        "crop_plan": crop_plan,
        "crop_info": crop_info,
        "rag_docs": docs,
        "curr_loc": location
    }

# === Feedback Loop ===
def revise_crop_plan(llm, original_plan: str, feedback: str) -> str:
    prompt = f"""
Original Crop Plan:
{original_plan}

Farmer's Feedback:
"{feedback}"

Revise the crop plan accordingly. Keep it structured and useful.
"""
    return llm.invoke(prompt).content

# === RAG with LLM Fallback ===
class RAGWithLLMFallback:
    def __init__(self, llm, documents):
        self.llm = llm
        self.vectorstore = FAISS.from_documents(documents, embedding_model)
        self.retriever = self.vectorstore.as_retriever()
        self.qa = RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever)

    def invoke(self, query: str) -> str:
        try:
            result = self.qa.invoke({"query": query})
            if not result.get("result") or "i don't know" in result.get("result", "").lower():
                raise ValueError("RAG fallback")
            return result["result"]
        except Exception:
            return self.llm.invoke(f"Answer this farming query directly:\n{query}").content
