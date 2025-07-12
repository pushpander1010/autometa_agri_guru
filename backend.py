from typing import List, Dict, TypedDict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from tools import (
    get_weather_and_soil_data,
    get_farm_prices,
    get_soil_properties,
    get_seasonal_weather_data
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

# === Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Supported LLMs ===
GROQ_MODELS = [
    "allam-2-7b",
    "compound-beta",
    "compound-beta-mini",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "mistral-saba-24b",
    "qwen-qwq-32b",
    "qwen/qwen3-32b"
]

# === LLM Initialization ===
def get_llm(model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    return ChatGroq(
        model=model_name,
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )

# === Pydantic output parser schema ===
class CropListOutput(BaseModel):
    crops: List[str] = Field(description="List of crop names")

# === LangGraph state schema ===
class CropPlanState(TypedDict, total=False):
    location: Dict[str, Any]
    weather_soil_summary: str
    crop_plan: str
    extracted_crops: List[str]
    mandi_prices: str
    soil_info_api: str
    soil_info_llm: str
    seasonal_weather: str
    crop_info: str
    rag_docs: List[Document]

llm = get_llm()

# === Graph Nodes ===
def get_weather_soil_node(state: CropPlanState) -> CropPlanState:
    summary = get_weather_and_soil_data.invoke(state["location"])
    return {**state, "weather_soil_summary": summary}

def generate_crop_plan_node(state: CropPlanState) -> CropPlanState:
    prompt = f"""Based on this weather and soil summary:\n\n{state["weather_soil_summary"]}\n\n
    Suggest the best seasonal crops with water needs, temperature range, soil compatibility, etc."""
    plan = llm.invoke(prompt).content
    return {**state, "crop_plan": plan}

def extract_crop_names_node(state: CropPlanState) -> CropPlanState:
    parser = PydanticOutputParser(pydantic_object=CropListOutput)
    prompt = f"""
Here is a crop plan:

{state['crop_plan']}

Extract only the names of the crops mentioned in this plan. 
Return them as a list of strings in the format: {{ "crops": ["Wheat", "Rice", "Mustard"] }}
"""
    parsed = parser.invoke(llm.invoke(prompt).content)
    return {**state, "extracted_crops": parsed.crops}

def fetch_mandi_prices_node(state: CropPlanState) -> CropPlanState:
    region = state["location"]["state"]
    mandi_data = []
    for crop in state["extracted_crops"]:
        prices = get_farm_prices.invoke({"stateName": region, "commodity": crop})
        if "No mandi price" not in prices:
            mandi_data.append(prices)
    summary = "\n\n".join(mandi_data) if mandi_data else "No recent mandi price data found for your region."
    return {**state, "mandi_prices": summary}

def enrich_crop_info_node(state: CropPlanState) -> CropPlanState:
    lat, lon = state["location"]["latitude"], state["location"]["longitude"]
    region = state["location"]["state"]
    soil_api = get_soil_properties.invoke({"latitude": lat, "longitude": lon})
    soil_llm = llm.invoke(f"Give expert soil summary for {region}").content
    seasonal_weather = get_seasonal_weather_data.invoke({"latitude": lat, "longitude": lon})
    full_prompt = f"""
Crop Plan:
{state['crop_plan']}

Location: {state['location']}
Soil Info: {soil_api} {soil_llm}
Seasonal Weather: {seasonal_weather}
Mandi Prices: {state['mandi_prices']}

Now generate:
- Common diseases & treatments
- Fertilizer types & usage
- Growing instructions
- Subsidies
- public Image URLs
"""
    crop_info = llm.invoke(full_prompt).content
    return {
        **state,
        "soil_info_api": soil_api,
        "soil_info_llm": soil_llm,
        "seasonal_weather": seasonal_weather,
        "crop_info": crop_info
    }

def prepare_rag_docs_node(state: CropPlanState) -> CropPlanState:
    docs = [
        Document(page_content=state["crop_plan"], metadata={"type": "plan"}),
        Document(page_content=state["soil_info_llm"], metadata={"type": "info"}),
        Document(page_content=state["crop_info"], metadata={"type": "info"}),
        Document(page_content=state["weather_soil_summary"], metadata={"type": "summary"}),
        Document(page_content=state["mandi_prices"], metadata={"type": "prices"})
    ]
    return {**state, "rag_docs": docs}

# === Assemble LangGraph ===
graph = StateGraph(CropPlanState)
graph.add_node("weather_soil", RunnableLambda(get_weather_soil_node))
graph.add_node("generate_plan", RunnableLambda(generate_crop_plan_node))
graph.add_node("extract_crops", RunnableLambda(extract_crop_names_node))
graph.add_node("mandi_prices", RunnableLambda(fetch_mandi_prices_node))
graph.add_node("enrich_info", RunnableLambda(enrich_crop_info_node))
graph.add_node("prepare_docs", RunnableLambda(prepare_rag_docs_node))
graph.set_entry_point("weather_soil")
graph.add_edge("weather_soil", "generate_plan")
graph.add_edge("generate_plan", "extract_crops")
graph.add_edge("extract_crops", "mandi_prices")
graph.add_edge("mandi_prices", "enrich_info")
graph.add_edge("enrich_info", "prepare_docs")
graph.add_edge("prepare_docs", END)
crop_graph = graph.compile()

# === Optional: RAG fallback class ===
class RAGWithLLMFallback:
    def __init__(self, documents: List[Document]):
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
            return llm.invoke(f"Answer this farming query directly:\n{query}").content

# === Feedback-based revision ===
def revise_crop_plan(original_plan: str, feedback: str) -> str:
    prompt = f"""
Original Crop Plan:
{original_plan}

Farmer's Feedback:
"{feedback}"

Revise the crop plan accordingly. Keep it structured and useful.
"""
    return llm.invoke(prompt).content