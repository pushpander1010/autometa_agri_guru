from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from IPython.display import Image
from typing import TypedDict, Optional, Annotated
import os

# --- Tool imports ---
from tools import get_farm_prices, get_seasonal_weather_data, get_weather_and_soil_data, get_soil_properties, get_curr_loc_tool

# === LLM and Embedding setup ===
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === State ===
class AgriState(TypedDict):
    weather_and_soil_summary: Annotated[str, "Summary of past weather trends and current soil properties."]
    crop_plan: Annotated[str, "AI-suggested crops with timing, water/soil requirements, and seasonal suitability."]
    crop_info: Annotated[str, "Detailed information about selected crops including fertilizers, diseases, etc."]
    rag_docs: Annotated[list, "List of LangChain Document objects used for RAG-based QA."]
    curr_loc: Annotated[dict, "User's current location information (e.g., city, region, lat/lon) from IP."]
    query: Annotated[str, "The farmer's latest question or input for the system."]
    response: Annotated[str, "The system's response to the latest query, based on RAG or fallback LLM."]
    unknown_query: Annotated[Optional[str], "Stores a question the system couldn't answer with RAG."]

# === Nodes ===
def fetch_weather_and_soil(state):
    state["weather_and_soil_summary"] = get_weather_and_soil_data.invoke(None)
    state["curr_loc"] = "Haryana"
    return state

FetchWeatherAndSoilData = RunnableLambda(fetch_weather_and_soil)


GetCurrentLocation = RunnableLambda(
    lambda state: {
        "curr_loc": get_curr_loc_tool.invoke(None)
    }
)

AnalyzeSoilAndWeather = RunnableLambda(lambda state: {
    "crop_plan": llm.invoke(
        f"Based on this weather and soil summary:\n\n{state['weather_and_soil_summary']}\n\n"
        f"Suggest all suitable crops to grow in different seasons. Include timing, soil needs, water needs, and temperature range."
    )
})


def extract_crop_names(text: str) -> str:
    result = llm.invoke(f"From this crop plan, extract the main crops names only (1 word):\n\n{text}")
    return result.content.strip().split("\n")[0].split(",")[0]

def get_soil_location(location: str) -> str:
    return llm.invoke(f"Get the soil properties in region {location}").content

def enrich_crop_info(state):
    crop_plan = state["crop_plan"].content
    loc = state.get("curr_loc", {})
    soil_info_api = get_soil_properties.invoke(None)
    soil_info_llm = get_soil_location(loc.get("regionName", "Haryana"))
    seasonal_weather = get_seasonal_weather_data.invoke(None)

    prompt = f"""
Crop Plan:
{crop_plan}

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

    return {"crop_info": llm.invoke(prompt)}

EnrichWithCropInfo = RunnableLambda(enrich_crop_info)

BuildRAGDocs = RunnableLambda(lambda state: {
    "rag_docs": [
        Document(page_content=state["crop_plan"].content, metadata={"type": "plan"}),
        Document(page_content=state["crop_info"].content, metadata={"type": "info"}),
        Document(page_content=state["weather_and_soil_summary"], metadata={"type": "summary"})
    ]
})

def answer_query(state):
    vectorstore = FAISS.from_documents(
        [Document(page_content=doc.page_content) for doc in state["rag_docs"]],
        embedding_model
    )
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    try:
        result = rag_chain.invoke({"query": state["query"]})
        answer = result.get("result", "").strip()
        if not answer or answer.lower() in ["i don't know", "not sure"]:
            return {"unknown_query": state["query"]}
        return {"response": answer}
    except Exception:
        return {"unknown_query": state["query"]}

AnswerFarmerQuery = RunnableLambda(answer_query)

def handle_pure_llm_answer(state):
    response = llm.invoke(f"Answer this farming query directly using your knowledge:\n\n{state['unknown_query']}").content
    state["unknown_query"] = None  # Reset after use
    return {"response": response}

PureLLMAnswer = RunnableLambda(handle_pure_llm_answer)

# === Graph ===
graph = StateGraph(AgriState)

# Nodes
graph.add_node("fetch_weather_soil", FetchWeatherAndSoilData)
graph.add_node("analyze_data", AnalyzeSoilAndWeather)
graph.add_node("get_location", GetCurrentLocation)
graph.add_node("generate_crop_info", EnrichWithCropInfo)
graph.add_node("build_docs", BuildRAGDocs)
graph.add_node("query_response", AnswerFarmerQuery)
graph.add_node("pure_llm_answer", PureLLMAnswer)
graph.add_node("error", RunnableLambda(lambda _: {"response": "Something went wrong."}))

#tool conditions
def weather_success(state): 
    return "weather_and_soil_summary" in state and "Error" not in state["weather_and_soil_summary"]

def crop_plan_success(state): 
    return "crop_plan" in state and "Error" not in state["crop_plan"]

def location_success(state): 
    return "curr_loc" in state and "Error" not in state["curr_loc"]

def rag_failed(state): 
    return "unknown_query" in state and state["unknown_query"] is not None



# Edges
# Set entry point
graph.set_entry_point("fetch_weather_soil")

# Weather tool succeeded → analyze
graph.add_conditional_edges("fetch_weather_soil", weather_success, {
    True: "analyze_data",
    False: "error"
})
graph.add_conditional_edges("analyze_data", location_success, {
    True: "get_location",
    False: "error"
})
graph.add_edge("get_location","generate_crop_info")

# Analyze tool succeeded → get location
graph.add_edge("generate_crop_info","build_docs")
# Location tool succeeded → generate crop info

graph.add_edge("build_docs", "query_response")

# If RAG fails, fallback to pure LLM
graph.add_conditional_edges("query_response", rag_failed, {
    True: "pure_llm_answer",
    False: "__end__"
})

# Mark valid end points
graph.set_finish_point(["pure_llm_answer", "query_response"])


# Compile & Visualize
agri_agent = graph.compile()
# # Get the image binary
# img_bytes = agri_agent.get_graph().draw_mermaid_png()

# # Save to file
# with open("agriguru_graph.png", "wb") as f:
#     f.write(img_bytes)

def revise_crop_plan(original_plan: str, feedback: str):
    revised = llm.invoke(
        f"""Here is the original crop plan:\n\n{original_plan}\n\n
        The farmer now says:\n"{feedback}"\n\n
        Please revise the crop plan accordingly. Output in similar format."""
    )
    return revised.content

class RAGWithLLMFallback:
    def __init__(self, llm, retriever):
        self.rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        self.llm = llm

    def invoke(self, query):
        try:
            rag_result = self.rag_chain.invoke({"query": query})
            if not rag_result["result"].strip():
                raise ValueError("Empty result from RAG")
            return rag_result["result"]
        except Exception:
            return self.llm.invoke(f"Answer this farming query directly:\n{query}").content