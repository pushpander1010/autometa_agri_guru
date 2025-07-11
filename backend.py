from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from typing import TypedDict,Optional
import os

# --- Your own tools ---
from tools import get_farm_prices,get_seasonal_weather_data,get_weather_and_soil_data,get_soil_properties,get_curr_loc_tool

# Groq LLM setup (replace with your model and API key)
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#define state
from typing import TypedDict, Optional, Annotated

class AgriState(TypedDict):
    weather_and_soil_summary: Annotated[str, "Summary of past weather trends and current soil properties."]
    crop_plan: Annotated[str, "AI-suggested crops with timing, water/soil requirements, and seasonal suitability."]
    crop_info: Annotated[str, "Detailed information about selected crops including fertilizers, diseases, etc."]
    rag_docs: Annotated[list, "List of LangChain Document objects used for RAG-based QA."]
    curr_loc: Annotated[dict, "User’s current location information (e.g., city, region, lat/lon) from IP."]
    query: Annotated[str, "The farmer's latest question or input for the system."]
    response: Annotated[str, "The system's response to the latest query, based on RAG or fallback LLM."]
    unknown_query: Annotated[Optional[str], "Stores a question the system couldn't answer with RAG for later analysis or retry."]


#make Nodes
FetchWeatherAndSoilData = RunnableLambda(lambda state: {
    "weather_and_soil_summary": get_weather_and_soil_data.invoke(None)
})
GetCurrentLocation=RunnableLambda(lambda state:{"curr_loc":get_curr_loc_tool.invoke(None)})

PureLLMAnswer = RunnableLambda(lambda state: {
    "response": llm.invoke(
        f"Answer this farming query directly using your knowledge:\n\n{state['unknown_query']}"
    ).content
})


AnalyzeSoilAndWeather = RunnableLambda(lambda state: {
    "crop_plan": llm.invoke(
        f"Based on this weather and soil summary:\n\n{state['weather_and_soil_summary']}\n\n"
        f"Suggest all suitable crops to grow in different seasons. Include timing, soil needs, water needs, and temperature range."
    )
})

def extract_crop_name(text: str) -> str:
    result = llm.invoke(f"From this crop plan, extract the main crop name only (1 word):\n\n{text}")
    return result.content.strip().split("\n")[0].split(",")[0]

def revise_crop_plan(original_plan: str, feedback: str):
    revised = llm.invoke(
        f"""Here is the original crop plan:\n\n{original_plan}\n\n
        The farmer now says:\n"{feedback}"\n\n
        Please revise the crop plan accordingly. Output in similar format."""
    )
    return revised.content


def enrich_crop_info(state):
    crop_plan = state["crop_plan"].content
    top_crop = extract_crop_name(crop_plan)
    mandi_info = get_farm_prices.invoke(top_crop)
    loc = state.get("curr_loc", {})
    stateName = loc.get("regionName", "Haryana") 
    soil_info = get_soil_properties.invoke(None)
    seasonal_weather = get_seasonal_weather_data.invoke(None)

    combined_prompt = f"""
Crop Plan:
{crop_plan}

Top Crop Mandi Price:
{mandi_info}

Soil Info:
{soil_info}

Seasonal Weather Info:
{seasonal_weather}

Now generate:
- Common diseases & treatments
- Fertilizer types & usage
- Step-by-step growing instructions
- Tips for small/marginal farmers
"""

    return {
        "crop_info": llm.invoke(combined_prompt)
    }

EnrichWithCropInfo = RunnableLambda(enrich_crop_info)


#Build Embedding
def build_rag_docs(state):
    docs = [
        Document(page_content=state["crop_plan"].content, metadata={"type": "plan"}),
        Document(page_content=state["crop_info"].content, metadata={"type": "info"}),
        Document(page_content=state["weather_and_soil_summary"], metadata={"type": "summary"})
    ]

    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()

    return {
        "rag_docs": docs,
        "retriever": retriever
    }


BuildRAGDocs = RunnableLambda(build_rag_docs)

#Query Answering with RAG

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
        if not answer or answer.lower() in ["i don't know", "not sure", "unknown"]:
            return {"unknown_query": state["query"]}  # trigger fallback
        return {"response": answer}
    except Exception:
        return {"unknown_query": state["query"]}  # trigger fallback in case of error

AnswerFarmerQuery = RunnableLambda(answer_query)


#conditional routing
def weather_success(state): return "weather_and_soil_summary" in state and "Error" not in state["weather_and_soil_summary"]
def crop_plan_success(state): return "crop_plan" in state and  "Error" not in state["crop_plan"]
def get_curr_location(state): return "curr_loc" in state and "Error" not in state["curr_loc"]
def needs_pure_llm(state):
    return "unknown_query" in state and state["unknown_query"] is not None

#build graph
graph = StateGraph(AgriState)

graph.add_node("fetch_weather_soil", FetchWeatherAndSoilData)
graph.add_node("analyze_data", AnalyzeSoilAndWeather)
graph.add_node("generate_crop_info", EnrichWithCropInfo)
graph.add_node("build_docs", BuildRAGDocs)
graph.add_node("get_location", GetCurrentLocation)
graph.add_node("pure_llm_answer", PureLLMAnswer)
graph.add_node("query_response", AnswerFarmerQuery)
graph.add_node("error", RunnableLambda(lambda _: {"response": "Something went wrong in the planning process."}))

#  Entry point
graph.set_entry_point("fetch_weather_soil")

#  Weather and Soil → Analyze
graph.add_conditional_edges("fetch_weather_soil", weather_success, {
    True: "analyze_data",
    False: "error"
})

#  Analyze → Get Location
graph.add_conditional_edges("analyze_data", crop_plan_success, {
    True: "get_location",
    False: "error"
})

#  Location → Enrich → RAG Docs → Query
graph.add_edge("get_location", "generate_crop_info")
graph.add_edge("generate_crop_info", "build_docs")
graph.add_edge("build_docs", "query_response")

#  Final Conditional: If RAG fails, go to LLM
graph.add_conditional_edges("query_response", needs_pure_llm, {
    True: "pure_llm_answer",
    False: "__end__"  # Mark as finished
})

# Mark both as possible finish points
graph.set_finish_point(["pure_llm_answer", "__end__"])

# Compile with safe recursion
agri_agent = graph.compile()


# ✅ Step 1: Run the agent and get initial result
result = agri_agent.invoke({
    "query": "Which crops should I grow?"
})
original_crop_plan = result["crop_plan"].content
rag_docs = result["rag_docs"]

# Initial vectorstore
vectorstore = FAISS.from_documents(
    [Document(page_content=doc.page_content) for doc in rag_docs],
    embedding_model
)
retriever = vectorstore.as_retriever()
smart_rag = RAGWithLLMFallback(llm, retriever)