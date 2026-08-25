import os
from dotenv import load_dotenv

load_dotenv()
import operator
from typing import TypedDict, Annotated, List, Literal, Dict, Any
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Literal
import fastf1
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END

fastf1.Cache.enable_cache('C:/Users/Asus/Desktop/ML_Projects/F1_project')

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("llm_key"),
    temperature=0
)


class F1AgentState(TypedDict):
    user_query: str
    year: int
    event: str
    session_type: str
    context: Annotated[List[str], operator.add] 
    next_node: str
    needs_clarification: bool
    clarification_question: Optional[str]

class SupervisorOutput(BaseModel):
    year: Optional[int] = Field(default=None, description="The 4-digit race year extracted from the query, or None if unknown.")
    event: Optional[str] = Field(default=None, description="The clean name of the F1 race weekend event (e.g., 'Monaco', 'Miami'), or None if unknown.")
    session_type: Optional[Literal['R', 'Q', 'S']] = Field(default=None, description="The F1 session type: 'R' for Race, 'Q' for Qualifying, 'S' for Sprint. Default to 'R' if unsure.")
    next_node: Literal["ScheduleWorker", "ResultsWorker", "LapsWorker", "TelemetryWorker", "FINISH"] = Field(
        description="The next specialized worker node to call based on what data is required, or 'FINISH' if the final answer is ready or if clarification is needed."
    )
    visual_data: Optional[Dict[str, Any]] = Field(default=None, description="If FINISH, output visualization payload here. Contains 'type' (table, pie_chart, bar_chart), 'title', and 'data'.")
    needs_clarification: bool = Field(default=False, description="Set to True if the query is missing critical info (year, event) or is ambiguous and needs user clarification before proceeding.")
    clarification_question: Optional[str] = Field(default=None, description="A natural follow-up question to ask the user when needs_clarification is True.")

def create_specialized_f1_agent(agent_type: str, year: int, event: str, session_str: str):
    """
    Dynamically fetches data from FastF1 APIs and wraps it into a customized Pandas Dataframe Agent.
    """
    if agent_type == 'schedule':
        df = pd.DataFrame(fastf1.get_event_schedule(year))
        specialized_instructions = (
            "You are looking at the season schedule. Look at the 'EventFormat' column "
            "to find Sprint weekends (formats like 'sprint' or 'sprint_qualifying'). "
            "Use 'OfficialEventName' or 'Location' to filter circuits."
        )
        
    elif agent_type == 'results':
        session = fastf1.get_session(year, event, session_str)
        session.load(laps=False, telemetry=False, weather=False)
        df = pd.DataFrame(session.results)
        specialized_instructions = (
            "You are analyzing race/session results. Useful columns include:\n"
            "- 'Position': Final finishing position.\n"
            "- 'GridPosition': Starting position.\n"
            "- 'Points': Points scored.\n"
            "- 'Status': Indicates finishes, accidents, mechanical DNFs, or laps down.\n"
            "- 'Time': Total gap or finishing time."
        )
        
    elif agent_type == 'laps':
        session = fastf1.get_session(year, event, session_str)
        session.load(laps=True, telemetry=False, weather=False)
        df = pd.DataFrame(session.laps)
        specialized_instructions = (
            "You are analyzing lap-by-lap timing. Crucial rules:\n"
            "- 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time' are timedelta objects. "
            "To find the fastest time, use .min(). To sort times, convert or use pandas timedelta methods.\n"
            "- 'Compound' indicates tire choices ('SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET').\n"
            "- 'Stint' tracks tire stints sequentially.\n"
            "- Use 'Driver' (e.g., 'VER', 'HAM') to filter down to specific drivers."
        )
        
    elif agent_type == 'telemetry':
        session = fastf1.get_session(year, event, session_str)
        session.load(laps=True, telemetry=True, weather=False)
        df = session.laps.pick_fastest().get_telemetry() 
        specialized_instructions = (
            "You are handling high-frequency car telemetry (Speed, RPM, Throttle, Brake, Gear, DRS).\n"
            "- 'Speed' is in km/h.\n"
            "- 'Throttle' is 0-100.\n"
            "- 'Brake' is a boolean (True/False).\n"
            "- 'Distance' measures progression down the track matrix. Use it as your X-axis reference."
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    dataframe_agent_system_prompt = f"""
You are a senior F1 Data Analyst working with a specialized pandas dataframe (`df`) containing {agent_type.upper()} data.

---
SPECIALIZED AGENT KNOWLEDGE:
{specialized_instructions}
---

Columns available in this specific Dataframe:
{list(df.columns)}

Sample reference rows:
{df.head(2).to_string()}

Your job is to answer the user's specific question by writing and executing Python pandas code.

IMPORTANT CODE RULES:
1. NEVER rely on your pre-trained knowledge. If the data is not in the dataframe, do not make it up.
2. PARTIAL ANSWERS ARE REQUIRED: If the user asks a multi-part question and your dataframe only contains data for ONE part (e.g., you have results data but no tire data), answer the part you CAN solve using python and explicitly state "I do not have the data for the rest" so the supervisor knows to route to another worker.
3. You MUST use the python_repl_ast tool to execute code.
4. NEVER explain code before executing it.
5. Store the final numerical/text result in variable `result`.
6. The last line of code must be exactly `result`.
"""

    agent = create_pandas_dataframe_agent(
        llm,
        df, 
        verbose=False,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        handle_parsing_errors=True,
        prefix=dataframe_agent_system_prompt
    )
    return agent, df



def supervisor_node(state: F1AgentState):
    """
    Acts as the orchestrator. Extracts parameters and routes to the correct specialized worker node.
    """
    supervisor_prompt = """
You are the F1 Data Orchestrator, an intelligent supervisor agent tasked with answering complex Formula 1 questions.
Your job is to break down the user's request, identify the required data types, route execution to the appropriate worker nodes, analyze the data added to the shared state, and synthesize a final answer when all required information is gathered.

1. 'ScheduleWorker': For season formats, race counts, tracking race venues/dates.
2. 'ResultsWorker': For finishing positions, points, grids, and statuses (DNFs).
3. 'LapsWorker': For lap times, sector times, tire compounds, and stint timelines.
4. 'TelemetryWorker': For speed traps, throttle profiles, braking points, and gears.

BEFORE routing to any worker, you MUST check if the query has enough info. If ANY of the following are true, set needs_clarification=True, next_node='FINISH', and provide a clarification_question:

1. **Year is missing**: The user does not specify a year (e.g., "Who won Monaco GP?"). Ask which year.
2. **Event is missing or too vague**: The user does not name a specific Grand Prix (e.g., "What was the fastest lap?"). Ask which race.
3. **Country is ambiguous**: The user mentions a country that hosts multiple GPs:
   - USA/United States → Miami GP, United States GP (Austin/COTA), Las Vegas GP
   - Italy → Italian GP (Monza), Emilia Romagna GP (Imola)
   - If the user says "US Grand Prix" or "American race", ask which one.
4. **Event name is unclear**: The user says something like "the European race" or "the street circuit" which could match multiple events.

Do NOT ask for clarification if:
- The query is about the full season schedule (ScheduleWorker handles all events for a year).
- The user explicitly names both year and a specific event.
- The context already contains enough data to answer.

You must output structured data with:
- "year", "event", "session_type": Extracted parameters (null if unknown).
- "next_node": The worker to route to, or 'FINISH' if done or clarification needed.
- "needs_clarification": True/False.
- "clarification_question": The question to ask (only when needs_clarification is True).

User Query: {user_query}
Current State Context: {context}
"""
    structured_llm = llm.with_structured_output(SupervisorOutput)
    
    formatted_prompt = supervisor_prompt.format(
        user_query=state["user_query"],
        context="\n".join(state["context"])
    )
    
    response = structured_llm.invoke([HumanMessage(content=formatted_prompt)])
    
    if response.needs_clarification:
        return {
            "next_node": "FINISH",
            "needs_clarification": True,
            "clarification_question": response.clarification_question
        }
    
    return {
        "year": response.year or state.get("year", 2024),
        "event": response.event or state.get("event", "Monaco"),
        "session_type": response.session_type or state.get("session_type", "R"),
        "next_node": response.next_node,
        "needs_clarification": False,
        "clarification_question": None
    }

def supervisor_router(state: F1AgentState) -> Literal["ScheduleWorker", "ResultsWorker", "LapsWorker", "TelemetryWorker", "__end__"]:
    """
    Conditional edge router that reads state['next_node'] to determine graph direction.
    """
    next_step = state["next_node"]
    if next_step == "FINISH":
        return END
    return next_step


def schedule_worker_node(state: F1AgentState):
    agent, df = create_specialized_f1_agent('schedule', state['year'], state['event'], state['session_type'])
    res = agent.invoke({"input": state["user_query"]})
    return {"context": [f"[Schedule Data]: {res['output']}"]}

def results_worker_node(state: F1AgentState):
    agent, df = create_specialized_f1_agent('results', state['year'], state['event'], state['session_type'])
    res = agent.invoke({"input": state["user_query"]})
    return {"context": [f"[Results Data]: {res['output']}"]}

def laps_worker_node(state: F1AgentState):
    agent, df = create_specialized_f1_agent('laps', state['year'], state['event'], state['session_type'])
    res = agent.invoke({"input": state["user_query"]})
    return {"context": [f"[Laps Data]: {res['output']}"]}

def telemetry_worker_node(state: F1AgentState):
    agent, df = create_specialized_f1_agent('telemetry', state['year'], state['event'], state['session_type'])
    res = agent.invoke({"input": state["user_query"]})
    return {"context": [f"[Telemetry Data]: {res['output']}"]}


workflow = StateGraph(F1AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("ScheduleWorker", schedule_worker_node)
workflow.add_node("ResultsWorker", results_worker_node)
workflow.add_node("LapsWorker", laps_worker_node)
workflow.add_node("TelemetryWorker", telemetry_worker_node)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    supervisor_router,
    {
        "ScheduleWorker": "ScheduleWorker",
        "ResultsWorker": "ResultsWorker",
        "LapsWorker": "LapsWorker",
        "TelemetryWorker": "TelemetryWorker",
        END: END
    }
)

workflow.add_edge("ScheduleWorker", "Supervisor")
workflow.add_edge("ResultsWorker", "Supervisor")
workflow.add_edge("LapsWorker", "Supervisor")
workflow.add_edge("TelemetryWorker", "Supervisor")


f1_chatbot = workflow.compile()


if __name__ == "__main__":
    initial_state = {
        "user_query": "Who won the 2025 Las Vegas Grand Prix, and what compound of tires did they complete their fastest lap on?",
        "context": [],
        "next_node": "Supervisor"
    }
    
    print("--- Starting F1 Multi-Agent Engine ---")
    final_output = f1_chatbot.invoke(
        initial_state, 
        config={"recursion_limit": 10}
)
    
    print("\n--- Final Aggregated Insights gathered by Supervisor ---")
    for update in final_output["context"]:
        print(update)