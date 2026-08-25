import asyncio
import json
import re
import uuid
import requests
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import fastf1
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

import chatbot_integration
import db

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user: str = "guest"

class VisualData(BaseModel):
    type: str  
    title: str
    data: Any  

class ChatResponse(BaseModel):
    session_id: str
    text: str
    citations: List[str]
    visual_data: Optional[VisualData] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

@app.get("/api/schedule")
async def get_schedule(year: int = 2024):
    """
    Returns the schedule for the requested year.
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        df = schedule[['RoundNumber', 'Country', 'Location', 'OfficialEventName', 'EventDate', 'EventFormat']]
        records = df.fillna("").to_dict(orient="records")
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/standings")
async def get_standings(year: int = 2024):
    """
    Returns simplified driver standings fetched from the Ergast API.
    """
    try:
        loop = asyncio.get_running_loop()
        url = f"http://api.jolpi.ca/ergast/f1/{year}/driverStandings.json"
        
        response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=10))
        response.raise_for_status()
        data = response.json()
        
        standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
        
        formatted_standings = []
        for s in standings:
            formatted_standings.append({
                "position": int(s["position"]),
                "driver": s["Driver"].get("code", s["Driver"]["familyName"][:3].upper()),
                "points": float(s["points"])
            })
            
        return formatted_standings
        
    except Exception as e:
        print(f"Error fetching live standings: {e}")
        return [
            {"position": 1, "driver": "VER", "points": 393},
            {"position": 2, "driver": "NOR", "points": 331},
            {"position": 3, "driver": "LEC", "points": 307},
            {"position": 4, "driver": "PIA", "points": 262},
            {"position": 5, "driver": "SAI", "points": 244}
        ]

@app.get("/api/telemetry")
async def get_telemetry(year: int = 2024, event: str = "Monaco", session: str = "R", drivers: str = None, lap: str = None):
    """
    Returns raw telemetry (throttle, lap times, speed) for charting on the dashboard.
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=True, telemetry=True, weather=False, messages=False)
        
        telemetry_data = {}
        driver_list = drivers.split(',') if drivers else sess.results.head(2)['Abbreviation'].tolist()
        
        for driver in driver_list:
            if driver not in sess.results['Abbreviation'].values:
                continue
                
            driver_laps = sess.laps.pick_driver(driver)
            if len(driver_laps) == 0:
                continue
            
            if lap and lap.isdigit():
                selected_lap = driver_laps[driver_laps['LapNumber'] == int(lap)]
                if len(selected_lap) == 0:
                    continue
                selected_lap = selected_lap.iloc[0]
            else:
                selected_lap = driver_laps.pick_fastest()
                
            if not pd.isnull(selected_lap.get('LapTime', pd.NaT)):
                tel = selected_lap.get_telemetry()
                tel = tel.iloc[::10, :]
                telemetry_data[driver] = {
                    "distance": tel['Distance'].tolist(),
                    "throttle": tel['Throttle'].tolist(),
                    "speed": tel['Speed'].tolist()
                }
                
        return telemetry_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/drivers")
async def get_drivers(year: int = 2024, event: str = "Monaco", session: str = "R"):
    """
    Returns a list of all drivers in the session to populate the UI dropdowns.
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=False, telemetry=False, weather=False, messages=False)
        
        results = sess.results
        drivers = []
        for _, row in results.iterrows():
            drivers.append({
                "abbreviation": row["Abbreviation"],
                "full_name": row["FullName"]
            })
        return drivers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/laps")
async def get_laps(year: int = 2024, event: str = "Monaco", session: str = "R"):
    """
    Returns the total number of laps in the session for the lap selector.
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=True, telemetry=False, weather=False, messages=False)
        
        total_laps = int(sess.laps['LapNumber'].max()) if not sess.laps.empty else 0
        return {"total_laps": total_laps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Executes the LangGraph chatbot flow. Monitors for client disconnects
    to cancel long-running FastF1 processing.
    """
    session_id = body.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    subquery_id = str(uuid.uuid4())
    agent_question = body.question     

    history = db.get_history(session_id, limit=3)
    history_context = [
        f"[Chat History]: User: {pair['user']} | Assistant: {pair['assistant']}"
        for pair in history
    ]


    initial_state = {
        "user_query": body.question,
        "context": history_context,
        "next_node": "Supervisor",
        "needs_clarification": False,
        "clarification_question": None
    }

    loop = asyncio.get_running_loop()
    
    task = loop.run_in_executor(
        None, 
        lambda: chatbot_integration.f1_chatbot.invoke(initial_state, {"recursion_limit": 10})
    )

    while not task.done():
        if await request.is_disconnected():
            task.cancel()
            raise HTTPException(status_code=499, detail="Client Closed Request")
        await asyncio.sleep(0.5)

    try:
        result = await task
    except asyncio.CancelledError:
         raise HTTPException(status_code=499, detail="Client Closed Request")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    raw_context = "\n".join(result.get("context", []))

    agents_used = [
        ctx.split("]")[0].replace("[", "").strip()
        for ctx in result.get("context", [])
        if "]: " in ctx and "Chat History" not in ctx
    ]

    if result.get("needs_clarification"):
        clarification_q = result.get("clarification_question", "Could you please provide more details?")
        db.save_message(
            subquery_id=subquery_id,
            session_id=session_id,
            message_id=message_id,
            user=body.user,
            question=body.question,
            agent_question=agent_question,
            response=clarification_q,
            needs_clarification=True
        )
        return ChatResponse(
            session_id=session_id,
            text=clarification_q,
            citations=[],
            needs_clarification=True,
            clarification_question=clarification_q
        )
    
    system_prompt = """
You are a helpful F1 Chatbot Assistant.
You have been provided with raw data and context gathered by specialized agents.
Your task is to answer the user's query based ONLY on this context.

1. Provide a concise, natural-language summary of the answer.
2. Include citations like [1] referring to 'FastF1 API'.
3. IMPORTANT: If the context contains ANY structured data (lap times, positions, standings, driver stats, tire strategies, sector times, etc.), you MUST also output a JSON block wrapped in ```json with this exact structure:
{
  "type": "table" | "pie_chart" | "bar_chart",
  "title": "Title of the visual",
  "data": [{"column1": "value1", "column2": "value2", ...}]
}
For example, if asked about fastest laps, output a table with columns like Driver, LapTime, Compound, etc.
Always prefer "table" type unless the data is clearly better as a chart (e.g., percentage breakdowns → pie_chart, comparisons → bar_chart).
If the data is purely descriptive with no tabular structure, do NOT output the JSON block.
"""
    prompt = f"User Query: {body.question}\n\nContext:\n{raw_context}"
    
    try:
        llm_response = await loop.run_in_executor(
            None,
            lambda: chatbot_integration.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
        )
        raw_content = llm_response.content
        if isinstance(raw_content, list):
            final_text = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            final_text = raw_content
    except Exception as e:
        final_text = f"An error occurred while formatting the answer: {str(e)}\n\nRaw Context: {raw_context}"
        
    visual_data = None
    json_match = re.search(r'```json\n(.*?)\n```', final_text, re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(1))
            visual_data = VisualData(**parsed_json)
            final_text = final_text.replace(json_match.group(0), "").strip()
        except Exception:
            pass

    db.save_message(
        subquery_id=subquery_id,
        session_id=session_id,
        message_id=message_id,
        user=body.user,
        question=body.question,
        agent_question=agent_question,
        response=final_text,
        citations=["FastF1 API"],
        agents_used=agents_used,
        visual_data=visual_data.model_dump() if visual_data else None
    )

    return ChatResponse(
        session_id=session_id,
        text=final_text,
        citations=["FastF1 API"],
        visual_data=visual_data
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
