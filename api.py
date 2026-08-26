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
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

import chatbot_integration
import db
import race_replay

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


@app.get("/api/weather")
async def get_weather(year: int = 2024, event: str = "Monaco", session: str = "R"):
    """
    Returns weather data throughout the session (temperature, humidity, wind, rainfall).
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=False, telemetry=False, weather=True, messages=False)

        weather = sess.weather_data
        if weather is None or weather.empty:
            return []

        records = []
        for _, row in weather.iterrows():
            records.append({
                "time": str(row.get("Time", "")),
                "air_temp": round(float(row["AirTemp"]), 1) if pd.notna(row.get("AirTemp")) else None,
                "track_temp": round(float(row["TrackTemp"]), 1) if pd.notna(row.get("TrackTemp")) else None,
                "humidity": round(float(row["Humidity"]), 1) if pd.notna(row.get("Humidity")) else None,
                "wind_speed": round(float(row["WindSpeed"]), 1) if pd.notna(row.get("WindSpeed")) else None,
                "wind_direction": int(row["WindDirection"]) if pd.notna(row.get("WindDirection")) else None,
                "rainfall": bool(row["Rainfall"]) if pd.notna(row.get("Rainfall")) else None,
                "pressure": round(float(row["Pressure"]), 1) if pd.notna(row.get("Pressure")) else None,
            })
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pit-stops")
async def get_pit_stops(year: int = 2024, event: str = "Monaco", session: str = "R", driver: str = None):
    """
    Returns pit stop events and stint timelines per driver.
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = sess.laps
        driver_list = [driver] if driver else laps['Driver'].unique().tolist()

        result = {}
        for drv in driver_list:
            driver_laps = laps.pick_driver(drv)
            if driver_laps.empty:
                continue

            # Build stints
            stints = []
            for stint_num in sorted(driver_laps['Stint'].dropna().unique()):
                stint_laps = driver_laps[driver_laps['Stint'] == stint_num]
                compound = str(stint_laps['Compound'].iloc[0]) if not stint_laps.empty else "UNKNOWN"
                start_lap = int(stint_laps['LapNumber'].min())
                end_lap = int(stint_laps['LapNumber'].max())
                stints.append({
                    "stint": int(stint_num),
                    "compound": compound,
                    "start_lap": start_lap,
                    "end_lap": end_lap,
                    "laps": end_lap - start_lap + 1,
                })

            # Build pit stop events (laps where PitInTime is not NaT)
            stops = []
            pit_laps = driver_laps[driver_laps['PitInTime'].notna()]
            for _, pit_lap in pit_laps.iterrows():
                lap_num = int(pit_lap['LapNumber'])
                compound_before = str(pit_lap['Compound'])

                # Find the next stint's compound
                next_laps = driver_laps[driver_laps['LapNumber'] > lap_num]
                compound_after = str(next_laps['Compound'].iloc[0]) if not next_laps.empty else "N/A"

                pit_in = pit_lap['PitInTime']
                pit_out = pit_lap['PitOutTime']
                duration = None
                if pd.notna(pit_in) and pd.notna(pit_out):
                    duration = round((pit_out - pit_in).total_seconds(), 3)

                stops.append({
                    "lap": lap_num,
                    "pit_duration_seconds": duration,
                    "compound_before": compound_before,
                    "compound_after": compound_after,
                    "tyre_life_before": int(pit_lap['TyreLife']) if pd.notna(pit_lap.get('TyreLife')) else None,
                })

            result[drv] = {"stops": stops, "stints": stints}

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/race-control")
async def get_race_control(year: int = 2024, event: str = "Monaco", session: str = "R"):
    """
    Returns race control messages: safety car, VSC, flags, penalties, deleted laps.
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=False, telemetry=False, weather=False, messages=True)

        messages = sess.race_control_messages
        if messages is None or messages.empty:
            return []

        records = []
        for _, row in messages.iterrows():
            records.append({
                "time": str(row.get("Time", "")),
                "lap_number": int(row["Lap"]) if pd.notna(row.get("Lap")) else None,
                "category": str(row.get("Category", "")),
                "message": str(row.get("Message", "")),
                "flag": str(row.get("Flag", "")) if pd.notna(row.get("Flag")) else None,
                "driver_number": str(row.get("RacingNumber", "")) if pd.notna(row.get("RacingNumber")) else None,
            })
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tire-degradation")
async def get_tire_degradation(year: int = 2024, event: str = "Monaco", session: str = "R", drivers: str = None):
    """
    Returns lap time vs tire age per driver per stint — ready for degradation curve plotting.
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = sess.laps
        driver_list = drivers.split(',') if drivers else laps['Driver'].unique().tolist()

        result = {}
        for drv in driver_list:
            driver_laps = laps.pick_driver(drv)
            if driver_laps.empty:
                continue

            driver_stints = []
            for stint_num in sorted(driver_laps['Stint'].dropna().unique()):
                stint_laps = driver_laps[driver_laps['Stint'] == stint_num]
                compound = str(stint_laps['Compound'].iloc[0]) if not stint_laps.empty else "UNKNOWN"

                lap_data = []
                for _, lap in stint_laps.iterrows():
                    if pd.notna(lap.get('LapTime')):
                        lap_data.append({
                            "lap": int(lap['LapNumber']),
                            "tyre_life": int(lap['TyreLife']) if pd.notna(lap.get('TyreLife')) else None,
                            "lap_time_seconds": round(lap['LapTime'].total_seconds(), 3),
                        })

                if lap_data:
                    driver_stints.append({
                        "stint": int(stint_num),
                        "compound": compound,
                        "laps": lap_data,
                    })

            if driver_stints:
                result[drv] = driver_stints

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/constructor-standings")
async def get_constructor_standings(year: int = 2024):
    """
    Returns constructor (team) championship standings from the Ergast API.
    """
    try:
        loop = asyncio.get_running_loop()
        url = f"http://api.jolpi.ca/ergast/f1/{year}/constructorStandings.json"

        response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=10))
        response.raise_for_status()
        data = response.json()

        standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]

        formatted = []
        for s in standings:
            formatted.append({
                "position": int(s["position"]),
                "constructor": s["Constructor"]["name"],
                "nationality": s["Constructor"].get("nationality", ""),
                "points": float(s["points"]),
                "wins": int(s.get("wins", 0)),
            })

        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qualifying")
async def get_qualifying(year: int = 2024, event: str = "Monaco"):
    """
    Returns Q1/Q2/Q3 qualifying times for all drivers with gap-to-pole and elimination info.
    """
    try:
        sess = fastf1.get_session(year, event, "Q")
        sess.load(laps=False, telemetry=False, weather=False, messages=False)

        results = sess.results
        if results is None or results.empty:
            return []

        # Determine pole time for gap calculation
        pole_time = None
        q3_times = results['Q3'].dropna()
        if not q3_times.empty:
            pole_time = q3_times.min()

        formatted = []
        for _, row in results.iterrows():
            q1 = row.get('Q1')
            q2 = row.get('Q2')
            q3 = row.get('Q3')

            # Determine elimination stage
            eliminated_in = None
            if pd.isna(q3) and pd.notna(q2):
                eliminated_in = "Q2"
            elif pd.isna(q2) and pd.notna(q1):
                eliminated_in = "Q1"
            elif pd.isna(q1):
                eliminated_in = "DNS"

            # Calculate gap to pole (using best qualifying time)
            best_time = q3 if pd.notna(q3) else (q2 if pd.notna(q2) else q1)
            gap = None
            if best_time is not None and pole_time is not None and pd.notna(best_time) and pd.notna(pole_time):
                gap = round((best_time - pole_time).total_seconds(), 3)

            formatted.append({
                "position": int(row["Position"]) if pd.notna(row.get("Position")) else None,
                "driver": row.get("Abbreviation", ""),
                "full_name": row.get("FullName", ""),
                "team": row.get("TeamName", ""),
                "q1": str(q1) if pd.notna(q1) else None,
                "q2": str(q2) if pd.notna(q2) else None,
                "q3": str(q3) if pd.notna(q3) else None,
                "gap_to_pole": gap,
                "eliminated_in": eliminated_in,
            })

        formatted.sort(key=lambda x: x["position"] if x["position"] is not None else 999)
        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sector-times")
async def get_sector_times(year: int = 2024, event: str = "Monaco", session: str = "R", drivers: str = None, lap: str = None):
    """
    Returns sector time breakdowns with session-best (purple sector) markers.
    """
    try:
        sess = fastf1.get_session(year, event, session)
        sess.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = sess.laps

        # Filter to laps with valid sector times
        valid_laps = laps[
            laps['Sector1Time'].notna() &
            laps['Sector2Time'].notna() &
            laps['Sector3Time'].notna()
        ]

        # Compute session best sectors
        session_best = {}
        for sector_col, sector_key in [('Sector1Time', 'sector_1'), ('Sector2Time', 'sector_2'), ('Sector3Time', 'sector_3')]:
            if not valid_laps.empty:
                best_idx = valid_laps[sector_col].idxmin()
                best_row = valid_laps.loc[best_idx]
                best_time = round(best_row[sector_col].total_seconds(), 3)
                session_best[sector_key] = {
                    "driver": best_row['Driver'],
                    "time_seconds": best_time,
                }

        # Get per-driver sector times
        driver_list = drivers.split(',') if drivers else valid_laps['Driver'].unique().tolist()

        driver_sectors = {}
        for drv in driver_list:
            drv_laps = valid_laps[valid_laps['Driver'] == drv]
            if drv_laps.empty:
                continue

            if lap and lap.isdigit():
                drv_laps = drv_laps[drv_laps['LapNumber'] == int(lap)]

            sector_data = []
            for _, row in drv_laps.iterrows():
                s1 = round(row['Sector1Time'].total_seconds(), 3)
                s2 = round(row['Sector2Time'].total_seconds(), 3)
                s3 = round(row['Sector3Time'].total_seconds(), 3)

                sector_data.append({
                    "lap": int(row['LapNumber']),
                    "sector_1": s1,
                    "sector_2": s2,
                    "sector_3": s3,
                    "is_best_s1": s1 == session_best.get("sector_1", {}).get("time_seconds"),
                    "is_best_s2": s2 == session_best.get("sector_2", {}).get("time_seconds"),
                    "is_best_s3": s3 == session_best.get("sector_3", {}).get("time_seconds"),
                })

            if sector_data:
                driver_sectors[drv] = sector_data

        return {
            "session_best_sectors": session_best,
            "drivers": driver_sectors,
        }
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


# ---------------------------------------------------------------------------
# Race Replay Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/race-replay/sessions")
async def get_replay_sessions(year: int = None):
    """
    Returns a list of available sessions for race replay.
    """
    try:
        sessions = race_replay.get_available_sessions(year)
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/race-replay/track")
async def get_replay_track(year: int = 2024, event: str = "Monaco", session: str = "R"):
    """
    Returns the circuit shape (X/Y coordinates), corner positions, and rotation.
    Used by the frontend to render the track map.
    """
    try:
        track = race_replay.get_track_shape(year, event, session)
        return track
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/race-replay/data")
async def get_replay_data(year: int = 2024, event: str = "Monaco", session: str = "R"):
    """
    Returns the full race replay dataset: all laps, all drivers,
    positions, timing, tires, pit events, and race control messages.
    This is the main payload the frontend uses to animate the race.
    """
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            None,
            lambda: race_replay.get_race_replay_data(year, event, session)
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/race-replay/positions")
async def get_replay_positions(year: int = 2024, event: str = "Monaco", session: str = "R"):
    """
    Returns per-driver, per-lap normalized track positions for animation.
    Each driver's lap has ~30 distance samples (0.0 → 1.0) that the frontend
    interpolates along the track shape to animate driver dots.

    NOTE: This endpoint is heavier than /data as it loads telemetry.
    Consider using /data for timing-only replays and this for full animation.
    """
    try:
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(
            None,
            lambda: race_replay.get_track_positions_for_replay(year, event, session)
        )
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/race-replay/stream")
async def replay_stream(websocket: WebSocket, session_key: str = None):
    """
    WebSocket endpoint for live race streaming (future OpenF1 integration).

    Currently returns a stub message. When OpenF1 live mode is implemented,
    this will push real-time car positions, intervals, and race events.
    """
    await websocket.accept()
    try:
        await websocket.send_json({
            "status": "stub",
            "message": "Live streaming is not yet implemented. "
                       "Use GET /api/race-replay/data for historical replays.",
            "future_features": [
                "Real-time car positions via OpenF1 /v1/location",
                "Live intervals via OpenF1 /v1/intervals",
                "Live position changes via OpenF1 /v1/position",
                "Team radio messages via OpenF1 /v1/team_radio",
            ]
        })
        # Keep connection alive until client disconnects
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"echo": data, "status": "stub"})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
