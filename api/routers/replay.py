import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from services import replay_service

router = APIRouter()

@router.get("/api/race-replay/sessions")
def get_replay_sessions(year: int = None):
    try:
        sessions = replay_service.get_available_sessions(year)
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/race-replay/track")
def get_replay_track(year: int = 2024, event: str = "Monaco", session: str = "R"):
    try:
        track = replay_service.get_track_shape(year, event, session)
        return track
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/race-replay/data")
def get_replay_data(year: int = 2024, event: str = "Monaco", session: str = "R"):
    try:
        data = replay_service.get_race_replay_data(year, event, session)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/race-replay/positions")
def get_replay_positions(year: int = 2024, event: str = "Monaco", session: str = "R"):
    try:
        positions = replay_service.get_track_positions_for_replay(year, event, session)
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from starlette.concurrency import run_in_threadpool

@router.websocket("/api/race-replay/stream")
async def replay_stream(websocket: WebSocket, year: int = 2024, event: str = "Bahrain", session: str = "R"):
    await websocket.accept()
    try:
        # Load the data in a threadpool so it doesn't block the ASGI event loop
        positions_data = await run_in_threadpool(
            replay_service.get_track_positions_for_replay, year, event, session
        )
        laps_data = await run_in_threadpool(
            replay_service.get_race_replay_data, year, event, session
        )
        
        total_laps = positions_data.get("total_laps", 0)
        drivers = positions_data.get("drivers", [])
        driver_positions = positions_data.get("driver_positions", {})
        lap_stats = laps_data.get("laps", {})
        
        # Send initial metadata
        await websocket.send_json({
            "type": "metadata",
            "total_laps": total_laps,
            "drivers": drivers,
            "track_shape": positions_data.get("track_shape", {})
        })
        
        # Stream lap by lap
        for current_lap in range(1, total_laps + 1):
            lap_update = {
                "type": "lap_update",
                "lap": current_lap,
                "positions": {},
                "stats": lap_stats.get(current_lap, {})
            }
            
            for drv, drv_laps in driver_positions.items():
                if current_lap in drv_laps:
                    lap_update["positions"][drv] = drv_laps[current_lap]
                    
            await websocket.send_json(lap_update)
            # Sleep to simulate live streaming (1.5 seconds per lap)
            await asyncio.sleep(1.5)
            
        await websocket.send_json({"type": "end_of_session"})
        
        # Wait for client to close, or echo any incoming messages
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"echo": data, "message": "Replay finished"})
            
    except WebSocketDisconnect:
        print("Client disconnected from replay stream")
    except Exception as e:
        print(f"Error in websocket stream: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except:
            pass
