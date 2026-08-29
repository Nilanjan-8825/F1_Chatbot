import requests
import pandas as pd
from fastapi import APIRouter, HTTPException
from services.fastf1_service import FastF1Service

router = APIRouter()

@router.get("/api/schedule")
def get_schedule(year: int = 2024):
    try:
        schedule = FastF1Service.get_schedule(year)
        df = schedule[['RoundNumber', 'Country', 'Location', 'OfficialEventName', 'EventDate', 'EventFormat']]
        records = df.fillna("").to_dict(orient="records")
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/standings")
def get_standings(year: int = 2024):
    try:
        url = f"http://api.jolpi.ca/ergast/f1/{year}/driverStandings.json"
        response = requests.get(url, timeout=10)
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

@router.get("/api/drivers")
def get_drivers(year: int = 2024, event: str = "Monaco", session: str = "R"):
    try:
        sess = FastF1Service.get_session(year, event, session)
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

@router.get("/api/laps")
def get_laps(year: int = 2024, event: str = "Monaco", session: str = "R"):
    try:
        sess = FastF1Service.get_session(year, event, session)
        total_laps = int(sess.laps['LapNumber'].max()) if not sess.laps.empty else 0
        return {"total_laps": total_laps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/constructor-standings")
def get_constructor_standings(year: int = 2024):
    try:
        url = f"http://api.jolpi.ca/ergast/f1/{year}/constructorStandings.json"
        response = requests.get(url, timeout=10)
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