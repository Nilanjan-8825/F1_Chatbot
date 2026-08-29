import pandas as pd
from fastapi import APIRouter, HTTPException
from services.fastf1_service import FastF1Service

router = APIRouter()

@router.get("/api/weather")
def get_weather(year: int = 2024, event: str = "Monaco", session: str = "R"):
    try:
        sess = FastF1Service.get_session(year, event, session, with_weather=True)
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

@router.get("/api/pit-stops")
def get_pit_stops(year: int = 2024, event: str = "Monaco", session: str = "R", driver: str = None):
    try:
        sess = FastF1Service.get_session(year, event, session)
        laps = sess.laps
        driver_list = [driver] if driver else laps['Driver'].unique().tolist()

        result = {}
        for drv in driver_list:
            driver_laps = laps.pick_drivers(drv)
            if driver_laps.empty:
                continue

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

            stops = []
            pit_laps = driver_laps[driver_laps['PitInTime'].notna()]
            for _, pit_lap in pit_laps.iterrows():
                lap_num = int(pit_lap['LapNumber'])
                compound_before = str(pit_lap['Compound'])

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

@router.get("/api/race-control")
def get_race_control(year: int = 2024, event: str = "Monaco", session: str = "R"):
    try:
        sess = FastF1Service.get_session(year, event, session, with_messages=True)
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
