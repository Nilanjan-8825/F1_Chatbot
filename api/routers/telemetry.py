import pandas as pd
from fastapi import APIRouter, HTTPException
from services.fastf1_service import FastF1Service

router = APIRouter()

@router.get("/api/telemetry")
def get_telemetry(year: int = 2024, event: str = "Monaco", session: str = "R", drivers: str = None, lap: str = None):
    try:
        sess = FastF1Service.get_session(year, event, session, with_telemetry=True)
        telemetry_data = {}
        driver_list = drivers.split(',') if drivers else sess.results.head(2)['Abbreviation'].tolist()
        
        for driver in driver_list:
            if driver not in sess.results['Abbreviation'].values:
                continue
                
            driver_laps = sess.laps.pick_drivers(driver)
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

@router.get("/api/tire-degradation")
def get_tire_degradation(year: int = 2024, event: str = "Monaco", session: str = "R", drivers: str = None):
    try:
        sess = FastF1Service.get_session(year, event, session)
        laps = sess.laps
        driver_list = drivers.split(',') if drivers else laps['Driver'].unique().tolist()

        result = {}
        for drv in driver_list:
            driver_laps = laps.pick_drivers(drv)
            if driver_laps.empty:
                continue

            driver_stints = []
            for stint_num in sorted(driver_laps['Stint'].dropna().unique()):
                stint_laps = driver_laps[driver_laps['Stint'] == stint_num]
                compound = str(stint_laps['Compound'].iloc[0]) if not stint_laps.empty else "UNKNOWN"

                lap_data = []
                for _, lap_row in stint_laps.iterrows():
                    if pd.notna(lap_row.get('LapTime')):
                        lap_data.append({
                            "lap": int(lap_row['LapNumber']),
                            "tyre_life": int(lap_row['TyreLife']) if pd.notna(lap_row.get('TyreLife')) else None,
                            "lap_time_seconds": round(lap_row['LapTime'].total_seconds(), 3),
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

@router.get("/api/qualifying")
def get_qualifying(year: int = 2024, event: str = "Monaco"):
    try:
        sess = FastF1Service.get_session(year, event, "Q")
        results = sess.results
        if results is None or results.empty:
            return []

        pole_time = None
        q3_times = results['Q3'].dropna()
        if not q3_times.empty:
            pole_time = q3_times.min()

        formatted = []
        for _, row in results.iterrows():
            q1 = row.get('Q1')
            q2 = row.get('Q2')
            q3 = row.get('Q3')

            eliminated_in = None
            if pd.isna(q3) and pd.notna(q2):
                eliminated_in = "Q2"
            elif pd.isna(q2) and pd.notna(q1):
                eliminated_in = "Q1"
            elif pd.isna(q1):
                eliminated_in = "DNS"

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

@router.get("/api/sector-times")
def get_sector_times(year: int = 2024, event: str = "Monaco", session: str = "R", drivers: str = None, lap: str = None):
    try:
        sess = FastF1Service.get_session(year, event, session)
        laps = sess.laps

        valid_laps = laps[
            laps['Sector1Time'].notna() &
            laps['Sector2Time'].notna() &
            laps['Sector3Time'].notna()
        ]

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
