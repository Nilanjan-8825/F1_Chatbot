"""
Race Replay Backend Module

Handles heavy data processing for the race replay feature:
- Track shape extraction from FastF1 position data
- Per-driver, per-lap position data computation
- Race event timeline construction (SC, VSC, flags, pit stops)
- Driver color mapping for visualization
- In-memory caching to avoid re-processing

Designed for historical replay via FastF1 (Phase 1).
Structured for future OpenF1 live integration (Phase 2).
"""

import math
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple

import fastf1
import numpy as np
import pandas as pd


from services.fastf1_service import FastF1Service


# ---------------------------------------------------------------------------
# Track shape extraction
# ---------------------------------------------------------------------------

def get_track_shape(year: int, event: str, session_type: str = "R") -> Dict[str, Any]:
    """
    Extract the circuit shape from the fastest lap's position data.

    Returns a dict with:
      - track_points: list of {x, y} coordinates (normalized & rotated)
      - corners: list of corner positions with number and letter
      - rotation: rotation angle applied (degrees)
      - circuit_name: name of the circuit
    """
    sess = FastF1Service.get_session(year, event, session_type, with_telemetry=True)

    # Get track coordinates from the fastest lap
    fastest_lap = sess.laps.pick_fastest()
    pos_data = fastest_lap.get_pos_data()

    if pos_data is None or pos_data.empty:
        raise ValueError(f"No position data available for {year} {event} {session_type}")

    x = pos_data['X'].values
    y = pos_data['Y'].values

    # Get circuit info for rotation and corners
    circuit_info = sess.get_circuit_info()
    rotation_angle = circuit_info.rotation if hasattr(circuit_info, 'rotation') else 0

    # Apply rotation to align the track
    x_rot, y_rot = _rotate_points(x, y, rotation_angle)

    # Normalize to 0-1000 range for consistent rendering
    x_norm, y_norm = _normalize_coordinates(x_rot, y_rot, target_size=1000)

    # Build track points (downsample to ~200 points for performance)
    total_points = len(x_norm)
    step = max(1, total_points // 200)
    track_points = [
        {"x": round(float(x_norm[i]), 1), "y": round(float(y_norm[i]), 1)}
        for i in range(0, total_points, step)
    ]

    # Extract corner positions
    corners = []
    if hasattr(circuit_info, 'corners') and circuit_info.corners is not None:
        for _, corner in circuit_info.corners.iterrows():
            cx, cy = _rotate_points(
                np.array([corner.get('X', 0)]),
                np.array([corner.get('Y', 0)]),
                rotation_angle
            )
            cx_norm, cy_norm = _normalize_coordinates_with_ref(
                cx, cy, x_rot, y_rot, target_size=1000
            )
            corners.append({
                "number": int(corner.get('Number', 0)),
                "letter": str(corner.get('Letter', '')),
                "x": round(float(cx_norm[0]), 1),
                "y": round(float(cy_norm[0]), 1),
            })

    return {
        "track_points": track_points,
        "corners": corners,
        "rotation": round(rotation_angle, 2),
        "circuit_name": getattr(circuit_info, 'circuitName', event) if hasattr(circuit_info, 'circuitName') else event,
    }


# ---------------------------------------------------------------------------
# Race replay data computation
# ---------------------------------------------------------------------------

def get_race_replay_data(year: int, event: str, session_type: str = "R") -> Dict[str, Any]:
    """
    Compute the full race replay dataset: all laps, all drivers,
    positions, timing, tires, pit events, and race control messages.

    Returns a dict with:
      - total_laps: int
      - drivers: list of driver info (code, color, team, number)
      - laps: dict keyed by lap number, each containing per-driver data
      - events: list of race control events (SC, VSC, flags)
    """
    # Load session with laps + messages (no telemetry for performance)
    sess = FastF1Service.get_session(year, event, session_type, with_messages=True)

    all_laps = sess.laps
    if all_laps is None or all_laps.empty:
        raise ValueError(f"No lap data available for {year} {event} {session_type}")

    total_laps = int(all_laps['LapNumber'].max())

    # Build driver info
    drivers = get_driver_info(sess)

    # Build per-lap data
    laps_data = {}
    for lap_num in range(1, total_laps + 1):
        lap_records = all_laps[all_laps['LapNumber'] == lap_num]
        drivers_lap_data = {}

        for _, row in lap_records.iterrows():
            drv = row['Driver']

            lap_time = None
            lap_time_seconds = None
            if pd.notna(row.get('LapTime')):
                lap_time_seconds = round(row['LapTime'].total_seconds(), 3)
                lap_time = _format_timedelta(row['LapTime'])

            drivers_lap_data[drv] = {
                "position": int(row['Position']) if pd.notna(row.get('Position')) else None,
                "lap_time": lap_time,
                "lap_time_seconds": lap_time_seconds,
                "compound": str(row['Compound']) if pd.notna(row.get('Compound')) else None,
                "tyre_life": int(row['TyreLife']) if pd.notna(row.get('TyreLife')) else None,
                "stint": int(row['Stint']) if pd.notna(row.get('Stint')) else None,
                "pitted": bool(pd.notna(row.get('PitInTime'))),
                "sector_1": round(row['Sector1Time'].total_seconds(), 3) if pd.notna(row.get('Sector1Time')) else None,
                "sector_2": round(row['Sector2Time'].total_seconds(), 3) if pd.notna(row.get('Sector2Time')) else None,
                "sector_3": round(row['Sector3Time'].total_seconds(), 3) if pd.notna(row.get('Sector3Time')) else None,
            }

        laps_data[lap_num] = drivers_lap_data

    # Compute cumulative gaps to leader for each lap
    laps_data = _compute_gaps(laps_data, total_laps)

    # Build race events timeline
    events = _extract_race_events(sess)

    return {
        "total_laps": total_laps,
        "drivers": drivers,
        "laps": laps_data,
        "events": events,
    }


def get_track_positions_for_replay(year: int, event: str, session_type: str = "R") -> Dict[str, Any]:
    """
    Compute per-driver, per-lap track positions for animation.

    For each driver on each lap, compute normalized distance progression
    (0.0 → 1.0 around the track) at regular time intervals. This allows
    the frontend to interpolate driver positions along the track shape.

    Returns:
      - track_shape: the circuit outline (X/Y points)
      - driver_positions: dict keyed by driver code, containing per-lap
        normalized distance samples
    """
    sess = FastF1Service.get_session(year, event, session_type, with_telemetry=True)

    all_laps = sess.laps
    total_laps = int(all_laps['LapNumber'].max())

    # Get track shape (reuse the cached function)
    track = get_track_shape(year, event, session_type)

    # Compute per-driver per-lap positions
    driver_positions = {}
    drivers = all_laps['Driver'].unique()

    for drv in drivers:
        drv_laps = all_laps.pick_drivers(drv)
        drv_positions = {}

        for _, lap_row in drv_laps.iterrows():
            lap_num = int(lap_row['LapNumber'])

            try:
                tel = lap_row.get_telemetry()
                if tel is None or tel.empty or 'Distance' not in tel.columns:
                    continue

                # Normalize distance to 0.0 → 1.0
                max_dist = tel['Distance'].max()
                if max_dist <= 0:
                    continue

                rel_distances = (tel['Distance'] / max_dist).values

                # Downsample to ~30 points per lap
                total = len(rel_distances)
                step = max(1, total // 30)
                sampled = [round(float(rel_distances[i]), 4) for i in range(0, total, step)]

                # Ensure we end at 1.0
                if sampled[-1] < 0.99:
                    sampled.append(1.0)

                drv_positions[lap_num] = sampled

            except Exception:
                # Some laps may not have telemetry (pit laps, retirements)
                continue

        if drv_positions:
            driver_positions[drv] = drv_positions

    return {
        "track_shape": track,
        "total_laps": total_laps,
        "drivers": get_driver_info(sess),
        "driver_positions": driver_positions,
    }


# ---------------------------------------------------------------------------
# Driver info and colors
# ---------------------------------------------------------------------------

def get_driver_info(session) -> List[Dict[str, Any]]:
    """
    Extract driver info including team colors for visualization.
    """
    results = session.results
    if results is None or results.empty:
        return []

    drivers = []
    for _, row in results.iterrows():
        color = None
        if pd.notna(row.get('TeamColor')):
            color = f"#{row['TeamColor']}"

        drivers.append({
            "code": row.get("Abbreviation", ""),
            "full_name": row.get("FullName", ""),
            "team": row.get("TeamName", ""),
            "number": int(row["DriverNumber"]) if pd.notna(row.get("DriverNumber")) else None,
            "color": color,
            "grid_position": int(row["GridPosition"]) if pd.notna(row.get("GridPosition")) else None,
        })

    return drivers


def get_available_sessions(year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of available sessions (from FastF1 schedule).
    """
    years = [year] if year else [2023, 2024, 2025]
    sessions = []

    for y in years:
        try:
            schedule = FastF1Service.get_schedule(y)
            for _, event in schedule.iterrows():
                if pd.notna(event.get('EventDate')):
                    sessions.append({
                        "year": y,
                        "round": int(event['RoundNumber']) if pd.notna(event.get('RoundNumber')) else None,
                        "event_name": str(event.get('OfficialEventName', '')),
                        "country": str(event.get('Country', '')),
                        "location": str(event.get('Location', '')),
                        "date": str(event.get('EventDate', '')),
                        "format": str(event.get('EventFormat', '')),
                        "session_types": ["R", "Q", "FP1", "FP2", "FP3"],
                    })
        except Exception:
            continue

    return sessions


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _rotate_points(x: np.ndarray, y: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate X/Y coordinates by the given angle (in degrees)."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    return x_rot, y_rot


def _normalize_coordinates(x: np.ndarray, y: np.ndarray,
                           target_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize coordinates to fit within [0, target_size] preserving aspect ratio."""
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    # Preserve aspect ratio
    scale = target_size / max(x_range, y_range)
    margin = target_size * 0.05  # 5% margin

    x_norm = (x - x_min) * scale + margin
    y_norm = (y - y_min) * scale + margin

    return x_norm, y_norm


def _normalize_coordinates_with_ref(x: np.ndarray, y: np.ndarray,
                                    ref_x: np.ndarray, ref_y: np.ndarray,
                                    target_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize coordinates using a reference coordinate set (for corners etc.)."""
    x_min, x_max = ref_x.min(), ref_x.max()
    y_min, y_max = ref_y.min(), ref_y.max()

    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    scale = target_size / max(x_range, y_range)
    margin = target_size * 0.05

    x_norm = (x - x_min) * scale + margin
    y_norm = (y - y_min) * scale + margin

    return x_norm, y_norm


def _format_timedelta(td) -> str:
    """Format a pandas Timedelta into M:SS.mmm string."""
    if pd.isna(td):
        return None
    total_seconds = td.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:06.3f}"


def _compute_gaps(laps_data: Dict, total_laps: int) -> Dict:
    """
    Compute cumulative time gaps to the leader for each lap.

    Uses cumulative lap times to approximate the gap to P1.
    """
    cumulative_times = {}  # driver -> total race time in seconds

    for lap_num in range(1, total_laps + 1):
        if lap_num not in laps_data:
            continue

        lap_drivers = laps_data[lap_num]

        for drv, data in lap_drivers.items():
            if data.get("lap_time_seconds") is not None:
                if drv not in cumulative_times:
                    cumulative_times[drv] = 0.0
                cumulative_times[drv] += data["lap_time_seconds"]

        # Find the leader (driver in P1 or with the lowest cumulative time)
        leader = None
        leader_time = float('inf')
        for drv, data in lap_drivers.items():
            pos = data.get("position")
            if pos == 1:
                leader = drv
                leader_time = cumulative_times.get(drv, 0)
                break

        if leader is None and cumulative_times:
            # Fallback: use lowest cumulative time
            leader = min(
                (d for d in lap_drivers if d in cumulative_times),
                key=lambda d: cumulative_times[d],
                default=None
            )
            if leader:
                leader_time = cumulative_times[leader]

        # Compute gaps
        for drv, data in lap_drivers.items():
            if drv == leader:
                data["gap_to_leader"] = 0.0
            elif drv in cumulative_times and leader_time < float('inf'):
                data["gap_to_leader"] = round(cumulative_times[drv] - leader_time, 3)
            else:
                data["gap_to_leader"] = None

    return laps_data


def _extract_race_events(session) -> List[Dict[str, Any]]:
    """Extract race control events (SC, VSC, flags, penalties) from session messages."""
    events = []

    try:
        messages = session.race_control_messages
        if messages is None or messages.empty:
            return events

        for _, row in messages.iterrows():
            category = str(row.get("Category", ""))
            # Filter to important events (skip routine messages)
            if category in ("SafetyCar", "Flag", "Drs", "") or \
               "SAFETY CAR" in str(row.get("Message", "")).upper() or \
               "VIRTUAL SAFETY" in str(row.get("Message", "")).upper() or \
               "RED FLAG" in str(row.get("Message", "")).upper() or \
               "PENALTY" in str(row.get("Message", "")).upper():
                events.append({
                    "time": str(row.get("Time", "")),
                    "lap": int(row["Lap"]) if pd.notna(row.get("Lap")) else None,
                    "category": category,
                    "message": str(row.get("Message", "")),
                    "flag": str(row.get("Flag", "")) if pd.notna(row.get("Flag")) else None,
                })
    except Exception:
        pass

    return events
