import fastf1
import os
from typing import Dict, Any

if not os.path.exists('fastf1_cache'):
    os.makedirs('fastf1_cache')
fastf1.Cache.enable_cache('fastf1_cache')

class FastF1Service:
    _session_cache: Dict[str, Any] = {}
    _schedule_cache: Dict[int, Any] = {}

    @classmethod
    def get_session(cls, year: int, event: str, session: str, with_telemetry: bool = False, with_weather: bool = False, with_messages: bool = False):
        cache_key = f"{year}_{event}_{session}_tel_{with_telemetry}_wea_{with_weather}_msg_{with_messages}"
        if cache_key in cls._session_cache:
            return cls._session_cache[cache_key]

        sess = fastf1.get_session(year, event, session)
        sess.load(laps=True, telemetry=with_telemetry, weather=with_weather, messages=with_messages)
        
        cls._session_cache[cache_key] = sess
        return sess

    @classmethod
    def get_schedule(cls, year: int):
        if year in cls._schedule_cache:
            return cls._schedule_cache[year]
            
        schedule = fastf1.get_event_schedule(year)
        cls._schedule_cache[year] = schedule
        return schedule
