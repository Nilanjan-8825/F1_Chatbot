# F1 AI Chatbot & Data API

An intelligent, multi-agent conversational AI and REST API for Formula 1 data analysis. Built with FastAPI, LangGraph, and FastF1, this project allows users to ask complex questions about F1 races, driver telemetry, lap times, and season schedules using natural language, as well as providing raw endpoints for dashboard visualizations.

## Features

- **Multi-Agent Conversational AI**: Powered by `langchain`, `langgraph`, and Google's Gemini (`gemini-3.1-flash-lite`), the chatbot uses a supervisor agent to route natural language queries to specialized pandas dataframe workers:
  - **Schedule Worker**: Season formats, race venues, and dates.
  - **Results Worker**: Finishing positions, points, grids, and race statuses.
  - **Laps Worker**: Lap-by-lap timing, sector times, and tire compounds.
  - **Telemetry Worker**: High-frequency car telemetry (speed, throttle, braking, DRS).
  - **Weather Worker**: Session weather conditions (temperature, humidity, wind, rainfall).
  - **Strategy Worker**: Tire strategy, pit stops, stint analysis, compound choices, and degradation patterns.
- **FastF1 Integration**: Leverages the `fastf1` library to pull historical and session data directly from F1 APIs.
- **RESTful API Endpoints**: Provides structured data endpoints for building custom frontends or dashboards.
- **Race Replay Backend**: Pre-computes lap-by-lap position data, track shapes, and race events for animated race replays. WebSocket stub ready for future live streaming via OpenF1.
- **Persistent Chat History**: Stores user conversations, agent routing decisions, and visual data requests in a MySQL database.
- **Dynamic Visualizations**: The AI automatically detects when tabular data should be represented visually and returns structured JSON for rendering charts (tables, pie charts, bar charts).

## Prerequisites

- Python 3.8+
- MySQL Server
- numpy (`pip install numpy`)

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd F1_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your credentials:
   ```env
   llm_key=your_google_gemini_api_key
   MYSQL_HOST=localhost
   MYSQL_USER=root
   MYSQL_PASSWORD=your_password
   MYSQL_DATABASE=f1_chatbot
   ```

4. **Database Setup:**
   Ensure you have a MySQL database named `f1_chatbot` running with the appropriate table (`ai_assistant_messages`). 
   
   *Note: The `db.py` file expects a table named `ai_assistant_messages` with columns: `subquery_id`, `session_id`, `message_id`, `user`, `question`, `agent_question`, `response`, `citations`, `agents_used`, `visual_data`, and `needs_clarification`.
   Also if desired the database names and table names can be changed but please ensure that the query , table and database changes in the codebase is done appropriately.*

5. **Run the Application:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### Core Endpoints
- `GET /api/schedule?year={year}` — Season schedule for a given year.
- `GET /api/standings?year={year}` — Driver championship standings.
- `GET /api/constructor-standings?year={year}` — Constructor (team) championship standings.
- `GET /api/drivers?year={year}&event={event}&session={session}` — Lists all drivers in a session.
- `GET /api/laps?year={year}&event={event}&session={session}` — Total number of laps in a session.
- `POST /api/chat` — Natural language queries via the multi-agent chatbot.

### Telemetry & Analysis
- `GET /api/telemetry?year={year}&event={event}&session={session}&drivers={drivers}&lap={lap}` — Raw telemetry (throttle, speed, distance).
- `GET /api/sector-times?year={year}&event={event}&session={session}&drivers={drivers}&lap={lap}` — Sector breakdown with purple (session best) markers.
- `GET /api/qualifying?year={year}&event={event}` — Q1/Q2/Q3 times with gap to pole and elimination info.

### Weather & Strategy
- `GET /api/weather?year={year}&event={event}&session={session}` — Weather data (temperature, humidity, wind, rainfall).
- `GET /api/pit-stops?year={year}&event={event}&session={session}&driver={driver}` — Pit stop events and stint timelines per driver.
- `GET /api/race-control?year={year}&event={event}&session={session}` — Race control messages (safety car, VSC, flags, penalties).
- `GET /api/tire-degradation?year={year}&event={event}&session={session}&drivers={drivers}` — Lap time vs tire age per stint for degradation curves.

### Race Replay
- `GET /api/race-replay/sessions?year={year}` — Available sessions for replay.
- `GET /api/race-replay/track?year={year}&event={event}&session={session}` — Circuit shape (X/Y coordinates), corners, and rotation angle.
- `GET /api/race-replay/data?year={year}&event={event}&session={session}` — Full race replay dataset (all laps, all drivers, positions, timing, tires, events).
- `GET /api/race-replay/positions?year={year}&event={event}&session={session}` — Per-driver, per-lap normalized track positions for dot animation.
- `WS /api/race-replay/stream?session_key={key}` — WebSocket for future live streaming (currently returns stub).

## Architecture

The AI chatbot uses a **Supervisor Agent** pattern:
1. The user sends a query to `/api/chat`.
2. The `Supervisor` node evaluates the query, extracts entities (Year, Event, Session), and determines which specialized worker needs to handle it.
3. The query is routed to the appropriate Pandas Dataframe Agent (Schedule, Results, Laps, Telemetry, Weather, or Strategy).
4. The worker dynamically writes and executes pandas code against the FastF1 data to find the answer.
5. The final insights are synthesized by an LLM and returned to the user, complete with data for frontend visualizations if applicable.

<<<<<<< HEAD
### Race Replay Architecture

The race replay backend (`race_replay.py`) pre-computes all data needed for animated replays:
- **Track Shape**: Extracted from the fastest lap's GPS position data, rotated and normalized.
- **Lap Data**: Per-driver, per-lap position, timing, tire compound, stint, pit events.
- **Gap Computation**: Cumulative time gaps to the race leader.
- **Race Events**: Safety car, VSC, red flags, penalties extracted from race control messages.
- **Animation Positions**: Normalized distance progression (0.0→1.0) around the track for smooth dot animation.

Data is cached in memory to avoid re-processing expensive FastF1 session loads.

## Project Structure

```
F1_project/
├── api.py                  # FastAPI application with all REST endpoints
├── chatbot_integration.py  # LangGraph multi-agent chatbot (Supervisor + 6 Workers)
├── race_replay.py          # Race replay data processing module
├── db.py                   # MySQL database layer for chat history
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, DB credentials)
└── README.md               # This file
```
=======
Coming releases will add more feature and functionalities to the chatbot.
>>>>>>> 8d70e90742a38b7bd0921af0c2b3d687c6cf3b7a
