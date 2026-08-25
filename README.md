# F1 AI Chatbot & Data API

An intelligent, multi-agent conversational AI and REST API for Formula 1 data analysis. Built with FastAPI, LangGraph, and FastF1, this project allows users to ask complex questions about F1 races, driver telemetry, lap times, and season schedules using natural language, as well as providing raw endpoints for dashboard visualizations.

## Features

- **Multi-Agent Conversational AI**: Powered by `langchain`, `langgraph`, and Google's Gemini (`gemini-3.1-flash-lite`), the chatbot uses a supervisor agent to route natural language queries to specialized pandas dataframe workers:
  - **Schedule Worker**: Season formats, race venues, and dates.
  - **Results Worker**: Finishing positions, points, grids, and race statuses.
  - **Laps Worker**: Lap-by-lap timing, sector times, and tire compounds.
  - **Telemetry Worker**: High-frequency car telemetry (speed, throttle, braking, DRS).
- **FastF1 Integration**: Leverages the `fastf1` library to pull historical and session data directly from F1 APIs.
- **RESTful API Endpoints**: Provides structured data endpoints (`/api/schedule`, `/api/standings`, `/api/telemetry`, etc.) for building custom frontends or dashboards.
- **Persistent Chat History**: Stores user conversations, agent routing decisions, and visual data requests in a MySQL database.
- **Dynamic Visualizations**: The AI automatically detects when tabular data should be represented visually and returns structured JSON for rendering charts (tables, pie charts, bar charts).

## Prerequisites

- Python 3.8+
- MySQL Server

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

- `GET /api/schedule?year={year}`: Returns the schedule for a given year.
- `GET /api/standings?year={year}`: Returns driver standings.
- `GET /api/telemetry?year={year}&event={event}&session={session}&drivers={drivers}&lap={lap}`: Returns raw telemetry for charting.
- `GET /api/drivers?year={year}&event={event}&session={session}`: Lists all drivers in a session.
- `GET /api/laps?year={year}&event={event}&session={session}`: Returns the total number of laps in a session.
- `POST /api/chat`: The main endpoint for natural language queries.

## Architecture

The AI chatbot uses a **Supervisor Agent** pattern:
1. The user sends a query to `/api/chat`.
2. The `Supervisor` node evaluates the query, extracts entities (Year, Event, Session), and determines which specialized worker needs to handle it.
3. The query is routed to the appropriate Pandas Dataframe Agent (Schedule, Results, Laps, or Telemetry).
4. The worker dynamically writes and executes pandas code against the FastF1 data to find the answer.
5. The final insights are synthesized by an LLM and returned to the user, complete with data for frontend visualizations if applicable.

Coming releases will add more feature and functionalities to the chatbot.
