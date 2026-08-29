from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import core, telemetry, strategy, replay, chat
import db
import os

app = FastAPI(title="F1 Race Dashboard API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
def startup_event():
    if not os.path.exists('data'):
        os.makedirs('data')

# Include routers
app.include_router(core.router)
app.include_router(telemetry.router)
app.include_router(strategy.router)
app.include_router(replay.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
