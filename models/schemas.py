from typing import Optional, List, Dict, Any
from pydantic import BaseModel

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

class SupervisorOutput(BaseModel):
    next_node: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
