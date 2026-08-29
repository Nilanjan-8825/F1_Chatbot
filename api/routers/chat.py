import asyncio
import json
import re
import uuid
from fastapi import APIRouter, HTTPException, Request
from models.schemas import ChatRequest, ChatResponse, VisualData
from services import chat_service
import db

router = APIRouter()

@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, body: ChatRequest):
    session_id = body.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    subquery_id = str(uuid.uuid4())
    agent_question = body.question     

    history = db.get_history(session_id, limit=3)
    history_context = [
        f"[Chat History]: User: {pair['user']} | Assistant: {pair['assistant']}"
        for pair in history
    ]

    initial_state = {
        "user_query": body.question,
        "context": history_context,
        "next_node": "Supervisor",
        "needs_clarification": False,
        "clarification_question": None
    }

    loop = asyncio.get_running_loop()
    
    task = loop.run_in_executor(
        None, 
        lambda: chat_service.f1_chatbot.invoke(initial_state, {"recursion_limit": 10})
    )

    while not task.done():
        if await request.is_disconnected():
            task.cancel()
            raise HTTPException(status_code=499, detail="Client Closed Request")
        await asyncio.sleep(0.5)

    try:
        result = await task
    except asyncio.CancelledError:
         raise HTTPException(status_code=499, detail="Client Closed Request")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    raw_context = "\n".join(result.get("context", []))

    agents_used = [
        ctx.split("]")[0].replace("[", "").strip()
        for ctx in result.get("context", [])
        if "]: " in ctx and "Chat History" not in ctx
    ]

    if result.get("needs_clarification"):
        clarification_q = result.get("clarification_question", "Could you please provide more details?")
        db.save_message(
            subquery_id=subquery_id,
            session_id=session_id,
            message_id=message_id,
            user=body.user,
            question=body.question,
            agent_question=agent_question,
            response=clarification_q,
            needs_clarification=True
        )
        return ChatResponse(
            session_id=session_id,
            text=clarification_q,
            citations=[],
            needs_clarification=True,
            clarification_question=clarification_q
        )
    
    system_prompt = """
You are a helpful F1 Chatbot Assistant.
You have been provided with raw data and context gathered by specialized agents.
Your task is to answer the user's query based ONLY on this context.

1. Provide a concise, natural-language summary of the answer.
2. Include citations like [1] referring to 'FastF1 API'.
3. IMPORTANT: If the context contains ANY structured data (lap times, positions, standings, driver stats, tire strategies, sector times, etc.), you MUST also output a JSON block wrapped in ```json with this exact structure:
{
  "type": "table" | "pie_chart" | "bar_chart",
  "title": "Title of the visual",
  "data": [{"column1": "value1", "column2": "value2", ...}]
}
For example, if asked about fastest laps, output a table with columns like Driver, LapTime, Compound, etc.
Always prefer "table" type unless the data is clearly better as a chart (e.g., percentage breakdowns → pie_chart, comparisons → bar_chart).
If the data is purely descriptive with no tabular structure, do NOT output the JSON block.
"""
    prompt = f"User Query: {body.question}\n\nContext:\n{raw_context}"
    
    try:
        llm_response = await loop.run_in_executor(
            None,
            lambda: chat_service.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
        )
        raw_content = llm_response.content
        if isinstance(raw_content, list):
            final_text = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            final_text = raw_content
    except Exception as e:
        final_text = f"An error occurred while formatting the answer: {str(e)}\n\nRaw Context: {raw_context}"
        
    visual_data = None
    json_match = re.search(r'```json\n(.*?)\n```', final_text, re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(1))
            visual_data = VisualData(**parsed_json)
            final_text = final_text.replace(json_match.group(0), "").strip()
        except Exception:
            pass

    db.save_message(
        subquery_id=subquery_id,
        session_id=session_id,
        message_id=message_id,
        user=body.user,
        question=body.question,
        agent_question=agent_question,
        response=final_text,
        citations=["FastF1 API"],
        agents_used=agents_used,
        visual_data=visual_data.model_dump() if visual_data else None
    )

    return ChatResponse(
        session_id=session_id,
        text=final_text,
        citations=["FastF1 API"],
        visual_data=visual_data
    )
