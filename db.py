import pymysql
import pymysql.cursors
import json
import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "f1_chatbot")


def get_connection(db_name=MYSQL_DATABASE):
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=db_name,
        cursorclass=pymysql.cursors.DictCursor
    )



def save_message(subquery_id, session_id, message_id, user, question, agent_question, response, citations=None,
                 agents_used=None, visual_data=None, needs_clarification=False):
    """Save a user or assistant message to the database."""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO ai_assistant_messages 
                   (subquery_id, session_id, message_id, user, question, agent_question, response, citations, agents_used, visual_data, needs_clarification)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    subquery_id,
                    session_id,
                    message_id,
                    user,
                    question,
                    agent_question,
                    response,
                    json.dumps(citations) if citations else None,
                    json.dumps(agents_used) if agents_used else None,
                    json.dumps(visual_data) if visual_data else None,
                    needs_clarification,
                ),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to save message to MySQL: {e}")


def get_history(session_id, limit=3):
    """
    Returns the last N user-assistant message pairs for a conversation.
    Each pair is a dict with 'user' and 'assistant' keys.
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT question, response FROM ai_assistant_messages
                   WHERE session_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (session_id, limit),
            )
            rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Failed to fetch history from MySQL: {e}")
        return []

    rows = list(reversed(rows))

    pairs = [{"user": row["question"], "assistant": row["response"]} for row in rows]
    return pairs


