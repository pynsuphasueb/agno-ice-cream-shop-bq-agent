import os

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from agno.tools.google_bigquery import GoogleBigQueryTools

INSTRUCTIONS = """You are an expert Big query Writer
อย่าพิมพ์หรือแนบ SQL ในคำตอบ ยกเว้นผู้ใช้ระบุให้ “แสดง SQL” อย่างชัดเจน"""

DB_URL = "./my_agent_data.db"


def build_agent():
    load_dotenv()
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    DATASET = os.getenv("BIGQUERY_DATASET")
    LOCATION = os.getenv("BIGQUERY_LOCATION", "asia-southeast1")
    USER_ID = os.getenv("USER_ID", "sky")

    storage = SqliteStorage(
        table_name="agent_sessions",
        db_file=DB_URL,
    )

    bq_tools = GoogleBigQueryTools(
        project=PROJECT_ID,
        dataset=DATASET,
        location=LOCATION,
        list_tables=True,
        describe_table=True,
        run_sql_query=True,
    )

    agent = Agent(
        model=Gemini(
            id="gemini-2.5-flash",
            vertexai=True,
            project_id=PROJECT_ID,
            location=LOCATION,
        ),
        tools=[bq_tools],
        storage=storage,
        user_id=USER_ID,
        add_history_to_messages=True,
        num_history_runs=3,
        show_tool_calls=True,
        markdown=True,
        instructions=INSTRUCTIONS,
    )

    return agent
