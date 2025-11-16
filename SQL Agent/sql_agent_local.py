from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.messages import HumanMessage
import re
load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

def _safe_sql(query: str) -> str:
    deny_re = re.compile(
        r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b", re.I
    )
    HAS_LIMIT_TAIL_RE = re.compile(r"(?is)\blimit\b\s+\d+(\s*,\s*\d+)?\s*;?\s*$")

    query = query.strip()
    if query.count(";") > 1 or (query.endswith(";") and ";" in query[:-1]):
        return "Error: Multiple statements are not allowed."
    query = query.rstrip(";").strip()

    if not query.lower().startswith("select"):
        return "Error: Only SELECT queries are allowed."
    if deny_re.search(query):
        return "Error: Query contains forbidden operations."
    
    # append LIMIT only if not already present at the end (robust to whitespace/newlines)
    if not HAS_LIMIT_TAIL_RE.search(query):
        query += " LIMIT 5"
    return query


@tool
def execute_query(query: str) -> str:
    """Execute a READ-ONLY SQLite SELECT query and return results."""
    query = _safe_sql(query)
    if query.startswith("Error: "):
        return query
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"
    

def main():
    model = init_chat_model("groq:moonshotai/kimi-k2-instruct-0905")
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    schema = db.get_table_info()

    sys_prompt = f"""You're a careful SQLite analyst.
    Authoritative schema (do not invent columns / tables):
    {schema}

    Rules:
    - Think step-by-step.
    - When you need data, call the tool [execute_query] with one SELECT query.
    - Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
    - Limit to 5 rows of output unless the user explicitly asks otherwise.
    - If the tool returns 'Error:', revise the SQL and try again.
    - Limit number of attempts to 5.
    - If you are not successful after 5 attempts, return a note to the user.
    - Prefer explicit column lists; avoid SELECT *.
    """
    agent = create_agent(
        model=model,
        tools=[execute_query],
        system_prompt=sys_prompt,
    )

    # question = HumanMessage("Which table has the biggest amount of entries")

    # for m in agent.stream(
    #     question,
    #     stream_mode="values",
    # ):
    #     m["messages"][-1].pretty_print()
    question = HumanMessage("How many tables are there?")
    for step in agent.stream({"messages": question}, stream_mode="values"):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()