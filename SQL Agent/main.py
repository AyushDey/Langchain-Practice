from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

_ = load_dotenv()

# Connect to the Chinook DB
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.table_info)


@dataclass
class RuntimeContext:
    db: SQLDatabase


@tool
def query_tool(query: str) -> str:
    """Execute SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


SYS_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- Follow the instructions exactly.
- The name of the database is 'Chinook.db'.
- When you need data, directly call the tool `query_tool` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""
agent = create_agent(
    model="groq:moonshotai/kimi-k2-instruct-0905",
    tools=[query_tool],
    system_prompt=SYS_PROMPT,
    context_schema=RuntimeContext,
)

question = "Which table has the largest number of entries and how many?"

for step in agent.stream(
    {"messages": question},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
