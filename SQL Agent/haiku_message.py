from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()


@tool
def check_haiku(text: str):
    """
    Check if a given haiku has only 3 lines.
    Returns None if it's correct, otherwise an error message.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 3:
        return f"Incorrect! This haiku has {len(lines)} lines. It should have 3 lines."
    return "Correct, this has 3 lines!"


def main():
    # Configure model with some randomness (temperature) to avoid identical outputs
    model = init_chat_model("ollama:phi4-mini", temperature=0.8)

    prompt_text = input("Enter your haiku prompt: ")
    message = HumanMessage(prompt_text)
    sys_prompt = """
    You are a creative haiku writer. You will write haiku based on user input.
    Your haiku must have exactly 3 lines. Use the [check_haiku] tool to verify the haiku's line count.
    If the haiku does not have 3 lines, rewrite it until it does.
    Don't add the tool's output with the model output.
    Only output the haiku.
    """
    agent = create_agent(model=model, tools=[check_haiku], system_prompt=sys_prompt)
    
    result = agent.invoke({'messages': message})
    print(result["messages"][-1].content)

    # for m in agent.stream(
    #     {'messages': message},
    #     stream_mode="values",
    # ):
    #     m["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
