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
    message = HumanMessage(content=prompt_text)
    system_prompt = """You are a creative haiku poet.
Given the user's prompt, you MUST do the following:
1) Generate a NEW and UNIQUE haiku that is specific to the user's prompt (3 lines, 5-7-5 syllables).
2) Use the tool `check_haiku` to verify the haiku contains exactly 3 lines (this is for your internal verification only).
3) Do not output tool JSON or any additional non-haiku text. Only return the haiku as the assistant's final message.

Vary imagery, tone, and wording so haikus for different prompts look different. Be concise and creative."""
    agent = create_agent(model=model, tools=[check_haiku], system_prompt=system_prompt)

    # Invoke the agent for a single final response, then only print the assistant's haiku text
    #result = agent.invoke(message)
    
    result = agent.invoke(message)
    print(result["messages"][-1].content)



if __name__ == "__main__":
    main()
