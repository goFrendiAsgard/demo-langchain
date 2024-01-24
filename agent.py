import boto3
import os
import sys
from typing import Any, Sequence

from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Bedrock
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class StreamingStdErrCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self._is_first_token = True

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        shown_text = token.replace("\n", "\n    🖳 ")
        if self._is_first_token:
            shown_text = "    🖳" + shown_text
        sys.stderr.write(shown_text)
        sys.stderr.flush()
        self._is_first_token = False


def get_llm(llm_provider: str) -> BaseLanguageModel:
    if llm_provider == "bedrock":
        bedrock_runtime = boto3.client(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )
        return Bedrock(
            client=bedrock_runtime,
            model_id="anthropic.claude-v2",
            streaming=True,
            callback_manager=CallbackManager([StreamingStdErrCallbackHandler()]),
        )
    if llm_provider == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            streaming=True,
            callback_manager=CallbackManager([StreamingStdErrCallbackHandler()]),
        )
    return ChatOllama(
        model="mistral",
        callback_manager=CallbackManager([StreamingStdErrCallbackHandler()]),
        temperature=0.9,
    )


def get_tools() -> Sequence[BaseTool]:
    return [
        Tool(
            name="Search",
            func=DuckDuckGoSearchAPIWrapper().run,
            description="Search engine to answer questions about current events",
        )
    ]


def get_prompt() -> BasePromptTemplate:
    # ReAct prompt
    return PromptTemplate.from_template(
        "\n".join(
            [
                "You are a helpful assistant.",
                "You have access to the following tools:",
                "{tools}",
                "To use a tool, please use the following format:",
                "```",
                "Thought: Do I need to use a tool? Yes",
                "Action: the action to take, should be one of [{tool_names}]",
                "Action Input: the input to the action",
                "Observation: the result of the action",
                "```",
                "When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:",  # noqa
                "```",
                "Thought: Do I need to use a tool? No",
                "Final Answer: [your response here]",
                "```",
                "Begin!",
                "Previous conversation history:",
                "{chat_history}",
                "New input: {input}",
                "{agent_scratchpad}",
            ]
        )
    )


def get_chat_history() -> str:
    if os.path.isfile("history.txt"):
        with open("history.txt", "r") as history_file:
            return history_file.read()
    return ""


def save_chat_history(human_message: str, ai_message: str):
    with open("history.txt", "a") as history_file:
        history_file.write(f"Human: {human_message}\n")
        history_file.write(f"Assistant: {ai_message}\n")


if __name__ == "__main__":
    llm = get_llm(os.getenv("LLM_PROVIDER"))
    tools = get_tools()
    prompt = get_prompt()
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
    )
    chat_history = get_chat_history()
    human_message = " ".join(sys.argv[1:])
    result = agent_executor.invoke(
        {
            "input": human_message,
            "chat_history": chat_history,
        }
    )
    ai_message = result["output"]
    print("")
    print(ai_message)
    save_chat_history(human_message, ai_message)
