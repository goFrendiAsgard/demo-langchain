import boto3
import sys
from typing import Any

from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Bedrock
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class StreamingStdErrCallbackHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        sys.stderr.write(token)
        sys.stderr.flush()


tools = [
    Tool(
        name="Search",
        func=DuckDuckGoSearchAPIWrapper().run,
        description="Search engine to answer questions about current events",
    )
]


# https://smith.langchain.com/hub/hwchase17/react-chat
# from langchain import hub
# prompt = hub.pull("hwchase17/react-chat")

prompt = PromptTemplate.from_template(
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
            "When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:",
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

llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    callback_manager=CallbackManager([StreamingStdErrCallbackHandler()]),
)

# bedrock_runtime = boto3.client(
#     service_name="bedrock-runtime",
#     region_name="us-east-1",
# )
# model_id = "anthropic.claude-v2"
# model_kwargs = {
#     "max_tokens_to_sample": 4096,
#     "temperature": 0.5,
#     "top_k": 250,
#     "top_p": 1,
#     "stop_sequences": ["\n\nHuman"],
# }
# llm = Bedrock(
#     client=bedrock_runtime,
#     model_id=model_id,
#     model_kwargs=model_kwargs
# )

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
)

result = agent_executor.invoke(
    {
        "input": "Who am I?",
        # "input": "How many people live in Canada right now?",
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)

print('')
print(result["output"])
