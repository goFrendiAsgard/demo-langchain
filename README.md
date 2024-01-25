# Langchain
LangChain is a framework for developing applications powered by language models. It enables applications that:

- Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
- Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

# Element of Langchain ReAct Agent

- Language Model (LLM)
- Prompt
- Tool
- Agent (LLM + Prompt + Tool)
- AgentExecutor

# Tinkering with Langchain

We do some experiment in [scratchpad.py](scratchpad.py). Feel free to take a look.

## LLM

### OpenAI

OpenAI is the easiest to setup, you just need an OpenAI API Key. It offers `gpt-3.5` and `gpt-4`

```python
import os

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0,
    streaming=True,
)
```

### Ollama

Ollama helps you install LLM model locally. Most model performs better if you have metal or CUDA (NVIDIA) installed.

```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="mistral",
    temperature=0.9,
)
```

### Bedrock

Bedrock offers variaous LLM model like `titan` or `claude`. As per this tutorial, it is currently only available at us-east-1 region.

```python
import boto3
import os

from langchain_community.llms import Bedrock

bedrock_runtime = boto3.client(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    service_name="bedrock-runtime",
    region_name="us-east-1",
)
llm = Bedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-v2",
    streaming=True,
)
```

## Prompt

Prompt is a template used by your LLM. You can pull prompt from langchain hub as follow:

```python
# https://smith.langchain.com/hub/hwchase17/react-chat
from langchain import hub
prompt = hub.pull("hwchase17/react-chat")
```

Or you can declare a custom prompt. Make sure that your prompt is compatible with your agent/chain/LLM

```python
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
```

## Tool

You can create custom tool to be used by your LLM. Langchain provides some common tools like DuckDuckGoSearch.

```python
from langchain.agents import Tool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

tool = Tool(
    name="Search",
    func=DuckDuckGoSearchAPIWrapper().run,
    description="Search engine to answer questions about current events",
)
```

Notice that you can turn any function into a tool.


```python
from langchain.agents import Tool

def word_count(sentence: str) -> int:
    return len(sentence.split(" "))

tool = Tool(
    name="WordCount",
    func=word_count,
    description="Count how many word in a text",
)
```

## Create and Invoke Agent

```python
from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
)

result = agent_executor.invoke(
    {
        # "input": "Who am I?",
        "input": "How many people live in Canada right now?",
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)

print(result["output"])
```

# Creating Virtual Environment

Whenever you work with Python, it is better to use virtual environments.

Virtual environments help you to isolate Python packages for different projects.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Prepare Environment Variables

Aside from virtual environments, it is also better to set your configuration as environment variables rather than hard-code it.

```bash
if [ ! f ".env" ]
then
    cp template.env .env
fi
```

Set the following variables if you have any:
- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY`
- `AWS_SECRET_ACCESS_KEY`

# Load Environment File

```bash
source .env
```

# Using OpenAI model

Prerequisites

- Make sure you have access to create an OpenAI API Key
- Make sure you have the `OPENAI_API_KEY` variable set

```bash
export LLM_PROVIDER=openai
python agent.py "Hello, I'm Go Frendi, I live in Indonesia"
```

```
    🖳 Thought: Do I need to use a tool? No
    🖳 Final Answer: Hello Go Frendi! How can I assist you today?

Hello Go Frendi! How can I assist you today?
```

```bash
python agent.py "How many people live in my country right now?"
```

```
    🖳 Thought: Do I need to use a tool? Yes
    🖳 Action: Search
    🖳 Action Input: "current population of Indonesia"Do I need to use a tool? No
    🖳 Final Answer: Indonesia has a population of over 270 million people.

Indonesia has a population of over 270 million people.
```


# Using Ollama (Local)

Prerequisites

- Make sure you have [Ollama](https://ollama.ai/) installed on your computer. If you are using Linux/WSL, you can invoke the following command:
    ```bash
    curl https://ollama.ai/install.sh | sh
    ```
- Download your favorite model using the `ollama pull` command.
    ```bash
    ollama pull mistral
    ```
- You can try to chat with the model using the `ollama run` command.
    ```bash
    ollama run mistral
    ```
Demo

```bash
export LLM_PROVIDER=ollama
python agent.py "Hello, I'm Go Frendi, I live in Indonesia"
```

```
    🖳 Thought: Do I need to use a tool? No
    🖳 Final Answer: Hi Go Frendi, nice to meet you! Where in Indonesia are you from?

Hi Go Frendi, nice to meet you! Where in Indonesia are you from?
```

```bash
python agent.py "How many people live in my country right now?"
```

```
    🖳 Thought: Do I need to use a tool? Yes
    🖳 Action: Search
    🖳 Action Input: population of Indonesia The search results indicate that the population of Indonesia is approximately 275.7 million people.
    🖳
    🖳 Final Answer: The population of your country, Indonesia, is estimated to be around 275.7 million people.

The population of your country, Indonesia, is estimated to be around 275.7 million people.
```

# Using Bedrock

Prerequisites

- Ensure you have an IAM user with full access to Bedrock (i.e., `BedrockFullAccess`). Using IAM is a good practice to limit security breaches. See the following documentation:
    - [Create IAM user](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html).
    - [Adding permission](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_change-permissions.html).
    - [Create access key](https://docs.aws.amazon.com/cli/latest/userguide/cli-authentication-user.html)
- Make sure you have saved your credentials somewhere. Don't share the credentials with anyone.
    ```
    Login: https://<some-random-numbers>.signin.aws.amazon.com/console
    IAM user: gofrendi_demo
    Password: <some-random-characters>
    Access Key: AKIAIOSFODNN7EXAMPLE
    Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    ```
- Set up the following variable:
    - `AWS_ACCESS_KEY`
    - `AWS_SECRET_ACCESS_KEY`
- CLI Access (Optional).
    - You can install AWS CLI at your computer. See the details [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
    - Perform `aws configure`. See the details [here](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html#sso-configure-profile-prereqs).

        ```bash
        aws configure --profile default
        AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
        AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
        Default region name [None]: us-east-1
        Default output format [None]: json
        ```
        Once you set the configuration, AWS CLI will store your credentials on `~/.aws/credentials`.
- Set up model access by accessing [model access page](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess). The models will be available by request.


Demo

```bash
export LLM_PROVIDER=bedrock
python agent.py "Hello, I'm Go Frendi, I live in Indonesia"
```

```
    🖳 Thought: Do I need to use a tool? No
    🖳
    🖳 Final Answer: Hello! It's nice to meet you Go Frendi. Living in Indonesia must be really interesting. I don't know much about life there but I'd love to learn more about your home country sometime.

Hello! It's nice to meet you Go Frendi. Living in Indonesia must be really interesting. I don't know much about life there but I'd love to learn more about your home country sometime.
```

```bash
python agent.py "How many people live in my country right now?"
```

```
    🖳 Thought: Do I need to use a tool to find out how many people currently live in Indonesia? Yes
    🖳
    🖳 Action: Search
    🖳
    🖳 Action Input: current population of indonesia
    🖳  No, I do not need to use any other tools to answer this question.
    🖳
    🖳 Final Answer: Based on my search, the current population of Indonesia is around 274-275 million people. Indonesia is the fourth most populous country in the world, with a large and relatively young population. The majority of Indonesians are Muslim, making it the country with the largest Muslim population globally. The population is projected to continue growing, potentially reaching 320 million by 2045 according to some estimates.

Based on my search, the current population of Indonesia is around 274-275 million people. Indonesia is the fourth most populous country in the world, with a large and relatively young population. The majority of Indonesians are Muslim, making it the country with the largest Muslim population globally. The population is projected to continue growing, potentially reaching 320 million by 2045 according to some estimates.
```

# Further Reading

- [More tutorials about Bedrock](https://medium.com/@dminhk/amazon-bedrock-workshop-getting-started-ffcf77982857)
- [LLM Intuition](https://docs.google.com/presentation/d/1suUTHmvPULCWvAI3bKv2JF92C8zaEc2QXf1qhUxILmY). Highly recommended. No Math required.
- [The Ilustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Recommended if you know a bit about Neural Network.
- [LLM Visualized](https://bbycroft.net/llm). Interactive illustration. Quite fun.
- [Ollama](https://ollama.ai/). Run LLM locally, well documented. Start here for simple use cases.
- [Langchain](https://www.langchain.com/). The LLM Framework, has massive integrations. Currently under development. Can be intimidating for beginners.
- [Agent with Bedrock](https://github.com/build-on-aws/amazon-bedrock-custom-langchain-agent)
