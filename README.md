# Element of Langchain

- Language Model (LLM)
- Memory
- Prompt
- Tool
- Agent
- AgentExecutor

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
python agent.py
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

```bash
export LLM_PROVIDER=ollama
python agent.py
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

```bash
export LLM_PROVIDER=openai
python agent.py
```

# Further Reading

- [More tutorials about Bedrock](https://medium.com/@dminhk/amazon-bedrock-workshop-getting-started-ffcf77982857)
- [LLM Intuition](https://docs.google.com/presentation/d/1suUTHmvPULCWvAI3bKv2JF92C8zaEc2QXf1qhUxILmY). Highly recommended. No Math required.
- [The Ilustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Recommended if you know a bit about Neural Network.
- [LLM Visualized](https://bbycroft.net/llm). Interactive illustration. Quite fun.
- [Ollama](https://ollama.ai/). Run LLM locally, well documented. Start here for simple use cases.
- [Langchain](https://www.langchain.com/). The LLM Framework, has massive integrations. Currently under development. Can be intimidating for beginners.
- [Agent with Bedrock](https://github.com/build-on-aws/amazon-bedrock-custom-langchain-agent)
