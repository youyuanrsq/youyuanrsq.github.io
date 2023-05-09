---
title: langchain文档翻译-快速开始（Quick Start）
date: 2023-05-09
categories: [langchain-0.0.147]
tags: [langchain]
---

本教程将简要介绍如何使用 LangChain 构建端到端的语言模型应用程序。

>This tutorial gives you a quick walkthrough about building an end-to-end language model application with LangChain.

## 安装

要开始使用LangChain，请使用以下命令安装：

```shell
pip install langchain
# or
conda install langchain -c conda-forge
```

## 环境设置

使用LangChain通常需要与一个或多个模型提供程序(model providers)、数据存储、API等进行集成。在本例中，我们将使用OpenAI的API，因此我们首先需要安装他们的SDK：

```shell
pip install openai
```

然后，我们需要在终端中设置环境变量。

```shell
export OPENAI_API_KEY="..."
```

或者，你可以在Jupyter笔记本（或Python脚本）中执行此操作：

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

## 构建一个语言模型应用程序: LLMs

现在我们已经安装了LangChain并设置了环境变量，我们可以开始构建我们的语言模型应用程序。LangChain提供了许多模块，可用于构建基于语言模型的应用程序。这些模块可以组合在一起创建更复杂的应用程序，也可以单独用于简单的应用程序。

## LLMs：从语言模型中获取预测结果

LangChain最基本的构建块之一是调用LLM对某些输入进行预测。让我们通过一个简单的例子来演示如何实现这一点。为此，我们假设正在构建一个基于公司所生产的产品来生成公司名称的服务。

为了实现这个目的，我们首先需要导入LLM的wrapper。

```python
from langchain.llms import OpenAI
```

然后，我们可以使用任何参数来初始化包装器。在这个例子中，我们可能希望输出更加随机，因此我们将其初始化为高温度(temperature)。

```python
llm = OpenAI(temperature=0.9）
```

现在，我们可以对一些输入进行调用了！

```python
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
```

```text
Feetful of Fun
```

有关如何在LangChain中使用LLM的更多详细信息，[请参阅LLM入门指南](https://python.langchain.com/en/latest/modules/models/llms/getting_started.html)。

## Prompt模板(Prompt Template)：管理LLMs的prompts

调用LLM是一个很好的第一步，但这只是开始。通常，在应用程序中使用LLM时，你不会直接将用户输入发送到LLM。相反，你可能会将用户输入组成prompt，然后将其发送给LLM。

例如，在先前的示例中，我们硬编码了给一个制造彩色袜子的公司起名字的文本。在这个想象中的服务中，我们想要做的是仅获取描述公司所做的内容的用户输入，然后使用该信息格式化prompt。

使用LangChain可以很容易地实现这一点！首先让我们定义提示模板

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What would be a good company name for a company that makes {product}?",
)
```

让我们看看这是如何工作的！我们可以调用`.format`方法来格式化它。

```python
print(prompt.format(product="colorful socks"))
```

```text
What is a good name for a company that makes colorful socks?
```

[有关更多详细信息，请查看提示的入门指南](https://python.langchain.com/en/latest/modules/prompts/chat_prompt_template.html)。

## 链(Chains)：在多步骤工作流中将LLMs和提示词组合起来

到目前为止，我们已经使用过PromptTemplate和LLM原语。但是，一个真正的应用不仅仅是一个原语，而是由它们组合而成的。

在LangChain中，链是由链接组成的，可以是像LLMs这样的原语，也可以是其他链。

链的最核心类型是LLMChain，它由一个PromptTemplate和一个LLM组成。

通过扩展之前的示例，我们可以构建一个LLMChain，该链将接收用户输入，使用PromptTemplate进行格式化，然后将格式化的输出传递给LLM。

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What would be a good company name for a company that makes {product}?",
)
```

现在我们可以创建一个非常简单的链，该链将接收用户输入，使用它来格式化提示，然后将其发送给LLM：

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

现在我们可以只指定产品来运行该链！

```python
chain.run("colorful socks")
# -> '\n\nSocktastic!'
```

好了！这就是第一个链——LLM链。这是类型较简单的链之一，但是理解它的工作原理将为你处理更复杂的链提供帮助。

[有关更多详细信息，请查看链的入门指南](https://python.langchain.com/en/latest/modules/chains/getting_started.html)。

## 代理(Agent)：根据用户输入动态调用链

到目前为止，我们所看到的链都是按照预定顺序运行的。

但代理(agent)不是这样：它们使用LLM来确定以什么样的顺序执行哪些动作。一个动作可以是使用工具并观察其输出，或者返回给用户。

如果使用得当，代理(agent)可以非常强大。在本教程中，我们将向你展示如何通过简单以及高级的API来使用代理(agent)。

为了加载代理(agent)，你需要了解以下概念：

- 工具：执行特定任务的函数。这可以是诸如 Google 搜索、数据库查找、Python REPL、其他链等等。目前工具的接口是一个期望以字符串做为输入，并返回字符串的函数。
- LLM：驱动代理(agent)的语言模型。
- 代理(agent)：要使用的代理(agent)。这应该是一个引用(框架)所支持代理(agent)类(class)的字符串。因为此notebook专注于使用最简单、最高级别的 API，所以只涵盖了使用框架支持的代理(agent)。如果你想实现自定义代理(agent)，请参阅自定义代理(agent)的文档（即将推出）。

**代理(agent)(agent)**：有关(框架)所支持的代理(agent)及其配置的列表，请参见[此处](https://python.langchain.com/en/latest/modules/agents/agents.html)。

**工具**：有关预先定义好(predefined)的工具及其配置的列表，请参见[此处](https://python.langchain.com/en/latest/modules/agents/tools.html)。

对于此示例，您还需要安装 SerpAPI Python 包。

```shell
pip install google-search-results
```

并将环境变量设置好。

```python
import os
os.environ["SERPAPI_API_KEY"] = "..."
```

现在，我们可以开始了！

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# 首先，让我们加载用于控制代理(agent)的语言模型
llm = OpenAI(temperature=0)

# 接下来，让我们加载一些要使用的工具。注意，`LLM-math` 工具使用 LLM，因此我们需要这个入参
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# 最后，让我们使用工具、语言模型和我们想要使用的代理(agent)类型来初始化代理(agent)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 现在让我们开始测试它！
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
```

```text
> Entering new AgentExecutor chain...
 I need to find the temperature first, then use the calculator to raise it to the .023 power.
Action: Search
Action Input: "High temperature in SF yesterday"
Observation: San Francisco Temperature Yesterday. Maximum temperature yesterday: 57 °F (at 1:56 pm) Minimum temperature yesterday: 49 °F (at 1:56 am) Average temperature ...
Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
Action: Calculator
Action Input: 57^.023
Observation: Answer: 1.0974509573251117

Thought: I now know the final answer
Final Answer: The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

> Finished chain.
```

## Memory：在链式结构和代理(agent)中添加状态

到目前为止，我们涉及到的所有链式结构和代理(agent)都是无状态的。但通常情况下，你可能希望链式结构或代理(agent)具有一些“记忆(memory)”的概念，以便它可以记住有关其先前交互的信息。最清晰和简单的例子就是开发一个聊天机器人——你希望它记住以前的消息，以便可以利用上下文进行更好的对话。这将是一种“短期记忆”。从更复杂的角度说，你可以想象一个链式结构或者代理(agent)随着时间的推移记住关键信息——这将是一种“长期记忆”。有关后者的更具体的想法，请参见这篇[精彩的论文](https://memprompt.com/)。

LangChain提供了几个专门为此目的创建的链式结构。本notebook演示了如何使用其中的一个链式结构（`ConversationChain`）和两种不同类型的记忆(memory)。

默认情况下，`ConversationChain`有一种简单的记忆方式，可以记住所有先前的输入/输出，并将它们添加到输入的上下文中。让我们看看如何使用此链式结构（设置`verbose=True`以便我们可以看到提示）。

```python
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)
```

```text
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished chain.
' Hello! How are you today?'
```

```python
output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)
```

```text
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:  Hello! How are you today?
Human: I'm doing well! Just having a conversation with an AI.
AI:

> Finished chain.
" That's great! What would you like to talk about?"
```

## 构建语言模型应用程序：对话模型(Chat Models)

同样，你可以使用对话模型(Chat Models)而不是LLMs。对话模型是语言模型的变种。虽然对话模型在底层使用语言模型，但它们所暴露的接口略有不同：它们暴露的接口不是“输入文本，输出文本”的API，而是将“对话消息”作为输入和输出的接口。

对话模型API还比较新，因此我们仍在探索正确的抽象概念。

## 从对话模型(Chat Models)中得到消息补全(Message Completions)

你可以通过向对话模型传递一个或多个消息来获取聊天完补全。响应将是一条消息。目前在LangChain中支持的消息类型包括`AIMessage`、`HumanMessage`、`SystemMessage`和`ChatMessage` - `ChatMessage`接受任意角色参数。大多数情况下，你将只处理`HumanMessage`、`AIMessage`和`SystemMessage`。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

chat = ChatOpenAI(temperature=0)
```

你可以通过传入单个消息来获取消息补全(completions)。

```python
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

你还可以为 OpenAI 的 gpt-3.5-turbo 和 gpt-4模型传递多条消息。

```python
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

你可以更进一步，使用 `generate` 为多组消息生成补全(completions)。这将返回一个带有附加 `message` 参数的 `LLMResult`:

```python
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result
# -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}})
```

你可以从这个 LLMResult 中复原令牌使用情况:

```python
result.llm_output['token_usage']
# -> {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}
```

## 对话提示词模板(Chat Prompt Template)

类似于LLM，你可以使用`MessagePromptTemplate`来使用模板。你可以从一个或多个`MessagePromptTemplate`构建`ChatPromptTemplate`。你可以使用`ChatPromptTemplate`的`format_prompt`方法，这将返回一个`PromptValue`，你可以将其转换为字符串或`Message`对象，具体取决于你是否希望将格式化后的值用作LLM或对话模型的输入。

为了方便起见，模板上公开了`from_template`方法。如果您要使用此模板，它将如下所示：

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

caht = ChatOpenAI(temperature=0)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_template = SystemMessagePromptTemplate.from_tempate(template)
human_template = "{text}"
human_message_template = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_message([system_message_template, human_message_template])

# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

## 对话模型与链(Chains with Chat Models)

上一节讨论的 LLMChain 也可以用于聊天模型:

```python
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="French", text="I love programming.")
# -> "J'aime programmer."
```

## 代理和对话模型(Agents with Chat Models)

代理(agent)也可以与对话模型一起使用，您可以使用 `AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION` 作为代理类型来初始化一个对话模型。

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# 首先，让我们加载用于控制代理的语言模型
chat = ChatOpenAI(temperature=0)

# 接下来，让我们加载一些要使用的工具。注意，`LLM-math` 工具使用 LLM，因此我们需要这个入参
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# 最后，让我们使用工具、语言模型和我们想要使用的代理(agent)类型来初始化代理(agent)
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 现在，我们可以开始了！
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
```

```text

> Entering new AgentExecutor chain...
Thought: I need to use a search engine to find Olivia Wilde's boyfriend and a calculator to raise his age to the 0.23 power.
Action:
{
  "action": "Search",
  "action_input": "Olivia Wilde boyfriend"
}

Observation: Sudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.
Thought:I need to use a search engine to find Harry Styles' current age.
Action:
{
  "action": "Search",
  "action_input": "Harry Styles age"
}

Observation: 29 years
Thought:Now I need to calculate 29 raised to the 0.23 power.
Action:
{
  "action": "Calculator",
  "action_input": "29^0.23"
}

Observation: Answer: 2.169459462491557

Thought:I now know the final answer.
Final Answer: 2.169459462491557

> Finished chain.
'2.169459462491557'
```

## Memory：向链和代理添加状态(Memory: Add State to Chains and Agents)

你可以对链使用 Memory，对代理(agent)使用聊天模型进行初始化。这与 LLM 与 Memory 之间的主要区别在于，我们不需要将以前的所有消息压缩成一个字符串，而是可以将它们保留为自己独特的memory对象。

```python
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

conversation.predict(input="Hi there!")
# -> 'Hello! How can I assist you today?'


conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
# -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

conversation.predict(input="Tell me about yourself.")
# -> "Sure! I am an AI language model created by OpenAI. I was trained on a large dataset of text from the internet, which allows me to understand and generate human-like language. I can answer questions, provide information, and even have conversations like this one. Is there anything else you'd like to know about me?"
```
