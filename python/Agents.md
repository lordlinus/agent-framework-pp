# Microsoft Agent Framework - Developer Guide

This guide provides comprehensive instructions for developers to get started with the Microsoft Agent Framework for Python. Whether you're new to AI agents or looking to build sophisticated multi-agent systems, this guide will help you understand the framework and build your first agents.

## Table of Contents
- [Dev Environment Tips](#dev-environment-tips)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Installation](#installation)
- [Setting Up Your Environment](#setting-up-your-environment)
- [Creating Your First Agent](#creating-your-first-agent)
- [Working with Chat Clients](#working-with-chat-clients)
- [Using Tools and Functions](#using-tools-and-functions)
- [Building Workflows](#building-workflows)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
- [Middleware and Context Providers](#middleware-and-context-providers)
- [DevUI for Testing and Debugging](#devui-for-testing-and-debugging)
- [Common Examples and Scenarios](#common-examples-and-scenarios)
- [Troubleshooting](#troubleshooting)

## Dev Environment Tips

### Quick Navigation
- Use `uv run poe` to discover available development tasks for the project
- Navigate to the `python` folder as your workspace root for development
- Check the `samples/getting_started` directory for working examples of agents, workflows, and integrations
- Use `view` command to inspect file contents: `uv run python -c "from pathlib import Path; print(Path('path/to/file').read_text())"`

### Development Tools
- **Virtual Environment**: Run `uv venv --python 3.10` (or 3.11, 3.12, 3.13) to create a virtual environment
- **Install Dependencies**: Run `uv sync --dev` to install all dependencies including development tools
- **Setup Everything**: Run `uv run poe setup -p 3.13` to setup venv, install packages, and pre-commit hooks
- **Run Tests**: Use `uv run poe test` to run the test suite
- **Code Quality**: Use `uv run poe lint` to run linters and formatters
- **Documentation**: Use `uv run poe docs-serve` to build and serve documentation locally

### VSCode Setup
1. Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
2. Open the `python` folder as your workspace (not the root folder)
3. Run `Python: Select Interpreter` from command palette and select the `.venv` virtual environment
4. Create a `.env` file in the `python` directory with your API keys (see [Setting Up Your Environment](#setting-up-your-environment))

## Quick Start

### Installation (2 Options)

**Option 1: Full Installation (Recommended for getting started)**
```bash
pip install agent-framework --pre
```
This installs the core framework and all integration packages.

**Option 2: Selective Installation (For lightweight environments)**
```bash
# Core only (includes Azure OpenAI and OpenAI support)
pip install agent-framework-core --pre

# Core + specific integrations
pip install agent-framework-azure-ai --pre
pip install agent-framework-copilotstudio --pre
```

### Your First Agent (30 seconds)
```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant."
    )
    
    result = await agent.run("What is the capital of France?")
    print(result)

asyncio.run(main())
```

## Core Concepts

The Agent Framework is built around several key concepts:

### 1. **Chat Clients**
Chat clients are the interface to LLM providers (OpenAI, Azure OpenAI, Azure AI, etc.). They handle communication with the AI models.

**Available Chat Clients:**
- `OpenAIChatClient` - OpenAI models
- `OpenAIResponsesClient` - OpenAI with responses API
- `OpenAIAssistantsClient` - OpenAI Assistants API
- `AzureOpenAIChatClient` - Azure OpenAI models
- `AzureOpenAIResponsesClient` - Azure OpenAI with responses API
- `AzureOpenAIAssistantsClient` - Azure OpenAI Assistants API
- `AzureAIChatClient` - Azure AI Foundry models
- `AnthropicChatClient` - Anthropic models (via OpenAI compatibility)
- `CopilotStudioChatClient` - Microsoft Copilot Studio

### 2. **Agents**
Agents are autonomous entities that can reason, make decisions, and take actions. The framework provides `ChatAgent` as the primary agent type.

**Key Features:**
- **Instructions**: Define agent behavior and personality
- **Tools**: Functions the agent can call to perform actions
- **Memory**: Conversation history and context management
- **Streaming**: Real-time response streaming
- **Middleware**: Intercept and modify agent behavior

### 3. **Tools and Functions**
Tools are Python functions that agents can call to interact with external systems, databases, APIs, or perform computations.

**Tool Types:**
- Regular Python functions with type annotations
- Async functions for I/O operations
- Pydantic models for structured outputs
- MCP (Model Context Protocol) servers

### 4. **Workflows**
Workflows enable orchestration of multiple agents and executors in graph-based patterns.

**Key Features:**
- **Graph-based execution**: Define nodes (executors) and edges (data flow)
- **Streaming**: Real-time event streaming from workflow execution
- **Checkpointing**: Save and resume workflow state
- **Human-in-the-loop**: Pause for human input or approval
- **Parallelism**: Execute multiple nodes concurrently

### 5. **Middleware**
Middleware intercepts and modifies behavior at different execution stages:
- **Agent Middleware**: Intercept agent runs
- **Function Middleware**: Intercept function calls
- **Chat Middleware**: Intercept chat requests to AI models

### 6. **Context Providers**
Context providers dynamically inject information into agent conversations based on the current context.

## Installation

### System Requirements
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OS**: Windows, macOS, or Linux
- **Package Manager**: `pip` or `uv` (recommended for development)

### Installation Options

#### Option 1: Full Installation
Install everything including all integrations:
```bash
pip install agent-framework --pre
```

#### Option 2: Core Only
Install just the core framework:
```bash
pip install agent-framework-core --pre
```

#### Option 3: Selective Integrations
Install specific integration packages:
```bash
# Azure AI Foundry integration
pip install agent-framework-azure-ai --pre

# Microsoft Copilot Studio integration
pip install agent-framework-copilotstudio --pre

# DevUI for testing and debugging
pip install agent-framework-devui --pre

# Redis context provider
pip install agent-framework-redis --pre

# Mem0 context provider
pip install agent-framework-mem0 --pre
```

#### Development Installation
For contributing to the framework:
```bash
# Clone the repository
git clone https://github.com/microsoft/agent-framework.git
cd agent-framework/python

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Setup development environment
uv run poe setup -p 3.12
```

## Setting Up Your Environment

### API Keys and Configuration

The framework uses environment variables for configuration. You can set them directly or use a `.env` file.

#### Option 1: Create a `.env` file
Create a `.env` file in your project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL_ID=gpt-4o-mini

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-05-01-preview

# Azure AI Foundry Configuration
AZURE_AI_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4o-mini

# Copilot Studio Configuration
COPILOT_STUDIO_ENDPOINT=https://your-copilot.microsoft.com/api/v1
```

#### Option 2: Set Environment Variables
```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."
export OPENAI_CHAT_MODEL_ID="gpt-4o-mini"

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_CHAT_MODEL_ID="gpt-4o-mini"
```

#### Option 3: Pass Configuration Directly
```python
from agent_framework.azure import AzureOpenAIChatClient

chat_client = AzureOpenAIChatClient(
    api_key='your-key',
    endpoint='https://your-resource.openai.azure.com/',
    deployment_name='gpt-4o-mini',
    api_version='2024-05-01-preview',
)
```

### Import Structure

The framework uses a flat import structure:

```python
# Core components
from agent_framework import ChatAgent, ChatMessage, ai_function

# Chat clients
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AzureOpenAIChatClient

# Workflows
from agent_framework import (
    WorkflowBuilder,
    Executor,
    WorkflowContext,
    executor,
    handler,
)

# Middleware
from agent_framework import AgentMiddleware, FunctionMiddleware

# Context providers
from agent_framework.redis import RedisContextProvider
from agent_framework.mem0 import Mem0ContextProvider
```

## Creating Your First Agent

### Basic Agent
```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create an agent
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="AssistantBot",
        instructions="You are a helpful assistant that provides concise answers."
    )
    
    # Run the agent
    result = await agent.run("What is Python?")
    print(result)

asyncio.run(main())
```

### Agent with Streaming
```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="StreamingBot",
        instructions="You are a helpful assistant."
    )
    
    # Stream responses
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream("Tell me a short story about AI."):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()

asyncio.run(main())
```

### Agent with Instructions
```python
async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="ThreeLawsBot",
        instructions="""
        1) A robot may not injure a human being...
        2) A robot must obey orders given it by human beings...
        3) A robot must protect its own existence...
        
        Give me the TLDR in exactly 5 words.
        """
    )
    
    result = await agent.run("Summarize the Three Laws of Robotics")
    print(result)
    # Output: Protect humans, obey, self-preserve, prioritized.

asyncio.run(main())
```

## Working with Chat Clients

You can use chat clients directly without wrapping them in an agent:

### Direct Chat Client Usage
```python
import asyncio
from agent_framework import ChatMessage
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()
    
    messages = [
        ChatMessage(role="system", text="You are a helpful assistant."),
        ChatMessage(role="user", text="Write a haiku about Agent Framework.")
    ]
    
    response = await client.get_response(messages)
    print(response.messages[0].text)

asyncio.run(main())
```

### Creating Agents from Chat Clients
```python
# Chat clients provide a convenience method to create agents
agent = OpenAIChatClient().create_agent(
    name="WeatherAgent",
    instructions="You are a helpful weather agent.",
    tools=[get_weather]
)
```

### Available Chat Clients

#### OpenAI Chat Client
```python
from agent_framework.openai import OpenAIChatClient

agent = OpenAIChatClient().create_agent(
    name="OpenAIAgent",
    instructions="You are a helpful assistant."
)
```

#### Azure OpenAI Chat Client
```python
from agent_framework.azure import AzureOpenAIChatClient

agent = AzureOpenAIChatClient().create_agent(
    name="AzureAgent",
    instructions="You are a helpful assistant."
)
```

#### Azure AI Foundry Chat Client
```python
from agent_framework.azure_ai import AzureAIChatClient

agent = AzureAIChatClient().create_agent(
    name="AzureAIAgent",
    instructions="You are a helpful assistant."
)
```

## Using Tools and Functions

Tools enable agents to perform actions beyond text generation.

### Basic Function Tool
```python
import asyncio
from random import randint
from typing import Annotated
from pydantic import Field
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")]
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful weather assistant.",
        tools=[get_weather]
    )
    
    response = await agent.run("What's the weather in Seattle?")
    print(response)

asyncio.run(main())
```

### Multiple Tools
```python
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    return f"Weather in {location}: 72°F and sunny"

def get_menu_specials() -> str:
    """Get today's menu specials."""
    return """
    Special Soup: Clam Chowder
    Special Salad: Cobb Salad
    Special Drink: Chai Tea
    """

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that can provide weather and restaurant information.",
        tools=[get_weather, get_menu_specials]
    )
    
    response = await agent.run("What's the weather in Amsterdam and what are today's specials?")
    print(response)

asyncio.run(main())
```

### Async Tools
```python
import httpx

async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        tools=[fetch_data]
    )
    
    result = await agent.run("Fetch the content from https://example.com")
    print(result)

asyncio.run(main())
```

### Structured Output with Pydantic
```python
from pydantic import BaseModel, Field

class WeatherInfo(BaseModel):
    location: str = Field(description="The location")
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition")

def get_weather_structured(location: str) -> WeatherInfo:
    """Get structured weather information."""
    return WeatherInfo(
        location=location,
        temperature=22.5,
        condition="sunny"
    )
```

## Building Workflows

Workflows enable graph-based orchestration of multiple agents and executors.

### Basic Workflow (Executors and Edges)
```python
import asyncio
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
)
from typing_extensions import Never

# Option 1: Class-based Executor
class UpperCase(Executor):
    def __init__(self, id: str):
        super().__init__(id=id)
    
    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        """Convert text to uppercase and send to next node."""
        result = text.upper()
        await ctx.send_message(result)

# Option 2: Function-based Executor
@executor(id="reverse_text_executor")
async def reverse_text(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """Reverse text and yield as workflow output."""
    result = text[::-1]
    await ctx.yield_output(result)

async def main():
    upper_case = UpperCase(id="upper_case_executor")
    
    # Build workflow with edges
    workflow = (
        WorkflowBuilder()
        .add_edge(upper_case, reverse_text)
        .set_start_executor(upper_case)
        .build()
    )
    
    # Run workflow
    events = await workflow.run("hello world")
    print(events.get_outputs())  # ['DLROW OLLEH']
    print("Final state:", events.get_final_state())

asyncio.run(main())
```

### Workflow with Agents
```python
import asyncio
from agent_framework import ChatAgent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create specialized agents
    writer = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="Writer",
        instructions="You are a creative content writer."
    )
    
    reviewer = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="Reviewer",
        instructions="You are a critical reviewer who provides constructive feedback."
    )
    
    # Build workflow
    workflow = (
        WorkflowBuilder()
        .add_edge(writer, reviewer)
        .set_start_executor(writer)
        .build()
    )
    
    # Run workflow
    events = await workflow.run("Write a slogan for an electric car")
    print(events.get_outputs())

asyncio.run(main())
```

### Streaming Workflows
```python
async def main():
    workflow = build_workflow()  # Your workflow
    
    # Stream events during execution
    async for event in workflow.run_stream("input"):
        if hasattr(event, 'text') and event.text:
            print(event.text, end="", flush=True)

asyncio.run(main())
```

### Checkpointing and Resume
```python
from agent_framework import CheckpointStore

async def main():
    checkpoint_store = CheckpointStore()
    workflow = build_workflow()
    
    # Run with checkpointing
    events = await workflow.run("input", checkpoint_store=checkpoint_store)
    
    # Get checkpoint
    checkpoint = checkpoint_store.get_checkpoint()
    
    # Resume from checkpoint
    resumed_events = await workflow.run(checkpoint=checkpoint)

asyncio.run(main())
```

### Human-in-the-Loop
```python
from agent_framework import RequestInfoMessage

async def main():
    workflow = build_workflow_with_hitl()
    
    # Start workflow
    result = await workflow.run("input")
    
    # Check for pending human input requests
    pending_requests = result.get_pending_requests()
    
    # Respond to request
    for request in pending_requests:
        await request.respond("user input")

asyncio.run(main())
```

## Multi-Agent Orchestration

### Sequential Orchestration
```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create specialized agents
    writer = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="Writer",
        instructions="You are a creative content writer."
    )
    
    reviewer = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="Reviewer",
        instructions="You are a critical reviewer."
    )
    
    # Sequential workflow: Writer -> Reviewer -> Writer
    task = "Create a slogan for an electric SUV"
    
    # Step 1: Writer creates initial slogan
    initial_result = await writer.run(task)
    print(f"Writer: {initial_result}")
    
    # Step 2: Reviewer provides feedback
    feedback = await reviewer.run(f"Please review this slogan: {initial_result}")
    print(f"Reviewer: {feedback}")
    
    # Step 3: Writer refines based on feedback
    final_result = await writer.run(
        f"Please refine this slogan based on the feedback: {initial_result}\nFeedback: {feedback}"
    )
    print(f"Final Slogan: {final_result}")

asyncio.run(main())
```

### Concurrent Orchestration
```python
import asyncio
from agent_framework import WorkflowBuilder, ConcurrentBuilder
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create multiple specialized agents
    agents = [
        OpenAIChatClient().create_agent(name=f"Agent{i}", instructions=f"You are expert {i}")
        for i in range(3)
    ]
    
    # Build concurrent workflow
    workflow = (
        ConcurrentBuilder()
        .add_participants(agents)
        .build()
    )
    
    # Run agents concurrently
    events = await workflow.run("Analyze this topic from different perspectives")
    print(events.get_outputs())

asyncio.run(main())
```

## Middleware and Context Providers

### Function-Based Middleware
```python
from agent_framework import ChatAgent, FunctionMiddleware
from agent_framework.openai import OpenAIChatClient

async def log_function_calls(context):
    """Log all function calls."""
    print(f"Calling function: {context.function_name}")
    await context.next()
    print(f"Function completed: {context.function_name}")

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        tools=[get_weather],
        middleware=[log_function_calls]
    )
    
    result = await agent.run("What's the weather in Seattle?")
    print(result)

asyncio.run(main())
```

### Class-Based Middleware
```python
from agent_framework import AgentMiddleware

class SecurityMiddleware(AgentMiddleware):
    async def on_run_start(self, context):
        """Validate request before processing."""
        if "sensitive" in context.input.lower():
            context.terminate = True
            context.result = "Request blocked for security reasons"
            return
        await context.next()

agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    middleware=[SecurityMiddleware()]
)
```

### Context Providers
```python
from agent_framework import ChatAgent, ContextProvider
from agent_framework.openai import OpenAIChatClient

class DateContextProvider(ContextProvider):
    async def provide_context(self, messages):
        """Add current date to context."""
        from datetime import datetime
        return f"Current date: {datetime.now().strftime('%Y-%m-%d')}"

agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    context_providers=[DateContextProvider()]
)
```

## DevUI for Testing and Debugging

DevUI is a web interface for testing agents and workflows interactively.

### Installation
```bash
pip install agent-framework-devui --pre
```

### Launch DevUI Programmatically
```python
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from agent_framework.devui import serve

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 72°F and sunny"

# Create your agent
agent = ChatAgent(
    name="WeatherAgent",
    chat_client=OpenAIChatClient(),
    tools=[get_weather]
)

# Launch DevUI
serve(entities=[agent], auto_open=True)
# → Opens browser to http://localhost:8080
```

### Launch DevUI from CLI
```bash
# Launch with directory-based discovery
devui ./agents --port 8080

# Launch with tracing enabled
devui ./agents --tracing framework

# Launch API only (no UI)
devui ./agents --headless
```

### Directory Structure for Discovery
DevUI can automatically discover agents and workflows from a directory structure:

```
agents/
├── weather_agent/
│   ├── __init__.py      # Must export: agent = ChatAgent(...)
│   ├── agent.py
│   └── .env             # Optional: API keys
├── my_workflow/
│   ├── __init__.py      # Must export: workflow = WorkflowBuilder()...
│   ├── workflow.py
│   └── .env
└── .env                 # Optional: shared environment variables
```

### DevUI Features
- **Interactive Chat**: Test agents with conversational interface
- **Workflow Visualization**: See workflow execution in real-time
- **Event Streaming**: Monitor agent and workflow events
- **Trace Viewer**: Analyze OpenTelemetry traces
- **OpenAI-Compatible API**: Use standard OpenAI API format

### OpenAI-Compatible API
```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-framework",
    "input": "What is the weather in Seattle?",
    "extra_body": {"entity_id": "weather_agent"}
  }'
```

## Common Examples and Scenarios

### Example 1: Customer Support Agent
```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

def check_order_status(order_id: str) -> str:
    """Check the status of an order."""
    # In real scenario, query database
    return f"Order {order_id} is in transit and will arrive in 2 days."

def initiate_return(order_id: str) -> str:
    """Initiate a return for an order."""
    # In real scenario, process return
    return f"Return initiated for order {order_id}. You will receive a return label via email."

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="CustomerSupportAgent",
        instructions="""
        You are a helpful customer support agent for an e-commerce platform.
        You can help customers check order status and initiate returns.
        Be friendly and professional.
        """,
        tools=[check_order_status, initiate_return]
    )
    
    # Example customer queries
    queries = [
        "What is the status of my order #12345?",
        "I want to return order #67890"
    ]
    
    for query in queries:
        print(f"\nCustomer: {query}")
        response = await agent.run(query)
        print(f"Agent: {response}")

asyncio.run(main())
```

### Example 2: Data Analysis Assistant
```python
import asyncio
from typing import List
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

def calculate_average(numbers: List[float]) -> float:
    """Calculate the average of a list of numbers."""
    return sum(numbers) / len(numbers)

def find_max_min(numbers: List[float]) -> dict:
    """Find the maximum and minimum values in a list."""
    return {"max": max(numbers), "min": min(numbers)}

async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="DataAnalyst",
        instructions="""
        You are a data analysis assistant that can perform calculations
        and provide insights on numerical data.
        """,
        tools=[calculate_average, find_max_min]
    )
    
    response = await agent.run(
        "What is the average and max/min of these sales figures: [120, 450, 230, 890, 340, 560]?"
    )
    print(response)

asyncio.run(main())
```

### Example 3: Content Moderation Workflow
```python
import asyncio
from agent_framework import ChatAgent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create specialized agents
    spam_detector = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="SpamDetector",
        instructions="Classify if content is spam or not. Reply with just 'SPAM' or 'NOT_SPAM'."
    )
    
    content_analyzer = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="ContentAnalyzer",
        instructions="Analyze content quality and provide a score from 1-10."
    )
    
    final_processor = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="FinalProcessor",
        instructions="Summarize the analysis and make a final recommendation."
    )
    
    # Build sequential workflow
    workflow = (
        WorkflowBuilder()
        .add_edge(spam_detector, content_analyzer)
        .add_edge(content_analyzer, final_processor)
        .set_start_executor(spam_detector)
        .build()
    )
    
    # Process content through workflow
    events = await workflow.run("Check this message: Hello, I'd like to inquire about your services.")
    print(events.get_outputs())

asyncio.run(main())
```

### Example 4: Research Assistant with Web Search
```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

# Note: Requires web search tool integration
async def main():
    agent = OpenAIChatClient().create_agent(
        name="ResearchAssistant",
        instructions="""
        You are a research assistant that can search the web for information.
        Provide comprehensive answers with sources.
        """,
        # Web search tool would be added here
    )
    
    response = await agent.run("What are the latest developments in quantum computing?")
    print(response)

asyncio.run(main())
```

### Example 5: Multimodal Agent (Image Analysis)
```python
import asyncio
from agent_framework import ChatAgent, ChatMessage
from agent_framework.openai import OpenAIResponsesClient

async def main():
    agent = OpenAIResponsesClient().create_agent(
        name="ImageAnalyzer",
        instructions="You are an expert at analyzing images and describing their contents."
    )
    
    # Create message with image
    message = ChatMessage(
        role="user",
        text="What do you see in this image?",
        images=["path/to/image.jpg"]  # or URL
    )
    
    response = await agent.run(message)
    print(response)

asyncio.run(main())
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Import Errors
**Problem**: `ModuleNotFoundError: No module named 'agent_framework'`

**Solution**:
```bash
# Install the framework
pip install agent-framework --pre

# Or install specific package
pip install agent-framework-core --pre
```

#### Issue: API Key Not Found
**Problem**: `ValueError: API key not found`

**Solution**:
1. Create a `.env` file with your API keys
2. Or set environment variables:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. Or pass keys directly to the client:
   ```python
   client = OpenAIChatClient(api_key="sk-...")
   ```

#### Issue: Azure Authentication Errors
**Problem**: `Authentication failed`

**Solution**:
```bash
# Install Azure CLI and login
az login

# Use Azure CLI credentials
from azure.identity import AzureCliCredential

client = AzureOpenAIChatClient(credential=AzureCliCredential())
```

#### Issue: Workflow Not Completing
**Problem**: Workflow hangs or doesn't complete

**Solution**:
1. Ensure terminal nodes call `ctx.yield_output()` to signal completion
2. Check that all edges are properly connected
3. Verify no infinite loops in workflow logic
4. Enable debug logging to see execution flow

#### Issue: Tool Not Being Called
**Problem**: Agent doesn't call the tool function

**Solution**:
1. Ensure tool has proper docstring (used by LLM to understand purpose)
2. Add type annotations with descriptions:
   ```python
   from typing import Annotated
   from pydantic import Field
   
   def get_weather(
       location: Annotated[str, Field(description="The location to get weather for")]
   ) -> str:
       """Get the weather for a given location."""
       ...
   ```
3. Make instructions clearer about when to use tools

#### Issue: Slow Response Times
**Problem**: Agent takes too long to respond

**Solution**:
1. Use streaming for better perceived performance:
   ```python
   async for chunk in agent.run_stream(query):
       print(chunk.text, end="", flush=True)
   ```
2. Reduce max_tokens parameter
3. Use faster models (e.g., gpt-4o-mini instead of gpt-4)
4. Optimize tool execution time

#### Issue: Memory/Context Issues
**Problem**: Agent forgets previous conversation

**Solution**:
1. Use threads for persistent conversations:
   ```python
   from agent_framework import Thread
   
   thread = Thread()
   agent.run(query, thread=thread)
   ```
2. Implement context providers for long-term memory
3. Use vector stores for RAG patterns

#### Issue: DevUI Not Starting
**Problem**: `devui` command not found or fails to start

**Solution**:
```bash
# Install DevUI package
pip install agent-framework-devui --pre

# Check if it's in PATH
which devui

# Or run via module
python -m agent_framework_devui ./agents
```

#### Issue: Rate Limiting
**Problem**: `RateLimitError` from API

**Solution**:
1. Add retry logic with exponential backoff
2. Use middleware to implement rate limiting
3. Consider using batching for multiple requests
4. Upgrade API tier for higher limits

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Or for specific loggers
logging.getLogger("agent_framework").setLevel(logging.DEBUG)
```

### Getting Help

- **Documentation**: [https://learn.microsoft.com/agent-framework/](https://learn.microsoft.com/agent-framework/)
- **GitHub Issues**: [https://github.com/microsoft/agent-framework/issues](https://github.com/microsoft/agent-framework/issues)
- **Discord**: [https://discord.gg/b5zjErwbQM](https://discord.gg/b5zjErwbQM)
- **Samples**: [https://github.com/microsoft/agent-framework/tree/main/python/samples](https://github.com/microsoft/agent-framework/tree/main/python/samples)

### Useful Resources

- **Design Documents**: [https://github.com/microsoft/agent-framework/tree/main/docs/design](https://github.com/microsoft/agent-framework/tree/main/docs/design)
- **Python Package Documentation**: [https://github.com/microsoft/agent-framework/tree/main/python](https://github.com/microsoft/agent-framework/tree/main/python)
- **Development Setup Guide**: [Python DEV_SETUP.md](./DEV_SETUP.md)
- **Sample Guidelines**: [samples/SAMPLE_GUIDELINES.md](./samples/SAMPLE_GUIDELINES.md)
- **Video Tutorial**: [Agent Framework Introduction (30 min)](https://www.youtube.com/watch?v=AAgdMhftj8w)
- **DevUI Demo**: [DevUI in Action (1 min)](https://www.youtube.com/watch?v=mOAaGY4WPvc)

---

## Next Steps

Now that you understand the basics:

1. **Explore Samples**: Browse the [samples directory](./samples/getting_started/) for complete working examples
2. **Build Your First Project**: Start with a simple agent and gradually add features
3. **Join the Community**: Connect with other developers on [Discord](https://discord.gg/b5zjErwbQM)
4. **Contribute**: Read the [Contributing Guide](../CONTRIBUTING.md) to contribute to the framework

Happy building! 🚀
