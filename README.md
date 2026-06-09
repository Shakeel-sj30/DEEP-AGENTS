# Deep Agents Chatbot

Deep Agents Chatbot is a Streamlit demo and learning project built around LangChain, LangGraph, Groq, and the `deepagents` ecosystem. It shows how to build a conversational agent that can reason, search the web, keep thread-based memory, work with virtual files, and demonstrate context engineering patterns across a set of guided notebooks.

Live app: https://deep-agents-ddnorp9gvxqwbjgnapp4e7v.streamlit.app/

## Overview

The project combines two parts:

1. A Streamlit chatbot in `streamlit_app.py`.
2. A notebook-based course in `deepagentsdemo/` that walks through the main concepts step by step.

The app is intentionally lightweight for Groq deployment, while the notebooks preserve the richer deep-agent examples for learning and experimentation.

## Features

- Groq-powered chat models
- Internet search with Tavily
- Thread-based memory using a LangGraph checkpointer
- Optional shared store backend
- Step-by-step agent reasoning and tool usage display
- Notebook examples for:
	- basic deep agent setup
	- context engineering with `AGENTS.md`, memory, and skills
	- backend options
	- subagents and structured outputs

## Tech Stack

- Python 3.13
- Streamlit
- LangChain
- LangGraph
- Groq
- Tavily
- `deepagents`

## Project Structure

- `streamlit_app.py` - main Streamlit chatbot application
- `main.py` - project entry point helper
- `deepagentsdemo/1-basicsdeepagent.ipynb` - basic agent setup and tools
- `deepagentsdemo/2-contextengineering.ipynb` - context engineering and memory
- `deepagentsdemo/3-backends.ipynb` - backend/storage examples
- `deepagentsdemo/4-subagents.ipynb` - subagents and structured output
- `deepagentsdemo/skills/` - reusable skills and examples
- `README.md` - project documentation

## Notebooks Guide

### 1. Basics

Shows how to create a deep agent, connect a model, add tools such as web search, and use built-in planning features.

### 2. Context Engineering

Demonstrates how `AGENTS.md`, memory, checkpointers, thread IDs, and skills shape agent behavior.

### 3. Backends

Explains how different storage layers work, including state, filesystem, and store-style backends.

### 4. Subagents

Covers custom subagents and structured output flows for richer multi-step workflows.

## Requirements

- Python 3.13 or later
- A Groq API key
- Optional Tavily API key for web search

## Environment Variables

Create a `.env` file locally or configure Streamlit secrets with:

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

`GROQ_API_KEY` is required to start the app. `TAVILY_API_KEY` is optional, but web search will be disabled without it.

## Local Setup

1. Create and activate a virtual environment.
2. Install the dependencies.
3. Add your API keys.
4. Run the app.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## How the App Works

The Streamlit app lets you:

- choose a Groq model
- set a system prompt
- send questions in a chat UI
- inspect intermediate tool calls and planning steps
- keep conversation state across turns with a checkpointer

The runtime uses Groq for inference and Tavily for search when available. If the Groq key is missing, the app stops early with a clear error message instead of failing later during model initialization.

## Deployment

The app is deployed on Streamlit Cloud here:

https://deep-agent-m7wsjnbgwz3z2c3lxfpwxk.streamlit.app/

If you deploy your own copy, make sure the following secrets are configured:

- `GROQ_API_KEY`
- `TAVILY_API_KEY` if you want web search

## Notes

- The repository is focused on Groq-based examples and no longer depends on OpenAI credentials for the app workflow.
- The notebooks are still useful as a learning resource even if the Streamlit runtime is kept intentionally lightweight for deployment stability.

## License

See `LICENSE` for licensing details.
