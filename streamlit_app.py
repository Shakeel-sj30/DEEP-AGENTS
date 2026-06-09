"""
Deep Agents Chatbot — Streamlit app
====================================
A conversational chatbot built on the `deepagents` library that demonstrates
EVERY feature covered in the deepagentsdemo notebooks:

1-basicsdeepagent.ipynb   -> create_deep_agent, custom model, custom system
                             prompt, custom tools (Tavily web search),
                             built-in planning (write_todos) + virtual files
2-contextengineering.ipynb -> AGENTS.md context file, memory=, checkpointer +
                             thread_id conversation memory, Skills (/skills/)
3-backends.ipynb          -> StateBackend / FilesystemBackend / StoreBackend
4-subagents.ipynb         -> custom subagents (research-agent) + structured
                             output subagent (Pydantic response_format)

Run with:  streamlit run streamlit_app.py
"""

import os
import uuid
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Environment (notebook 1: load API keys from .env)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DEMO_DIR = ROOT_DIR / "deepagentsdemo"

load_dotenv(ROOT_DIR / ".env")

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from tavily import TavilyClient


def get_secret_or_env(key: str) -> str | None:
    """Read a credential from Streamlit secrets first, then the environment."""
    if key in st.secrets:
        value = st.secrets.get(key)
        if value:
            os.environ[key] = str(value)
            return str(value)
    return os.getenv(key)

# ---------------------------------------------------------------------------
# Custom tool (notebook 1: Tavily internet search)
# ---------------------------------------------------------------------------
groq_api_key = get_secret_or_env("GROQ_API_KEY")
tavily_api_key = get_secret_or_env("TAVILY_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it to Streamlit Cloud secrets or your local .env file."
    )
    st.stop()

tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    if tavily_client is None:
        return {"error": "TAVILY_API_KEY missing — web search is disabled."}
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# ---------------------------------------------------------------------------
# Structured output schema (notebook 4: structured output with subagents)
# ---------------------------------------------------------------------------
class ResearchFindings(BaseModel):
    """Structured findings from a research task."""

    summary: str = Field(description="Summary of findings")
    confidence: float = Field(description="Confidence score from 0 to 1")
    sources: list[str] = Field(description="List of source URLs")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Agent factory — lightweight runtime to stay within Groq TPM limits
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI assistant. Use internet_search when needed and "
    "answer concisely with practical steps."
)

SUBAGENT_DOC = "Lightweight mode is enabled to avoid Groq TPM errors."


def build_agent(cfg: dict):
    """Create a lightweight LangChain agent that stays under Groq limits."""
    kwargs = dict(
        model=cfg["model"],
        tools=[internet_search],
        system_prompt=cfg["system_prompt"],
        checkpointer=st.session_state.checkpointer,
    )
    if cfg["backend"].startswith("StoreBackend"):
        kwargs["store"] = st.session_state.store

    return create_agent(**kwargs), {}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def extract_text(content) -> str:
    """AIMessage.content may be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def render_steps(messages):
    """Show the agent's intermediate work: tool calls, todos, subagent tasks."""
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name, args = tc["name"], tc["args"]
                if name == "write_todos":
                    with st.expander("📋 Planning — write_todos", expanded=False):
                        for todo in args.get("todos", []):
                            icon = {"pending": "⬜", "in_progress": "🔄",
                                    "completed": "✅"}.get(todo.get("status"), "⬜")
                            st.markdown(f"{icon} {todo.get('content', todo)}")
                elif name == "task":
                    with st.expander(
                        f"🤖 Subagent — {args.get('subagent_type', 'task')}",
                        expanded=False,
                    ):
                        st.markdown(args.get("description", ""))
                elif name == "internet_search":
                    with st.expander(
                        f"🔎 Web search — “{args.get('query', '')}”", expanded=False
                    ):
                        st.json(args)
                elif name in ("write_file", "edit_file", "read_file", "ls",
                              "glob", "grep"):
                    label = args.get("file_path") or args.get("path") or ""
                    with st.expander(f"📁 File system — {name} {label}",
                                     expanded=False):
                        st.json(args)
                else:
                    with st.expander(f"🛠️ Tool — {name}", expanded=False):
                        st.json(args)
        elif msg_type == "tool":
            text = extract_text(msg.content)
            if len(text) > 700:
                text = text[:700] + " …(truncated)"
            with st.expander(f"↩️ Result — {getattr(msg, 'name', 'tool')}",
                             expanded=False):
                st.code(text)


def render_files(files: dict):
    if not files:
        return
    with st.expander(f"🗂️ Virtual files in state ({len(files)})", expanded=False):
        for path, data in files.items():
            content = data.get("content", "") if isinstance(data, dict) else str(data)
            st.markdown(f"**`{path}`**")
            st.code(content[:1500] + (" …(truncated)" if len(content) > 1500 else ""))


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Deep Agents Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Deep Agents Chatbot")
st.caption(
    "Planning • Virtual file system • Context engineering (AGENTS.md + memory) • "
    "Skills • Subagents (incl. structured output) • Swappable backends • "
    "Thread memory via checkpointer"
)

# --- session state init ------------------------------------------------------
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "store" not in st.session_state:
    st.session_state.store = InMemoryStore()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, text, steps_messages, files)]

# --- sidebar configuration ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Agent configuration")

    model = st.selectbox(
        "Model",
        [
            "groq:llama-3.1-8b-instant",
            "groq:openai/gpt-oss-20b",
            "groq:llama-3.3-70b-versatile",
        ],
        index=0,
        help="Notebook 1: customizing the deep agent's model "
             "with a Groq provider:model string.",
    )

    backend = st.radio(
        "Backend (where files/memory live)",
        [
            "StoreBackend (cross-thread store)",
        ],
        help="Uses a persistent LangGraph store across threads.",
    )

    st.subheader("Features")
    use_agents_md = st.checkbox(
        "Load AGENTS.md context (memory=)", value=False,
        help="Disabled in lightweight mode to keep requests under Groq TPM.",
    )
    use_skills = st.checkbox(
        "Skills (/skills/)", value=False,
        help="Disabled in lightweight mode to keep requests under Groq TPM.",
    )
    use_subagents = st.checkbox(
        "Subagents", value=False,
        help="Disabled in lightweight mode to keep requests under Groq TPM.",
    )
    if use_subagents:
        st.warning(SUBAGENT_DOC)

    system_prompt = st.text_area(
        "System prompt", DEFAULT_SYSTEM_PROMPT, height=160,
        help="Kept intentionally short to avoid request-too-large errors.",
    )

    st.divider()
    st.caption(f"🧵 Thread: `{st.session_state.thread_id[:8]}…`")
    col1, col2 = st.columns(2)
    if col1.button("🆕 New thread", use_container_width=True,
                   help="Same agent + store, fresh conversation. With "
                        "StoreBackend, files written earlier are still there!"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    if col2.button("🗑️ Reset all", use_container_width=True,
                   help="Wipe checkpointer, store, and chat."):
        for k in ("checkpointer", "store", "store_seeded", "thread_id", "history"):
            st.session_state.pop(k, None)
        st.rerun()

    # missing-key warnings
    if not tavily_api_key:
        st.warning("TAVILY_API_KEY missing — web search will fail")

cfg = {
    "model": model,
    "backend": backend,
    "use_agents_md": use_agents_md,
    "use_skills": use_skills,
    "use_subagents": use_subagents,
    "system_prompt": system_prompt,
}

# Rebuild the agent only when the configuration changes
cfg_key = str(sorted(cfg.items()))
if st.session_state.get("cfg_key") != cfg_key:
    st.session_state.agent, st.session_state.seed_files = build_agent(cfg)
    st.session_state.cfg_key = cfg_key

# --- replay chat history -----------------------------------------------------
for role, text, steps, files in st.session_state.history:
    with st.chat_message(role):
        if steps:
            render_steps(steps)
        st.markdown(text)
        if files:
            render_files(files)

# --- chat input / agent invocation -------------------------------------------
if prompt := st.chat_input("Ask me anything — research, code, AWS, LangGraph…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.history.append(("user", prompt, None, None))

    payload = {"messages": [{"role": "user", "content": prompt}]}
    # StateBackend: seed AGENTS.md + skills into this thread's virtual FS
    if st.session_state.seed_files:
        payload["files"] = st.session_state.seed_files

    config = {"configurable": {"thread_id": st.session_state.thread_id},
              "recursion_limit": 100}

    with st.chat_message("assistant"):
        with st.spinner("🧠 Deep agent planning, researching, delegating…"):
            try:
                result = st.session_state.agent.invoke(payload, config=config)
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.stop()

        # Only render the messages generated for THIS turn
        all_msgs = result["messages"]
        turn_start = max(
            (i for i, m in enumerate(all_msgs)
             if getattr(m, "type", "") == "human"),
            default=0,
        )
        new_msgs = all_msgs[turn_start + 1:]

        render_steps(new_msgs)
        answer = extract_text(all_msgs[-1].content) or "*(no text response)*"
        st.markdown(answer)

        # Virtual files (StateBackend) — notebook 3 backend check
        files = {
            p: d for p, d in result.get("files", {}).items()
            if p not in st.session_state.seed_files
        }
        render_files(files)

    st.session_state.history.append(("assistant", answer, new_msgs, files))
