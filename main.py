import contextlib
import io
import os
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from agent import build_agent
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from rich.console import Console
from rich.theme import Theme

from agno.agent import Agent

load_dotenv()


console = Console(
    theme=Theme(
        {
            "tool.name": "bold cyan",
            "tool.ok": "green",
            "tool.err": "bold red",
            "tool.args": "magenta",
            "tool.sid": "dim cyan",
            "tool.ts": "dim",
        }
    )
)


def env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    return (
        (str(val).strip().lower() in {"1", "true", "t", "yes", "y", "on"})
        if val is not None
        else default
    )


LOG_TOOL_LIVE = env_flag("LOG_TOOL_LIVE", default=False)
LOG_TOOL_SUMMARY = env_flag("LOG_TOOL_SUMMARY", default=True)
LOG_DEBUG = env_flag("LOG_DEBUG", default=False)
LOG_TOOL_PREVIEW = env_flag("LOG_TOOL_PREVIEW", default=False)

STATIC_DIR = Path("static")


app = FastAPI(title="Agno BigQuery Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


AGENT: Agent = build_agent()
if hasattr(AGENT, "show_tool_calls"):
    try:
        AGENT.show_tool_calls = False
    except Exception:
        pass

TOOL_LOGS: Dict[str, List[Dict[str, Any]]] = {}
TOOL_LOGS_LOCK = Lock()


def _logs_reset(session_id: str) -> None:
    with TOOL_LOGS_LOCK:
        TOOL_LOGS[session_id] = []


def _logs_append(session_id: str, entry: Dict[str, Any]) -> None:
    with TOOL_LOGS_LOCK:
        TOOL_LOGS.setdefault(session_id, []).append(entry)


def _logs_get(session_id: str) -> List[Dict[str, Any]]:
    with TOOL_LOGS_LOCK:
        return list(TOOL_LOGS.get(session_id, []))


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


def capture_tool_calls(
    agent: Agent, function_name: str, function_call: Callable, arguments: Dict[str, Any]
):
    sid = (getattr(agent, "context", {}) or {}).get("_log_key", "default")
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        result = function_call(**arguments)
        entry = {
            "time": ts,
            "name": function_name,
            "args": arguments,
            "result_preview": repr(result)[:800],
        }
        _logs_append(sid, entry)

        if LOG_TOOL_LIVE:
            console.print(
                f"[tool.ts]{ts}[/tool.ts] [tool.sid]({sid})[/tool.sid] "
                f"[tool.name]{function_name}[/tool.name] [tool.ok]OK[/tool.ok]"
            )
            console.print(f"  [tool.args]args[/tool.args]= {arguments}")
            if LOG_TOOL_PREVIEW:
                console.print(f"  preview= {entry['result_preview']}")
        return result

    except Exception as e:
        entry = {"time": ts, "name": function_name, "args": arguments, "error": repr(e)}
        _logs_append(sid, entry)

        if LOG_TOOL_LIVE:
            console.print(
                f"[tool.ts]{ts}[/tool.ts] [tool.sid]({sid})[/tool.sid] "
                f"[tool.name]{function_name}[/tool.name] [tool.err]ERROR[/tool.err] {e!r}"
            )
            console.print(f"  [tool.args]args[/tool.args]= {arguments}")
        raise


AGENT.tool_hooks = (getattr(AGENT, "tool_hooks", []) or []) + [capture_tool_calls]


def choose_session_id(agent: Agent, requested: Optional[str]) -> str:
    env_sid = (os.getenv("SESSION_ID") or "").strip()
    req_sid = (requested or "").strip()
    return env_sid or req_sid or f"{agent.user_id}-{uuid.uuid4().hex[:8]}"


def run_agent(agent: Agent, message: str, session_id: str) -> str:
    agent.context = {"_log_key": session_id}
    _logs_reset(session_id)

    for method_name in ("create_response", "get_response", "run", "respond"):
        method = getattr(agent, method_name, None)
        if callable(method):
            if LOG_DEBUG:
                console.print(
                    f"[tool.ts]{datetime.now():%H:%M:%S}[/tool.ts] use [tool.name]{method_name}[/tool.name]"
                )
            try:
                result = method(message, session_id=session_id)
                if isinstance(result, str):
                    return result
                text = getattr(result, "content", None) or getattr(result, "text", None)
                if text is not None:
                    return str(text)
            except Exception as e:
                if LOG_DEBUG:
                    console.print(
                        f"[tool.err]method {method_name} raised: {e!r}[/tool.err]"
                    )

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            agent.print_response(message, session_id=session_id, stream=False)
        return buf.getvalue().strip()
    finally:
        buf.close()


def print_tool_summary(session_id: str) -> None:
    calls = _logs_get(session_id)
    if not (LOG_TOOL_SUMMARY and calls):
        return
    console.rule(f"[tool.name]Tool calls[/tool.name] [tool.sid]{session_id}[/tool.sid]")
    for c in calls:
        if "error" in c:
            console.print(
                f"[tool.ts]{c['time']}[/tool.ts] [tool.name]{c['name']}[/tool.name] [tool.err]ERROR[/tool.err]"
            )
            console.print(f"  [tool.args]args[/tool.args]= {c['args']}")
            console.print(f"  [tool.err]{c['error']}[/tool.err]")
        else:
            console.print(
                f"[tool.ts]{c['time']}[/tool.ts] [tool.name]{c['name']}[/tool.name] [tool.ok]OK[/tool.ok]"
            )
            console.print(f"  [tool.args]args[/tool.args]= {c['args']}")
            if LOG_TOOL_PREVIEW:
                console.print(f"  preview= {c['result_preview']}")
    console.rule()


@app.post("/api/chat")
def chat(payload: ChatIn):
    session_id = choose_session_id(AGENT, payload.session_id)
    try:
        answer = run_agent(AGENT, payload.message, session_id)
        print_tool_summary(session_id)
        return JSONResponse(
            {
                "session_id": session_id,
                "answer": answer,
                "tool_calls": _logs_get(session_id),
            }
        )
    except Exception as e:
        console.print(f"[tool.err]Unhandled error in /api/chat: {e!r}[/tool.err]")
        return JSONResponse(
            {
                "session_id": session_id,
                "error": "Internal error while generating response.",
            },
            status_code=500,
        )
