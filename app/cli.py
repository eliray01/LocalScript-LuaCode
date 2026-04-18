import argparse
import json
from collections import deque
from typing import NoReturn

from app.graph import CHAT_HISTORY_TURNS, run_agent_stream_with_trace
from app.ollama_ready import wait_for_ollama_model
from app.state import MemoryTurn

_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
}
_RESET = "\033[0m"


def _sanitize_text(value: str) -> str:
    """Ensure text is safe for UTF-8 JSON transport."""
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _print_now(message: str = "", color: str | None = None) -> None:
    if color and color in _COLORS:
        print(f"{_COLORS[color]}{message}{_RESET}", flush=True)
    else:
        print(message, flush=True)


# 5×5 block letters (█ / space) for the REPL title.
_BANNER_GLYPHS: dict[str, tuple[str, str, str, str, str]] = {
    " ": ("      ", "      ", "      ", "      ", "      "),
    "A": (" ███ ", "█   █", "█████", "█   █", "█   █"),
    "C": (" ███ ", "█    ", "█    ", "█    ", " ███ "),
    "E": ("█████", "█    ", "████ ", "█    ", "█████"),
    "G": (" ███ ", "█    ", "█  ██", "█   █", " ███ "),
    "I": (" ███ ", "  █  ", "  █  ", "  █  ", " ███ "),
    "L": ("█    ", "█    ", "█    ", "█    ", "█████"),
    "N": ("█   █", "██  █", "█ █ █", "█  ██", "█   █"),
    "O": (" ███ ", "█   █", "█   █", "█   █", " ███ "),
    "P": ("████ ", "█  █ ", "████ ", "█    ", "█    "),
    "R": ("████ ", "█  █ ", "████ ", "█ █  ", "█  █ "),
    "S": (" ███ ", "█    ", " ███ ", "    █", " ███ "),
    "T": ("█████", "  █  ", "  █  ", "  █  ", "  █  "),
}


def _banner_word_lines(word: str) -> list[str]:
    w = word.upper()
    rows: list[list[str]] = [[] for _ in range(5)]
    for i, ch in enumerate(w):
        g = _BANNER_GLYPHS.get(ch, _BANNER_GLYPHS[" "])
        for r in range(5):
            rows[r].append(g[r])
            if i < len(w) - 1:
                rows[r].append(" ")
    return ["".join(row) for row in rows]


def _print_localscript_cli_agent_banner() -> None:
    """Big multi-line title: LocalScript CLI agent (as block █ letters)."""
    parts = ("LOCALSCRIPT", "CLI", "AGENT")
    for j, part in enumerate(parts):
        for line in _banner_word_lines(part):
            _print_now(line, color="magenta")
        if j < len(parts) - 1:
            _print_now()


def _normalize_command_text(text: str) -> str:
    # Normalize common slash variants and hidden separators from terminal input.
    cleaned = text.replace("\u200b", "").replace("\ufeff", "").strip()
    if cleaned.startswith("／"):
        cleaned = "/" + cleaned[1:]
    return cleaned


def _command_name(text: str) -> str | None:
    normalized = _normalize_command_text(text)
    if not normalized.startswith("/"):
        return None
    return normalized.split(maxsplit=1)[0].lower()


def _run_once(
    prompt: str, pretty: bool, chat_history: list[MemoryTurn] | None = None
) -> tuple[int, dict[str, str], str, str]:
    _print_now("Prompt received", color="blue")
    final: dict = {}
    intent_printed = False
    chat_printed = False
    routing_printed = False
    clarifying_printed = False
    planning_printed = False
    generating_printed = False
    validating_printed = False
    last_reported_attempts = 0

    for state in run_agent_stream_with_trace(prompt, chat_history=chat_history):
        final = dict(state)
        route = str(final.get("route", "complex")).strip().lower() or "complex"
        intent = str(final.get("intent", "")).strip().lower()
        if intent and not intent_printed:
            _print_now(f"Intent ({intent})", color="cyan")
            intent_printed = True

        if (
            intent == "chat"
            and str(final.get("nl_response", "")).strip()
            and not chat_printed
        ):
            _print_now("Answering (no Lua generation)", color="green")
            _print_now(str(final.get("nl_response", "")).strip())
            chat_printed = True

        if final.get("needs_clarification") and not clarifying_printed:
            _print_now("Clarifying", color="yellow")
            question = str(final.get("clarification_question", "")).strip()
            if question:
                _print_now(question)
            else:
                _print_now("Please clarify missing requirements.", color="yellow")
            clarifying_printed = True

        if (
            not routing_printed
            and final.get("route")
            and not final.get("needs_clarification")
        ):
            _print_now(f"Routing ({route})", color="blue")
            routing_printed = True

        if routing_printed and not planning_printed and not final.get("needs_clarification"):
            if route == "easy":
                _print_now("Planning", color="yellow")
                _print_now("(skipped for easy task)", color="yellow")
                planning_printed = True
            else:
                plan = str(final.get("plan", "")).strip()
                if plan:
                    _print_now("Planning", color="yellow")
                    _print_now(plan)
                    planning_printed = True

        attempts = int(final.get("attempts", 0) or 0)
        if attempts > 0 and not generating_printed:
            _print_now("Generating", color="magenta")
            generating_printed = True
        if attempts > 1 and attempts != last_reported_attempts:
            _print_now(f"Generation attempts: {attempts}", color="yellow")
        last_reported_attempts = attempts

        has_validation_step = bool(final.get("bindings")) or bool(final.get("validation_error"))
        if has_validation_step and not validating_printed:
            _print_now("Validating", color="cyan")
            validating_printed = True

    if final.get("intent") == "chat":
        text = str(final.get("nl_response", "")).strip()
        if not chat_printed:
            _print_now("Answering (no Lua generation)", color="green")
            _print_now(text or "(empty response)")
        return 0, {}, text, "chat"

    if final.get("needs_clarification"):
        question = str(final.get("clarification_question", "")).strip()
        if not clarifying_printed:
            _print_now("Clarifying", color="yellow")
            _print_now(question or "Please clarify missing requirements.", color="yellow")
        return 0, {}, question or "Please clarify missing requirements.", "clarify"

    if not intent_printed and str(final.get("intent", "")).strip():
        _print_now(f"Intent ({str(final.get('intent', '')).strip().lower()})", color="cyan")
    if not planning_printed:
        _print_now("Planning", color="yellow")
        _print_now("(plan is empty)", color="yellow")
    if not generating_printed:
        _print_now("Generating", color="magenta")
    if not validating_printed:
        _print_now("Validating", color="cyan")

    bindings = dict(final.get("bindings") or {})
    if not bindings:
        err = str(final.get("validation_error", "")).strip()
        if err:
            _print_now(f"Validation error: {err}", color="red")
        _print_now(
            "Validation failed: model did not return valid LowCode JSON "
            "(one or more tags with lua{...}lua values).",
            color="red",
        )
        return 1, {}, "", ""
    if pretty:
        assistant_text = json.dumps(bindings, ensure_ascii=False, indent=2)
        _print_now(assistant_text, color="green")
    else:
        assistant_text = json.dumps(bindings, ensure_ascii=False)
        _print_now(assistant_text, color="green")
    return 0, bindings, assistant_text, "bindings"


def _repl(pretty: bool) -> NoReturn:
    _print_localscript_cli_agent_banner()
    _print_now(
        "Type your task or answers (multiline supported). Always submit with an empty line.",
        color="cyan",
    )
    _print_now("Commands: /help, /cancel, /memory, /reset, /exit, /quit", color="yellow")
    _print_now(f"Memory: keeps last {CHAT_HISTORY_TURNS} turns.", color="cyan")
    multiline_mode = False
    multiline_buffer: list[str] = []
    history: deque[MemoryTurn] = deque(maxlen=CHAT_HISTORY_TURNS)
    while True:
        try:
            prompt_raw = _sanitize_text(input("... " if multiline_mode else "localscript> "))
        except EOFError:
            _print_now()
            raise SystemExit(0)
        except KeyboardInterrupt:
            _print_now()
            multiline_mode = False
            multiline_buffer = []
            continue

        prompt = _sanitize_text(prompt_raw.strip())
        cmd = _command_name(prompt)

        if not prompt:
            if multiline_mode:
                payload = _sanitize_text("\n".join(multiline_buffer).strip())
                multiline_mode = False
                multiline_buffer = []
                if not payload:
                    continue
                status, bindings, assistant_text, turn_mode = _run_once(
                    payload, pretty=pretty, chat_history=list(history)
                )
                if status == 0 and assistant_text:
                    turn: MemoryTurn = {"user": payload, "assistant": assistant_text}
                    if turn_mode:
                        turn["mode"] = turn_mode
                    history.append(turn)
                continue
            continue
        if cmd in {"/exit", "/quit"}:
            raise SystemExit(0)
        if cmd == "/help":
            _print_now("Enter a natural-language task. The agent returns JSON bindings.", color="cyan")
            _print_now(
                "Submit every message (including one line) with an empty line after your text.",
                color="cyan",
            )
            _print_now(
                "Clarifying answers use the same rules as tasks — no auto-send on punctuation.",
                color="cyan",
            )
            _print_now("Use /cancel to clear current input; /reset clears memory.", color="yellow")
            _print_now("Use /memory to inspect context, /reset to clear memory.", color="yellow")
            continue
        if cmd == "/cancel":
            multiline_mode = False
            multiline_buffer = []
            _print_now("Input buffer cleared.", color="green")
            continue
        if cmd == "/memory":
            if not history:
                _print_now("Memory is empty.", color="yellow")
                continue
            _print_now(f"Memory turns: {len(history)}", color="cyan")
            for idx, turn in enumerate(history, start=1):
                _print_now(f"Turn {idx} user:", color="blue")
                _print_now(turn["user"])
                _print_now(f"Turn {idx} assistant:", color="green")
                _print_now(turn["assistant"])
            continue
        if cmd == "/reset":
            history.clear()
            multiline_mode = False
            multiline_buffer = []
            _print_now("Memory cleared.", color="green")
            continue
        if cmd is not None:
            _print_now(f"Unknown command: {cmd}. Use /help.", color="red")
            continue

        if multiline_mode:
            multiline_buffer.append(_sanitize_text(prompt_raw))
            continue

        multiline_mode = True
        multiline_buffer = [_sanitize_text(prompt_raw)]
        continue

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LocalScript CLI agent. Run with --prompt for one-shot mode "
            "or without --prompt for interactive REPL."
        )
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="One-shot prompt. If omitted, interactive REPL starts.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    wait_for_ollama_model()

    if args.prompt:
        status, _, _, _ = _run_once(args.prompt, pretty=args.pretty, chat_history=[])
        return status

    _repl(pretty=args.pretty)


if __name__ == "__main__":
    raise SystemExit(main())
