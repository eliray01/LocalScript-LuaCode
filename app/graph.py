from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
import json

from app.extract import extract_json_object
from app.llm import get_llm
from app.state import GraphState, MemoryTurn
from app.validate_lua import validate_lowcode_bindings_json

CHAT_HISTORY_TURNS = 6

INTENT_SYSTEM = """Classify ONLY the single user message below. You do not see prior conversation turns — use this message's text alone (no memory of earlier tasks, bindings, or clarification).

Return ONLY JSON:
{"intent": "generate"|"chat"}

Definitions:
- generate: the user wants executable LowCode Lua bindings for a workflow task (transform data, compute values, filter/clean structures, use wf context, implement concrete logic, or asks you to write/produce lua{...}lua output in this message).
- chat: general or educational only (explain concepts, what does X mean, how something works, review/discuss style, compare approaches), greetings, thanks, or agreeing/naming a topic without asking you to implement or emit new binding JSON here (e.g. "yes", "sure", "tell me more", "Lua LowCode bindings" as a topic continuation).

Rules:
- If the message mixes both, prefer generate if they ask you to implement or produce bindings in this same message.
- If they only pasted wf JSON and ask "what is this?" → chat.
- If unsure, prefer generate.

No markdown outside JSON."""

NL_CHAT_SYSTEM = """You are a helpful assistant for Lua LowCode workflow bindings (Lua 5.5).

The user may ask in Russian or English.

Rules:
- Answer in plain natural language only (no JSON, no lua{...}lua wrappers, no full binding solutions unless they only asked for a snippet concept).
- Be concise unless the user asks for detail.
- You may explain: wf.vars / wf.initVariables access, typical patterns, allowed helpers like _utils.array.new and _utils.array.markAsArray, and restrictions (no require, no JsonPath, etc.).
- If the user actually needs a full workflow implementation for their app, tell them briefly to phrase it as an implementation task so the agent can generate bindings."""

PLAN_SYSTEM = """You are a planning assistant for Lua generation tasks.

Given the user request and optional wf context, produce a concise implementation plan that will be used to generate Lua code.

Rules:
- Return plain text only (no JSON, no markdown code fences).
- Keep it short: 3-7 bullet points.
- Focus on output tags, data access paths, conditions/loops, and final return values.
- If task is ambiguous, include one safe assumption.
"""

ROUTER_SYSTEM = """Classify the task complexity for Lua generation.

Return exactly one word:
- EASY: the task can be solved with a trivial one-line `return ...` expression.
- COMPLEX: any task that needs multi-line logic, conditions, loops, intermediate variables, or multiple steps.
"""

CLARIFY_SYSTEM = """Decide if the user provided enough information to generate correct Lua bindings.

You must be strict: if any critical detail is missing, do not guess — ask for it.

Return ONLY JSON with this shape:
{
  "needs_clarification": true|false,
  "question": "one short question, or empty string"
}

Rules:
- If needs_clarification is false, question must be "" (empty string).
- If needs_clarification is true, ask exactly ONE question: the single most important missing piece of information for this step. Never ask multiple questions in one response; the user will answer, then you will be invoked again.
- Prefer concrete questions (paths under wf.vars, output binding tag names, edge cases, data shape).
- Use conversation history: the latest user message may be an answer to your previous question — merge it and treat it as resolved context. If the user already answered your last question, do NOT ask that same question again; either proceed (needs_clarification false) or ask one different missing item.
- If the task is to compute or derive a new value (totals, discounts, transformations), do NOT ask which existing field "already contains" the computed result unless the task text clearly states that such a field exists. In that case ask for the output binding name (JSON key) to return under and where input data lives in wf.vars (path or example JSON).
- A short reply like a single identifier (e.g. total_price, result) is often the output tag name — accept it and move on unless the task still cannot be implemented.
- Do not include markdown or extra text outside JSON.
"""

SYSTEM = """You generate Lua 5.5 code for a LowCode workflow engine. The user may write in Russian or English.

## Required response shape (STRICT)
Return ONLY one valid UTF-8 JSON object with one or more top-level keys.

Shape:
{
  "<output_tag_1>": "lua{ <lua code> }lua",
  "<output_tag_2>": "lua{ <lua code> }lua"
}

Rules:
- Root must be a JSON object with at least one property.
- Each property value must be a string in the exact form `lua{...}lua`.
- No wrapper keys like `output`.
- No explanations, no prose, no markdown outside the JSON.
- Do not wrap the answer in code fences.

## Variable access rules (STRICT)
All input data must be accessed ONLY through:
- wf.vars.<name>
- wf.initVariables.<name>

Never use JsonPath.
Never use bare variables like:
score, status, mode, quantity, user, value, age, phone, discount, priority.
If needed, first assign:
local score = wf.vars.score

## Allowed Lua / environment
Use simple Lua 5.5 constructs:
- if ... then ... else
- for / ipairs / pairs
- while / repeat
- local functions
- tonumber, tostring, type
- string.sub, string.format, string.match
- table.insert
- math.* only when clearly needed

## Arrays
Allowed array helpers:
- _utils.array.new()
- _utils.array.markAsArray(arr)

Do NOT use any other _utils.array helpers.
Forbidden:
- _utils.array.push
- _utils.array.filter
- _utils.array.map
- _utils.array.sum
- _utils.array.any
- _utils.array.some
- _utils.array.contains
- _utils.array.average
- _utils.array.sort
- _utils.array.reduce
- _utils.array.unique
- _utils.array.isArray

When creating a new result array, use:
local result = _utils.array.new()
table.insert(result, value)

## Forbidden APIs
Do NOT use:
- os.time
- unpack
- require, dofile, loadfile, load
- io.*, os.*, debug.*, package.*

## Tag naming
Infer the output tag from the task.
If the task explicitly names the target variable, use that exact name.

## Logic requirements
The code must solve the exact task.
Examples:
- If asked to count matching items, count only matching items.
- If asked to return only emails, return emails, not full objects.
- If asked for the first matching element, search for the first matching element.
- If asked to normalize a field into an array, preserve the original value inside the array when needed.
- If asked to convert date/time to ISO 8601, return a real ISO 8601 string, not simple concatenation unless that exactly matches the required format.

## Style
- Prefer explicit, deterministic code over clever shortcuts.
- Prefer multiline readable Lua for anything beyond a trivial one-line return.
- Use local variables where appropriate.

## Examples
{"lastEmail":"lua{return wf.vars.emails[#wf.vars.emails]}lua"}
{"try_count_n":"lua{return wf.vars.try_count_n + 1}lua"}
{"result":"lua{local r = wf.vars.RESTbody.result\\nfor _, e in pairs(r) do\\n  for k in pairs(e) do\\n    if k ~= 'ID' and k ~= 'ENTITY_ID' and k ~= 'CALL' then\\n      e[k] = nil\\n    end\\n  end\\nend\\nreturn r\\n}lua"}
{"result":"lua{local result = _utils.array.new()\\nfor _, item in ipairs(wf.vars.parsedCsv) do\\n  if (item.Discount ~= '' and item.Discount ~= nil) or (item.Markdown ~= '' and item.Markdown ~= nil) then\\n    table.insert(result, item)\\n  end\\nend\\nreturn result\\n}lua"}

If the user message includes JSON with wf, use it only as context.

Before answering, silently verify:
- all inputs use wf.vars or wf.initVariables
- no forbidden helpers or APIs are used
- the logic matches the task exactly
- the final answer is valid JSON only
"""


def _normalize_for_question_match(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _intent_entry(state: GraphState) -> dict:
    """Skip intent LLM when the user is continuing a generate+clarify thread."""
    history = state.get("chat_history") or []
    if history and str(history[-1].get("mode", "")).lower() == "clarify":
        return {"intent": "generate"}
    return {}


def _route_intent_entry(state: GraphState):
    history = state.get("chat_history") or []
    if history and str(history[-1].get("mode", "")).lower() == "clarify":
        return "clarify"
    return "intent_classify"


def _history_block(state: GraphState) -> str:
    history = state.get("chat_history", [])
    if not history:
        return ""
    lines: list[str] = []
    for idx, turn in enumerate(history[-CHAT_HISTORY_TURNS:], start=1):
        lines.append(f"Turn {idx} user:\n{turn['user']}")
        lines.append(f"Turn {idx} assistant:\n{turn['assistant']}")
    return "\n\n".join(lines)


def _intent_classify(state: GraphState) -> dict:
    llm = get_llm()
    messages = [
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(
            content=(
                "User message to classify (may include JSON context with wf):\n"
                f"{state['prompt']}"
            )
        ),
    ]
    out = llm.invoke(messages)
    raw = out.content if isinstance(out.content, str) else str(out.content)
    intent = "generate"
    try:
        parsed = json.loads(extract_json_object(raw))
        if isinstance(parsed, dict):
            v = str(parsed.get("intent", "")).strip().lower()
            if v in {"chat", "generate"}:
                intent = v
    except json.JSONDecodeError:
        pass
    return {"intent": intent}


def _route_after_intent(state: GraphState):
    if state.get("intent") == "chat":
        return "nl_answer"
    return "clarify"


def _nl_answer(state: GraphState) -> dict:
    llm = get_llm()
    messages = [SystemMessage(content=NL_CHAT_SYSTEM)]
    history_block = _history_block(state)
    if history_block:
        messages.append(
            HumanMessage(
                content=(
                    f"Previous conversation context (up to {CHAT_HISTORY_TURNS} turns):\n\n"
                    f"{history_block}"
                )
            )
        )
    messages.append(
        HumanMessage(
            content=(
                f"User question (may include JSON context with wf):\n{state['prompt']}"
            )
        )
    )
    out = llm.invoke(messages)
    text = out.content if isinstance(out.content, str) else str(out.content)
    return {"nl_response": text.strip()}


def _router(state: GraphState) -> dict:
    llm = get_llm()
    messages = [SystemMessage(content=ROUTER_SYSTEM)]
    history_block = _history_block(state)
    if history_block:
        messages.append(
            HumanMessage(
                content=(
                    f"Previous conversation context (up to {CHAT_HISTORY_TURNS} turns):\n\n"
                    f"{history_block}"
                )
            )
        )
    messages.append(
        HumanMessage(
            content=(
                f"User task (may include JSON context with wf):\n{state['prompt']}\n\n"
                "Return only EASY or COMPLEX."
            )
        )
    )
    out = llm.invoke(messages)
    raw = out.content if isinstance(out.content, str) else str(out.content)
    decision = raw.strip().upper()
    route = "easy" if "EASY" in decision and "COMPLEX" not in decision else "complex"
    return {"route": route}


def _clarify(state: GraphState) -> dict:
    llm = get_llm()
    messages = [SystemMessage(content=CLARIFY_SYSTEM)]
    history_block = _history_block(state)
    if history_block:
        messages.append(
            HumanMessage(
                content=(
                    f"Previous conversation context (up to {CHAT_HISTORY_TURNS} turns):\n\n"
                    f"{history_block}"
                )
            )
        )
    messages.append(
        HumanMessage(
            content=(
                f"Latest user message (task, answer to a prior question, or JSON context with wf):\n"
                f"{state['prompt']}\n\n"
                "Assess completeness for safe Lua generation. "
                "If the message is an answer to your previous clarifying question, merge it with prior turns before deciding."
            )
        )
    )
    out = llm.invoke(messages)
    raw = out.content if isinstance(out.content, str) else str(out.content)

    needs_clarification = False
    question_text = ""
    parsed = {}
    try:
        parsed = json.loads(extract_json_object(raw))
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        needs_clarification = bool(parsed.get("needs_clarification"))
        q = parsed.get("question", "")
        if isinstance(q, str) and q.strip():
            question_text = q.strip()
        # Back-compat: older prompts returned "questions": [...]
        if needs_clarification and not question_text:
            raw_questions = parsed.get("questions", [])
            if isinstance(raw_questions, list) and raw_questions:
                first = raw_questions[0]
                if isinstance(first, str) and first.strip():
                    question_text = first.strip()

    if needs_clarification and not question_text:
        question_text = "Уточните, пожалуйста, недостающую информацию для задачи."

    history = state.get("chat_history") or []
    user_latest = str(state.get("prompt", "")).strip()
    if history and needs_clarification and user_latest:
        last_assistant = str(history[-1].get("assistant", "")).strip()
        if last_assistant:
            same_as_last = _normalize_for_question_match(
                question_text
            ) == _normalize_for_question_match(last_assistant)
            # Model repeated the same clarifying question after the user already replied.
            if same_as_last:
                needs_clarification = False
                question_text = ""

    return {
        "needs_clarification": needs_clarification,
        "clarification_question": question_text,
    }


def _route_after_clarify(state: GraphState):
    if state.get("needs_clarification"):
        return END
    return "router"


def _route_after_router(state: GraphState):
    if state.get("route") == "easy":
        return "generate"
    return "plan"


def _plan(state: GraphState) -> dict:
    llm = get_llm()
    messages = [SystemMessage(content=PLAN_SYSTEM)]
    history_block = _history_block(state)
    if history_block:
        messages.append(
            HumanMessage(
                content=(
                    f"Previous conversation context (up to {CHAT_HISTORY_TURNS} turns):\n\n"
                    f"{history_block}"
                )
            )
        )
    messages.append(
        HumanMessage(
            content=(
                f"User task (may include JSON context with wf):\n{state['prompt']}\n\n"
                "Create a short plan for implementing this in Lua."
            )
        )
    )
    out = llm.invoke(messages)
    plan = out.content if isinstance(out.content, str) else str(out.content)
    return {"plan": plan.strip()}


def _generate(state: GraphState) -> dict:
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM),
    ]
    history_block = _history_block(state)
    if history_block:
        messages.append(
            HumanMessage(
                content=(
                    f"Context from previous conversation turns (up to {CHAT_HISTORY_TURNS}):\n\n"
                    f"{history_block}"
                )
            )
        )
    plan = state.get("plan", "").strip()
    plan_section = (
        f"Implementation plan to follow:\n{plan}\n\n" if plan else "No explicit plan provided.\n\n"
    )
    messages.append(
        HumanMessage(
            content=(
                f"{plan_section}"
                f"User task (may include JSON context with wf):\n{state['prompt']}\n\n"
                "Return exactly one JSON object with one or more output-tag keys. "
                "Each value must be a lua{...}lua string. Use multiline readable Lua by default. "
                "Use one-line format only for trivial single 'return ...' cases."
            )
        )
    )
    if state.get("validation_error"):
        messages.append(
            HumanMessage(
                content=(
                    "Your previous answer failed validation. Fix and reply again with ONLY the JSON object.\n\n"
                    f"Error:\n{state['validation_error']}\n\n"
                    f"Previous output:\n{state.get('raw_output', '')}"
                )
            )
        )
    out = llm.invoke(messages)
    raw = out.content if isinstance(out.content, str) else str(out.content)
    raw_output = extract_json_object(raw)
    return {
        "raw_output": raw_output,
        "attempts": state.get("attempts", 0) + 1,
    }


def _validate(state: GraphState) -> dict:
    ok, err, bindings = validate_lowcode_bindings_json(state.get("raw_output", ""))
    if ok:
        return {"validation_error": "", "bindings": bindings}
    return {"validation_error": err, "bindings": {}}


def _route_after_validate(state: GraphState):
    if not state.get("validation_error"):
        return END
    if state.get("attempts", 0) >= 2:
        return END
    return "generate"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("intent_entry", _intent_entry)
    g.add_node("intent_classify", _intent_classify)
    g.add_node("nl_answer", _nl_answer)
    g.add_node("clarify", _clarify)
    g.add_node("router", _router)
    g.add_node("plan", _plan)
    g.add_node("generate", _generate)
    g.add_node("validate", _validate)
    g.set_entry_point("intent_entry")
    g.add_conditional_edges("intent_entry", _route_intent_entry)
    g.add_conditional_edges("intent_classify", _route_after_intent)
    g.add_edge("nl_answer", END)
    g.add_conditional_edges("clarify", _route_after_clarify)
    g.add_conditional_edges("router", _route_after_router)
    g.add_edge("plan", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", _route_after_validate)
    return g.compile()


_graph = None


def _initial_state(prompt: str, chat_history: list[MemoryTurn] | None = None) -> GraphState:
    return {
        "prompt": prompt,
        "chat_history": (chat_history or [])[-CHAT_HISTORY_TURNS:],
        "intent": "",
        "nl_response": "",
        "route": "",
        "needs_clarification": False,
        "clarification_question": "",
        "plan": "",
        "raw_output": "",
        "bindings": {},
        "validation_error": "",
        "attempts": 0,
    }

def run_agent_with_trace(prompt: str, chat_history: list[MemoryTurn] | None = None) -> GraphState:
    global _graph
    if _graph is None:
        _graph = build_graph()
    initial: GraphState = _initial_state(prompt, chat_history=chat_history)
    final: GraphState = _graph.invoke(initial)
    return final


def run_agent_stream_with_trace(
    prompt: str, chat_history: list[MemoryTurn] | None = None
):
    global _graph
    if _graph is None:
        _graph = build_graph()
    initial: GraphState = _initial_state(prompt, chat_history=chat_history)
    yield from _graph.stream(initial, stream_mode="values")


def run_agent(prompt: str, chat_history: list[MemoryTurn] | None = None) -> dict[str, str]:
    final = run_agent_with_trace(prompt, chat_history=chat_history)
    return dict(final.get("bindings") or {})

