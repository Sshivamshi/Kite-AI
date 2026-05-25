"""
Kite AI — main Streamlit entry point
====================================
Stage 0 (InputGuard) → Stage 1 (LLM) → Stage 2 (IR builder)

Run:  streamlit run main.py
"""

from __future__ import annotations

import streamlit as st

from config import (
    DEFAULT_MODEL,
    FAMILY_LABELS,
    MAX_CLARIFICATION_ROUNDS,
    OLLAMA_LOCAL_HOST,
    format_scope_selection_llm,
    format_scope_selection_merchant,
    format_stage2_catalog_reply,
    get_store_context,
    get_cart_attribute_choices,
    get_collection_choices,
    get_currency_choices,
    get_customer_segment_choices,
    get_customer_tag_choices,
    get_gift_product_choices,
    get_market_choices,
    get_product_choices,
    get_trigger_scope_choices,
    humanize_confirmed_fields,
    infer_stage2_widget_type,
    TIER_BEHAVIOR_CHOICES,
)
from stage0 import (
    SCOPE_CORPUS,
    SCOPE_EMBEDDING_MODEL,
    SCOPE_THRESHOLD,
    bypass_guard_result,
    check_prompt,
)
from stage1 import (
    Stage1Classifier,
    format_clarify_message,
    format_pass_message,
    format_unsupported_message,
    get_draft_prompt,
    session_from_dict as stage1_session_from_dict,
    session_to_dict as stage1_session_to_dict,
)
from stage2 import (
    Stage2Classifier,
    format_clarifying_message,
    format_complete_message,
    session_from_dict as stage2_session_from_dict,
    session_to_dict as stage2_session_to_dict,
)

st.set_page_config(
    page_title="Kite AI — Promotion Pipeline",
    page_icon="🛍️",
    layout="wide",
)


def _get_stage2_payload() -> dict | None:
    """Return validated Stage 2 dict; clear legacy string payloads from old sessions."""
    raw = st.session_state.get("stage2_payload")
    if isinstance(raw, dict) and raw.get("prompt"):
        return raw

    session = st.session_state.get("stage1_session") or {}
    nested = session.get("stage2_payload")
    if isinstance(nested, dict) and nested.get("prompt"):
        st.session_state.stage2_payload = nested
        return nested

    if isinstance(raw, str):
        st.session_state.stage2_payload = None
    return None


def _get_stage1_handoff() -> dict | None:
    """Stage 1 pass payload: prompt + family + family_label → feeds Stage 2."""
    return _get_stage2_payload()


def _clear_stage2() -> None:
    st.session_state.stage2_session = None
    st.session_state.stage2_chat = []
    st.session_state.stage2_ir = None
    st.session_state.stage2_last_status = None
    st.session_state.stage2_last_result = None
    st.session_state.stage2_pick_reply = None


def _append_stage2_assistant(chat: list, turn) -> None:
    if turn.status == "clarifying":
        chat.append({
            "role": "assistant",
            "content": format_clarifying_message(turn.result),
            "status": "clarifying",
            "result": turn.result,
        })
    elif turn.status == "complete":
        family = (turn.ir or {}).get("feature") or ""
        chat.append({
            "role": "assistant",
            "content": format_complete_message(family),
            "status": "complete",
            "ir": turn.ir,
        })
    else:
        chat.append({
            "role": "assistant",
            "content": f"Error: {turn.error}",
            "status": "error",
        })


def _start_stage2(stage1_payload: dict) -> None:
    classifier = Stage2Classifier()
    session = classifier.new_session(stage1_payload)
    prompt = stage1_payload.get("prompt", "")
    chat = [{"role": "user", "content": prompt, "kind": "stage1_prompt"}]
    with st.spinner("Stage 2 building IR…"):
        turn = classifier.process_turn(session, prompt)
    _append_stage2_assistant(chat, turn)
    st.session_state.stage2_session = stage2_session_to_dict(session)
    st.session_state.stage2_chat = chat
    st.session_state.stage2_last_status = turn.status
    st.session_state.stage2_last_result = turn.result if turn.status == "clarifying" else None
    st.session_state.stage2_ir = turn.ir if turn.status == "complete" else None


def _send_stage2_message(user_text: str, catalog_resolution: str | None = None) -> None:
    session = stage2_session_from_dict(st.session_state.stage2_session)
    classifier = Stage2Classifier()
    chat = st.session_state.stage2_chat
    chat.append({"role": "user", "content": user_text})
    with st.spinner("Stage 2 thinking…"):
        turn = classifier.process_turn(session, user_text, catalog_resolution=catalog_resolution)
    _append_stage2_assistant(chat, turn)
    st.session_state.stage2_session = stage2_session_to_dict(session)
    st.session_state.stage2_chat = chat
    st.session_state.stage2_last_status = turn.status
    st.session_state.stage2_last_result = turn.result if turn.status == "clarifying" else session.last_result
    if turn.status == "complete":
        st.session_state.stage2_ir = turn.ir


def _render_stage2_question_widget(q: dict, idx: int, key_prefix: str) -> tuple[str, str] | None:
    """Render one catalog-backed widget; return (merchant line, LLM resolution) or None."""
    qid = q.get("id") or f"q{idx}"
    qtext = q.get("question") or f"Question {idx + 1}"
    widget = infer_stage2_widget_type(qid, qtext)
    base = f"{key_prefix}_{qid}_{idx}"

    st.markdown(f"**{qtext}**")

    if widget == "tier_behavior":
        labels = [c["label"] for c in TIER_BEHAVIOR_CHOICES]
        pick = st.radio("Choose one", labels, key=f"{base}_radio", label_visibility="collapsed")
        choice = next(c for c in TIER_BEHAVIOR_CHOICES if c["label"] == pick)
        merchant = f"When multiple tiers match: {pick.split(' — ')[0]}"
        return merchant, f"tier_behavior: {choice['id']}"

    if widget == "currency":
        opts = get_currency_choices()
        pick = st.selectbox("Currency", [o["label"] for o in opts], key=f"{base}_cur")
        code = next(o["id"] for o in opts if o["label"] == pick)
        return f"Currency: {pick}", f"currency: {code}"

    if widget == "market":
        opts = get_market_choices()
        pick = st.selectbox("Country / market", [o["label"] for o in opts], key=f"{base}_mkt")
        code = next(o["id"] for o in opts if o["label"] == pick)
        if code == "ALL":
            return "Available in: all countries", "customer_eligibility market: all markets (no restriction)"
        return f"Available in: {pick}", f"customer_eligibility market: {code}"

    if widget == "customer_tags":
        opts = get_customer_tag_choices()
        pick = st.selectbox("Who qualifies", [o["label"] for o in opts], key=f"{base}_tag")
        val = next(o["id"] for o in opts if o["label"] == pick)
        if val == "ALL":
            return "Who qualifies: all customers", "customer_eligibility: all customers (empty tag list)"
        if val == "logged_in":
            return "Who qualifies: logged-in customers only", "customer_eligibility: logged_in customers only"
        return f"Who qualifies: {pick}", f"customer_eligibility customer_tag: {val}"

    if widget == "customer_segments":
        opts = get_customer_segment_choices()
        pick = st.selectbox("Customer segment", [o["label"] for o in opts], key=f"{base}_seg")
        val = next(o["id"] for o in opts if o["label"] == pick)
        if val == "ALL":
            return "Customer segment: all segments", "customer_eligibility: no customer_segment restriction"
        return f"Customer segment: {pick}", f"customer_eligibility customer_segment: {val}"

    if widget == "cart_attributes":
        opts = get_cart_attribute_choices()
        pick = st.selectbox("Cart requirements", [o["label"] for o in opts], key=f"{base}_cartattr")
        val = next(o["id"] for o in opts if o["label"] == pick)
        if val == "NONE":
            return "Cart requirements: none", "cart_attributes: none (no special cart requirements)"
        return f"Cart must include: {pick}", f"cart_attributes: {val}"

    if widget == "gift_products":
        opts = get_gift_product_choices()
        pick = st.selectbox("Free gift product", [o["label"] for o in opts], key=f"{base}_gift")
        pid = next(o["id"] for o in opts if o["label"] == pick)
        title = next(o["title"] for o in opts if o["id"] == pid)
        return f"Free gift: {title}", f"gift_product: {title} ({pid})"

    if widget == "collection":
        opts = get_collection_choices()
        pick = st.selectbox("Collection", [o["label"] for o in opts], key=f"{base}_coll")
        cid = next(o["id"] for o in opts if o["label"] == pick)
        title = next(o["title"] for o in opts if o["id"] == cid)
        return f"Applies to collection: {title}", f"trigger_scope: collection {title} ({cid})"

    if widget == "trigger_scope":
        scope_opts = get_trigger_scope_choices()
        labels = [o["label"] for o in scope_opts]
        pick_label = st.selectbox("Apply promotion to", labels, key=f"{base}_scope")
        scope_key = next(o["key"] for o in scope_opts if o["label"] == pick_label)

        picked_ids: list[str] = []
        if scope_key == "pick_products":
            all_prods = get_product_choices()
            picked = st.multiselect(
                "Select product(s)",
                options=[p["id"] for p in all_prods],
                format_func=lambda pid: next(p["title"] for p in all_prods if p["id"] == pid),
                key=f"{base}_prods",
            )
            picked_ids = list(picked)
            if not picked_ids:
                st.caption("Pick at least one product.")
                return None
        merchant = format_scope_selection_merchant(scope_key, picked_ids or None)
        llm = format_scope_selection_llm(scope_key, picked_ids or None)
        return merchant, llm

    options = q.get("options") or []
    if not options:
        return None
    pick = st.radio("Choose one", options, key=f"{base}_opt", label_visibility="collapsed")
    return pick, f"{qid}: {pick}"


def _render_stage2_catalog_form(last_result: dict | None, key_prefix: str) -> None:
    if not last_result or last_result.get("status") != "clarifying":
        return
    questions = last_result.get("questions") or []
    if not questions:
        return

    confirmed = last_result.get("confirmed_fields") or {}
    if confirmed:
        st.markdown("**What we know so far**")
        for line in humanize_confirmed_fields(confirmed):
            st.markdown(f"- {line}")

    with st.expander("Your store catalog", expanded=False):
        st.caption(
            "Products, categories, collections, markets, customer groups, segments, "
            "cart attributes, and currencies you can reference when answering Stage 2 questions."
        )
        st.json(get_store_context())

    st.markdown("**Answer with pickers** *(products, categories, markets, customer groups)*")

    with st.form(key=f"{key_prefix}_form", clear_on_submit=True):
        merchant_lines: list[str] = []
        llm_lines: list[str] = []
        for i, q in enumerate(questions):
            pair = _render_stage2_question_widget(q, i, key_prefix)
            if pair:
                merchant_lines.append(pair[0])
                llm_lines.append(pair[1])
        submitted = st.form_submit_button("Submit selections", type="primary", use_container_width=True)

    if submitted:
        if len(merchant_lines) < len(questions):
            st.warning("Complete all pickers above first.")
        else:
            merchant_msg, llm_note = format_stage2_catalog_reply(merchant_lines, llm_lines)
            st.session_state.stage2_pick_reply = merchant_msg
            st.session_state.stage2_pick_resolution = llm_note
            st.rerun()


def _maybe_start_stage2() -> None:
    handoff = _get_stage1_handoff()
    if handoff and not st.session_state.get("stage2_session"):
        _start_stage2(handoff)


def _stage1_status_label(status: str, last_verdict: str | None) -> str:
    if status == "passed":
        return "Ready for Stage 2"
    if status == "rejected":
        return "Unsupported"
    if last_verdict == "clarify":
        return "In conversation"
    return "Active"


def _render_chat_message(msg: dict, *, show_draft_action: bool = False, action_key: str = "") -> None:
    st.write(msg["content"])

    result = msg.get("result") or {}
    verdict = msg.get("verdict")

    if verdict == "clarify":
        draft = get_draft_prompt(result)
        if draft:
            family = (result.get("inferred_family") or "").strip()
            label = FAMILY_LABELS.get(family, "") if family else ""
            caption = "Draft promotion"
            if label:
                caption += f" · {label}"
            st.caption(caption + " *(updates each turn)*")
            st.code(draft, language=None)
            if show_draft_action and st.button(
                "Use this draft",
                key=f"{action_key}_use_draft",
                use_container_width=True,
            ):
                st.session_state.stage1_quick_reply = "yes"
                st.rerun()

    elif verdict == "unsupported":
        draft = get_draft_prompt(result) or (result.get("supported_alternative") or "").strip()
        if draft:
            st.caption("Suggested alternative")
            st.code(draft, language=None)


def _append_stage1_assistant(chat: list, turn) -> None:
    if turn.verdict == "clarify":
        chat.append({
            "role": "assistant",
            "content": format_clarify_message(turn.result),
            "verdict": "clarify",
            "result": turn.result,
        })
    elif turn.verdict == "unsupported":
        chat.append({
            "role": "assistant",
            "content": format_unsupported_message(turn.result),
            "verdict": "unsupported",
            "result": turn.result,
        })
    elif turn.verdict == "pass":
        payload = turn.stage2_payload or {}
        chat.append({
            "role": "assistant",
            "content": format_pass_message(payload) if payload else "Promotion accepted.",
            "verdict": "pass",
            "stage2_payload": payload,
        })
    else:
        chat.append({
            "role": "assistant",
            "content": f"Error: {turn.error}",
            "verdict": "error",
        })


def _start_stage1(stage0_prompt: str) -> None:
    classifier = Stage1Classifier()
    session = classifier.new_session(stage0_prompt)
    chat = [{"role": "user", "content": stage0_prompt}]
    with st.spinner("Stage 1 thinking…"):
        turn = classifier.process_turn(session, stage0_prompt)
    _append_stage1_assistant(chat, turn)
    st.session_state.stage1_session = stage1_session_to_dict(session)
    st.session_state.stage1_chat = chat
    st.session_state.stage1_last_verdict = turn.verdict
    st.session_state.stage2_payload = turn.stage2_payload if turn.verdict == "pass" else None
    if turn.verdict == "pass":
        _clear_stage2()
        if turn.stage2_payload:
            _start_stage2(turn.stage2_payload)


def _send_stage1_message(user_text: str) -> None:
    session = stage1_session_from_dict(st.session_state.stage1_session)
    classifier = Stage1Classifier()
    chat = st.session_state.stage1_chat
    chat.append({"role": "user", "content": user_text})
    with st.spinner("Stage 1 thinking…"):
        turn = classifier.process_turn(session, user_text)
    _append_stage1_assistant(chat, turn)
    st.session_state.stage1_session = stage1_session_to_dict(session)
    st.session_state.stage1_chat = chat
    st.session_state.stage1_last_verdict = turn.verdict
    if turn.verdict == "pass":
        st.session_state.stage2_payload = turn.stage2_payload
        _clear_stage2()
        if turn.stage2_payload:
            _start_stage2(turn.stage2_payload)


# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("guard_results", None),
    ("last_guard_result", None),
    ("last_prompt", ""),
    ("stage1_session", None),
    ("stage1_chat", []),
    ("stage1_last_verdict", None),
    ("stage2_payload", None),
    ("stage1_quick_reply", None),
    ("stage2_session", None),
    ("stage2_chat", []),
    ("stage2_ir", None),
    ("stage2_last_status", None),
    ("stage2_last_result", None),
    ("stage2_pick_reply", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🛍️ Kite AI — Promotion Pipeline")
st.caption("Stage 0 guard → Stage 1 classifier → Stage 2 IR builder")

# ── Stage 0 ────────────────────────────────────────────────────────────────
st.subheader("🛡️ Stage 0 — Input guard")

bypass_stage0 = st.checkbox(
    "Skip Stage 0 guard (send directly to Stage 1)",
    value=st.session_state.get("bypass_stage0", False),
    help=(
        "Bypasses embedding scope checks and injection patterns. Use when Ollama "
        f"embeddings are unavailable on this server (run `ollama pull {SCOPE_EMBEDDING_MODEL}` locally to enable the guard)."
    ),
    key="bypass_stage0",
)

if bypass_stage0:
    st.info(
        "Stage 0 guard is **off** — prompts go straight to Stage 1. "
        "Only basic length checks (5–1000 characters) still apply."
    )

scope_threshold = st.slider(
    "Scope threshold",
    min_value=0.40,
    max_value=0.90,
    value=SCOPE_THRESHOLD,
    step=0.05,
    help="Minimum cosine similarity to accept a prompt as promotion-related.",
    disabled=bypass_stage0,
)

user_input = st.text_area(
    "Describe your promotion",
    height=100,
    placeholder="e.g. Spend $100 and get a free gift",
)

col_send, col_clear = st.columns([1, 5])
with col_send:
    send_label = "Send to Stage 1" if bypass_stage0 else "Run Stage 0"
    send = st.button(send_label, type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear Stage 0"):
        st.session_state.last_guard_result = None
        st.session_state.last_prompt = ""

if send and user_input.strip():
    with st.spinner("InputGuard…" if not bypass_stage0 else "Sending to Stage 1…"):
        if bypass_stage0:
            result = bypass_guard_result(user_input.strip())
        else:
            result = check_prompt(user_input.strip(), scope_threshold=scope_threshold)
        st.session_state.last_guard_result = result
        st.session_state.last_prompt = user_input.strip()
        if result.passed:
            st.session_state.stage2_payload = None
            _clear_stage2()
            st.session_state.stage1_session = None
            st.session_state.stage1_chat = []
            _start_stage1(user_input.strip())

if st.session_state.last_guard_result is not None:
    res = st.session_state.last_guard_result
    prompt = st.session_state.last_prompt

    if res.passed:
        if res.scope_backend == "bypassed":
            st.warning("**BYPASS** — Stage 0 skipped; sent directly to Stage 1")
        else:
            st.success(
                f"**PASS** — sent to Stage 1  ·  "
                f"best cosine **{res.scope_score:.0%}** (≥ {scope_threshold:.0%})"
            )
    else:
        verdict = res.rejection_type or "blocked"
        if verdict == "out_of_scope":
            st.warning(
                f"**OUT OF SCOPE** — best cosine **{res.scope_score:.0%}** "
                f"< threshold **{scope_threshold:.0%}**"
            )
        elif verdict == "injection":
            st.error(f"**INJECTION** — pattern `{res.rejection_label}`")
        else:
            st.error(f"**{verdict.upper()}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdict", "bypass" if res.scope_backend == "bypassed" else ("pass" if res.passed else res.rejection_type))
    m2.metric("Scope score", "—" if res.scope_backend == "bypassed" else (res.scope_score if res.scope_score is not None else "—"))
    m3.metric("Embeddings", "bypassed" if res.scope_backend == "bypassed" else (res.scope_backend or "—"))
    m4.metric("Latency", f"{res.latency_ms} ms")

    if res.rejection_reason:
        st.info(res.rejection_reason)

    if not res.embedding_available and not bypass_stage0 and res.scope_backend != "bypassed":
        st.error(
            f"Ollama embeddings not available — ensure Ollama is running at "
            f"`{OLLAMA_LOCAL_HOST}` and run `ollama pull {SCOPE_EMBEDDING_MODEL}`, "
            "or enable **Skip Stage 0 guard** above."
        )

    if res.scope_best_match:
        with st.expander("Why this score?"):
            st.write(
                f"**Best match** among {len(SCOPE_CORPUS)} corpus sentences — "
                f"cosine **{res.scope_score:.0%}** (need ≥ **{scope_threshold:.0%}**):"
            )
            st.code(res.scope_best_match)

st.divider()

# ── Stage 1 — LLM chat ─────────────────────────────────────────────────────
st.subheader("🤖 Stage 1 — Promotion chat")

if not st.session_state.stage1_session:
    st.caption(
        "Pass Stage 0, or enable **Skip Stage 0 guard** and submit your prompt, to start Stage 1."
    )
else:
    session = st.session_state.stage1_session

    status_label = _stage1_status_label(
        session.get("status", "active"),
        st.session_state.stage1_last_verdict,
    )

    s1, s2, s3 = st.columns(3)
    s1.metric("Status", status_label)
    s2.metric("Turns", session.get("clarification_rounds", 0))
    s3.metric("Last verdict", st.session_state.stage1_last_verdict or "—")

    if st.session_state.get("stage1_quick_reply"):
        reply = st.session_state.pop("stage1_quick_reply")
        _send_stage1_message(reply)
        st.rerun()

    chat = st.session_state.stage1_chat
    last_assistant_idx = max(
        (i for i, m in enumerate(chat) if m.get("role") == "assistant"),
        default=-1,
    )

    for i, msg in enumerate(chat):
        with st.chat_message(msg["role"]):
            _render_chat_message(
                msg,
                show_draft_action=(
                    i == last_assistant_idx
                    and msg.get("verdict") == "clarify"
                    and not _get_stage2_payload()
                ),
                action_key=f"chat_{i}",
            )

    if not _get_stage2_payload():
        reply = st.chat_input("Message…")
        if reply:
            _send_stage1_message(reply.strip())
            st.rerun()

    if _get_stage1_handoff():
        st.divider()
        st.success("Stage 1 complete — handoff ready")
        payload = _get_stage1_handoff()
        st.metric("Promotion type", payload.get("family_label", "—"))
        with st.expander("Stage 1 → Stage 2 handoff JSON"):
            st.json(payload)

st.divider()

# ── Stage 2 — IR builder chat ───────────────────────────────────────────────
st.subheader("🏗️ Stage 2 — IR builder")

if not _get_stage1_handoff():
    st.caption("Complete Stage 1 to start building the promotion IR.")
else:
    _maybe_start_stage2()

    if st.session_state.get("stage2_session"):
        s2session = st.session_state.stage2_session

        if st.session_state.get("stage2_pick_reply"):
            reply = st.session_state.pop("stage2_pick_reply")
            resolution = st.session_state.pop("stage2_pick_resolution", None)
            _send_stage2_message(reply, catalog_resolution=resolution)
            st.rerun()

        s1, s2, s3 = st.columns(3)
        s1.metric("Status", "Complete" if st.session_state.stage2_ir else "Building IR")
        s2.metric("Turns", s2session.get("turns", 0))
        family_id = s2session.get("family", "—")
        s3.metric("Promotion type", FAMILY_LABELS.get(family_id, family_id))

        for msg in st.session_state.stage2_chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        last_result = (
            st.session_state.get("stage2_last_result")
            or s2session.get("last_result")
            or {}
        )

        if not st.session_state.stage2_ir:
            _render_stage2_catalog_form(last_result, key_prefix="stage2_active")
            st.caption("Or type a custom answer below if none of the pickers fit.")
            reply2 = st.chat_input("Custom answer…", key="stage2_chat_input")
            if reply2:
                _send_stage2_message(reply2.strip())
                st.rerun()

        if st.session_state.stage2_ir:
            st.divider()
            st.success("Stage 2 complete — promotion IR ready")
            st.json(st.session_state.stage2_ir)

st.divider()

with st.expander("Pipeline reference"):
    st.markdown(f"""
**Stage 0 checks (in order):**
1. Length — min 5 / max 1000 characters
2. Injection — compiled regex patterns *(skipped when bypass enabled)*
3. Scope — embed prompt, max cosine vs all **{len(SCOPE_CORPUS)}** corpus sentences *(skipped when bypass enabled)*

**Bypass:** enable *Skip Stage 0 guard* when `ollama pull {SCOPE_EMBEDDING_MODEL}` is not available on the server.

**Stage 1:** `{DEFAULT_MODEL}` — classifies promotion family; outputs handoff JSON.

**Stage 2:** `{DEFAULT_MODEL}` — asks plain-language questions about your store catalog;
maps answers to the promotion IR internally until complete.
    """)
