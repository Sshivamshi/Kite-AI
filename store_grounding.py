"""
Stage 2 — Store Grounding
=========================
Maps Stage 1 LLM hints to store catalog IDs using fuzzy matching against
product_taxonomy / collection_product_index.

Stage 2 only asks for IR fields the merchant did NOT clearly specify.
Values extracted from the prompt (currency, amounts, thresholds) are auto-applied.
Catalog refs are auto-applied on a single confident match; ambiguous or broad
terms (e.g. "shirts") trigger multi-select UI. Market and customer tags are
asked only when eligibility requires them but the value is missing or vague.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

_CATALOG_PATH = Path(__file__).resolve().parent / "store_catalog.json"

with open(_CATALOG_PATH, encoding="utf-8") as _f:
    STORE_CATALOG: Dict[str, Any] = json.load(_f)

MATCH_THRESHOLD = 70
AMBIGUITY_GAP = 12

TRIGGER_TYPE_LABELS = {
    "cart_subtotal": "cart spend",
    "cart_quantity": "number of items",
    "collection_quantity": "items from the collection",
    "collection_subtotal": "spend on the collection",
    "product_quantity": "quantity of the product",
    "product_subtotal": "spend on the product",
}

TRIGGER_VALUE_PILLS: Dict[str, List[Dict[str, Any]]] = {
    "cart_subtotal": [
        {"id": "50", "label": "Spend $50", "value": 50},
        {"id": "100", "label": "Spend $100", "value": 100},
        {"id": "150", "label": "Spend $150", "value": 150},
        {"id": "200", "label": "Spend $200", "value": 200},
    ],
    "cart_quantity": [
        {"id": "2", "label": "Buy 2 items", "value": 2},
        {"id": "3", "label": "Buy 3 items", "value": 3},
        {"id": "5", "label": "Buy 5 items", "value": 5},
    ],
    "collection_quantity": [
        {"id": "2", "label": "Buy 2 from collection", "value": 2},
        {"id": "3", "label": "Buy 3 from collection", "value": 3},
    ],
    "product_quantity": [
        {"id": "2", "label": "Buy 2", "value": 2},
        {"id": "3", "label": "Buy 3", "value": 3},
    ],
}

REWARD_VALUE_PILLS: Dict[str, List[Dict[str, Any]]] = {
    "percentage_off": [
        {"id": "10", "label": "10% off", "value": 10},
        {"id": "15", "label": "15% off", "value": 15},
        {"id": "20", "label": "20% off", "value": 20},
        {"id": "25", "label": "25% off", "value": 25},
        {"id": "50", "label": "50% off", "value": 50},
    ],
    "fixed_amount_off": [
        {"id": "10", "label": "$10 off", "value": 10},
        {"id": "25", "label": "$25 off", "value": 25},
        {"id": "50", "label": "$50 off", "value": 50},
    ],
    "percentage_off_y": [
        {"id": "100", "label": "100% off (free)", "value": 100},
        {"id": "50", "label": "50% off", "value": 50},
    ],
}


@dataclass
class CatalogHit:
    id: str
    title: str
    score: float
    kind: str  # product | collection | customer_tag | currency
    category: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


def _score_term(query: str, term: str) -> float:
    q, t = query.lower().strip(), term.lower().strip()
    if not q or not t:
        return 0.0
    if q == t:
        return 100.0
    if q in t or t in q:
        return 92.0
    if _RAPIDFUZZ:
        return float(_rfuzz.token_set_ratio(q, t))
    return 80.0 if q.split()[0] == t.split()[0] else 0.0


def _best_hits(query: Optional[str], candidates: List[CatalogHit]) -> List[CatalogHit]:
    if not query or not query.strip():
        return []
    scored: List[CatalogHit] = []
    for c in candidates:
        terms = [c.title] + list(c.payload.get("aliases") or [])
        best = max(_score_term(query, term) for term in terms)
        if best >= MATCH_THRESHOLD - 15:
            scored.append(CatalogHit(
                id=c.id, title=c.title, score=best, kind=c.kind,
                category=c.category, payload=c.payload,
            ))
    scored.sort(key=lambda h: (-h.score, h.title))
    return scored


def _is_ambiguous(hits: List[CatalogHit]) -> bool:
    if len(hits) < 2:
        return False
    return hits[0].score - hits[1].score < AMBIGUITY_GAP and hits[1].score >= MATCH_THRESHOLD


def _all_products() -> List[CatalogHit]:
    out: List[CatalogHit] = []
    for cat_name, cat_data in STORE_CATALOG["product_taxonomy"].items():
        for prod in cat_data["products"]:
            out.append(CatalogHit(
                id=prod["id"], title=prod["title"], score=0.0, kind="product",
                category=cat_name,
                payload={"aliases": prod.get("aliases", []), "collection_id": cat_data["collection"]["id"]},
            ))
    return out


def _products_in_category(category: str) -> List[CatalogHit]:
    cat = STORE_CATALOG["product_taxonomy"].get(category)
    if not cat:
        return []
    return [
        CatalogHit(
            id=p["id"], title=p["title"], score=0.0, kind="product",
            category=category,
            payload={"aliases": p.get("aliases", []), "collection_id": cat["collection"]["id"]},
        )
        for p in cat["products"]
    ]


def _gift_products() -> List[CatalogHit]:
    return _products_in_category("Gift Items")


def _all_collections() -> List[CatalogHit]:
    return [
        CatalogHit(
            id=cid, title=meta["title"], score=0.0, kind="collection",
            payload={"aliases": meta.get("aliases", []), "product_ids": meta.get("product_ids", [])},
        )
        for cid, meta in STORE_CATALOG["collection_product_index"].items()
    ]


def _customer_tags() -> List[CatalogHit]:
    return [
        CatalogHit(
            id=t["id"], title=t["title"], score=0.0, kind="customer_tag",
            payload={"aliases": t.get("aliases", [])},
        )
        for t in STORE_CATALOG.get("customer_tag_catalog", [])
    ]


def _currencies() -> List[CatalogHit]:
    return [
        CatalogHit(
            id=c["code"], title=f"{c['label']} ({c['code']})", score=0.0, kind="currency",
            payload=c,
        )
        for c in STORE_CATALOG.get("currency_options", [])
    ]


def _product_by_id(pid: str) -> Optional[Dict[str, Any]]:
    for cat_data in STORE_CATALOG["product_taxonomy"].values():
        for prod in cat_data["products"]:
            if prod["id"] == pid:
                return {**prod, "category": cat_data["collection"]["title"]}
    return None


def _collection_by_id(cid: str) -> Optional[Dict[str, Any]]:
    meta = STORE_CATALOG["collection_product_index"].get(cid)
    if meta:
        return {"id": cid, **meta}
    return None


def _hits_to_options(
    hits: List[CatalogHit],
    field: str,
    recommended_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    seen: set = set()
    for h in hits[:8]:
        if h.id in seen:
            continue
        seen.add(h.id)
        opt: Dict[str, Any] = {
            "id": h.id,
            "label": h.title,
            "field": field,
            "kind": h.kind,
        }
        if h.category:
            opt["subtitle"] = h.category
        if recommended_id and h.id == recommended_id:
            opt["recommended"] = True
        options.append(opt)
    return options


def _detect_currency_from_text(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\brupees?\b|\brs\.?\b|\binr\b|₹", t):
        return "INR"
    if re.search(r"\b(usd|dollars?|\$\d)", t):
        return "USD"
    if re.search(r"\b(eur|euros?|€)\b", t):
        return "EUR"
    if re.search(r"\b(gbp|pounds?|£)\b", t):
        return "GBP"
    return None


def _currency_symbol(code: str) -> str:
    for c in STORE_CATALOG.get("currency_options", []):
        if c["code"] == code:
            return c.get("symbol", code)
    return "$"


def _resolve_selection(field: str, selections: Dict[str, Any]) -> Optional[Any]:
    """Return a resolved value only when the user explicitly picked it."""
    if field not in selections:
        return None
    sel = selections[field]
    if isinstance(sel, dict):
        if sel.get("kind") == "value":
            return sel.get("value", sel.get("id"))
        return sel.get("id") or sel.get("value")
    return sel


def _product_options_for_query(
    query: str,
    field: str,
    fallback_category: Optional[str] = None,
) -> Tuple[List[CatalogHit], Optional[str]]:
    """Return catalog hits + recommended product id for a scope/target query."""
    cat = _infer_category_from_query(query) or fallback_category
    hits = _best_hits(query, _products_in_category(cat)) if cat else []
    if not hits:
        hits = _best_hits(query, _all_products())
    # Broad terms like "shirts" / "apparel" → show full category for user to pick
    if cat and len(hits) <= 2:
        hits = list(_products_in_category(cat))
        hits.sort(key=lambda h: -max(
            _score_term(query, h.title),
            *[_score_term(query, a) for a in h.payload.get("aliases", [])],
        ))
    recommended = hits[0].id if hits else None
    return hits, recommended


def _collection_options_for_query(query: str, field: str) -> Tuple[List[CatalogHit], Optional[str]]:
    hits = _best_hits(query, _all_collections())
    if not hits:
        hits = _all_collections()[:8]
        for h in hits:
            h.score = 50.0
    recommended = hits[0].id if hits else None
    return hits, recommended


def _value_confirm_options(
    field: str,
    suggested: Optional[float],
    pills: List[Dict[str, Any]],
    symbol: str = "",
    unit: str = "",
) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    if suggested is not None:
        label = f"{symbol}{int(suggested) if suggested == int(suggested) else suggested}{unit} (from your prompt)"
        options.append({
            "id": str(suggested),
            "label": label,
            "field": field,
            "kind": "value",
            "value": suggested,
            "recommended": True,
        })
    for p in pills:
        if suggested is not None and p["value"] == suggested:
            continue
        options.append({
            "id": p["id"],
            "label": p["label"],
            "field": field,
            "kind": "value",
            "value": p["value"],
        })
    return options


def _markets() -> List[CatalogHit]:
    return [
        CatalogHit(
            id=m["id"], title=m["title"], score=0.0, kind="market",
            payload={"aliases": m.get("aliases", [])},
        )
        for m in STORE_CATALOG.get("market_catalog", [])
    ]


def _is_broad_catalog_query(query: str, hits: List[CatalogHit]) -> bool:
    """True when the merchant used a category-level term, not one exact product."""
    if not query or not hits:
        return False
    q = query.lower().strip()
    broad = {
        "shirt", "shirts", "apparel", "clothes", "clothing", "accessories",
        "skincare", "products", "items", "gifts", "gift", "electronics",
        "fitness", "collection",
    }
    if q in broad:
        return True
    if len(hits) >= 2 and _is_ambiguous(hits):
        return True
    if len(hits) >= 2 and hits[0].score < 95:
        return True
    return False


def _auto_resolve_catalog(
    field: str,
    hits: List[CatalogHit],
    selections: Dict[str, Any],
    llm_res: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve a catalog field from user pick, LLM id, or single confident match."""
    if field in selections:
        sel = selections[field]
        if isinstance(sel, dict) and sel.get("kind") == "multi":
            items = []
            for pid in sel.get("ids", []):
                prod = _product_by_id(str(pid))
                if prod:
                    items.append({"id": pid, "title": prod["title"], "status": "resolved"})
                else:
                    coll = _collection_by_id(str(pid))
                    if coll:
                        items.append({"id": pid, "title": coll["title"], "status": "resolved", "kind": "collection"})
            return {"multi": items} if items else None
        raw = sel.get("id") if isinstance(sel, dict) else sel
        prod = _product_by_id(str(raw))
        if prod:
            return {"id": str(raw), "title": prod["title"], "status": "resolved"}
        coll = _collection_by_id(str(raw))
        if coll:
            return {"id": str(raw), "title": coll["title"], "status": "resolved", "kind": "collection"}
        tag = next((t for t in STORE_CATALOG.get("customer_tag_catalog", []) if t["id"] == raw), None)
        if tag:
            return {"id": tag["id"], "title": tag["title"], "status": "resolved"}
        mkt = next((m for m in STORE_CATALOG.get("market_catalog", []) if m["id"] == raw), None)
        if mkt:
            return {"id": mkt["id"], "title": mkt["title"], "status": "resolved"}

    llm_id = llm_res.get(field)
    if llm_id and hits:
        for h in hits:
            if h.id == llm_id:
                return {"id": h.id, "title": h.title, "status": "resolved"}

    if hits and len(hits) == 1 and hits[0].score >= MATCH_THRESHOLD:
        return {"id": hits[0].id, "title": hits[0].title, "status": "resolved"}

    if hits and not _is_ambiguous(hits) and hits[0].score >= MATCH_THRESHOLD + 10:
        return {"id": hits[0].id, "title": hits[0].title, "status": "resolved"}

    return None


def _mentions_market(text: str, eligibility_hints: List[Dict[str, Any]]) -> bool:
    blob = text.lower()
    if any(w in blob for w in ("market", "country", "region", "india only", "us only", "uk only")):
        return True
    for eh in eligibility_hints:
        t = (eh.get("type_hint") or "").lower()
        if "market" in t or "country" in t or "region" in t:
            return True
    return False


def _product_clarification_options(
    query: str,
    hits: List[CatalogHit],
    field: str,
    include_collection: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """Build options; use multi_select when query is broad or ambiguous."""
    cat = _infer_category_from_query(query)
    if _is_broad_catalog_query(query, hits) and cat:
        hits = list(_products_in_category(cat))
        hits.sort(key=lambda h: -max(
            _score_term(query, h.title),
            *[_score_term(query, a) for a in h.payload.get("aliases", [])],
        ))
        options = _hits_to_options(hits, field, recommended_id=hits[0].id if hits else None)
        if include_collection and cat:
            coll_id = STORE_CATALOG["product_taxonomy"][cat]["collection"]["id"]
            coll = _collection_by_id(coll_id)
            if coll:
                options.insert(0, {
                    "id": coll_id,
                    "label": f"All {coll['title']}",
                    "subtitle": "collection",
                    "field": field,
                    "kind": "collection",
                })
        return options, "multi_select"
    if _is_ambiguous(hits):
        return _hits_to_options(hits, field, recommended_id=hits[0].id if hits else None), "multi_select"
    return _hits_to_options(hits, field, recommended_id=hits[0].id if hits else None), "cards"


def _write_catalog_resolution(
    result: GroundingResult,
    field: str,
    resolved: Optional[Dict[str, Any]],
) -> None:
    if not resolved:
        return
    if resolved.get("multi"):
        result.resolved[field] = resolved["multi"]
        return
    if field == "reward.y_target":
        kind = resolved.get("kind", "product")
        key = "reward.y_target.collection" if kind == "collection" else "reward.y_target.product"
        result.resolved[key] = resolved
    else:
        result.resolved[field] = resolved


def _apply_scalar_selections(selections: Dict[str, Any], result: GroundingResult) -> None:
    """Apply user-confirmed scalar fields (currency, numeric values)."""
    if "currency" in selections:
        raw = _resolve_selection("currency", selections)
        if raw:
            result.resolved["currency"] = str(raw)
    for fld in ("trigger.value", "reward.value"):
        if fld in selections:
            raw = _resolve_selection(fld, selections)
            if raw is not None:
                result.resolved[fld] = float(raw)


def _make_clarification(
    qid: str,
    field: str,
    question: str,
    options: List[Dict[str, Any]],
    ui_type: str = "cards",
    required: bool = True,
    fallback_message: Optional[str] = None,
) -> Dict[str, Any]:
    q: Dict[str, Any] = {
        "id": qid,
        "field": field,
        "question": question,
        "ui_type": ui_type,
        "options": options,
        "required": required,
    }
    if fallback_message:
        q["fallback_message"] = fallback_message
    return q


def _infer_category_from_query(query: str) -> Optional[str]:
    """Pick the product taxonomy category whose aliases/products best match."""
    best_cat, best_score = None, 0.0
    for cat_name, cat_data in STORE_CATALOG["product_taxonomy"].items():
        terms = [cat_name, cat_data["collection"]["title"]]
        idx = STORE_CATALOG["collection_product_index"].get(cat_data["collection"]["id"], {})
        terms.extend(idx.get("aliases", []))
        for prod in cat_data["products"]:
            terms.append(prod["title"])
            terms.extend(prod.get("aliases", []))
        score = max(_score_term(query, t) for t in terms)
        if score > best_score:
            best_score, best_cat = score, cat_name
    return best_cat if best_score >= MATCH_THRESHOLD - 20 else None


def _eligibility_mentions_tag(hints_eligibility: List[Dict[str, Any]], merchant_text: str) -> bool:
    blob = merchant_text.lower()
    tag_words = ("vip", "wholesale", "gold", "loyalty", "member", "customer group", "eligible")
    if any(w in blob for w in tag_words):
        return True
    for eh in hints_eligibility:
        t = (eh.get("type_hint") or "").lower()
        if "tag" in t or "customer" in t or "vip" in t:
            return True
    return False


@dataclass
class GroundingResult:
    resolved: Dict[str, Any] = field(default_factory=dict)
    clarifications: List[Dict[str, Any]] = field(default_factory=list)
    admin_selections_needed: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    stage2_notes: Optional[str] = None


def apply_user_selections(
    hints: Any,
    family: str,
    currency: str,
    selections: Dict[str, Any],
    merchant_text: str = "",
    trigger_type: Optional[str] = None,
) -> GroundingResult:
    """Re-run grounding with user picks from clarification UI."""
    return run_store_grounding(
        family=family,
        hints=hints,
        currency=currency,
        merchant_text=merchant_text,
        user_selections=selections,
        skip_missing_checks_for_fields=set(selections.keys()),
        trigger_type=trigger_type,
    )


def run_store_grounding(
    family: str,
    hints: Any,
    currency: str,
    merchant_text: str = "",
    user_selections: Optional[Dict[str, Any]] = None,
    skip_missing_checks_for_fields: Optional[set] = None,
    stage2_llm_resolutions: Optional[Dict[str, Any]] = None,
    trigger_type: Optional[str] = None,
) -> GroundingResult:
    """
    Stage 2 — fill IR gaps only.

    Auto-applies values the merchant already specified (currency from rs/rupees,
    numeric thresholds, discount amounts). Asks only for missing or ambiguous
    catalog refs: products, collections, customer tags, markets.
    """
    selections = dict(user_selections or {})
    skip = skip_missing_checks_for_fields or set()
    llm_res = stage2_llm_resolutions or {}
    result = GroundingResult()
    schema = STORE_CATALOG.get("clarification_schema", {})
    mapped_trigger = trigger_type or (hints.trigger_type_hint or "cart_subtotal")
    eligibility = hints.customer_eligibility_hints or []

    _apply_scalar_selections(selections, result)

    # ── Currency: auto from prompt text, never ask if detected ───────────────
    if "currency" not in result.resolved:
        detected = _detect_currency_from_text(merchant_text)
        if detected:
            result.resolved["currency"] = detected
            result.assumptions.append(f"Currency inferred from prompt: {detected}")
        elif currency and "currency" not in selections:
            result.resolved["currency"] = currency

    sym = _currency_symbol(result.resolved.get("currency") or currency)

    # ── Trigger / reward values: auto from Stage 1 hints ─────────────────────
    if hints.trigger_value_hint is not None and "trigger.value" not in result.resolved:
        result.resolved["trigger.value"] = hints.trigger_value_hint
    if hints.reward_value_hint is not None and "reward.value" not in result.resolved:
        result.resolved["reward.value"] = hints.reward_value_hint

    def _ask_or_resolve_catalog(
        field: str,
        query: str,
        hits: List[CatalogHit],
        qid: str,
        question: str,
        *,
        fallback: Optional[str] = None,
        include_collection: bool = True,
    ) -> None:
        if field in skip or field in result.resolved:
            return
        if field == "reward.y_target" and (
            "reward.y_target.product" in result.resolved
            or "reward.y_target.collection" in result.resolved
        ):
            return
        resolved = _auto_resolve_catalog(field, hits, selections, llm_res)
        if resolved and not resolved.get("multi"):
            if not _is_broad_catalog_query(query, hits) and not _is_ambiguous(hits):
                _write_catalog_resolution(result, field, resolved)
                return
        if not query and not hits:
            return
        options, ui_type = _product_clarification_options(
            query, hits, field, include_collection=include_collection,
        )
        if not options:
            result.admin_selections_needed.append(f"{field} — no catalog match for {query!r}")
            return
        result.clarifications.append(_make_clarification(
            qid, field, question, options, ui_type=ui_type, fallback_message=fallback,
        ))

    # ── Free gift product ────────────────────────────────────────────────────
    if family == "free_gift":
        gift_query = hints.reward_target_hint or ""
        if not gift_query or gift_query.lower() in ("free gift", "gift", "freebie"):
            if "reward.gift_product" not in skip:
                tpl = schema.get("missing_gift_product", {})
                result.clarifications.append(_make_clarification(
                    "missing_gift_product",
                    "reward.gift_product",
                    tpl.get("question_template", "Which product should be the free gift?"),
                    _hits_to_options(_gift_products(), "reward.gift_product"),
                    fallback_message=tpl.get("fallback_message"),
                ))
        else:
            hits, _ = _product_options_for_query(gift_query, "reward.gift_product", "Gift Items")
            tpl = schema.get("missing_gift_product", {})
            _ask_or_resolve_catalog(
                "reward.gift_product", gift_query, hits or _gift_products(),
                "missing_gift_product",
                tpl.get("question_template", "Which product should be the free gift?"),
                fallback=tpl.get("fallback_message"),
                include_collection=False,
            )

    # ── Trigger scope (product/collection triggers only) ─────────────────────
    scope_hint = hints.trigger_scope_hint
    scope_needs_product = mapped_trigger in ("product_quantity", "product_subtotal")
    scope_needs_collection = mapped_trigger in ("collection_quantity", "collection_subtotal")

    if scope_hint and scope_needs_product:
        hits, _ = _product_options_for_query(scope_hint, "trigger.scope.productRef", "Apparel")
        tpl = schema.get("ambiguous_trigger_product", {})
        _ask_or_resolve_catalog(
            "trigger.scope.productRef", scope_hint, hits,
            "ambiguous_trigger_product",
            tpl.get("question_template", "Which product should trigger this promotion?"),
        )

    if scope_hint and scope_needs_collection:
        hits, _ = _collection_options_for_query(scope_hint)
        tpl = schema.get("ambiguous_trigger_collection", {})
        resolved = _auto_resolve_catalog("trigger.scope.collectionRef", hits, selections, llm_res)
        if resolved and not _is_ambiguous(hits):
            _write_catalog_resolution(result, "trigger.scope.collectionRef", resolved)
        elif "trigger.scope.collectionRef" not in skip:
            options = _hits_to_options(hits, "trigger.scope.collectionRef", hits[0].id if hits else None)
            ui = "multi_select" if _is_ambiguous(hits) or len(hits) > 1 else "cards"
            result.clarifications.append(_make_clarification(
                "ambiguous_trigger_collection",
                "trigger.scope.collectionRef",
                tpl.get("question_template", "Which collection should trigger this promotion?"),
                options, ui_type=ui,
            ))

    # ── Missing trigger value (only when not in prompt) ──────────────────────
    if hints.trigger_value_hint is None and "trigger.value" not in skip and "trigger.value" not in result.resolved:
        tpl = schema.get("missing_trigger_value", {})
        label = TRIGGER_TYPE_LABELS.get(mapped_trigger, mapped_trigger.replace("_", " "))
        question = tpl.get("question_template", "What is the minimum threshold?").format(
            trigger_type_label=label,
        )
        pills = TRIGGER_VALUE_PILLS.get(mapped_trigger, TRIGGER_VALUE_PILLS["cart_subtotal"])
        options = [
            {"id": p["id"], "label": p["label"].replace("$", sym), "field": "trigger.value", "kind": "value", "value": p["value"]}
            for p in pills
        ]
        result.clarifications.append(_make_clarification(
            "missing_trigger_value", "trigger.value", question, options, ui_type="pills",
        ))

    # ── Buy X Get Y — Y target & missing reward value ────────────────────────
    if family == "buy_x_get_y":
        y_hint = hints.reward_target_hint or (
            hints.trigger_scope_hint if mapped_trigger in ("cart_subtotal", "cart_quantity") else None
        )
        if y_hint:
            hits, _ = _product_options_for_query(y_hint, "reward.y_target", "Apparel")
            if not hits:
                hits, _ = _collection_options_for_query(y_hint)
            tpl = schema.get("ambiguous_y_target", {})
            _ask_or_resolve_catalog(
                "reward.y_target", y_hint, hits,
                "ambiguous_y_target",
                tpl.get("question_template", "Which product or collection receives the discount?"),
            )

        reward_type = (hints.reward_type_hint or "").lower()
        needs_val = any(x in reward_type for x in ("percent", "fixed", "amount", "off"))
        if hints.reward_value_hint is None and needs_val and "reward.value" not in skip:
            is_fixed = any(x in reward_type for x in ("fixed", "amount", "rupee"))
            rt_key = "fixed_amount_off" if is_fixed else "percentage_off_y"
            tpl = schema.get("missing_reward_value", {})
            pills = REWARD_VALUE_PILLS.get(rt_key, REWARD_VALUE_PILLS["percentage_off"])
            options = [
                {"id": p["id"], "label": p["label"].replace("$", sym), "field": "reward.value", "kind": "value", "value": p["value"]}
                for p in pills
            ]
            result.clarifications.append(_make_clarification(
                "missing_reward_value", "reward.value",
                tpl.get("question_template", "What discount should the customer receive?"),
                options, ui_type="pills",
            ))

    # ── Tiered discount gaps ───────────────────────────────────────────────
    if family == "tiered_discount":
        tier_count = len(hints.tier_hints or [])
        if tier_count < 2 and "tier.1" not in skip:
            tpl = schema.get("ambiguous_tier_count", {})
            result.clarifications.append({
                "id": "ambiguous_tier_count",
                "field": "tier.1",
                "question": tpl.get("question_template", "Tiered discounts need at least 2 thresholds."),
                "ui_type": "text_hint",
                "options": [],
                "required": True,
                "example_question": tpl.get("example_question"),
            })

    # ── Customer tags — ask only when missing or ambiguous ─────────────────
    for i, eh in enumerate(eligibility):
        field = f"customer_eligibility.{i}.value"
        e_type = (eh.get("type_hint") or "").lower()
        if "market" in e_type or "country" in e_type:
            continue
        val_hint = eh.get("value_hint") or ""
        tag_hits = _best_hits(val_hint, _customer_tags()) if val_hint else []
        resolved = _auto_resolve_catalog(field, tag_hits, selections, llm_res)
        if resolved and val_hint:
            result.resolved[field] = resolved
        elif field not in skip and field not in result.resolved:
            tpl = schema.get("missing_customer_tag", {})
            hits = tag_hits or _customer_tags()
            ui = "multi_select" if _is_ambiguous(hits) else "cards"
            result.clarifications.append(_make_clarification(
                f"missing_customer_tag_{i}", field,
                tpl.get("question_template", "Which customer group should qualify?"),
                _hits_to_options(hits[:6], field, hits[0].id if hits else None),
                ui_type=ui,
            ))

    if _eligibility_mentions_tag(eligibility, merchant_text) and not eligibility:
        field = "customer_eligibility.0.value"
        if field not in skip and field not in result.resolved:
            tpl = schema.get("missing_customer_tag", {})
            result.clarifications.append(_make_clarification(
                "missing_customer_tag_0", field,
                tpl.get("question_template", "Which customer group should qualify?"),
                _hits_to_options(_customer_tags(), field),
            ))

    # ── Market — ask when mentioned but not resolved ─────────────────────────
    for i, eh in enumerate(eligibility):
        e_type = (eh.get("type_hint") or "").lower()
        if "market" not in e_type and "country" not in e_type:
            continue
        field = f"customer_eligibility.{i}.market"
        val_hint = eh.get("value_hint") or ""
        mkt_hits = _best_hits(val_hint, _markets()) if val_hint else []
        resolved = _auto_resolve_catalog(field, mkt_hits, selections, llm_res)
        if resolved and val_hint:
            result.resolved[field] = resolved
        elif field not in skip:
            tpl = schema.get("missing_market", {})
            hits = mkt_hits or _markets()
            result.clarifications.append(_make_clarification(
                f"missing_market_{i}", field,
                tpl.get("question_template", "Which market should this promotion apply to?"),
                _hits_to_options(hits, field, hits[0].id if hits else None),
                ui_type="multi_select" if len(hits) != 1 else "cards",
            ))

    if _mentions_market(merchant_text, eligibility) and not any(
        "market" in (e.get("type_hint") or "").lower() for e in eligibility
    ):
        field = "customer_eligibility.0.market"
        if field not in skip and field not in result.resolved:
            tpl = schema.get("missing_market", {})
            result.clarifications.append(_make_clarification(
                "missing_market_0", field,
                tpl.get("question_template", "Which market should this promotion apply to?"),
                _hits_to_options(_markets(), field),
                ui_type="multi_select",
            ))

    # Apply user catalog selections not yet written
    for fld in (
        "reward.gift_product", "trigger.scope.productRef", "trigger.scope.collectionRef",
        "reward.y_target",
    ):
        if fld in selections and fld not in result.resolved:
            hits: List[CatalogHit] = []
            resolved = _auto_resolve_catalog(fld, hits, selections, llm_res)
            if resolved:
                _write_catalog_resolution(result, fld, resolved)

    for key in list(selections.keys()):
        if key.startswith("customer_eligibility.") and key not in result.resolved:
            kind = "market" if key.endswith(".market") else "tag"
            hits = _markets() if kind == "market" else _customer_tags()
            resolved = _auto_resolve_catalog(key, hits, selections, llm_res)
            if resolved:
                result.resolved[key] = resolved

    # Remove answered clarifications
    def _field_done(fld: str) -> bool:
        if fld in skip or fld in selections:
            return True
        if fld in result.resolved:
            return True
        if fld == "reward.y_target" and (
            "reward.y_target.product" in result.resolved
            or "reward.y_target.collection" in result.resolved
            or "reward.y_target" in result.resolved
        ):
            return True
        return False

    result.clarifications = [c for c in result.clarifications if not _field_done(c["field"])]

    return result


def apply_grounding_to_ir(ir: Dict[str, Any], grounding: GroundingResult, currency: str) -> Dict[str, Any]:
    """Patch IR skeleton with grounded catalog IDs."""
    ir = json.loads(json.dumps(ir))  # deep copy
    resolved = grounding.resolved

    if resolved.get("currency"):
        cur = resolved["currency"]
        if "trigger" in ir and ir["trigger"].get("currency") is not None:
            ir["trigger"]["currency"] = cur
        if "tiers" in ir:
            for tier in ir["tiers"]:
                if tier.get("trigger", {}).get("currency") is not None:
                    tier["trigger"]["currency"] = cur

    if resolved.get("trigger.value") is not None and "trigger" in ir:
        ir["trigger"]["value"] = resolved["trigger.value"]

    gift = resolved.get("reward.gift_product")
    if gift and "reward" in ir:
        ir["reward"]["gift_product"] = {
            "status": "resolved",
            "resolved_id": gift["id"],
            "title": gift["title"],
            "query": ir["reward"]["gift_product"].get("query"),
        }

    prod_ref = resolved.get("trigger.scope.productRef")
    if prod_ref and "trigger" in ir:
        scope = ir["trigger"].setdefault("scope", {})
        scope["type"] = "product"
        scope["productRef"] = {
            "status": "resolved",
            "resolved_id": prod_ref["id"],
            "title": prod_ref["title"],
            "query": scope.get("productRef", {}).get("query"),
        }
        scope["productTitles"] = [prod_ref["title"]]

    coll_ref = resolved.get("trigger.scope.collectionRef")
    if coll_ref and "trigger" in ir:
        scope = ir["trigger"].setdefault("scope", {})
        scope["type"] = "collection"
        scope["collectionRef"] = {
            "status": "resolved",
            "resolved_id": coll_ref["id"],
            "title": coll_ref["title"],
            "query": scope.get("collectionRef", {}).get("query"),
        }
        scope["collectionTitles"] = [coll_ref["title"]]

    for key, val in resolved.items():
        if key.startswith("reward.y_target."):
            if "reward" not in ir:
                continue
            y = ir["reward"].setdefault("y_target", {})
            y["type"] = "product" if "product" in key else "collection"
            y["status"] = "resolved"
            y["resolved_id"] = val["id"]
            y["title"] = val["title"]

    # Multi-select product/collection lists
    if isinstance(resolved.get("reward.y_target"), list) and "reward" in ir:
        items = resolved["reward.y_target"]
        ir["reward"]["y_target"] = {
            "type": "product" if all(i.get("kind") != "collection" for i in items) else "collection",
            "status": "resolved",
            "resolved_ids": [i["id"] for i in items],
            "titles": [i["title"] for i in items],
        }

    if isinstance(resolved.get("trigger.scope.productRef"), list) and "trigger" in ir:
        items = resolved["trigger.scope.productRef"]
        scope = ir["trigger"].setdefault("scope", {})
        scope["type"] = "product"
        scope["productRef"] = {
            "status": "resolved",
            "resolved_ids": [i["id"] for i in items],
            "titles": [i["title"] for i in items],
        }
        scope["productTitles"] = [i["title"] for i in items]

    if isinstance(resolved.get("trigger.scope.collectionRef"), list) and "trigger" in ir:
        items = resolved["trigger.scope.collectionRef"]
        scope = ir["trigger"].setdefault("scope", {})
        scope["type"] = "collection"
        scope["collectionRef"] = {
            "status": "resolved",
            "resolved_ids": [i["id"] for i in items],
            "titles": [i["title"] for i in items],
        }

    if resolved.get("reward.value") is not None and "reward" in ir:
        ir["reward"]["value"] = resolved["reward.value"]

    for key, val in resolved.items():
        if key.startswith("customer_eligibility.") and isinstance(val, dict):
            idx_part = key.split(".")[1]
            try:
                idx = int(idx_part)
            except ValueError:
                continue
            elig = ir.get("customer_eligibility") or []
            if idx >= len(elig):
                continue
            entry = elig[idx]
            if key.endswith(".market"):
                entry["type"] = "market"
                entry["value"] = val["title"]
                entry["resolved_id"] = val["id"]
            elif entry.get("type") == "customer_tag":
                entry["value"] = val["title"]
                entry["resolved_id"] = val["id"]

    tag = resolved.get("customer_eligibility.0.value")
    if tag and ir.get("customer_eligibility"):
        for entry in ir["customer_eligibility"]:
            if entry.get("type") == "customer_tag" and not entry.get("resolved_id"):
                entry["value"] = tag["title"]
                entry["resolved_id"] = tag["id"]
                break

    return ir


STAGE2_GROUNDING_PROMPT = """You are Stage 2 of a promotion pipeline. Stage 1 identified a supported promotion and extracted loose hints.

Your job: map hints to exact store catalog IDs using ONLY the catalog below. Never invent IDs.

Rules:
- If one clear match exists, return resolved field paths with catalog IDs.
- If ambiguous or missing, leave field out of resolved and list it in needs_clarification.
- promotion_family is already confirmed — use it to pick relevant catalog sections.

Return ONLY JSON:
{
  "resolved": {
    "reward.gift_product": "p1",
    "trigger.scope.productRef": "p11",
    "trigger.value": 100,
    "currency": "USD"
  },
  "needs_clarification": ["reward.gift_product"],
  "notes": "brief reasoning"
}
"""


def catalog_excerpt_for_family(family: str) -> Dict[str, Any]:
    """Compact catalog slice for Stage 2 LLM context."""
    excerpt: Dict[str, Any] = {
        "collections": STORE_CATALOG["collection_product_index"],
        "customer_tags": STORE_CATALOG.get("customer_tag_catalog", []),
        "currencies": [c["code"] for c in STORE_CATALOG.get("currency_options", [])],
    }
    if family == "free_gift":
        excerpt["gift_products"] = STORE_CATALOG["product_taxonomy"]["Gift Items"]["products"]
    elif family == "buy_x_get_y":
        excerpt["products"] = {
            k: v["products"] for k, v in STORE_CATALOG["product_taxonomy"].items()
        }
    else:
        excerpt["products"] = {
            k: v["products"] for k, v in STORE_CATALOG["product_taxonomy"].items()
        }
    return excerpt


def build_stage2_user_prompt(
    family: str,
    hints_dict: Dict[str, Any],
    merchant_text: str,
) -> str:
    return (
        f"Promotion family: {family}\n"
        f"Merchant input: {json.dumps(merchant_text)}\n"
        f"Stage 1 hints: {json.dumps(hints_dict, indent=2)}\n"
        f"Store catalog excerpt: {json.dumps(catalog_excerpt_for_family(family), indent=2)}\n"
        "Return ONLY the JSON object."
    )


def parse_stage2_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON in Stage 2 response")
    data = json.loads(text[start: end + 1])
    if not isinstance(data.get("resolved"), dict):
        data["resolved"] = {}
    return data
