"""Shared configuration and labels."""

import json
from pathlib import Path
from typing import Any, Dict, List

# Config
# ═══════════════════════════════════════════════════════════════════════════

OLLAMA_API_KEY      = "fc30369802484bb5a1fd3a8e08a7cf08.Ej-syunN81aqQTD95ojud8w3"
OLLAMA_CLOUD_HOST   = "https://ollama.com"
OLLAMA_LOCAL_HOST   = "http://127.0.0.1:11434"
DEFAULT_MODEL       = "gpt-oss:120b"

# Stage 0 scope classifier — Ollama embeddings (local, free)
SCOPE_EMBED_MODEL   = "nomic-embed-text"   # ollama pull nomic-embed-text
SCOPE_THRESHOLD     = 0.60                 # tuned for nomic-embed-text (53/54 guard tests)
SCOPE_EMBED_CACHE   = ".cache"

# Stage 1 LLM conversation
MAX_CLARIFICATION_ROUNDS = 6
MIN_SUGGESTED_PROMOTIONS = 5
MAX_SUGGESTED_PROMOTIONS = 6
MIN_CLARIFY_SUGGESTIONS = 1
CLOUD_MODEL_ALIASES = ("gpt-oss:120b", "gpt-oss:120b-cloud")
DEFAULT_CURRENCY    = "USD"

VALID_FLAGS    = frozenset({"discount_code", "free_shipping", "usage_limit", "pos_only", "scheduling"})
VALID_VERDICTS = frozenset({"pass", "out_of_scope", "injection", "unsupported"})
VALID_FAMILIES = frozenset({"free_gift", "buy_x_get_y", "tiered_discount"})

VALID_TRIGGER_TYPES = frozenset({
    "cart_quantity", "cart_subtotal",
    "collection_quantity", "collection_subtotal",
    "product_quantity", "product_subtotal",
    "cart_attributes", "cart_currency",
})
SUBTOTAL_TRIGGERS = frozenset({"cart_subtotal", "collection_subtotal", "product_subtotal"})
CART_RULE_TRIGGERS = frozenset({"cart_quantity", "cart_subtotal", "cart_attributes", "cart_currency"})

VALID_ELIGIBILITY_TYPES = frozenset({
    "customer_tag", "logged_in", "market", "customer_segment",
})

FLAG_MESSAGES: Dict[str, str] = {
    "discount_code": (
        "Discount codes, coupons, and promo-code checkout flows are not supported. "
        "Please describe a promotion without coupon codes (e.g. spend threshold or buy-X-get-Y)."
    ),
    "free_shipping": (
        "Free shipping / delivery-fee waivers are not supported. "
        "Please describe a product or order discount instead."
    ),
    "usage_limit": (
        "Usage limits (first N customers, one use per person, redemption caps) are not supported. "
        "Please describe the promotion without per-customer or total-use limits."
    ),
    "pos_only": (
        "POS-only or in-store-only restrictions are not supported. "
        "Promotions must work for online and in-store customers."
    ),
    "scheduling": (
        "Scheduled start/end dates and time-limited flash sales are not supported. "
        "Please describe an always-on promotion without expiry dates."
    ),
}

VERDICT_MESSAGES: Dict[str, str] = {
    "out_of_scope": (
        "This doesn't look like a promotion request. "
        "Please describe a retail promotion (e.g. 'Spend $100 and get a free gift')."
    ),
    "injection": (
        "Your input contains content that cannot be processed. "
        "Please describe your promotion only."
    ),
    "pass": "Valid promotion request — IR skeleton built below.",
}

FLAG_LABELS: Dict[str, str] = {
    "discount_code": "Discount code / coupon",
    "free_shipping":  "Free shipping",
    "usage_limit":    "Usage limit",
    "pos_only":       "POS / in-store only",
    "scheduling":     "Scheduling / expiry",
}

FAMILY_LABELS: Dict[str, str] = {
    "free_gift":       "🎁 Free Gift",
    "buy_x_get_y":     "🛒 Buy X Get Y",
    "tiered_discount": "📊 Tiered Discount",
}

# Stage 2 input — validated JSON from Stage 1 pass
STAGE2_PAYLOAD_KEYS = frozenset({"prompt", "family", "family_label"})

# Stage 2 — authoritative store context (shared by LLM + merchant UI)
STORE_CONTEXT_DOCUMENT: Dict[str, Any] = {
    "store_context_schema": {
        "products": [
            {
                "category": "Gift Items",
                "supported_product_values": [
                    "Sample Gift",
                    "Mini Sample Pack",
                    "Mystery Gift Box",
                    "Free Trial Kit",
                    "Complimentary Gift",
                    "Welcome Gift",
                    "Free Tote Bag",
                    "Gift Hamper",
                ],
            },
            {
                "category": "Accessories",
                "supported_product_values": [
                    "Cap",
                    "Snapback Cap",
                    "Travel Tote Bag",
                    "Laptop Sleeve",
                    "Travel Backpack",
                    "Sunglasses",
                    "Leather Wallet",
                    "Crossbody Bag",
                    "Beanie",
                    "Belt",
                ],
            },
            {
                "category": "Apparel",
                "supported_product_values": [
                    "Shirt",
                    "Premium T-Shirt",
                    "Oversized Tee",
                    "Tank Top",
                    "Minimal Hoodie",
                    "Denim Jacket",
                    "Sweatshirt",
                    "Joggers",
                    "Cargo Pants",
                    "Polo Shirt",
                ],
            },
            {
                "category": "Skincare",
                "supported_product_values": [
                    "Face Wash",
                    "Vitamin C Serum",
                    "Night Cream",
                    "Body Lotion",
                    "Moisturizer",
                    "Sunscreen",
                    "Cleanser",
                    "Toner",
                    "Acne Gel",
                    "Eye Cream",
                ],
            },
            {
                "category": "Footwear",
                "supported_product_values": [
                    "Sneakers",
                    "Running Shoes",
                    "Sports Socks",
                    "Slides",
                    "Boots",
                    "Canvas Shoes",
                ],
            },
            {
                "category": "Electronics",
                "supported_product_values": [
                    "Wireless Earbuds",
                    "Bluetooth Speaker",
                    "Gaming Mouse",
                    "Mechanical Keyboard",
                    "USB Hub",
                    "Portable Charger",
                    "Smart Watch",
                    "Webcam",
                ],
            },
            {
                "category": "Fitness",
                "supported_product_values": [
                    "Yoga Mat",
                    "Resistance Bands",
                    "Water Bottle",
                    "Protein Powder",
                    "Gym Gloves",
                    "Skipping Rope",
                    "Foam Roller",
                ],
            },
            {
                "category": "Home Essentials",
                "supported_product_values": [
                    "Coffee Mug",
                    "Ceramic Plate Set",
                    "Water Jug",
                    "Storage Box",
                    "Desk Lamp",
                    "Kitchen Towels",
                ],
            },
            {
                "category": "Beauty",
                "supported_product_values": [
                    "Luxury Perfume",
                    "Lipstick Set",
                    "Foundation Kit",
                    "Makeup Brush Set",
                    "Hair Serum",
                    "Nail Polish Kit",
                ],
            },
        ],
        "collections": [
            "Summer Collection",
            "Winter Collection",
            "Skincare",
            "Accessories",
            "Apparel",
            "Footwear",
            "Electronics",
            "Fitness",
            "Home Essentials",
            "Gift Items",
            "Beauty",
            "Streetwear",
            "Gym Essentials",
            "Travel Collection",
            "Luxury Collection",
        ],
        "customer_tags": [
            "VIP",
            "Wholesale",
            "Gold",
            "Silver",
            "Employee",
            "Subscriber",
            "First Time Buyer",
            "Loyalty Member",
            "Skincare Club",
            "Summer Sale Access",
            "Premium Member",
            "Frequent Buyer",
            "Corporate Customer",
        ],
        "customer_segments": [
            "High Value Customers",
            "Repeat Buyers",
            "First-Time Customers",
            "At-Risk Customers",
            "Newsletter Subscribers",
            "Mobile App Users",
            "Abandoned Cart Segment",
            "Loyalty Program Members",
        ],
        "cart_attributes": [
            "Contains subscription product",
            "Gift wrap requested",
            "Contains preorder item",
            "Contains bundle",
            "B2B / wholesale order",
            "International shipping",
            "Express delivery selected",
        ],
        "markets": [
            {"country": "United States", "market_code": "US"},
            {"country": "United Kingdom", "market_code": "UK"},
            {"country": "India", "market_code": "IN"},
            {"country": "Germany", "market_code": "DE"},
            {"country": "Canada", "market_code": "CA"},
            {"country": "Australia", "market_code": "AU"},
            {"country": "France", "market_code": "FR"},
        ],
        "currencies": [
            {
                "currency_code": "USD",
                "symbols_and_aliases": ["$", "usd", "dollar", "dollars", "us dollar"],
            },
            {
                "currency_code": "INR",
                "symbols_and_aliases": ["₹", "inr", "rupee", "rupees", "indian rupee"],
            },
            {
                "currency_code": "GBP",
                "symbols_and_aliases": ["£", "gbp", "pound", "pounds", "british pound"],
            },
            {
                "currency_code": "EUR",
                "symbols_and_aliases": ["€", "eur", "euro", "euros"],
            },
            {
                "currency_code": "CAD",
                "symbols_and_aliases": ["cad", "canadian dollar"],
            },
            {
                "currency_code": "AUD",
                "symbols_and_aliases": ["aud", "australian dollar"],
            },
        ],
    },
}

# Optional legacy catalog file (store_grounding.py); pipeline uses STORE_CONTEXT_DOCUMENT
_CATALOG_PATH = Path(__file__).resolve().parent / "store_catalog.json"
if _CATALOG_PATH.is_file():
    with _CATALOG_PATH.open(encoding="utf-8") as _f:
        STORE_CATALOG: Dict[str, Any] = json.load(_f)
else:
    STORE_CATALOG = {}

_CURRENCY_BY_MARKET = {
    "US": "USD", "UK": "GBP", "GB": "GBP", "IN": "INR",
    "DE": "EUR", "CA": "CAD", "AU": "AUD", "FR": "EUR", "EU": "EUR",
}


def get_store_context_schema() -> Dict[str, Any]:
    """Inner store_context_schema block."""
    return STORE_CONTEXT_DOCUMENT["store_context_schema"]


def get_store_context() -> Dict[str, Any]:
    """Return the authoritative store context document (LLM + UI)."""
    return json.loads(json.dumps(STORE_CONTEXT_DOCUMENT))


def _build_store_context_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten store_context_schema for UI picker helpers."""
    products: List[Dict[str, str]] = []
    categories: List[Dict[str, str]] = []
    for block in schema.get("products") or []:
        cat_name = block["category"]
        categories.append({
            "name": cat_name,
            "collection_id": cat_name,
            "collection_title": cat_name,
        })
        for title in block.get("supported_product_values") or []:
            products.append({
                "id": title,
                "title": title,
                "category": cat_name,
                "aliases": [],
            })

    collections = [
        {"id": name, "title": name}
        for name in schema.get("collections") or []
    ]

    markets = []
    for m in schema.get("markets") or []:
        code = m.get("market_code", "")
        markets.append({
            "code": code,
            "country": m.get("country", code),
            "currency": _CURRENCY_BY_MARKET.get(code, DEFAULT_CURRENCY),
        })

    return {
        "products": products,
        "categories": categories,
        "collections": collections,
        "customer_tags": list(schema.get("customer_tags") or []),
        "customer_segments": list(schema.get("customer_segments") or []),
        "cart_attributes": list(schema.get("cart_attributes") or []),
        "markets": markets,
        "currencies": [c["currency_code"] for c in schema.get("currencies") or []],
    }


STORE_CONTEXT: Dict[str, Any] = _build_store_context_from_schema(get_store_context_schema())

# Aliases — same document for merchant UI and Stage 2 LLM
MERCHANT_STORE_CONTEXT: Dict[str, Any] = STORE_CONTEXT_DOCUMENT
LLM_STORE_CONTEXT: Dict[str, Any] = STORE_CONTEXT_DOCUMENT


def format_store_context_json(
    store: Dict[str, Any] | None = None,
) -> str:
    """Pretty JSON for UI display and Stage 2 system prompt."""
    return json.dumps(store or get_store_context(), indent=2, ensure_ascii=False)


def format_store_context_json_for_merchants(
    store: Dict[str, Any] | None = None,
) -> str:
    return format_store_context_json(store)


def format_store_context_json_for_llm(
    store: Dict[str, Any] | None = None,
) -> str:
    return format_store_context_json(store)


def get_merchant_store_context() -> Dict[str, Any]:
    return get_store_context()


def get_llm_store_context() -> Dict[str, Any]:
    return get_store_context()

# ── LLM-internal only (Stage 2 system prompt — never show to merchants) ───────

FAMILY_CAPABILITIES: Dict[str, Any] = {
    "supported_families": ["free_gift", "buy_x_get_y", "tiered_discount"],
    "free_gift": {
        "triggers": [
            "cart_quantity", "cart_subtotal", "cart_currency", "cart_attributes",
            "collection_quantity", "collection_subtotal",
            "product_quantity", "product_subtotal",
        ],
        "rewards": ["free_gift"],
        "eligibility": ["customer_tag", "logged_in", "market", "customer_segment"],
        "cart_rules_note": (
            "Subtotal triggers: one currency question → trigger.currency (threshold = cart currency). "
            "cart_currency trigger: currency-only promos with no spend/qty. "
            "cart_attributes: attribute names from catalog or confirm none"
        ),
    },
    "buy_x_get_y": {
        "triggers": [
            "cart_quantity", "cart_subtotal", "cart_currency", "cart_attributes",
            "collection_quantity", "collection_subtotal",
            "product_quantity", "product_subtotal",
        ],
        "rewards": ["percentage_off_y", "fixed_amount_off_y", "free_y"],
        "reward_note": "free_y is stored as percentage_off_y with value 100",
        "y_allocation": [
            "specific_product", "specific_collection",
            "cheapest_eligible", "same_item",
        ],
        "eligibility": ["customer_tag", "logged_in", "market", "customer_segment"],
        "cart_rules_note": (
            "Subtotal triggers: one currency question → trigger.currency (threshold = cart currency). "
            "cart_currency trigger: currency-only promos with no spend/qty. "
            "cart_attributes: attribute names from catalog or confirm none"
        ),
    },
    "tiered_discount": {
        "triggers": [
            "cart_quantity", "cart_subtotal", "cart_currency", "cart_attributes",
            "collection_quantity", "collection_subtotal",
            "product_quantity", "product_subtotal",
        ],
        "rewards": ["percentage_off", "fixed_amount_off"],
        "tier_behavior": ["best_tier_only", "all_matching_tiers"],
        "minimum_tiers": 2,
        "eligibility": ["customer_tag", "logged_in", "market", "customer_segment"],
        "cart_rules_note": (
            "Subtotal triggers: one currency question → trigger.currency (threshold = cart currency). "
            "cart_currency trigger: currency-only promos with no spend/qty. "
            "cart_attributes: attribute names from catalog or confirm none"
        ),
    },
}

# Merchant-facing labels (never expose raw IR / capability key names in UI)
TRIGGER_TYPE_MERCHANT: Dict[str, str] = {
    "cart_quantity": "Number of items in cart",
    "cart_subtotal": "Minimum cart value",
    "cart_currency": "Cart checkout currency",
    "cart_attributes": "Required cart attributes",
    "collection_quantity": "Items from a collection",
    "collection_subtotal": "Spend on a collection",
    "product_quantity": "Quantity of a product",
    "product_subtotal": "Spend on a product",
}

CONFIRMED_FIELD_MERCHANT: Dict[str, str] = {
    "trigger_type": "Qualification",
    "trigger_value": "Threshold",
    "trigger_scope": "Applies to",
    "currency": "Currency",
    "cart_currency": "Cart currency",
    "cart_attributes": "Cart must include",
    "reward_type": "Reward",
    "reward_value": "Discount amount",
    "gift_product": "Free gift",
    "gift_quantity": "Gift quantity",
    "customer_eligibility": "Who qualifies",
    "customer_segment": "Customer segment",
    "tier_count": "Tiers",
    "tier_behavior": "When multiple tiers match",
    "y_target": "Discount applies to",
    "y_quantity": "Quantity",
}


def humanize_confirmed_fields(confirmed: Dict[str, Any]) -> List[str]:
    """Plain-language progress lines for merchants — no IR key jargon."""
    if not confirmed:
        return []
    lines: List[str] = []
    for key, val in confirmed.items():
        label = CONFIRMED_FIELD_MERCHANT.get(key, key.replace("_", " ").title())
        text = str(val)
        for tech, friendly in TRIGGER_TYPE_MERCHANT.items():
            text = text.replace(tech, friendly)
        text = text.replace("best_tier_only", "Highest tier only")
        text = text.replace("all_matching_tiers", "All matching tiers stacked")
        text = text.replace("customer_tag:", "Customer group:")
        text = text.replace("customer_segment:", "Customer segment:")
        text = text.replace("product_quantity", TRIGGER_TYPE_MERCHANT["product_quantity"])
        lines.append(f"{label}: {text}")
    return lines


def format_store_context_for_merchants(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    """Readable store summary — products, categories, markets, tags."""
    from collections import defaultdict
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for p in store["products"]:
        by_cat[p["category"]].append(p["title"])
    parts = ["**Product categories**"]
    for cat, titles in by_cat.items():
        parts.append(f"- **{cat}:** {', '.join(titles)}")
    parts.append("")
    parts.append("**Collections:** " + ", ".join(c["title"] for c in store["collections"]))
    parts.append("")
    parts.append("**Customer groups:** " + ", ".join(store["customer_tags"]))
    parts.append("")
    if store.get("customer_segments"):
        parts.append("**Customer segments:** " + ", ".join(store["customer_segments"]))
        parts.append("")
    if store.get("cart_attributes"):
        parts.append("**Cart attributes:** " + ", ".join(store["cart_attributes"]))
        parts.append("")
    parts.append("**Markets:** " + ", ".join(m["country"] for m in store["markets"]))
    return "\n".join(parts)


STAGE2_STRICT_CONFIRMATION = """\
STRICT 100% CONFIRMATION POLICY — never break
────────────────────────────────────────────
• Return status "complete" ONLY when EVERY required IR field is explicitly
  confirmed by the merchant (stated in prompt OR chosen from your options).
• If ANY field is missing, ambiguous, or matches multiple catalog items →
  status MUST remain "clarifying".
• NEVER guess product, collection, category, market, currency, or customer tag.
• When merchant says "shirt", "gift", "skincare", etc. and multiple catalog
  items match → list the exact matches and ask WHICH ONE.
• When merchant says a country/region vaguely → list exact markets from catalog.
• When subtotal trigger and currency not explicit → ask ONCE with currency picklist;
  store as trigger.currency (threshold currency = cart currency). Never ask twice.
• When cart attribute requirements not stated → ask with cart_attributes picklist
  or confirm "No special cart requirements".
• Use cart_currency trigger type ONLY when promotion gates checkout currency with
  NO spend amount and NO quantity threshold (e.g. "USD carts only").
• Do NOT ask currency for cart_quantity unless merchant explicitly requires
  checkout-currency restriction in a multi-currency store.
• When eligibility implied (VIP, wholesale, market-specific, segment-specific) but
  tag/market/segment not exact → ask with customer_tag, market, or customer_segment picklist.
• admin_selection_required is ONLY allowed after merchant confirms none of the
  listed products/collections fit AND explicitly agrees to admin selection.
• Once a field is in confirmed_fields, never ask again unless merchant changes it.
• When merchant selects an ENTIRE CATEGORY (e.g. Apparel) or COLLECTION for trigger_scope,
  that confirms scope for ALL products in it — do NOT ask again for a single product SKU.
• When merchant selects tier stacking preference, apply it and do not re-ask.

MERCHANT-FACING LANGUAGE (questions & options shown to the merchant):
• NEVER use IR key names in questions or options: no trigger_type, cart_subtotal,
  tier_behavior, confirmed_fields, product_quantity, percentage_off, etc.
• Ask in plain English: "Which products does this apply to?", "Who can use this
  promotion?", "When someone qualifies for both tiers, which discount wins?"
• Options must name real catalog items: product titles, category names,
  collection names, countries, customer groups — never internal IDs or codes
  unless the merchant already used them (e.g. USD is fine for currency).
• confirmed_fields in your JSON is INTERNAL — use technical keys there for
  your own tracking, but the merchant never sees that object. Questions must
  be fully understandable without knowledge of the IR schema.
• Infer triggers, rewards, tier_behavior, and eligibility from the Stage 1
  family and merchant answers — the merchant does not know FAMILY_CAPABILITIES.
"""


def format_products_by_category(store: Dict[str, Any] = STORE_CONTEXT, *, include_ids: bool = True) -> str:
    from collections import defaultdict
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for p in store["products"]:
        if include_ids:
            by_cat[p["category"]].append(f"{p['title']} (id:{p['id']})")
        else:
            by_cat[p["category"]].append(p["title"])
    return "\n".join(f"  {cat}:\n    " + " | ".join(items) for cat, items in by_cat.items())


def format_category_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    return " | ".join(
        f"{c['name']} → collection {c['collection_title']} (id:{c['collection_id']})"
        for c in store["categories"]
    )


def format_product_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    return " | ".join(f"{p['title']} (id:{p['id']}, {p['category']})" for p in store["products"])


def format_collection_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    return " | ".join(f"{c['title']} (id:{c['id']})" for c in store["collections"])


def format_customer_tag_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    tags = list(store["customer_tags"]) + ["All customers (no tag restriction)"]
    return " | ".join(tags)


def format_customer_segment_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    segments = list(store.get("customer_segments") or []) + ["All segments (no segment restriction)"]
    return " | ".join(segments)


def format_cart_attribute_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    attrs = list(store.get("cart_attributes") or []) + ["No special cart requirements"]
    return " | ".join(attrs)


def format_market_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    return " | ".join(
        f"{m['country']} (code:{m['code']}, currency:{m['currency']})"
        for m in store["markets"]
    ) + " | All markets (no market restriction)"


def format_currency_picklist(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    schema = get_store_context_schema()
    parts = []
    for c in schema.get("currencies") or []:
        code = c["currency_code"]
        aliases = c.get("symbols_and_aliases") or []
        symbol = aliases[0] if aliases else code
        parts.append(f"{code} ({symbol})")
    return " | ".join(parts)


def format_clarification_schema() -> str:
    schema = STORE_CATALOG.get("clarification_schema") or {}
    if not schema:
        return json.dumps(
            {
                "note": "Use store_context_schema in STORE_CONTEXT_DOCUMENT for clarification options.",
            },
            indent=2,
        )
    return json.dumps(schema, indent=2)


def format_store_picklists_for_prompt(store: Dict[str, Any] = STORE_CONTEXT) -> str:
    """Dynamic option lists Stage 2 MUST use when asking clarification questions."""
    return f"""\
PRODUCT CATEGORIES (trigger scope / browsing):
  {format_category_picklist(store)}

ALL PRODUCTS (gift rewards, trigger product, Y target):
  {format_product_picklist(store)}

COLLECTIONS (trigger scope, collection_quantity/subtotal):
  {format_collection_picklist(store)}

CUSTOMER TAGS (eligibility — pick exact tag or all customers):
  {format_customer_tag_picklist(store)}

CUSTOMER SEGMENTS (eligibility — pick exact segment or all segments):
  {format_customer_segment_picklist(store)}

CART ATTRIBUTES (cart rules — pick required attributes or none):
  {format_cart_attribute_picklist(store)}

MARKETS (eligibility — pick exact market code or all markets):
  {format_market_picklist(store)}

CURRENCIES (subtotal triggers — pick exact code):
  {format_currency_picklist(store)}"""


# ── Stage 2 UI catalog widgets (Streamlit pickers) ───────────────────────────

TIER_BEHAVIOR_CHOICES: List[Dict[str, str]] = [
    {
        "id": "best_tier_only",
        "label": "Best tier only — apply the highest matching discount (e.g. buy 4 → INR 50 off only)",
    },
    {
        "id": "all_matching_tiers",
        "label": "All matching tiers — apply every tier the customer qualifies for (stacked)",
    },
]


def infer_stage2_widget_type(question_id: str, question_text: str) -> str:
    """Map a Stage 2 clarify question to a catalog widget type."""
    qid = (question_id or "").lower()
    text = (question_text or "").lower()

    if "tier_behavior" in qid or "which tier" in text or "both tiers" in text or "which discount should" in text:
        return "tier_behavior"
    if qid == "currency" or "currency" in qid or "which currency" in text:
        return "currency"
    if "market" in qid or "country" in text or "which market" in text:
        return "market"
    if "customer_eligibility" in qid or "customer group" in text or "who can trigger" in text or "who can" in text:
        return "customer_tags"
    if "customer_segment" in qid or "segment" in text:
        return "customer_segments"
    if "cart_attribute" in qid or "cart attribute" in text or "special cart" in text:
        return "cart_attributes"
    if qid in ("gift_product", "missing_gift_product") or "free gift" in text:
        return "gift_products"
    if "collection" in qid and "scope" not in qid:
        return "collection"
    if (
        "trigger_scope" in qid
        or "trigger_product" in qid
        or "ambiguous_trigger" in qid
        or "which product" in text
        or "which products" in text
        or "apply to" in text
        or "buy x" in text
        or "shirts" in text
        or "refer to" in text
    ):
        return "trigger_scope"
    return "options"


def get_trigger_scope_choices(store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    choices: List[Dict[str, str]] = [
        {"key": "all_products", "label": "All products in cart"},
    ]
    for cat in store["categories"]:
        prods = [p for p in store["products"] if p["category"] == cat["name"]]
        preview = ", ".join(p["title"] for p in prods[:4])
        if len(prods) > 4:
            preview += ", …"
        choices.append({
            "key": f"category:{cat['name']}",
            "label": f"All products in {cat['name']} — {preview}",
            "collection_id": cat["collection_id"],
            "collection_title": cat["collection_title"],
        })
    for coll in store["collections"]:
        choices.append({
            "key": f"collection:{coll['id']}",
            "label": f"Collection: {coll['title']}",
        })
    choices.append({"key": "pick_products", "label": "Select specific product(s)…"})
    return choices


def get_product_choices(category: str | None = None, store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    prods = store["products"]
    if category:
        prods = [p for p in prods if p["category"] == category]
    return [{"id": p["id"], "label": f"{p['title']} — {p['category']}", "title": p["title"]} for p in prods]


def get_gift_product_choices(store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    return get_product_choices("Gift Items", store)


def get_collection_choices(store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    return [{"id": c["id"], "label": c["title"], "title": c["title"]} for c in store["collections"]]


def get_market_choices(store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    items = [{"id": m["code"], "label": m["country"], "country": m["country"]} for m in store["markets"]]
    items.append({"id": "ALL", "label": "All countries"})
    return items


def get_currency_choices() -> List[Dict[str, str]]:
    schema = get_store_context_schema()
    items: List[Dict[str, str]] = []
    for c in schema.get("currencies") or []:
        code = c["currency_code"]
        aliases = c.get("symbols_and_aliases") or []
        symbol = aliases[0] if aliases else code
        label = f"{code} ({symbol})"
        items.append({"id": code, "label": label, "code": code})
    return items


def get_customer_tag_choices(store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    items = [{"id": tag, "label": tag} for tag in store["customer_tags"]]
    items.append({"id": "ALL", "label": "All customers"})
    items.append({"id": "logged_in", "label": "Logged-in customers only"})
    return items


def get_customer_segment_choices(store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    items = [{"id": seg, "label": seg} for seg in store.get("customer_segments") or []]
    items.append({"id": "ALL", "label": "All segments (no restriction)"})
    return items


def get_cart_attribute_choices(store: Dict[str, Any] = STORE_CONTEXT) -> List[Dict[str, str]]:
    items = [{"id": attr, "label": attr} for attr in store.get("cart_attributes") or []]
    items.append({"id": "NONE", "label": "No special cart requirements"})
    return items


def format_scope_selection_merchant(scope_key: str, picked_product_ids: List[str] | None = None) -> str:
    """Plain-language selection for chat — no IR keys or internal IDs."""
    store = STORE_CONTEXT
    if scope_key == "all_products":
        return "Apply to all products in the cart"

    if scope_key.startswith("category:"):
        cat_name = scope_key.split(":", 1)[1]
        prods = [p["title"] for p in store["products"] if p["category"] == cat_name]
        names = ", ".join(prods[:6])
        if len(prods) > 6:
            names += ", …"
        return f"Apply to all products in {cat_name}: {names}"

    if scope_key.startswith("collection:"):
        cid = scope_key.split(":", 1)[1]
        coll = next((c for c in store["collections"] if c["id"] == cid), None)
        title = coll["title"] if coll else cid
        return f"Apply to the {title} collection"

    if scope_key == "pick_products" and picked_product_ids:
        titles = []
        for pid in picked_product_ids:
            p = next((x for x in store["products"] if x["id"] == pid), None)
            if p:
                titles.append(p["title"])
        return f"Apply to: {', '.join(titles)}"

    return scope_key


def format_scope_selection_llm(scope_key: str, picked_product_ids: List[str] | None = None) -> str:
    """Structured resolution for LLM system note — not shown to merchant."""
    store = STORE_CONTEXT
    if scope_key == "all_products":
        return "trigger_scope: all_products (entire cart, any product counts toward threshold)"

    if scope_key.startswith("category:"):
        cat_name = scope_key.split(":", 1)[1]
        cat = next((c for c in store["categories"] if c["name"] == cat_name), None)
        prods = [p for p in store["products"] if p["category"] == cat_name]
        plist = ", ".join(p["title"] for p in prods)
        return (
            f"trigger_scope: entire {cat_name} category ({cat_name} collection). "
            f"All products in this category qualify: {plist}"
        )

    if scope_key.startswith("collection:"):
        cid = scope_key.split(":", 1)[1]
        coll = next((c for c in store["collections"] if c["id"] == cid), {"title": cid})
        return f"trigger_scope: collection {coll['title']}"

    if scope_key == "pick_products" and picked_product_ids:
        labels = []
        for pid in picked_product_ids:
            p = next((x for x in store["products"] if x["id"] == pid), None)
            if p:
                labels.append(p["title"])
        return f"trigger_scope: specific products — {', '.join(labels)}"

    return f"trigger_scope: {scope_key}"


def format_stage2_catalog_reply(merchant_lines: List[str], llm_resolutions: List[str]) -> tuple[str, str]:
    """Return (merchant-visible chat text, LLM resolution note)."""
    merchant_body = "\n".join(f"- {line}" for line in merchant_lines if line.strip())
    merchant_msg = merchant_body.strip()
    llm_note = (
        "[CATALOG RESOLUTION — map to IR fields internally; merchant did not author this block]\n"
        + "\n".join(f"- {r}" for r in llm_resolutions if r.strip())
    )
    return merchant_msg, llm_note
