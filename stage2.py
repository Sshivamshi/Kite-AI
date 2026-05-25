#!/usr/bin/env python3
"""
Stage 2 — IR Builder (CLI)
===========================
Receives the validated prompt + family from Stage 1.
Asks the merchant targeted questions to fill every required IR field.
Never assumes any value — always confirms with the merchant.
Outputs the complete, validated IR JSON once all fields are confirmed.

Input  (stdin or CLI arg):
    {"prompt": "Spend $100 get 10% off, spend $200 get 20% off",
     "family": "tiered_discount",
     "family_label": "📊 Tiered Discount"}

Output (stdout, last line):
    IR: <complete IR json>

Usage:
    python stage2.py                                     # reads stdin
    python stage2.py '{"prompt":"...", "family":"..."}'  # CLI arg
    echo '{"prompt":"...", "family":"..."}' | python stage2.py
"""

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ollama import Client

from config import (
    DEFAULT_MODEL,
    FAMILY_CAPABILITIES,
    FAMILY_LABELS,
    OLLAMA_API_KEY,
    OLLAMA_CLOUD_HOST,
    STAGE2_STRICT_CONFIRMATION,
    STORE_CONTEXT,
    SUBTOTAL_TRIGGERS,
    VALID_ELIGIBILITY_TYPES,
    VALID_FAMILIES,
    VALID_TRIGGER_TYPES,
    format_store_context_json_for_llm,
    format_store_picklists_for_prompt,
    get_store_context,
    get_store_context_schema,
    humanize_confirmed_fields,
)

OLLAMA_HOST = OLLAMA_CLOUD_HOST
OLLAMA_KEY = OLLAMA_API_KEY
MODEL = DEFAULT_MODEL

FAMILY_REFERENCE = """\
═══════════════════════════════════════════════════════════
PROMOTION FAMILY REFERENCE — what each family IS (from Stage 1)
═══════════════════════════════════════════════════════════

── free_gift ──
WHAT IT IS:
  Customer reaches a trigger threshold → receives a FREE PHYSICAL PRODUCT
  added to their cart. The reward is a product, not a discount.

TRIGGER (one of): cart_quantity | cart_subtotal | cart_currency | cart_attributes |
  collection_quantity | collection_subtotal | product_quantity | product_subtotal
REWARD: free_gift (one free product; exact SKU can be admin-selected later)
ELIGIBILITY: customer_tag | logged_in | market | customer_segment | everyone (empty [] = all)
CART RULES:
  • cart_subtotal / collection_subtotal / product_subtotal → ONE currency field on
    trigger (threshold currency = cart currency). Ask once. Store as trigger.currency.
  • cart_currency trigger → ONLY when promo gates checkout currency with NO spend/qty
    threshold (e.g. "USD carts only"). Uses trigger.type cart_currency + trigger.currency.
  • cart_quantity → do NOT ask currency unless merchant explicitly needs checkout-currency restriction.
  • cart_attributes → confirm none or pick from catalog when not stated.

KEY SIGNALS: "free gift", "free sample", "complimentary product", "get a [item] free"
NOT free_gift: 2+ threshold→reward pairs → tiered_discount; % or $ off → buy_x_get_y

── buy_x_get_y ──
WHAT IT IS:
  ONE trigger → ONE reward: "Do X → get Y". X = quantity/spend threshold;
  Y = discount or free item on a target (product, collection, cheapest, same item, order).

TRIGGER (one of): same 6 product/collection/cart types plus cart_currency | cart_attributes
REWARD (one of): percentage_off_y | fixed_amount_off_y | free_y (→ percentage_off_y value 100)
Y TARGET: specific_product | specific_collection | cheapest_eligible | same_item
ELIGIBILITY: customer_tag | logged_in | market | customer_segment | everyone

KEY SIGNALS: "Buy 2 get 1 free", "Spend $100 get 10% off", single X→Y deal
NOT buy_x_get_y: multiple escalating tiers → tiered_discount

── tiered_discount ──
WHAT IT IS:
  TWO OR MORE tiers in the SAME prompt. Each tier: threshold + discount on the
  SAME dimension (spend ladder or quantity ladder). Higher threshold → better deal.

TRIGGER (per tier, same type): same cart/collection/product types plus cart_currency | cart_attributes
REWARD (per tier): percentage_off | fixed_amount_off
TIER BEHAVIOR: best_tier_only | all_matching_tiers
MINIMUM: 2 tiers
ELIGIBILITY: customer_tag | logged_in | market | customer_segment | everyone

KEY SIGNALS: "Spend $100 get 10%, spend $200 get 20%", "Buy 2 get 10%, buy 4 get 20%"
NOT tiered_discount: single threshold + single reward → buy_x_get_y or free_gift
"""


# ── System Prompt Builder ───────────────────────────────────────────────────

def build_system_prompt(store_context: dict, family: str) -> str:
    store_doc = get_store_context()
    store_context_json = format_store_context_json_for_llm(store_doc)
    picklists = format_store_picklists_for_prompt(STORE_CONTEXT)
    schema = get_store_context_schema()
    currency_codes = " | ".join(
        c["currency_code"] for c in schema.get("currencies") or []
    )

    return f"""\
ROLE
────
You are Stage 2 of the Kite promotion builder.

Stage 1 has already confirmed the promotion is supported and identified its family.
Your job: ask the merchant targeted questions to fill EVERY required field in the IR.
You must NEVER assume any value — keep clarifying until you are 100% sure.
Only output status "complete" when every IR field is explicitly confirmed.

Output ONLY valid JSON. No markdown. No explanation outside JSON.

{STAGE2_STRICT_CONFIRMATION}

═══════════════════════════════════════════════════════════
STORE CONTEXT JSON — authoritative catalog for questions & IR
Use ONLY values from store_context_schema when building options and the IR.
Merchants know product names, categories, collections, countries, and customer tags.
═══════════════════════════════════════════════════════════

{store_context_json}

═══════════════════════════════════════════════════════════
DYNAMIC PICKLISTS — use EXACT strings in question options
═══════════════════════════════════════════════════════════

{picklists}

═══════════════════════════════════════════════════════════
HOW TO USE STORE CONTEXT
═══════════════════════════════════════════════════════════
• products[].category + supported_product_values — product & category scope options
• collections — collection scope options (collection_quantity / collection_subtotal)
• customer_tags — eligibility when merchant mentions VIP, wholesale, etc.
• customer_segments — eligibility when merchant mentions segments / cohorts
• cart_attributes — cart must include these (gift wrap, subscription, etc.)
• markets[].country + market_code — country/market eligibility
• currencies[].currency_code + symbols_and_aliases — threshold currency on subtotal triggers (one field, one question)
• Question options MUST be exact strings from this JSON — not open-ended
• Category or collection scope = ALL supported_product_values in that group
• Use product/collection/market/tag names in the IR — not internal system keys in questions

{FAMILY_REFERENCE}

═══════════════════════════════════════════════════════════
FAMILY CAPABILITIES MATRIX (LLM-INTERNAL) — infer from Stage 1 family
Merchants never see this. Map their plain-language answers to these values.
═══════════════════════════════════════════════════════════

{json.dumps(FAMILY_CAPABILITIES, indent=2)}

═══════════════════════════════════════════════════════════
HOW TO THINK — Stage 1 family → IR output
═══════════════════════════════════════════════════════════

1. Read family from Stage 1 input ({family}). Lock to that family only.
2. Look up triggers / rewards / eligibility in FAMILY CAPABILITIES MATRIX.
3. Parse the merchant prompt for explicit values (amounts, currency, quantities,
   product/collection names, customer tags, markets).
4. Build confirmed_fields from prompt + conversation — only mark confirmed when
   the merchant stated it or chose an option.
5. List UNCONFIRMED IR fields still needed for a complete IR for this family.
6. If anything unconfirmed → status "clarifying" with 1–2 questions (options from store).
7. If ALL required IR fields confirmed → status "complete" with full ir JSON.
8. Map capability values to IR:
   • triggers → ir.trigger.type (or each tier.trigger.type)
   • rewards → ir.reward.type and value (free_y → percentage_off_y value 100)
   • y_allocation → ir.reward.y_target.type (buy_x_get_y only)
   • tier_behavior → ir.tier_behavior (tiered_discount only)
   • eligibility → ir.customer_eligibility array (tag | logged_in | market | segment)

WHAT TO CLARIFY (priority order):
  1. trigger_type + trigger_value
  2. currency — ONLY when needed (see CURRENCY RULES below); never ask twice
  3. cart_attributes when not stated — confirm none or pick from catalog
  4. trigger_scope (all_products vs collection vs product) when not in prompt
  5. reward fields (gift product, discount %, Y target, tier values)
  6. tier_behavior (tiered_discount only, if not stated)
  7. customer_eligibility — tags, logged-in, market, customer segment (if not stated → ask; all = [])

═══════════════════════════════════════════════════════════
CURRENCY RULES — one question, one field, no double-asking
═══════════════════════════════════════════════════════════

Subtotal triggers (cart_subtotal | collection_subtotal | product_subtotal):
  • Merchant says "$100" / "₹500" / "spend 100" → ask ONE currency question (id: "currency").
  • Store answer as trigger.currency on the subtotal trigger. That IS the cart currency
    for the threshold — do NOT ask a separate "cart currency" question.
  • confirmed_fields key: "currency". IR field: trigger.currency (same value).

cart_currency trigger type (currency-only promos):
  • Use ONLY when the promotion has NO spend amount and NO quantity threshold —
    just "applies when cart checkout currency is X" (e.g. "USD carts only").
  • IR: {{"type":"cart_currency","operator":"equals","currency":"USD"}} — no value field.
  • Do NOT use cart_currency trigger when merchant also gave a spend/qty threshold;
    use cart_subtotal or cart_quantity instead with trigger.currency if needed.

cart_quantity trigger:
  • Default: no currency field. Item count is currency-agnostic.
  • Ask currency ONLY if merchant explicitly mentions multi-currency checkout restriction
    (e.g. "3 items in USD cart"). Then set optional trigger.currency — still ONE question.

NEVER in the same promotion:
  • Ask "which currency for $100?" AND a separate "which cart currency?" — redundant.
  • Use both cart_subtotal + cart_currency triggers for the same currency gate.

═══════════════════════════════════════════════════════════
CURRENT PROMOTION FAMILY: {family.upper()}
═══════════════════════════════════════════════════════════

══ FAMILY: free_gift ══════════════════════════════════════

A customer meets a threshold condition and receives a free physical product.

REQUIRED FIELDS you must confirm before outputting IR:
  1. trigger_type     → what the customer must do
                        VALUES: cart_quantity | cart_subtotal | cart_currency |
                                cart_attributes | collection_quantity | collection_subtotal |
                                product_quantity | product_subtotal
  2. trigger_value    → the numeric threshold (e.g. 100, 2, 3)
                        NOT required for cart_currency or cart_attributes triggers
  3. currency          → ONE field, ONE question when required:
                        • REQUIRED for cart_subtotal | collection_subtotal | product_subtotal
                          (threshold currency = cart currency → trigger.currency)
                        • OPTIONAL on cart_quantity only if merchant explicitly needs
                          checkout-currency restriction in a multi-currency store
                        • cart_currency trigger type: currency is the ONLY field (no value)
                        • NEVER ask currency twice for the same subtotal promotion
                        VALUES: {currency_codes}
  4. cart_attributes  → when trigger is cart_attributes OR merchant adds cart attribute rules
                        VALUES: exact strings from cart_attributes picklist | none confirmed
  5. trigger_scope    → what products does the trigger apply to?
                        VALUES: all_products | a specific collection (show list) | a specific product (show list)
  6. gift_product     → which product is the free gift?
                        VALUES: any product id from store (show list) | admin_selection_required
  7. gift_quantity    → how many free units? (default ask: usually 1)
  8. customer_eligibility → who can trigger this?
                        VALUES: all_customers | customer_tag (which tag?) | logged_in |
                                market (which market?) | customer_segment (which segment?)

IR SHAPE for free_gift:
{{
  "feature": "free_gift",
  "trigger": {{
    "type": "<trigger_type>",
    "operator": ">=",
    "value": <number>,
    "currency": "<USD|GBP|INR|EUR|AUD>",   // subtotal triggers ONLY (one field = threshold + cart currency)
    "cart_attributes": [],                 // optional; omit or [] if none required
    "scope": {{
      "type": "all_products"               // or "collection" or "product"
      // if collection: "collection_id": "<id>", "collection_title": "<title>"
      // if product:    "product_id": "<id>",    "product_title": "<title>"
    }}
  }},
  "reward": {{
    "type": "free_gift",
    "gift_product": {{
      "status": "resolved",                // or "admin_selection_required"
      "query": "<what merchant said>",
      "resolved_id": "<product_id>"        // null if admin_selection_required
    }},
    "quantity": <number>
  }},
  "customer_eligibility": [
    // empty [] if all customers (no tag, market, segment, or login restriction)
    // {{"type":"customer_tag","operator":"includes","value":"<TAG>"}}
    // {{"type":"customer_segment","operator":"includes","value":"<SEGMENT>"}}
    // {{"type":"logged_in"}} for members only
    // {{"type":"market","operator":"includes","value":"<MARKET_CODE>"}}
  ],
  "tier_behavior": null
}}

══ FAMILY: buy_x_get_y ════════════════════════════════════

A customer buys X qualifying items and Y items receive a discount.

REQUIRED FIELDS:
  1. trigger_type     → same 6 options as above
  2. trigger_value    → the X threshold
  3. currency           → only if subtotal trigger (one question → trigger.currency)
  4. trigger_scope    → which products/collection are the X items from?
  5. reward_type      → what discount does Y receive?
                        VALUES: percentage_off_y | fixed_amount_off_y | free_y (stored as percentage_off_y value:100)
  6. reward_value     → the percentage or fixed amount (not needed for free_y)
  7. y_target_type    → what are the Y items?
                        VALUES: specific_product | specific_collection | cheapest_eligible | same_item
  8. y_target_ref     → only if y_target_type is specific_product or specific_collection
                        (show product/collection list)
  9. y_quantity       → how many Y items get the reward?
  10. customer_eligibility → tags | logged_in | market | customer_segment (same as free_gift)

Cart-attribute-only trigger IR example:
{{
  "feature": "free_gift",
  "trigger": {{
    "type": "cart_attributes",
    "operator": "includes_all",
    "attributes": ["Gift wrap requested"]
  }},
  ...
}}

Cart-currency-only trigger (no spend/qty — rare):
{{
  "trigger": {{
    "type": "cart_currency",
    "operator": "equals",
    "currency": "USD"
  }}
}}

IR SHAPE for buy_x_get_y:
{{
  "feature": "buy_x_get_y",
  "trigger": {{
    "type": "<trigger_type>",
    "operator": ">=",
    "value": <number>,
    "currency": "<currency>",              // subtotal triggers only (one field, not separate cart_currency)
    "cart_attributes": [],                 // optional cart attribute requirements
    "scope": {{ ... }}                     // only if collection/product trigger
  }},
  "reward": {{
    "type": "percentage_off_y",            // or "fixed_amount_off_y"
    "value": <number>,                     // 100 if free_y
    "y_target": {{
      "type": "product",                   // or "collection" | "cheapest_eligible" | "same_item"
      "status": "resolved",               // or "admin_selection_required"
      "query": "<what merchant said>",
      "resolved_id": "<id>"              // null if cheapest_eligible/same_item/admin_selection
    }},
    "quantity": <number>
  }},
  "customer_eligibility": [ ... ],
  "tier_behavior": null
}}

══ FAMILY: tiered_discount ════════════════════════════════

Multiple thresholds, each tier unlocks a different discount.

REQUIRED FIELDS:
  1. tier_count         → confirmed from prompt (need at least 2 tiers)
  2. per_tier:
     a. trigger_type    → same 6 options (usually consistent across tiers)
     b. trigger_value   → the threshold value for each tier
     c. currency        → only if subtotal trigger (one question → each tier's trigger.currency)
     d. reward_type     → percentage_off | fixed_amount_off
     e. reward_value    → the discount value for each tier
  3. trigger_scope      → all_products | collection | product (same for all tiers)
  4. tier_behavior      → what happens when customer qualifies for multiple tiers?
                          VALUES: best_tier_only | all_matching_tiers
  5. customer_eligibility → tags | logged_in | market | customer_segment
  6. cart_attributes    → confirm none or pick from catalog when not stated (NOT currency — see CURRENCY RULES)

IR SHAPE for tiered_discount:
{{
  "feature": "tiered_discount",
  "tiers": [
    {{
      "trigger": {{
        "type": "<trigger_type>",
        "operator": ">=",
        "value": <number>,
        "currency": "<currency>",           // subtotal triggers
        "cart_attributes": []             // optional; same for all tiers when applicable
      }},
      "reward": {{
        "type": "percentage_off",          // or "fixed_amount_off"
        "value": <number>
      }}
    }}
    // one object per tier, sorted ascending by trigger value
  ],
  "tier_behavior": "best_tier_only",       // or "all_matching_tiers"
  "customer_eligibility": [ ... ]
}}

═══════════════════════════════════════════════════════════
YOUR TASK — step by step on every turn
═══════════════════════════════════════════════════════════

STEP 1 — Read the full conversation history.
          Build a "confirmed_fields" map of everything you now know.

STEP 2 — Check which required fields for the family are still UNCONFIRMED.
          A field is confirmed only when the merchant explicitly stated it.
          A field is NOT confirmed if:
            • The merchant did not mention it at all
            • The value is ambiguous (e.g. "$100" without saying USD)
            • A product/collection name matches multiple store items

STEP 3 — If ANY required field is unconfirmed → status MUST be "clarifying".
          Pick the 1–2 most important unconfirmed fields.
          Every question MUST include options copied from DYNAMIC PICKLISTS above.
          Examples:
            • gift product → list exact products (especially Gift Items category)
            • trigger scope → All products | [each collection title]
            • ambiguous "shirt" → list matching products: Shirt, Premium T-Shirt, Polo Shirt
            • market → list exact markets: India (IN), United States (US), …
            • customer group → VIP | Wholesale | Gold | … | All customers
            • customer segment → High Value Customers | Repeat Buyers | … | All segments
            • cart attributes → Gift wrap requested | Contains subscription | … | No special requirements
            • currency (subtotal only) → USD | GBP | INR | … — ONE question, stored as trigger.currency

STEP 4 — status "complete" ONLY when 100% of required IR fields are confirmed.
          Zero ambiguity allowed. Build full IR with exact catalog IDs.
          When a product/collection name is not in catalog after merchant review,
          use admin_selection_required with resolved_id null.

═══════════════════════════════════════════════════════════
RULES — never break these
═══════════════════════════════════════════════════════════

1. NEVER assume any field value — 100% confirmation required before complete.
   If merchant says "$100" on a subtotal trigger — ask currency ONCE using CURRENCIES
   picklist; store as trigger.currency. Do NOT ask a second cart-currency question.
   If scope unclear — ask with COLLECTIONS or PRODUCT CATEGORIES picklist.
   If product name ambiguous — list ALL matching products by exact title + id.
   If market/country vague — list exact MARKETS with codes.
   If VIP/wholesale/segment implied — list exact CUSTOMER TAGS, CUSTOMER SEGMENTS, or All customers.
   If cart attribute rules possible — list CART ATTRIBUTES or confirm no special requirements.
   Use cart_currency trigger type ONLY for currency-only promos with no spend/qty threshold.

2. ALWAYS include picklist options in every question (never open-ended).
   Pull options from DYNAMIC PICKLISTS — product titles, collection titles,
   market codes, customer tags, currencies. Merchant must pick exactly one
   (or "All customers" / "All markets" / "All products" when applicable).

3. When a product or collection NAME is mentioned in the prompt:
   - Exact single match in catalog → confirm in confirmed_fields, do not re-ask
   - Multiple matches → list each match with id in options, ask WHICH ONE
   - No match → offer closest catalog options + admin selection fallback

4. Resolve confirmed choices to exact IDs in the final IR (e.g. Cap → p6).

5. MAX 2 questions per turn. Priority: trigger → currency (if subtotal) → cart attributes → reward → scope → eligibility.

6. Never re-ask a field already in confirmed_fields.

7. status "complete" is FORBIDDEN if any required field is inferred but unconfirmed.

═══════════════════════════════════════════════════════════
OUTPUT SCHEMA — two formats only
═══════════════════════════════════════════════════════════

── When still clarifying ──
{{
  "status": "clarifying",
  "confirmed_fields": {{
    "<field>": "<confirmed_value>"
  }},
  "questions": [
    {{
      "id": "<field_id>",
      "question": "<specific question with context>",
      "options": ["<option 1>", "<option 2>"]
    }}
  ]
}}

── When all fields confirmed ──
{{
  "status": "complete",
  "ir": {{
    // complete IR JSON matching the shape for the family
    // use exact product IDs, collection IDs, currency codes from store context
  }}
}}

HARD RULES:
  status must be exactly "clarifying" or "complete"
  complete ONLY when 100% sure — every IR field explicitly confirmed
  questions: 1 or 2 items maximum, always include options from DYNAMIC PICKLISTS
  ir: must be complete — use exact product/collection/market/tag/currency IDs
  confirmed_fields: always present in "clarifying" responses
  ALL field values must come from the store catalog in config

═══════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
═══════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────
[tiered_discount] Turn 1
Input: {{"prompt": "Spend $100 get 10% off, spend $200 get 20% off", "family": "tiered_discount"}}
──────────────────────────────────────────────────────────
Analysis:
  Known from prompt: 2 tiers, subtotal trigger, values 100+200, rewards 10%+20%
  trigger_type: cart_subtotal (spend = subtotal) ← currency needed once (trigger.currency)
  trigger_value: 100, 200 ✓
  reward_type: percentage_off ✓
  reward_value: 10, 20 ✓
  UNCONFIRMED: currency (one question only), cart_attributes, trigger_scope, tier_behavior, customer_eligibility
Output:
{{"status":"clarifying","confirmed_fields":{{"tiers":"2 tiers confirmed: $100→10%, $200→20%","trigger_type":"cart_subtotal","reward_type":"percentage_off"}},"questions":[{{"id":"currency","question":"The thresholds $100 and $200 — which currency are these in?","options":["USD (US Dollar)","GBP (British Pound)","INR (Indian Rupee)","EUR (Euro)","AUD (Australian Dollar)"]}},{{"id":"cart_attributes","question":"Does the cart need any special attributes for this promotion to apply?","options":["No special cart requirements","Contains subscription product","Gift wrap requested","Contains bundle","B2B / wholesale order"]}}]}}

──────────────────────────────────────────────────────────
[tiered_discount] Turn 2 — merchant answers
Input: "USD, all products"
──────────────────────────────────────────────────────────
Analysis:
  Now confirmed: currency=USD, scope=all_products
  UNCONFIRMED: tier_behavior, customer_eligibility (segment if applicable)
Output:
{{"status":"clarifying","confirmed_fields":{{"tiers":"2 tiers: $100→10%, $200→20%","trigger_type":"cart_subtotal","reward_type":"percentage_off","currency":"USD","trigger_scope":"all_products","cart_attributes":"none"}},"questions":[{{"id":"tier_behavior","question":"When a customer qualifies for both tiers (e.g. spends $200), which tier should apply?","options":["Best tier only (apply the highest matching discount — e.g. 20% only)","All matching tiers (apply both 10% and 20% — stacked)"]}},{{"id":"customer_eligibility","question":"Who can trigger this promotion?","options":["All customers","VIP customers only","High Value Customers segment only","Logged-in customers only","United States market only","India market only"]}}]}}

──────────────────────────────────────────────────────────
[tiered_discount] Turn 3 — merchant answers
Input: "best tier only, all customers"
──────────────────────────────────────────────────────────
Analysis:
  All required fields now confirmed. Build complete IR.
Output:
{{"status":"complete","ir":{{"feature":"tiered_discount","tiers":[{{"trigger":{{"type":"cart_subtotal","operator":">=","value":100,"currency":"USD"}},"reward":{{"type":"percentage_off","value":10}}}},{{"trigger":{{"type":"cart_subtotal","operator":">=","value":200,"currency":"USD"}},"reward":{{"type":"percentage_off","value":20}}}}],"tier_behavior":"best_tier_only","customer_eligibility":[]}}}}

──────────────────────────────────────────────────────────
[free_gift] Turn 1
Input: {{"prompt": "VIP customers spend $100 and get a free gift", "family": "free_gift"}}
──────────────────────────────────────────────────────────
Analysis:
  Known: trigger=cart_subtotal($100), reward=free_gift, eligibility=VIP
  VIP confirmed against customer_tags list ✓
  UNCONFIRMED: currency, trigger_scope, gift_product, gift_quantity
Output:
{{"status":"clarifying","confirmed_fields":{{"trigger_type":"cart_subtotal","trigger_value":100,"customer_eligibility":"customer_tag: VIP"}},"questions":[{{"id":"currency","question":"The $100 threshold — which currency?","options":["USD","GBP","INR","EUR","AUD"]}},{{"id":"trigger_scope","question":"Does the $100 apply to all products, or only items from a specific collection?","options":["All products","Summer Collection","Skincare","Accessories","Apparel","Electronics","Fitness","Gift Items"]}}]}}

──────────────────────────────────────────────────────────
[free_gift] Turn 2
Input: "USD, all products"
──────────────────────────────────────────────────────────
Output:
{{"status":"clarifying","confirmed_fields":{{"trigger_type":"cart_subtotal","trigger_value":100,"currency":"USD","trigger_scope":"all_products","customer_eligibility":"customer_tag: VIP"}},"questions":[{{"id":"gift_product","question":"Which product should be given as the free gift?","options":["Sample Gift","Mini Sample Pack","Mystery Gift Box","Welcome Gift","Cap","Travel Tote Bag","Other product — let admin choose from full catalog"]}},{{"id":"gift_quantity","question":"How many units of the free gift should each qualifying customer receive?","options":["1 (most common)","2","3","Other number"]}}]}}

──────────────────────────────────────────────────────────
[free_gift] Turn 3
Input: "Sample Gift, 1"
──────────────────────────────────────────────────────────
Output:
{{"status":"complete","ir":{{"feature":"free_gift","trigger":{{"type":"cart_subtotal","operator":">=","value":100,"currency":"USD","scope":{{"type":"all_products"}}}},"reward":{{"type":"free_gift","gift_product":{{"status":"resolved","query":"Sample Gift","resolved_id":"p1"}},"quantity":1}},"customer_eligibility":[{{"type":"customer_tag","operator":"includes","value":"VIP"}}],"tier_behavior":null}}}}

──────────────────────────────────────────────────────────
[buy_x_get_y] Turn 1
Input: {{"prompt": "Buy 2 shirts and get 1 cap free", "family": "buy_x_get_y"}}
──────────────────────────────────────────────────────────
Analysis:
  Known: trigger_value=2, "shirts"→Shirt(p10) matches store, reward=free_y, y_target="cap"→Cap(p5) matches store, y_quantity=1
  trigger_type: product_quantity (buy N of specific product = product_quantity)
  UNCONFIRMED: customer_eligibility
  All core fields resolvable from store catalog.
Output:
{{"status":"clarifying","confirmed_fields":{{"trigger_type":"product_quantity","trigger_value":2,"trigger_scope":"Shirt (p10)","reward_type":"free_y","y_target":"Cap (p5)","y_quantity":1}},"questions":[{{"id":"customer_eligibility","question":"Who can trigger this promotion?","options":["All customers","VIP customers only","Wholesale customers only","Subscriber customers only","Premium customers only","New customers only","Logged-in customers only","Specific market (US / UK / IN / DE / AU)"]}}]}}

──────────────────────────────────────────────────────────
[buy_x_get_y] Turn 2
Input: "all customers"
──────────────────────────────────────────────────────────
Output:
{{"status":"complete","ir":{{"feature":"buy_x_get_y","trigger":{{"type":"product_quantity","operator":">=","value":2,"scope":{{"type":"product","product_id":"p10","product_title":"Shirt"}}}},"reward":{{"type":"percentage_off_y","value":100,"y_target":{{"type":"product","status":"resolved","query":"cap","resolved_id":"p5"}},"quantity":1}},"customer_eligibility":[],"tier_behavior":null}}}}
"""


# ── Ollama Client ────────────────────────────────────────────────────────────

def get_client() -> Client:
    return Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
    )


def call_llm(client: Client, system_prompt: str, history: list) -> dict:
    """
    Call Ollama with full conversation history.
    Returns parsed JSON dict.
    Raises ValueError on bad JSON or empty response.
    """
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
        ],
        format="json",
    )
    raw = (response.message.content or "").strip()

    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    if not cleaned:
        raise ValueError("LLM returned empty response")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nRaw: {raw[:400]}")


# ── Validation ───────────────────────────────────────────────────────────────

VALID_STATUSES = {"clarifying", "complete"}
VALID_REWARD_TYPES = {
    "free_gift", "percentage_off", "fixed_amount_off",
    "percentage_off_y", "fixed_amount_off_y",
}


def _validate_trigger(trigger: dict, label: str = "trigger") -> None:
    """Validate a single IR trigger object."""
    ttype = trigger.get("type")
    if ttype not in VALID_TRIGGER_TYPES:
        raise ValueError(f"{label} invalid trigger type: {ttype!r}")

    if ttype == "cart_attributes":
        attrs = trigger.get("attributes")
        if not isinstance(attrs, list) or not attrs:
            raise ValueError(f"{label}: cart_attributes trigger missing attributes list")
        return

    if ttype == "cart_currency":
        if not trigger.get("currency"):
            raise ValueError(f"{label}: cart_currency trigger missing currency")
        return

    if trigger.get("value") is None:
        raise ValueError(f"{label}.value is null")

    if ttype in SUBTOTAL_TRIGGERS and not trigger.get("currency"):
        raise ValueError(f"{label}: subtotal trigger missing currency")

    cart_attrs = trigger.get("cart_attributes")
    if cart_attrs is not None and not isinstance(cart_attrs, list):
        raise ValueError(f"{label}: cart_attributes must be an array when present")


def _validate_customer_eligibility(rules: Any) -> None:
    """Validate customer_eligibility array entries."""
    if not isinstance(rules, list):
        raise ValueError("customer_eligibility must be an array")
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"customer_eligibility[{i}] must be an object")
        rtype = rule.get("type")
        if rtype not in VALID_ELIGIBILITY_TYPES:
            raise ValueError(f"customer_eligibility[{i}] invalid type: {rtype!r}")
        if rtype in ("customer_tag", "market", "customer_segment") and not rule.get("value"):
            raise ValueError(f"customer_eligibility[{i}] ({rtype}) missing value")


def validate_result(result: dict, family: str) -> None:
    """
    Validate LLM response against expected schema.
    Raises ValueError on any violation.
    """
    status = result.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status!r}. Must be 'clarifying' or 'complete'.")

    if status == "clarifying":
        if "confirmed_fields" not in result:
            raise ValueError("clarifying response missing confirmed_fields")
        questions = result.get("questions", [])
        if not questions:
            raise ValueError("clarifying response has no questions")
        if len(questions) > 2:
            raise ValueError(f"Too many questions: {len(questions)} (max 2)")
        for q in questions:
            if not q.get("question"):
                raise ValueError("Question missing question text")
            if not q.get("options"):
                raise ValueError(f"Question '{q.get('id')}' missing options")
        return

    # status == "complete"
    ir = result.get("ir")
    if not ir:
        raise ValueError("complete response missing ir")
    if ir.get("feature") not in VALID_FAMILIES:
        raise ValueError(f"IR feature {ir.get('feature')!r} not in supported families")

    # Family-specific IR validation
    if family == "tiered_discount":
        tiers = ir.get("tiers")
        if not tiers or len(tiers) < 2:
            raise ValueError(f"tiered_discount requires ≥2 tiers, got {len(tiers or [])}")
        for i, tier in enumerate(tiers):
            _validate_trigger(tier.get("trigger", {}), f"Tier {i} trigger")
            r = tier.get("reward", {})
            if r.get("type") not in ("percentage_off", "fixed_amount_off"):
                raise ValueError(f"Tier {i} invalid reward type: {r.get('type')!r}")
            if r.get("value") is None:
                raise ValueError(f"Tier {i} reward.value is null")
        if not ir.get("tier_behavior"):
            raise ValueError("tiered_discount missing tier_behavior")
        _validate_customer_eligibility(ir.get("customer_eligibility", []))

    elif family == "free_gift":
        _validate_trigger(ir.get("trigger", {}))
        reward = ir.get("reward", {})
        if reward.get("type") != "free_gift":
            raise ValueError(f"free_gift family reward type must be 'free_gift', got {reward.get('type')!r}")
        gp = reward.get("gift_product", {})
        if not gp.get("status"):
            raise ValueError("gift_product missing status")
        _validate_customer_eligibility(ir.get("customer_eligibility", []))

    elif family == "buy_x_get_y":
        _validate_trigger(ir.get("trigger", {}))
        reward = ir.get("reward", {})
        if reward.get("type") not in VALID_REWARD_TYPES:
            raise ValueError(f"Invalid reward type: {reward.get('type')!r}")
        if reward.get("type") != "free_gift" and reward.get("value") is None:
            raise ValueError("reward.value is null")
        if not reward.get("y_target"):
            raise ValueError("buy_x_get_y missing y_target")
        _validate_customer_eligibility(ir.get("customer_eligibility", []))


# ── Display ──────────────────────────────────────────────────────────────────

LINE = "─" * 60

def _print(text: str = "", indent: int = 2) -> None:
    prefix = " " * indent
    for line in text.splitlines():
        print(f"{prefix}{line}")

def _input(prompt: str = "") -> str:
    try:
        if prompt:
            _print(prompt)
        sys.stdout.write("  > ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nExiting Stage 2.")
        sys.exit(0)

def show_clarifying(result: dict) -> None:
    confirmed = result.get("confirmed_fields", {})
    if confirmed:
        _print("What we know so far:")
        for line in humanize_confirmed_fields(confirmed):
            _print(f"  ✓ {line}")
        print()

    for q in result.get("questions", []):
        _print(q["question"])
        for i, opt in enumerate(q.get("options", []), 1):
            _print(f"  {i}. {opt}")

def show_complete(ir: dict, family: str) -> None:
    family_labels = dict(FAMILY_LABELS)
    print(LINE)
    _print(f"✓  IR complete — {family_labels.get(family, family)}")
    print()
    _print(json.dumps(ir, indent=2))
    print(LINE)


def format_clarifying_message(result: dict) -> str:
    parts: List[str] = []
    confirmed = result.get("confirmed_fields") or {}
    if confirmed:
        lines = humanize_confirmed_fields(confirmed)
        parts.append("What we know so far:\n" + "\n".join(f"  • {line}" for line in lines))
    for q in result.get("questions", []):
        qtext = (q.get("question") or "").strip()
        if qtext:
            parts.append(qtext)
    return "\n\n".join(parts).strip()


def format_complete_message(family: str) -> str:
    label = FAMILY_LABELS.get(family, family)
    return f"Promotion IR is complete — {label}"


# ── Session API (Streamlit / programmatic) ───────────────────────────────────

@dataclass
class Stage2TurnResult:
    status: str
    result: dict
    ir: Optional[dict] = None
    error: Optional[str] = None
    turns: int = 0


@dataclass
class Stage2Session:
    stage1_input: Dict[str, str]
    family: str
    history: List[Dict[str, str]] = field(default_factory=list)
    user_turns: List[str] = field(default_factory=list)
    turns: int = 0
    status: str = "active"
    ir: Optional[dict] = None
    last_result: Optional[dict] = None


def session_to_dict(session: Stage2Session) -> Dict[str, Any]:
    return {
        "stage1_input": session.stage1_input,
        "family": session.family,
        "history": session.history,
        "user_turns": session.user_turns,
        "turns": session.turns,
        "status": session.status,
        "ir": session.ir,
        "last_result": session.last_result,
    }


def session_from_dict(data: Dict[str, Any]) -> Stage2Session:
    return Stage2Session(
        stage1_input=dict(data["stage1_input"]),
        family=data["family"],
        history=list(data.get("history") or []),
        user_turns=list(data.get("user_turns") or []),
        turns=int(data.get("turns") or 0),
        status=data.get("status") or "active",
        ir=data.get("ir"),
        last_result=data.get("last_result"),
    )


class Stage2Classifier:
    """Turn-based Stage 2 IR builder for UI integration."""

    def __init__(
        self,
        client: Optional[Client] = None,
        store_context: Optional[dict] = None,
    ) -> None:
        self.client = client or get_client()
        self.store_context = store_context or STORE_CONTEXT

    def new_session(self, stage1_input: dict) -> Stage2Session:
        family = (stage1_input.get("family") or "").strip()
        if family not in VALID_FAMILIES:
            raise ValueError(f"Invalid family: {family!r}")
        if not (stage1_input.get("prompt") or "").strip():
            raise ValueError("stage1_input missing prompt")
        return Stage2Session(
            stage1_input={
                "prompt": stage1_input["prompt"].strip(),
                "family": family,
                "family_label": stage1_input.get("family_label") or FAMILY_LABELS.get(family, family),
            },
            family=family,
        )

    def process_turn(
        self,
        session: Stage2Session,
        user_input: str,
        catalog_resolution: str | None = None,
    ) -> Stage2TurnResult:
        user_input = user_input.strip()
        if not user_input:
            return Stage2TurnResult(
                status="error",
                result={},
                error="Empty message.",
                turns=session.turns,
            )

        session.user_turns.append(user_input)
        if session.turns == 0:
            content = json.dumps(session.stage1_input)
        else:
            content = user_input
        session.history.append({"role": "user", "content": content})
        session.turns += 1

        if catalog_resolution:
            session.history.append({
                "role": "user",
                "content": (
                    catalog_resolution
                    + "\n\nApply these exactly to confirmed_fields and advance the IR. "
                    "Category/collection scope = all products in that group — never re-ask for one SKU. "
                    "If all required fields are now confirmed → status complete with full ir JSON."
                ),
            })

        system_prompt = build_system_prompt(self.store_context, session.family)
        result, error = self._call_with_retry(system_prompt, session.history, session.family)
        if error:
            return Stage2TurnResult(
                status="error",
                result={},
                error=error,
                turns=session.turns,
            )

        session.history.append({"role": "assistant", "content": json.dumps(result)})
        session.last_result = result

        if result["status"] == "complete":
            session.status = "complete"
            session.ir = result["ir"]
            return Stage2TurnResult(
                status="complete",
                result=result,
                ir=result["ir"],
                turns=session.turns,
            )

        session.status = "active"
        return Stage2TurnResult(
            status="clarifying",
            result=result,
            turns=session.turns,
        )

    def _call_with_retry(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        family: str,
    ) -> tuple[Optional[dict], Optional[str]]:
        for attempt in range(2):
            try:
                raw = call_llm(self.client, system_prompt, history)
                validate_result(raw, family)
                return raw, None
            except ValueError as e:
                if attempt == 0:
                    history.append({"role": "assistant", "content": "{}"})
                    history.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM NOTE: Your last response failed validation: {e}. "
                            "Please respond again with valid JSON matching the schema exactly.]"
                        ),
                    })
                else:
                    return None, str(e)
            except Exception as e:
                return None, f"Connection error: {e}"
        return None, "Unknown LLM error"


# ── Core loop ─────────────────────────────────────────────────────────────────

def run(stage1_input: dict, store_context: dict = STORE_CONTEXT) -> dict:
    """
    Run the Stage 2 IR-building conversation (CLI).
    Receives stage1_input: {"prompt": "...", "family": "...", "family_label": "..."}
    Returns the complete IR dict.
    """
    prompt = stage1_input.get("prompt", "").strip()
    family = stage1_input.get("family", "").strip()
    label = stage1_input.get("family_label", family)

    classifier = Stage2Classifier(store_context=store_context)
    session = classifier.new_session(stage1_input)

    print(LINE)
    print(f"  KITE — Stage 2  ·  IR Builder  ·  {label}")
    print(f"  Prompt: \"{prompt}\"")
    print(LINE)

    user_input = json.dumps(session.stage1_input)

    while True:
        turn = classifier.process_turn(session, user_input)
        print()

        if turn.status == "error":
            _print(f"Error: {turn.error}")
            sys.exit(1)

        if turn.status == "complete":
            show_complete(turn.ir, family)
            return turn.ir

        show_clarifying(turn.result)
        print()
        user_input = _input()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Read stage1 input: CLI arg, stdin, or interactive
    raw_input: Optional[str] = None

    if len(sys.argv) > 1:
        raw_input = " ".join(sys.argv[1:]).strip()
    elif not sys.stdin.isatty():
        raw_input = sys.stdin.readline().strip()

    if raw_input:
        try:
            stage1_json = json.loads(raw_input)
        except json.JSONDecodeError as e:
            print(f"Error: invalid Stage 1 JSON input: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive fallback — ask for details manually
        print("Stage 1 JSON input not provided. Enter manually:")
        print('  Example: {"prompt": "Spend $100 get a free gift", "family": "free_gift", "family_label": "🎁 Free Gift"}')
        raw_input = input("  > ").strip()
        try:
            stage1_json = json.loads(raw_input)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    ir = run(stage1_json)

    # Machine-readable output for the next stage
    print(f"\nIR: {json.dumps(ir)}")