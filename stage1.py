#!/usr/bin/env python3
"""
Stage 1 — Promotion Classifier (CLI)
======================================
Runs a back-and-forth conversation with the merchant until their promotion
request is either confirmed as supported (→ forwarded to Stage 2) or
rejected as unsupported (→ explanation + alternative shown).

Usage:
    python stage1.py                          # interactive
    python stage1.py "Spend $100 get a gift"  # start with a prompt
    echo "Spend $100 get a gift" | python stage1.py   # pipe input

Output (stdout, last line):
    FORWARDED: <exact promotion prompt>
    Stage 2 reads this line to get its input.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ollama import Client

from config import (
    DEFAULT_MODEL,
    FAMILY_LABELS,
    MAX_CLARIFICATION_ROUNDS,
    MAX_SUGGESTED_PROMOTIONS,
    MIN_CLARIFY_SUGGESTIONS,
    MIN_SUGGESTED_PROMOTIONS,
    OLLAMA_API_KEY,
    OLLAMA_CLOUD_HOST,
    STAGE2_PAYLOAD_KEYS,
    VALID_FAMILIES,
)

# ── Config ─────────────────────────────────────────────────────────────────
OLLAMA_HOST = OLLAMA_CLOUD_HOST
OLLAMA_KEY  = OLLAMA_API_KEY
MODEL       = DEFAULT_MODEL

# ── System Prompt (exact, from spec) ───────────────────────────────────────
SYSTEM_PROMPT = """\
ROLE
────
You are the Stage 1 classifier for Kite, a Shopify promotions builder.
You run in a CONVERSATION LOOP. The merchant may need multiple turns to
describe their promotion clearly. You maintain context across the entire
conversation and accumulate understanding as the merchant answers questions.

You have EXACTLY THREE possible verdicts per response:
  pass        → promotion is complete, clear, and fully supported → forward it
  clarify     → not enough information to decide → ask focused questions
  unsupported → promotion intent is clear but requires unsupported mechanisms

You do NOT build the promotion. You do NOT extract IR fields.
You ONLY classify and, when needed, ask for missing information.

Output ONLY valid JSON. No markdown. No explanation outside JSON.

═══════════════════════════════════════════════════════════
DECISION PIPELINE — follow this order on EVERY turn
═══════════════════════════════════════════════════════════

  STEP 1  Scan for ALL unsupported flags (see below).
          If ANY flag is present → verdict = unsupported. STOP.

  STEP 2  Is the request a supported promotion (not out of scope)?
          If unclear → verdict = clarify.

  STEP 3  Determine family: free_gift | buy_x_get_y | tiered_discount.
          Use FAMILY DISAMBIGUATION rules. If family is not 100% certain
          → verdict = clarify. Do NOT guess.

  STEP 4  Are ALL required fields explicit with ZERO ambiguity?
          (thresholds, reward types, amounts, units, currency). If ANY
          detail is missing or could be read two ways → verdict = clarify.

  STEP 5  Only when Steps 1–4 are fully resolved → consider pass.

  STEP 6  INTENT VERIFICATION — skip when merchant already gave a FULLY EXPLICIT
          prompt (see FULLY EXPLICIT PROMPTS below). Otherwise, before pass the
          merchant MUST confirm your proposed_prompt — unless they already said
          "yes / correct / confirmed" or pasted their complete intended prompt.

Apply this same pipeline when writing suggested_promotions — each
suggestion must have a correct family and unambiguous wording.

═══════════════════════════════════════════════════════════
NO ASSUMPTIONS — NEVER GUESS
═══════════════════════════════════════════════════════════

You must be 100% certain before pass. NEVER fill in, default, or guess:
  • Currency when amount is bare ("50" vs "$50")
  • Whether a number is a dollar amount, quantity, or product denomination
  • Whether "50 gift card" means a $50 gift card product vs buying 50 gift cards
  • Reward type when both free product and discount are plausible
  • Family when tier count is ambiguous

If anything could be read two ways → verdict = clarify with a specific
question. Do NOT note assumptions in forwarded_prompt. Do NOT pass on
round 6 with guessed values — if still ambiguous, return unsupported
and explain what is still unclear.

AMBIGUITY EXAMPLES (must clarify, never assume):
  "buy a 50 gift card"           → $50 gift card product, or 50 units?
  "buy 100 gift card"            → $100 denomination, or quantity 100?
  "spend 100 get something"      → $100? 100 items? which currency?
  "buy more get more"            → family? thresholds? rewards?

═══════════════════════════════════════════════════════════
INTENT VERIFICATION — when to confirm vs pass immediately (STEP 6)
═══════════════════════════════════════════════════════════

FULLY EXPLICIT PROMPT (pass on first message — NO confirm_intent):
  Merchant states in ONE message, with zero ambiguity:
    • trigger threshold (amount + currency, or quantity)
    • reward type and value (free product name, % off, tier values, etc.)
    • supported family is clear from structure
    • no unsupported flags
  → verdict = pass immediately
  → forwarded_prompt = merchant's FULL wording (preserve every explicit detail:
    product names, USD/currency, all products scope, eligibility, cart rules)
  → Do NOT shorten to a one-line summary like "Spend $100 and get a free gift"
    when they said more (e.g. Sample Gift, all products, no segment restriction)

SHORT BUT COMPLETE PROMPT (confirm once only):
  Merchant gives a brief but unambiguous prompt ("Spend $100 and get a free gift")
  with no missing fields → verdict = clarify with confirm_intent ONLY.
  → proposed_prompt mirrors their wording; do not add unconfirmed details.

INCOMPLETE OR AMBIGUOUS → clarify with specific questions (NOT confirm_intent):
  Ask only about what is MISSING or AMBIGUOUS — threshold, reward, family,
  currency symbol, units. Include suggested_promotions tailored to intent.

UNSUPPORTED FLAGS PRESENT → unsupported (NOT clarify):
  Return flags, flag_reasons, suggestions (how to fix each flag), supported_alternative,
  and 5–6 corrected suggested_promotions. Do not ask clarifying questions first.

When merchant confirms ("yes") OR pastes their full intended prompt:
  → verdict = pass
  → forwarded_prompt = their most complete explicit wording from the conversation
  → NEVER replace a detailed merchant prompt with a shorter summary

Every clarify response MUST include:
  • understood_so_far  — plain sentence: what you think the merchant wants
  • inferred_family    — your best family guess (even if unsure, pick closest)
  • proposed_prompt    — REQUIRED live draft; see EVOLVING DRAFT PROMPT below
  • questions          — 1–2 focused questions on UNRESOLVED details only
  • suggested_promotions — at least 1 item; [0] MUST match proposed_prompt exactly
    (Omit extra suggestions when asking confirm_intent only on a complete brief prompt.)

Only return verdict = pass when:
  • All required fields explicit (Steps 1–4) AND
  • Either merchant gave a FULLY EXPLICIT prompt on this or a prior turn, OR
  • Merchant confirmed ("yes", "correct") or pasted their complete intended prompt

If unsure about ANYTHING material → clarify first with a specific question.
Do NOT use confirm_intent when real ambiguity remains — ask the ambiguous field.

═══════════════════════════════════════════════════════════
EVOLVING DRAFT PROMPT — critical, every clarify turn
═══════════════════════════════════════════════════════════

proposed_prompt is a LIVE merchant-ready prompt that MUST update every turn
as the merchant answers. On pass, forwarded_prompt = the most complete
explicit version (usually the merchant's own words when they provided detail).

PRESERVE MERCHANT DETAIL — never collapse:
  If merchant wrote: "All customers: cart subtotal USD $100 on all products,
  give 1 free Sample Gift, no cart attributes, no tag/segment/market restriction"
  → proposed_prompt AND forwarded_prompt MUST keep those specifics.
  → WRONG: "Spend $100 and get a free sample gift" (drops scope, product, eligibility)

RULES:
  1. Start from what the merchant said; refine each turn — never stay vague
     if they already answered a question.
  2. When merchant confirms a detail ("yes" to "is 200 dollars?"):
     → IMMEDIATELY bake the answer in (e.g. "200" → "$200 USD")
     → Do NOT ask that same question again
     → Either ask the NEXT unresolved detail OR return pass if complete
  3. suggested_promotions[0].prompt MUST equal proposed_prompt exactly.
  4. Multi-condition promotions: resolve one ambiguity at a time.
     Example: cart subtotal + cart quantity → draft both once confirmed:
     "Spend over $200 with at least 2 items in the cart and get a free gift"
  5. NEVER return the same question twice in a row if merchant answered it.
  6. Bare numbers without currency → ask once; after "yes" to dollar amount,
     use $ and USD (or stated currency) in proposed_prompt permanently.

LOOP PREVENTION:
  If merchant said "yes" / "correct" / "that's right" and you asked about
  currency, units, or confirmation → treat as confirmed and move forward.
  Re-asking an already-answered question is FORBIDDEN.

═══════════════════════════════════════════════════════════
5 UNSUPPORTED FLAGS — scan for ALL independently (STEP 1)
A valid core promotion + one flag = still unsupported.
Supported clauses do NOT cancel or reduce flags.
═══════════════════════════════════════════════════════════

discount_code:  customer must enter a code/coupon/voucher/token to activate
free_shipping:  reward is waiving delivery, shipping, or postage fees
usage_limit:    caps TOTAL redemptions or per-customer uses (not who qualifies, but how many)
pos_only:       restricts to physical store / POS (even "in-store AND online" = flag)
scheduling:     time-bounded: expiry, flash sale, "this weekend", "early access", start/end date

NOT a flag:
  "VIP customers"           → customer_tag eligibility (supported)
  "Wholesale customers"     → customer_tag (supported)
  "new customers"           → customer segment (supported)
  "first purchase"          → customer state, not a global cap (supported)
  "limited edition"         → product attribute, not promotion expiry (supported)
  "complimentary gift wrap" → physical product reward, not shipping (supported)

═══════════════════════════════════════════════════════════
SUPPORTED FEATURE SET — reference for all decisions (STEP 2–3)
═══════════════════════════════════════════════════════════

THREE families are supported. A complete prompt must fit one.
Classify the family correctly — especially when writing suggested_promotions.

── free_gift ──
WHAT IT IS:
  Customer reaches a trigger threshold → receives a FREE PHYSICAL PRODUCT
  added to their cart. The reward is a product, not a discount.

TRIGGER (one of):
  cart_quantity | cart_subtotal | collection_quantity | collection_subtotal |
  product_quantity | product_subtotal

REWARD:
  A free physical product (sample, gift, tote, bonus item). Can be vague
  ("a free gift") — exact SKU is chosen later in admin.

WHO (eligibility):
  customer_tag | logged_in | market | everyone

KEY SIGNALS:
  "free gift", "free sample", "complimentary product", "get a [item] free",
  "add a free [product] to cart", "bonus item", "freebie with purchase"

NOT free_gift:
  2+ threshold→reward pairs in the same prompt → tiered_discount
  (even when every reward is a free product or gift card)
  "10% off" or "$10 off" → that is buy_x_get_y or tiered_discount
  "free shipping" → unsupported flag (free_shipping)

EXAMPLES:
  ✓ "Spend $100 and get a free gift"
  ✓ "Buy 3 items from Skincare and get a free travel-size moisturizer"
  ✓ "VIP customers spend $150 and receive a complimentary tote bag"

── buy_x_get_y ──
WHAT IT IS:
  ONE trigger condition → ONE reward. A single deal statement:
    "Do X  →  get Y"
  where X is a quantity, spend, or subtotal threshold and Y is a discount
  or free item on a target (specific product, collection, cheapest item,
  same item, or the order).

TRIGGER (one of):
  cart_quantity | cart_subtotal | collection_quantity | collection_subtotal |
  product_quantity | product_subtotal

REWARD (exactly one):
  percentage_off_y | fixed_amount_off_y | free_y (100% off the Y item)

Y TARGET:
  specific product | specific collection | cheapest eligible item | same item

WHO:
  customer_tag | logged_in | market | everyone

KEY SIGNALS — single X→Y statement:
  "Buy 2 shirts and get 1 cap free"
  "Buy 3 from Collection A and get 50% off Collection B"
  "Spend $100 and get 10% off"          ← ONE spend threshold, ONE discount
  "Buy 2 get 1 free"                      ← ONE quantity threshold, ONE reward
  "Purchase 2 hoodies, get 50% off a cap"

STRUCTURE TEST:
  If the prompt describes only ONE threshold and ONE reward → buy_x_get_y.
  There is no second tier, no "spend more save more" escalation.

NOT buy_x_get_y:
  Multiple escalating thresholds in the same prompt → tiered_discount
  Reward is a free gift product with no discount mechanics → free_gift

EXAMPLES:
  ✓ "Buy 2 shirts and get 1 cap free"
  ✓ "Buy 2 get 50% off the cheapest item"
  ✓ "Spend $75 and get $10 off the order"
  ✓ "Buy 3 from Summer Collection, get 40% off Skincare"

── tiered_discount ──
WHAT IT IS:
  TWO OR MORE tiers in the SAME prompt. Each tier has its own threshold
  AND its own discount value. Tiers escalate on the SAME dimension —
  the same type of trigger (quantity, spend, or subtotal) applied to the
  same scope (cart, collection, or product). Higher threshold → better deal.

  Think: "buy X get Y, AND buy 2X get Z" or "spend $100 get 10%,
  spend $200 get 20%" — multiple rungs on the same ladder.

TRIGGER (one per tier, same type across tiers):
  cart_quantity | cart_subtotal | collection_quantity | collection_subtotal |
  product_quantity | product_subtotal

REWARD (one per tier):
  percentage_off | fixed_amount_off | free_product (free gift, gift card,
  sample, or bonus item added to cart — one free reward per tier)

MINIMUM: 2 tiers. A single "buy 2 get 10%" with no second tier is NOT
tiered_discount — that is buy_x_get_y. A single "buy $50 gift card get
$10 gift card free" with no second tier is free_gift, NOT tiered_discount.

WHO:
  customer_tag | logged_in | market | everyone

KEY SIGNALS — multiple thresholds, same dimension:
  "Buy 2 get 10%, buy 4 get 20%"
            "Spend $100 get 10% off, spend $200 get 20% off"
  "Buy 3 items get 5% off, buy 5 get 15% off, buy 10 get 25% off"
  "Buy a $50 gift card get a $10 gift card free, buy a $100 gift card
   get a $25 gift card free"  ← 2 tiers, same product line → tiered_discount
  "The more you spend, the more you save: $50→5%, $100→10%, $200→20%"
  "Volume discount tiers", "spend more save more", "buy more save more"

STRUCTURE TEST:
  Count distinct threshold→reward pairs in the prompt.
  • Exactly 1 pair → buy_x_get_y (or free_gift if reward is a free product)
  • 2+ pairs on the same trigger type/scope → tiered_discount
    (includes multi-tier free gift / gift card ladders — NOT free_gift)

NOT tiered_discount:
  "Buy 2 shirts get 1 cap free" → single pair → buy_x_get_y
  "Spend $100 and get a free gift" → single pair, free product → free_gift
  Trigger A → reward on B, where A and B are different products/collections
  with only ONE threshold → buy_x_get_y (cross-product deal, not tiered)

EXAMPLES:
  ✓ "Spend $100 get 10% off, spend $200 get 20% off"
  ✓ "Buy 2 get 10%, buy 4 get 20%, buy 6 get 30%"
  ✓ "Buy 3 items get $5 off, buy 6 items get $15 off"
  ✓ "Buy a $50 gift card get a $10 gift card free, buy a $100 gift card
     get a $25 gift card free"

═══════════════════════════════════════════════════════════
FAMILY DISAMBIGUATION — buy_x_get_y vs tiered_discount
(Most common confusion — apply this when classifying AND when
 labeling suggested_promotions)
═══════════════════════════════════════════════════════════

RULE 1 — COUNT THE TIERS:
  1 threshold + 1 reward  →  buy_x_get_y  (or free_gift)
  2+ thresholds + rewards on the SAME dimension  →  tiered_discount

RULE 2 — SAME DIMENSION FOR TIERED:
  Tiered tiers must escalate the SAME measure on the SAME scope:
    • spend $100 → 10%, spend $200 → 20%        (cart_subtotal tiers) ✓
    • buy 2 → 10%, buy 4 → 20%                 (cart_quantity tiers) ✓
    • buy 3 from Skincare → 5%, buy 5 → 15%    (collection_quantity) ✓
  Different products with ONE threshold each is NOT tiered:
    • "Buy 2 shirts get 1 cap free"            → buy_x_get_y (one deal)

RULE 3 — REWARD TYPE HELPS SEPARATE free_gift vs tiered vs buy_x_get_y:
  Single tier, free physical product as reward  →  free_gift
  Single tier, percentage or fixed $ off        →  buy_x_get_y
  2+ tiers (same dimension), any reward type
    (% off, $ off, OR free product/gift card per tier)  →  tiered_discount
  (then apply Rule 1 to confirm tier count)

SIDE-BY-SIDE:
  buy_x_get_y                          tiered_discount
  ─────────────────────────────────    ─────────────────────────────────
  "Buy 2 get 1 free"                   "Buy 2 get 10%, buy 4 get 20%"
  "Spend $100 get 10% off"             "Spend $100 get 10%, $200 get 20%"
  "Buy 3 hoodies, 50% off a cap"       "Buy 3 get 5% off, buy 6 get 15%"
  ONE statement, ONE deal              MULTIPLE tiers, SAME ladder

WHEN LABELING suggested_promotions:
  Double-check the family field using the rules above.
  Do NOT label "Buy 2 get 10%" as tiered_discount (only 1 tier).
  Do NOT label "Spend $100 get 10%, spend $200 get 20%" as buy_x_get_y
  (2 tiers on same spend dimension).
  Do NOT label "Spend $100 and get a free gift" as buy_x_get_y (free product).
  Do NOT label multi-tier gift card promos as free_gift — 2+ tiers = tiered_discount.

═══════════════════════════════════════════════════════════
WHAT IS REQUIRED FOR A COMPLETE PROMPT (STEP 4)
Only ask about REQUIRED missing fields — not nice-to-haves
═══════════════════════════════════════════════════════════

free_gift requires:
  ✦ REQUIRED:  trigger threshold (spend $X or buy N items)
  ✦ REQUIRED:  reward is a free gift (can be vague — specific product selected later)
  ✗ NOT required: exact product name, currency, operator, scope details

buy_x_get_y requires:
  ✦ REQUIRED:  trigger threshold
  ✦ REQUIRED:  reward type (% off Y / $off Y / free Y)
  ✦ REQUIRED:  reward value (the percentage or amount)
  ✗ NOT required: exact product/collection names (resolved later by admin)

tiered_discount requires:
  ✦ REQUIRED:  at least 2 tiers, each with a threshold AND a reward value
  ✦ REQUIRED:  reward type per tier (% off, $ off, or free product) explicit
  ✗ NOT required: tier behavior (default: best_tier_only), exact scope

Customer eligibility → NOT required if unspecified.
Currency, units, and amounts → REQUIRED if ambiguous — CLARIFY, never default.
Scope → can be vague if family and tiers are otherwise complete.

═══════════════════════════════════════════════════════════
WHEN TO CLARIFY vs WHEN TO DECIDE (STEP 4–5)
═══════════════════════════════════════════════════════════

ALWAYS CLARIFY when:
  • No trigger threshold exists ("give customers a discount" — how much/many?)
  • No reward type exists ("spend $100+" — what do they get?)
  • Family cannot be determined from context ("give VIP a deal")
  • Single tier detected for tiered_discount ("buy 2 get 10%" — only one tier;
    that is buy_x_get_y unless reward is a free product → free_gift)
  • Family unclear between buy_x_get_y and tiered_discount → apply
    FAMILY DISAMBIGUATION rules: count tiers, check same dimension
  • Amount or unit is ambiguous ("50 gift card" — $50 product or 50 units?)
  • Missing currency symbol where dollars are likely ("100" vs "$100")
  • Multi-tier structure visible but thresholds or rewards are incomplete
  • You would need to assume ANY detail to return pass

NEVER CLARIFY when:
  • Prompt contains any unsupported flag → go straight to unsupported with
    suggestions and corrected suggested_promotions
  • Merchant provided a FULLY EXPLICIT supported prompt → pass immediately

NEVER PASS when:
  • You are guessing currency, units, denomination, or family
  • forwarded_prompt would contain assumed or bracketed values
  • Required fields are still ambiguous and merchant has not supplied them
  • You would shorten a detailed merchant prompt into a vague summary

PASS IMMEDIATELY when:
  • Steps 1–4 satisfied with ZERO ambiguity in the merchant's own words
  • forwarded_prompt preserves ALL explicit details they stated

MAX 2 QUESTIONS PER TURN. Pick the most important missing fields.
UP TO 6 CLARIFICATION ROUNDS before you must decide. Use rounds 1–5 freely
to clarify — ask follow-ups, refine intent, and offer suggestions. Only on
round 6 (or earlier if intent is fully clear) return pass or unsupported.
Unsupported can be returned at ANY round when unsupported flags are detected —
do not waste rounds clarifying a promotion that will never be supported.
On round 6, if still ambiguous → unsupported (explain what is unclear).
Do NOT pass with assumptions on round 6.

═══════════════════════════════════════════════════════════
CONVERSATION CONTEXT RULE
═══════════════════════════════════════════════════════════

Read the ENTIRE conversation history on every turn.
Combine ALL information provided across all turns before deciding.
A later answer can complete an earlier partial prompt.

Example:
  Turn 1: "Give VIP customers a discount"          → clarify: what kind? what threshold?
  Turn 2: "tiered, spend more save more"           → clarify: what are the two tiers?
  Turn 3: "spend $100 get 10%, spend $200 get 20%" → clarify: verify intent
  Turn 4: "Yes, that's correct"                     → pass

The forwarded_prompt on pass must use the merchant's most complete explicit
wording — preserve product names, currency, scope, eligibility, and cart rules.
Do not replace a detailed prompt with a short summary. One or more sentences OK.
Use ONLY facts the merchant stated or confirmed — never insert assumed values.

═══════════════════════════════════════════════════════════
SUGGESTED PROMOTIONS — required in clarify AND unsupported
CLARIFY: at least 1 item; [0] MUST match proposed_prompt exactly (the live draft).
         Optional extras (up to 6) only when intent is vague.
UNSUPPORTED: 5–6 alternative prompts tailored to merchant intent.
Each suggestion must be a valid, supported promotion (no unsupported flags).

CRITICAL — label each suggestion's "family" using FAMILY DISAMBIGUATION:
  • 1 threshold + 1 reward  →  buy_x_get_y or free_gift
  • 2+ tiers, same dimension  →  tiered_discount
  Never mislabel single-deal prompts as tiered_discount.
  Never mislabel multi-tier spend/quantity ladders as buy_x_get_y.
═══════════════════════════════════════════════════════════

Examples of good suggested_promotions entries (note correct family labels):
  {"prompt": "Spend $100 and get a free gift", "family": "free_gift"}
  {"prompt": "Buy 2 shirts and get 1 cap free", "family": "buy_x_get_y"}
  {"prompt": "Spend $100 and get 10% off", "family": "buy_x_get_y"}
  {"prompt": "Buy 2 get 10%, buy 4 get 20%", "family": "tiered_discount"}
  {"prompt": "Spend $100 get 10% off, spend $200 get 20% off", "family": "tiered_discount"}

Common labeling mistakes to AVOID:
  ✗ {"prompt": "Buy 2 get 10%", "family": "tiered_discount"}     ← only 1 tier
  ✗ {"prompt": "Spend $100 get 10%, $200 get 20%", "family": "buy_x_get_y"}  ← 2 tiers
  ✗ {"prompt": "Spend $100 and get a free sample", "family": "buy_x_get_y"}  ← free product

Include 1–6 items in clarify (first = proposed_prompt); 5–6 in unsupported.
Each must have a non-empty "prompt" and a
"family" field (free_gift | buy_x_get_y | tiered_discount).

═══════════════════════════════════════════════════════════
SUGGESTIONS — required in every unsupported response
For each flag, tell the merchant what they CAN do instead
═══════════════════════════════════════════════════════════

discount_code:
  "Instead of a discount code, you can restrict this promotion to specific
   customer tags (e.g. VIP, Loyalty Member) or to logged-in customers only.
   They qualify automatically — no code entry needed."

free_shipping:
  "Free shipping is outside our scope. You can offer a free gift product,
   a percentage discount, or a fixed amount off orders over a threshold instead."

usage_limit:
  "Total redemption caps are not supported. To control who gets the deal,
   use customer tags (e.g. 'First Time Buyer', 'Loyalty Member') or restrict
   to logged-in customers. They act as a natural eligibility gate."

pos_only:
  "Channel restrictions are not supported — promotions apply across all
   channels. You can target in-store customers by assigning them a customer
   tag in Shopify and using that as the eligibility condition."

scheduling:
  "Time-bounded promotions are not supported. Consider an always-on promotion
   restricted to a specific customer segment instead. You can assign a tag
   like 'Summer Sale Access' to control exactly who sees the deal."

═══════════════════════════════════════════════════════════
OUTPUT SCHEMA — three exact formats, no other output
═══════════════════════════════════════════════════════════

── VERDICT: pass ──
{
  "verdict": "pass",
  "forwarded_prompt": "<clean synthesis confirmed by merchant>",
  "family": "free_gift | buy_x_get_y | tiered_discount",
  "understood_intent": "<one sentence: what merchant confirmed they want>"
}

On pass you MUST infer the promotion family using FAMILY DISAMBIGUATION rules.
Only use pass after merchant confirmation. The family field is required.

── VERDICT: clarify ──
{
  "verdict": "clarify",
  "understood_so_far": "<REQUIRED: what you think the merchant wants>",
  "inferred_family": "free_gift | buy_x_get_y | tiered_discount",
  "proposed_prompt": "<your best synthesis of the promotion so far>",
  "questions": [
    {
      "id": "confirm_intent",
      "question": "Is this what you want to set up?",
      "options": ["Yes, that's correct", "No — I'll clarify further"]
    }
  ],
  "suggested_promotions": [
    {"prompt": "<MUST match proposed_prompt exactly>", "family": "free_gift|buy_x_get_y|tiered_discount"},
    ... (1–6 total; first item is the live draft)
  ]
}

Include inferred_family and proposed_prompt in EVERY clarify response.
suggested_promotions[0] MUST duplicate proposed_prompt — this is the solid
suggestion shown to the merchant each turn.
When details are missing, ask about those AND still show an updated draft.
When details are complete but unconfirmed, ask confirm_intent only.

── VERDICT: unsupported ──
{
  "verdict": "unsupported",
  "flags": ["<flag_id>"],
  "flag_reasons": {
    "<flag_id>": "<one sentence: exactly what in the prompt triggered this>"
  },
  "suggestions": {
    "<flag_id>": "<one sentence: what the merchant can do instead>"
  },
  "supported_alternative": "<one valid version without unsupported parts>",
  "suggested_promotions": [
    {"prompt": "<complete supported alternative>", "family": "free_gift|buy_x_get_y|tiered_discount"},
    ... (5 or 6 total, tailored to merchant intent)
  ]
}

HARD RULES:
  verdict must be exactly "pass" | "clarify" | "unsupported"
  flags uses only: discount_code | free_shipping | usage_limit | pos_only | scheduling
  questions array: 1 or 2 items maximum
  forwarded_prompt: merchant's most complete explicit wording — never a shortened summary
  family: required on pass — exactly one of free_gift | buy_x_get_y | tiered_discount
  understood_so_far: REQUIRED in every clarify response
  inferred_family: REQUIRED in every clarify response
  proposed_prompt: REQUIRED in every clarify response
  suggestions: one entry per flag, always present in unsupported responses
  suggested_promotions: 1–6 items in clarify (first MUST match proposed_prompt);
                       5–6 items in unsupported responses
  each suggested_promotions item must have "prompt" and "family"

═══════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
═══════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
[FULLY EXPLICIT — pass immediately, no confirm_intent]
User: "All customers: when the cart subtotal reaches USD $100 on all products in the cart, give 1 free Sample Gift. Currency is US dollars. No special cart requirements. No customer tag, segment, or market restriction. Logged-in not required."
────────────────────────────────────────────────────────────
Reasoning:
  Flags: none. Family: free_gift. Threshold $100 USD, reward Sample Gift, scope all
  products, eligibility all stated. ZERO ambiguity → pass on first turn. Preserve FULL text.
Output:
{"verdict":"pass","forwarded_prompt":"All customers: when the cart subtotal reaches USD $100 on all products in the cart, give 1 free Sample Gift. Currency is US dollars. No special cart requirements. No customer tag, segment, or market restriction. Logged-in not required.","family":"free_gift","understood_intent":"Free Sample Gift when cart reaches $100 USD on all products for all customers"}

────────────────────────────────────────────────────────────
[BRIEF BUT COMPLETE — confirm_intent only]
User: "Spend $100 and get a free gift"
────────────────────────────────────────────────────────────
Reasoning:
  Flags: none. Family: free_gift. Complete but brief — merchant did not supply extra
  scope/product/eligibility detail. Ask confirm_intent ONLY (no other questions).
Output:
{"verdict":"clarify","understood_so_far":"Customer spends at least $100 and receives a free gift product added to their cart.","inferred_family":"free_gift","proposed_prompt":"Spend $100 and get a free gift","questions":[{"id":"confirm_intent","question":"Is this what you want to set up?","options":["Yes, that's correct","No — I'll clarify further"]}],"suggested_promotions":[{"prompt":"Spend $100 and get a free gift","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[MERCHANT CONFIRMS WITH FULL PROMPT — pass, preserve their words]
User: "All customers: when the cart subtotal reaches USD $100 on all products in the cart, give 1 free Sample Gift. Currency is US dollars. No special cart requirements. No customer tag, segment, or market restriction. Logged-in not required. this is what I want to use"
────────────────────────────────────────────────────────────
Reasoning:
  Merchant pasted complete explicit prompt as confirmation. Pass with THEIR full text.
  Do NOT shorten to "Spend $100 and get a free sample gift".
Output:
{"verdict":"pass","forwarded_prompt":"All customers: when the cart subtotal reaches USD $100 on all products in the cart, give 1 free Sample Gift. Currency is US dollars. No special cart requirements. No customer tag, segment, or market restriction. Logged-in not required.","family":"free_gift","understood_intent":"Free Sample Gift at $100 USD cart subtotal on all products for all customers"}

────────────────────────────────────────────────────────────
[MERCHANT CONFIRMS BRIEF DRAFT — pass with proposed wording]
User: "Yes, that's correct"
────────────────────────────────────────────────────────────
Output:
{"verdict":"pass","forwarded_prompt":"Spend $100 and get a free gift","family":"free_gift","understood_intent":"Spend $100 and get a free gift"}

────────────────────────────────────────────────────────────
[MULTI-TURN — ambiguous amount, merchant confirms currency]
Turn 1 — User: "free gift when cart exceeds 200 and has at least 2 products"
Output:
{"verdict":"clarify","understood_so_far":"Free gift when cart total exceeds 200 and cart has at least 2 items.","inferred_family":"free_gift","proposed_prompt":"Spend over $200 with at least 2 items in the cart and get a free gift","questions":[{"id":"currency_confirm","question":"Is the 200 value a dollar amount (e.g. $200 USD)?","options":["Yes, $200 USD","No — I'll clarify"]}],"suggested_promotions":[{"prompt":"Spend over $200 with at least 2 items in the cart and get a free gift","family":"free_gift"}]}

Turn 2 — User: "yes"
Reasoning: Merchant confirmed $200 USD. Draft is complete. Do NOT re-ask currency.
Output:
{"verdict":"clarify","understood_so_far":"Free gift when cart exceeds $200 USD and has at least 2 items.","inferred_family":"free_gift","proposed_prompt":"Spend over $200 with at least 2 items in the cart and get a free gift","questions":[{"id":"confirm_intent","question":"Is this what you want to set up?","options":["Yes","No — I'll clarify further"]}],"suggested_promotions":[{"prompt":"Spend over $200 with at least 2 items in the cart and get a free gift","family":"free_gift"}]}

Turn 3 — User: "yes"
Output:
{"verdict":"pass","forwarded_prompt":"Spend over $200 with at least 2 items in the cart and get a free gift","family":"free_gift","understood_intent":"Free gift when cart exceeds $200 with at least 2 items"}

────────────────────────────────────────────────────────────
[SINGLE TURN — MULTI-CLAUSE WITH FLAGS]
User: "VIP customers spend $100 and get a free gift with code SAVE10 only on POS"
────────────────────────────────────────────────────────────
Reasoning:
  VIP customers → customer_tag ✓   spend $100 → cart_subtotal ✓   free gift → reward ✓
  code SAVE10 → discount_code flag fires
  only on POS → pos_only flag fires
  Valid core + 2 flags → unsupported (supported clauses do not cancel flags)
Output:
{"verdict":"unsupported","flags":["discount_code","pos_only"],"flag_reasons":{"discount_code":"'Code SAVE10' requires the customer to enter a coupon code at checkout.","pos_only":"'Only on POS' restricts this promotion to physical store terminals."},"suggestions":{"discount_code":"Instead of a code, restrict this to VIP customers via their customer tag — they qualify automatically.","pos_only":"Remove the channel restriction — it runs on all channels. Assign in-store customers a Shopify tag and use that as eligibility."},"supported_alternative":"VIP customers spend $100 and get a free gift","suggested_promotions":[{"prompt":"VIP customers spend $100 and get a free gift","family":"free_gift"},{"prompt":"VIP customers spend $150 and get 15% off","family":"tiered_discount"},{"prompt":"VIP customers buy 3 items and get a free sample","family":"free_gift"},{"prompt":"VIP customers buy 2 get 1 free on selected items","family":"buy_x_get_y"},{"prompt":"VIP customers spend $200 get 10% off, spend $300 get 20% off","family":"tiered_discount"}]}

────────────────────────────────────────────────────────────
[SINGLE TURN — AMBIGUOUS: no reward specified]
User: "Spend $100+"
────────────────────────────────────────────────────────────
Output:
{"verdict":"clarify","understood_so_far":"Customer must spend at least $100.","questions":[{"id":"reward_type","question":"What should the customer receive after spending $100?","options":["A free gift product added to their cart","A percentage discount (e.g. 10% off)","A fixed amount off (e.g. $10 off)","A discount on specific items (Buy X Get Y)"]}],"suggested_promotions":[{"prompt":"Spend $100 and get a free gift","family":"free_gift"},{"prompt":"Spend $100 and get 10% off the order","family":"tiered_discount"},{"prompt":"Spend $100 and get $10 off","family":"tiered_discount"},{"prompt":"Spend $100 and buy 2 items get 1 free","family":"buy_x_get_y"},{"prompt":"Spend $100 on Skincare and get a free travel sample","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[MULTI-TURN — clarify until complete, turn 1]
User: "Give customers a deal"
────────────────────────────────────────────────────────────
Output:
{"verdict":"clarify","understood_so_far":"You want to create a promotion for your customers.","questions":[{"id":"promotion_family","question":"What kind of promotion would you like to create?","options":["Free gift: Customer spends or buys a certain amount and gets a free product","Buy X Get Y: Customer buys specific items and gets other items discounted or free","Tiered discount: Different discount levels based on how much they spend or buy"]}],"suggested_promotions":[{"prompt":"Spend $75 and get a free gift","family":"free_gift"},{"prompt":"Buy 2 items and get 10% off","family":"buy_x_get_y"},{"prompt":"Spend $100 get 10% off, spend $200 get 20% off","family":"tiered_discount"},{"prompt":"Buy 2 shirts and get 1 cap free","family":"buy_x_get_y"},{"prompt":"Orders over $50 qualify for a free sample","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[MULTI-TURN — turn 2]
User: "tiered, spend more get more"
────────────────────────────────────────────────────────────
Output:
{"verdict":"clarify","understood_so_far":"Tiered discount based on how much the customer spends.","questions":[{"id":"tiers","question":"What are the spend thresholds and discounts for each tier? Tiered discounts need at least two levels.","options":[]}]}

────────────────────────────────────────────────────────────
[MULTI-TURN — turn 3, now complete]
User: "$100 get 10% off, $200 get 20% off"
────────────────────────────────────────────────────────────
Output:
{"verdict":"clarify","understood_so_far":"Tiered discount: spend $100 get 10% off, spend $200 get 20% off.","inferred_family":"tiered_discount","proposed_prompt":"Spend $100 get 10% off, spend $200 get 20% off","questions":[{"id":"confirm_intent","question":"Is this what you want to set up?","options":["Yes, that's correct","No — I'll clarify further"]}],"suggested_promotions":[{"prompt":"Spend $100 get 10% off, spend $200 get 20% off","family":"tiered_discount"},{"prompt":"Buy 2 get 10%, buy 4 get 20%","family":"tiered_discount"},{"prompt":"Spend $50 get 5% off, spend $100 get 10% off","family":"tiered_discount"},{"prompt":"Buy 3 get 5% off, buy 6 get 15% off","family":"tiered_discount"},{"prompt":"Spend $100 and get a free gift","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[MULTI-TURN — turn 4, merchant confirms]
User: "Yes, that's correct"
────────────────────────────────────────────────────────────
Output:
{"verdict":"pass","forwarded_prompt":"Spend $100 get 10% off, spend $200 get 20% off","family":"tiered_discount","understood_intent":"Tiered spend discount: $100→10%, $200→20%"}

────────────────────────────────────────────────────────────
[MULTI-TURN — clarify then hits unsupported flag]
Turn 1 — User: "Something for my VIP customers when they buy a lot"
Output:
{"verdict":"clarify","understood_so_far":"A promotion for VIP customers triggered by a purchase quantity or amount.","questions":[{"id":"trigger","question":"What must VIP customers do to qualify?","options":["Spend a minimum amount (e.g. spend $100)","Buy a minimum number of items (e.g. buy 3 items)"]},{"id":"reward_type","question":"What should they receive as their reward?","options":["A free gift product","A percentage discount","A fixed amount off","A Buy X Get Y deal"]}]}

Turn 2 — User: "spend $150, free gift, but limit to first 50 VIP customers"
Output:
{"verdict":"unsupported","flags":["usage_limit"],"flag_reasons":{"usage_limit":"'Limit to first 50 customers' sets a total redemption cap, which is not supported."},"suggestions":{"usage_limit":"You cannot cap total redemptions, but you can restrict this to VIP customers using their customer tag — they qualify automatically and it acts as a natural gate."},"supported_alternative":"VIP customers spend $150 and get a free gift","suggested_promotions":[{"prompt":"VIP customers spend $150 and get a free gift","family":"free_gift"},{"prompt":"VIP customers spend $150 and get 15% off","family":"tiered_discount"},{"prompt":"VIP customers buy 3 items and get a free tote","family":"free_gift"},{"prompt":"VIP customers buy 2 get 1 free on accessories","family":"buy_x_get_y"},{"prompt":"VIP customers spend $100 get 10% off, spend $200 get 20% off","family":"tiered_discount"}]}

────────────────────────────────────────────────────────────
[AMBIGUOUS — gift card tiers, missing $ and units — must CLARIFY]
User: "buy a 50 gift card, get a 10$ gift card free and buy a 100 gift card, get a 25$ gift card free"
────────────────────────────────────────────────────────────
Reasoning:
  Flags: none.
  Structure: 2 tiers visible (50→$10 free, 100→$25 free) → likely tiered_discount
  NOT free_gift — multiple tiers on same product line.
  BUT "50" and "100" lack $ — could be $50/$100 denomination OR quantity 50/100.
  STEP 4 fails — ambiguous units. Must clarify before pass. NO assumptions.
Output:
{"verdict":"clarify","understood_so_far":"A two-tier promotion: buy a gift card at one level get a smaller gift card free, with a higher tier at 100.","questions":[{"id":"gift_card_denomination","question":"When you say '50 gift card' and '100 gift card', do you mean a $50 and $100 gift card product (denomination), or something else?","options":["$50 gift card product and $100 gift card product","A quantity of 50 or 100 gift cards","Something else — I'll describe"]},{"id":"confirm_tiers","question":"Just to confirm — is this a tiered deal where tier 1 is buy $50 gift card → get $10 gift card free, and tier 2 is buy $100 gift card → get $25 gift card free?","options":["Yes, that's correct","No — I'll clarify the tiers"]}],"suggested_promotions":[{"prompt":"Buy a $50 gift card get a $10 gift card free, buy a $100 gift card get a $25 gift card free","family":"tiered_discount"},{"prompt":"Buy a $50 gift card and get a $10 gift card free","family":"free_gift"},{"prompt":"Spend $50 get a free $10 gift card, spend $100 get a free $25 gift card","family":"tiered_discount"},{"prompt":"Buy 2 gift cards get 1 free","family":"buy_x_get_y"},{"prompt":"Spend $100 get 10% off, spend $200 get 20% off","family":"tiered_discount"}]}

────────────────────────────────────────────────────────────
[GIFT CARD TIERS — confirmed, now pass as tiered_discount]
User: "Yes — $50 gift card product gets $10 gift card free, $100 gift card gets $25 gift card free"
────────────────────────────────────────────────────────────
Output:
{"verdict":"pass","forwarded_prompt":"Buy a $50 gift card get a $10 gift card free, buy a $100 gift card get a $25 gift card free","family":"tiered_discount"}
"""


# ── LLM Client ──────────────────────────────────────────────────────────────

def get_client() -> Client:
    return Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
    )


def call_llm(client: Client, history: list) -> dict:
    """
    Call Ollama with the full conversation history.
    Returns parsed JSON dict.
    Raises ValueError if JSON cannot be parsed after cleaning.
    """
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
        ],
        format="json",
    )
    raw = (response.message.content or "").strip()

    # Strip markdown fences if the model wrapped its output
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")


# ── Validation ──────────────────────────────────────────────────────────────

VALID_VERDICTS = {"pass", "clarify", "unsupported"}
VALID_FLAGS    = {"discount_code", "free_shipping", "usage_limit", "pos_only", "scheduling"}


def validate_stage2_payload(data: Any) -> Dict[str, str]:
    """
    Validate the Stage 2 handoff JSON. Only allowed keys, valid family, non-empty prompt.
    Returns a normalized dict ready for Stage 2.
    """
    if not isinstance(data, dict):
        raise ValueError("Stage 2 payload must be a JSON object")

    keys = set(data.keys())
    unknown = keys - STAGE2_PAYLOAD_KEYS
    if unknown:
        raise ValueError(f"Stage 2 payload has unknown keys: {sorted(unknown)}")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Stage 2 payload 'prompt' must be a non-empty string")

    family = data.get("family")
    if family not in VALID_FAMILIES:
        raise ValueError(
            f"Stage 2 payload 'family' must be one of {sorted(VALID_FAMILIES)}, got {family!r}"
        )

    label = FAMILY_LABELS[family]
    if "family_label" in data and data["family_label"] != label:
        raise ValueError(
            f"Stage 2 payload 'family_label' must be {label!r} for family {family!r}"
        )

    return {
        "prompt": prompt.strip(),
        "family": family,
        "family_label": label,
    }


def build_stage2_payload(
    session: "Stage1Session",
    llm_result: dict,
    prompt_override: Optional[str] = None,
) -> Dict[str, str]:
    """Build and validate Stage 2 payload from session + LLM pass result."""
    family = (llm_result.get("family") or "").strip()
    if prompt_override:
        exact_prompt = prompt_override
    elif (llm_result.get("forwarded_prompt") or "").strip():
        exact_prompt = llm_result["forwarded_prompt"].strip()
    elif session.confirmed_prompt:
        exact_prompt = session.confirmed_prompt
    else:
        exact_prompt = session.user_turns[-1] if session.user_turns else session.stage0_prompt
    return validate_stage2_payload({
        "prompt": exact_prompt.strip(),
        "family": family,
        "family_label": FAMILY_LABELS.get(family, ""),
    })


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _prompts_match(a: str, b: str) -> bool:
    return _normalize_prompt(a) == _normalize_prompt(b)


def is_affirmative(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.match(r"^(yes|y|yeah|yep|yup|ok|okay|sure|correct|confirm(ed)?|1)[\s.,!?;:\"']*$", t, re.I):
        return True
    if re.match(r"^(yes|yeah|yep|yup|correct|that('?s|\s+is)\s*(right|correct|it))[\s.,!?;:\"']*$", t, re.I):
        return True
    return False


_CONFIRM_SUFFIXES = (
    "this is what i want to use",
    "this is what i want",
    "that's what i want to use",
    "that's what i want",
    "use this prompt",
    "use this",
    "go with this",
)


def strip_confirmation_suffix(text: str) -> str:
    """Remove trailing confirmation phrases; return the promotion wording."""
    t = (text or "").strip()
    lower = t.lower()
    for suffix in _CONFIRM_SUFFIXES:
        if lower.endswith(suffix):
            body = t[: len(t) - len(suffix)].strip(" .,\n-–—")
            if body:
                return body
        marker = f". {suffix}"
        if marker in lower:
            idx = lower.find(marker)
            body = t[:idx].strip(" .,\n-–—")
            if body:
                return body
    return t


def is_intent_restatement(text: str) -> bool:
    """True when merchant pastes their full prompt again to confirm intent."""
    t = (text or "").strip().lower()
    if is_affirmative(text):
        return False
    if len(t.split()) < 15:
        return False
    return any(s in t for s in _CONFIRM_SUFFIXES) or "this is what i want" in t


def _explicit_prompt_score(text: str) -> int:
    """Higher score = more explicit detail worth preserving for Stage 2."""
    t = (text or "").strip()
    if not t:
        return 0
    lower = t.lower()
    score = len(t.split())
    for marker in (
        "usd", "gbp", "inr", "eur", "$", "currency", "subtotal",
        "all products", "all customers", "no special cart",
        "sample gift", "logged-in", "segment", "market restriction",
        "cart subtotal", "free gift", "tier", "best tier",
    ):
        if marker in lower:
            score += 3
    return score


def is_fully_explicit_prompt(text: str) -> bool:
    """Heuristic: merchant supplied enough detail to pass without confirm_intent."""
    t = (text or "").strip().lower()
    if len(t.split()) < 18:
        return False
    has_currency = any(x in t for x in ("usd", "gbp", "inr", "eur", "aud", "$", "currency"))
    has_threshold = any(x in t for x in ("$", "spend", "subtotal", "buy ", "cart", "reach"))
    has_reward = any(x in t for x in ("free", "gift", "off", "%", "discount", "sample"))
    return has_currency and has_threshold and has_reward


def best_forwarded_prompt(session: "Stage1Session", llm_result: dict) -> str:
    """Prefer the most complete merchant-stated prompt over LLM summaries."""
    candidates: List[str] = []
    for turn in session.user_turns:
        cleaned = strip_confirmation_suffix(turn).strip()
        if not cleaned:
            continue
        if is_affirmative(cleaned) and len(cleaned.split()) < 8:
            continue
        candidates.append(cleaned)
    if fp := (llm_result.get("forwarded_prompt") or "").strip():
        candidates.append(fp)
    if pp := (llm_result.get("proposed_prompt") or "").strip():
        candidates.append(pp)
    if not candidates:
        return session.stage0_prompt.strip()
    return max(candidates, key=_explicit_prompt_score)


def is_clarify_loop(session: "Stage1Session", new_result: dict) -> bool:
    """True when LLM repeats the same question and draft after merchant affirmed."""
    last = session.last_result
    if not last or last.get("verdict") != "clarify" or new_result.get("verdict") != "clarify":
        return False
    if not session.user_turns or not is_affirmative(session.user_turns[-1]):
        return False
    last_qs = [q.get("question", "") for q in (last.get("questions") or [])]
    new_qs = [q.get("question", "") for q in (new_result.get("questions") or [])]
    if last_qs != new_qs:
        return False
    return _prompts_match(
        last.get("proposed_prompt") or "",
        new_result.get("proposed_prompt") or "",
    )


def _validate_suggested_promotions(
    promotions: Any,
    *,
    min_count: int = MIN_SUGGESTED_PROMOTIONS,
    max_count: int = MAX_SUGGESTED_PROMOTIONS,
) -> None:
    if not isinstance(promotions, list):
        raise ValueError("suggested_promotions must be a list")
    count = len(promotions)
    if count < min_count or count > max_count:
        raise ValueError(
            f"suggested_promotions must have {min_count}-"
            f"{max_count} items, got {count}"
        )
    for i, item in enumerate(promotions):
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"suggested_promotions[{i}] is empty")
            continue
        if not isinstance(item, dict):
            raise ValueError(f"suggested_promotions[{i}] must be an object")
        prompt = (item.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"suggested_promotions[{i}] missing prompt")
        family = (item.get("family") or "").strip()
        if family and family not in VALID_FAMILIES:
            raise ValueError(f"suggested_promotions[{i}] has invalid family: {family!r}")


def get_suggested_promotions(result: dict) -> List[Dict[str, str]]:
    """Normalize suggested_promotions from an LLM result."""
    raw = result.get("suggested_promotions") or []
    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"prompt": item.strip(), "family": ""})
        elif isinstance(item, dict):
            out.append({
                "prompt": (item.get("prompt") or "").strip(),
                "family": (item.get("family") or "").strip(),
            })
    return [s for s in out if s["prompt"]]


def validate_result(result: dict) -> None:
    """
    Raises ValueError if the LLM result violates schema rules.
    Called after every LLM response before acting on it.
    """
    verdict = result.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict!r}")

    if verdict == "pass":
        if not result.get("forwarded_prompt", "").strip():
            raise ValueError("pass verdict missing forwarded_prompt")
        family = (result.get("family") or "").strip()
        if family not in VALID_FAMILIES:
            raise ValueError(
                f"pass verdict missing or invalid family: {family!r} "
                f"(must be one of {sorted(VALID_FAMILIES)})"
            )

    elif verdict == "clarify":
        if not result.get("understood_so_far", "").strip():
            raise ValueError("clarify verdict missing understood_so_far")
        inferred = (result.get("inferred_family") or "").strip()
        if inferred and inferred not in VALID_FAMILIES:
            raise ValueError(f"clarify has invalid inferred_family: {inferred!r}")
        if not (result.get("proposed_prompt") or "").strip():
            raise ValueError("clarify verdict missing proposed_prompt")
        questions = result.get("questions", [])
        if not questions:
            raise ValueError("clarify verdict has no questions")
        if len(questions) > 2:
            raise ValueError(f"clarify has {len(questions)} questions (max 2)")
        _validate_suggested_promotions(
            result.get("suggested_promotions"),
            min_count=MIN_CLARIFY_SUGGESTIONS,
            max_count=MAX_SUGGESTED_PROMOTIONS,
        )
        proposed = (result.get("proposed_prompt") or "").strip()
        promos = get_suggested_promotions(result)
        if promos and not _prompts_match(promos[0]["prompt"], proposed):
            raise ValueError(
                "suggested_promotions[0].prompt must match proposed_prompt exactly"
            )

    elif verdict == "unsupported":
        flags = result.get("flags", [])
        if not flags:
            raise ValueError("unsupported verdict has empty flags list")
        bad = [f for f in flags if f not in VALID_FLAGS]
        if bad:
            raise ValueError(f"Unknown flags: {bad}")
        if not result.get("suggestions"):
            raise ValueError("unsupported verdict missing suggestions")
        _validate_suggested_promotions(result.get("suggested_promotions"))


# ── Session API (Streamlit / programmatic) ───────────────────────────────────

@dataclass
class Stage1TurnResult:
    verdict: str
    result: dict
    forwarded_prompt: Optional[str] = None
    stage2_payload: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    clarification_rounds: int = 0


@dataclass
class Stage1Session:
    """One Stage 1 conversation tied to a Stage 0 prompt."""

    stage0_prompt: str
    history: List[Dict[str, str]] = field(default_factory=list)
    user_turns: List[str] = field(default_factory=list)
    clarification_rounds: int = 0
    status: str = "active"          # active | passed | rejected
    forwarded_prompt: Optional[str] = None
    stage2_payload: Optional[Dict[str, str]] = None
    confirmed_prompt: Optional[str] = None
    pending_pass: Optional[dict] = None
    awaiting_verification: bool = False
    intent_confirmed: bool = False
    from_suggestion: bool = False
    last_result: Optional[dict] = None
    pending_alternative: Optional[str] = None


def md_escape(text: str) -> str:
    """Escape text for Streamlit st.markdown ($ triggers LaTeX math mode)."""
    if not text:
        return text
    return text.replace("\\", "\\\\").replace("$", "\\$")


def format_suggested_promotions(result: dict) -> str:
    """Plain-text fallback for CLI — UI renders suggestions separately."""
    suggestions = get_suggested_promotions(result)
    if not suggestions:
        return ""
    lines = ["Suggested promotions:", ""]
    for i, item in enumerate(suggestions, 1):
        family = item.get("family") or "promotion"
        lines.append(f"{i}. [{family}] {item['prompt']}")
    return "\n".join(lines)


def get_draft_prompt(result: dict) -> str:
    """Best merchant-ready draft from a clarify/unsupported result."""
    if proposed := (result.get("proposed_prompt") or "").strip():
        return proposed
    if forwarded := (result.get("forwarded_prompt") or "").strip():
        return forwarded
    promos = get_suggested_promotions(result)
    return promos[0]["prompt"] if promos else ""


def format_clarify_message(result: dict) -> str:
    """Plain-text chat message from a clarify JSON result."""
    parts: List[str] = []
    if understood := (result.get("understood_so_far") or "").strip():
        parts.append(understood)
    for q in result.get("questions", []):
        qtext = (q.get("question") or "").strip()
        if qtext:
            parts.append(qtext)
    return "\n\n".join(parts).strip()


def format_unsupported_message(result: dict) -> str:
    lines = ["This promotion cannot be built as described.", ""]
    for flag in result.get("flags", []):
        lines.append(f"[{flag}]")
        if reason := result.get("flag_reasons", {}).get(flag):
            lines.append(f"  Why: {reason}")
        if sug := result.get("suggestions", {}).get(flag):
            lines.append(f"  Instead: {sug}")
        lines.append("")
    alt = (result.get("supported_alternative") or "").strip()
    if alt:
        lines.append(f"Closest supported version: {alt}")
    suggested = format_suggested_promotions(result)
    if suggested:
        lines.append("")
        lines.append(suggested)
    return "\n".join(lines).strip()


def format_pass_message(stage2_payload: Dict[str, str]) -> str:
    family_label = stage2_payload.get("family_label", stage2_payload.get("family", ""))
    prompt = stage2_payload.get("prompt", "")
    return (
        f"Promotion verified and accepted.\n\n"
        f"Type: {family_label}\n\n"
        f"Your exact prompt (→ Stage 2):\n{prompt}"
    )


class Stage1Classifier:
    """Turn-based Stage 1 classifier for UI integration."""

    def __init__(self, client: Optional[Client] = None) -> None:
        self.client = client or get_client()

    def new_session(self, stage0_prompt: str) -> Stage1Session:
        return Stage1Session(stage0_prompt=stage0_prompt.strip())

    def process_turn(self, session: Stage1Session, user_input: str) -> Stage1TurnResult:
        user_input = user_input.strip()
        if not user_input:
            return Stage1TurnResult(
                verdict="error",
                result={},
                error="Empty message.",
                clarification_rounds=session.clarification_rounds,
            )

        session.user_turns.append(user_input)
        session.history.append({"role": "user", "content": user_input})
        session.clarification_rounds += 1

        cleaned_input = strip_confirmation_suffix(user_input)

        if is_fully_explicit_prompt(cleaned_input) and session.clarification_rounds == 1:
            session.history.append({
                "role": "user",
                "content": (
                    "[SYSTEM NOTE: Merchant provided a FULLY EXPLICIT supported promotion "
                    "in one message. If Steps 1–4 pass with ZERO ambiguity → verdict=pass "
                    "immediately. Do NOT ask confirm_intent. forwarded_prompt MUST preserve "
                    "ALL explicit merchant details verbatim — never shorten to a summary.]"
                ),
            })

        if (
            is_affirmative(user_input)
            and session.last_result
            and session.last_result.get("verdict") == "clarify"
        ):
            session.history.append({
                "role": "user",
                "content": (
                    "[SYSTEM NOTE: Merchant replied affirmatively. Apply every confirmed "
                    "detail into proposed_prompt (explicit $, USD, units). "
                    "suggested_promotions[0].prompt MUST match proposed_prompt. "
                    "Do NOT repeat any question they already answered. "
                    "If proposed_prompt is now complete → verdict=pass with "
                    "forwarded_prompt = most complete explicit wording. "
                    "Else ask only the NEXT unresolved detail.]"
                ),
            })

        if (
            is_intent_restatement(user_input)
            and session.last_result
            and session.last_result.get("verdict") == "clarify"
        ):
            session.history.append({
                "role": "user",
                "content": (
                    "[SYSTEM NOTE: Merchant confirmed by pasting their full intended prompt. "
                    "verdict=pass. forwarded_prompt MUST be their complete explicit wording "
                    f"(minus confirmation phrases) — e.g. start with: {cleaned_input[:200]}… "
                    "Do NOT shorten to a one-line summary.]"
                ),
            })

        if session.clarification_rounds > MAX_CLARIFICATION_ROUNDS:
            session.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM NOTE: Maximum clarification rounds ({MAX_CLARIFICATION_ROUNDS}) reached. "
                    "You MUST return verdict='pass' or verdict='unsupported' now. "
                    "Do NOT return verdict='clarify'. "
                    "Do NOT pass with assumptions — if any amount, unit, currency, "
                    "family, or reward is still ambiguous, return unsupported and "
                    "explain what remains unclear. Only pass if 100% certain. "
                    "If unsupported flags exist, return unsupported with suggested_promotions.]"
                ),
            })

        result, error = self._call_with_retry(session.history)
        if error:
            session.history = []
            session.clarification_rounds = 0
            session.user_turns = []
            session.pending_alternative = None
            return Stage1TurnResult(
                verdict="error",
                result={},
                error=error,
                clarification_rounds=0,
            )

        if is_clarify_loop(session, result):
            session.history.append({
                "role": "user",
                "content": (
                    "[SYSTEM NOTE: You repeated the same question and draft after the "
                    "merchant already answered. This is forbidden. Update proposed_prompt "
                    "with resolved values and either return pass (if complete) or ask a "
                    "DIFFERENT question about the next missing detail.]"
                ),
            })
            result, error = self._call_with_retry(session.history)
            if error:
                return Stage1TurnResult(
                    verdict="error",
                    result={},
                    error=error,
                    clarification_rounds=session.clarification_rounds,
                )

        session.history.append({"role": "assistant", "content": json.dumps(result)})
        session.last_result = result
        verdict = result["verdict"]

        if verdict == "pass":
            forwarded = best_forwarded_prompt(session, result)
            stage2_payload = build_stage2_payload(session, result, prompt_override=forwarded)
            session.status = "passed"
            session.forwarded_prompt = forwarded
            session.confirmed_prompt = forwarded
            session.stage2_payload = stage2_payload
            session.pending_alternative = None
            return Stage1TurnResult(
                verdict="pass",
                result=result,
                forwarded_prompt=forwarded,
                stage2_payload=stage2_payload,
                clarification_rounds=session.clarification_rounds,
            )

        if verdict == "unsupported":
            session.status = "rejected"
            suggestions = get_suggested_promotions(result)
            alt = (result.get("supported_alternative") or "").strip()
            session.pending_alternative = (
                alt or (suggestions[0]["prompt"] if suggestions else None)
            )
            return Stage1TurnResult(
                verdict="unsupported",
                result=result,
                clarification_rounds=session.clarification_rounds,
            )

        session.status = "active"
        session.pending_alternative = None
        return Stage1TurnResult(
            verdict="clarify",
            result=result,
            clarification_rounds=session.clarification_rounds,
        )

    def restart_with_prompt(self, session: Stage1Session, new_prompt: str) -> Stage1Session:
        return Stage1Session(stage0_prompt=new_prompt.strip())

    def _call_with_retry(self, history: List[Dict[str, str]]) -> tuple[Optional[dict], Optional[str]]:
        for attempt in range(2):
            try:
                raw_result = call_llm(self.client, history)
                validate_result(raw_result)
                return raw_result, None
            except ValueError as e:
                if attempt == 0:
                    history.append({"role": "assistant", "content": "{}"})
                    history.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM NOTE: Your last response had an error: {e}. "
                            "Please respond again with valid JSON matching the schema exactly.]"
                        ),
                    })
                else:
                    return None, str(e)
            except Exception as e:
                return None, f"Connection error: {e}"
        return None, "Unknown LLM error"


def session_to_dict(session: Stage1Session) -> Dict[str, Any]:
    return {
        "stage0_prompt": session.stage0_prompt,
        "history": session.history,
        "user_turns": session.user_turns,
        "clarification_rounds": session.clarification_rounds,
        "status": session.status,
        "forwarded_prompt": session.forwarded_prompt,
        "stage2_payload": session.stage2_payload,
        "confirmed_prompt": session.confirmed_prompt,
        "pending_pass": session.pending_pass,
        "awaiting_verification": session.awaiting_verification,
        "intent_confirmed": session.intent_confirmed,
        "from_suggestion": session.from_suggestion,
        "last_result": session.last_result,
        "pending_alternative": session.pending_alternative,
    }


def session_from_dict(data: Dict[str, Any]) -> Stage1Session:
    return Stage1Session(
        stage0_prompt=data["stage0_prompt"],
        history=list(data.get("history") or []),
        user_turns=list(data.get("user_turns") or []),
        clarification_rounds=int(data.get("clarification_rounds") or 0),
        status=data.get("status") or "active",
        forwarded_prompt=data.get("forwarded_prompt"),
        stage2_payload=data.get("stage2_payload"),
        confirmed_prompt=data.get("confirmed_prompt"),
        pending_pass=data.get("pending_pass"),
        awaiting_verification=bool(data.get("awaiting_verification")),
        intent_confirmed=bool(data.get("intent_confirmed")),
        from_suggestion=bool(data.get("from_suggestion")),
        last_result=data.get("last_result"),
        pending_alternative=data.get("pending_alternative"),
    )


# ── Display helpers (CLI) ────────────────────────────────────────────────────

LINE = "─" * 58

def _print(text: str = "", indent: int = 2) -> None:
    prefix = " " * indent
    for line in text.splitlines():
        print(f"{prefix}{line}")

def _input(prompt: str = "") -> str:
    """Read one line from stdin. Exits cleanly on Ctrl-C or EOF."""
    try:
        if prompt:
            _print(prompt)
        sys.stdout.write("  > ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:           # EOF (pipe ended)
            raise EOFError
        return line.strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nExiting Stage 1.")
        sys.exit(0)

def show_clarify(result: dict) -> None:
    _print(f"Understood: {result['understood_so_far']}")
    print()
    for q in result["questions"]:
        _print(q["question"])
        for i, opt in enumerate(q.get("options") or [], 1):
            _print(f"  {i}. {opt}")
        print()
    suggestions = get_suggested_promotions(result)
    if suggestions:
        _print("Suggested promotions:")
        for i, item in enumerate(suggestions, 1):
            family = item.get("family") or "promotion"
            _print(f"  {i}. [{family}] {item['prompt']}")
        print()

def show_unsupported(result: dict) -> None:
    _print("✗  This promotion cannot be built as described.\n")
    flags    = result.get("flags", [])
    reasons  = result.get("flag_reasons", {})
    suggests = result.get("suggestions", {})

    for flag in flags:
        _print(f"[{flag}]")
        if reason := reasons.get(flag):
            _print(f"  Why:     {reason}")
        if sug := suggests.get(flag):
            _print(f"  Instead: {sug}")
        print()

    suggestions = get_suggested_promotions(result)
    if suggestions:
        _print("Suggested promotions:")
        for i, item in enumerate(suggestions, 1):
            family = item.get("family") or "promotion"
            _print(f"  {i}. [{family}] {item['prompt']}")
        print()

def show_alternative(alt: str) -> None:
    _print("Supported version of your idea:")
    _print(f'  → "{alt}"')


# ── Core conversation loop ───────────────────────────────────────────────────

def run(initial_prompt: Optional[str] = None) -> str:
    """
    Run the Stage 1 conversation loop (CLI).

    Returns the forwarded_prompt string once the LLM accepts the promotion.
    This string is what Stage 2 receives as its input.
    """
    classifier = Stage1Classifier()

    print(LINE)
    print("  KITE — Stage 1  ·  Promotion Classifier")
    print("  Confirm your promotion is supported before building.")
    print(LINE)

    if initial_prompt:
        user_input = initial_prompt.strip()
        _print(f'Prompt: "{user_input}"')
        session = classifier.new_session(user_input)
    else:
        user_input = _input("Describe your promotion:")
        session = classifier.new_session(user_input)

    while True:
        turn = classifier.process_turn(session, user_input)
        print()

        if turn.verdict == "error":
            _print(f"Error: {turn.error}")
            _print("Please try rephrasing your promotion.")
            user_input = _input("Describe your promotion:")
            session = classifier.new_session(user_input)
            continue

        result = turn.result
        verdict = turn.verdict

        if verdict == "pass":
            forwarded = turn.forwarded_prompt or ""
            print(LINE)
            _print("✓  Promotion accepted.\n")
            _print(f'Forwarding to Stage 2: "{forwarded}"')
            print(LINE)
            return forwarded

        if verdict == "unsupported":
            print(LINE)
            show_unsupported(result)
            alt = session.pending_alternative or ""

            if alt:
                show_alternative(alt)
                print()
                _print("Options:")
                _print("  1. Use the supported version above")
                _print("  2. Describe a different promotion")
                _print("  3. Exit")
                choice = _input("Your choice (1 / 2 / 3 or type a new prompt):")

                if choice == "1":
                    user_input = alt
                    session = classifier.restart_with_prompt(session, alt)
                    _print(f'\nRetrying with: "{alt}"')
                    continue

                if choice == "3":
                    _print("Exiting.")
                    sys.exit(0)

                new = choice if len(choice) > 1 and choice != "2" else _input(
                    "Describe your promotion:"
                )
                user_input = new
                session = classifier.new_session(new)
                continue

            user_input = _input("Describe a different promotion (or press Ctrl+C to exit):")
            session = classifier.new_session(user_input)
            continue

        if verdict == "clarify":
            show_clarify(result)
            print()
            user_input = _input()
            continue

        _print(f"Unexpected verdict '{verdict}'. Please try again.")
        user_input = _input("Describe your promotion:")
        session = classifier.new_session(user_input)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Accept an optional initial prompt as a command-line argument
    # e.g.  python stage1.py "Spend $100 and get a free gift"
    initial = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None

    forwarded_prompt = run(initial)

    # Print the final forwarded prompt in a machine-readable format
    # Stage 2 can grep for this line or import run() directly
    print(f"\nFORWARDED: {forwarded_prompt}")