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
          Use FAMILY DISAMBIGUATION rules (tiered vs BXGY AND free_gift vs BXGY).
          If family is not 100% certain → verdict = clarify. Do NOT guess.

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
  • Family when free_gift vs buy_x_get_y is ambiguous → use promotion_family question
    FIRST (before confirm_intent or sub-questions). Tiered discount is separate — easy
    to tell apart. Never ask "what happens to the extra item" before family is chosen.

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
  ONLY after promotion_family is resolved (or promotion is tiered_discount).
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

SHORT BUT COMPLETE PROMPT:
  Even brief prompts ("Spend $100 and get a free gift", "Buy 2 get 1 free") MUST
  go through promotion_family FIRST (unless tiered_discount). Then confirm_intent
  or pass — never skip the family question.

INCOMPLETE OR AMBIGUOUS → clarify with specific questions (NOT confirm_intent):
  promotion_family is still FIRST for any non-tiered promotion; then ask threshold,
  reward, currency, etc. as needed.

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
  • promotion_family resolved for EVERY non-tiered promotion (merchant must choose
    free_gift or buy_x_get_y before confirm_intent or pass)
  • Either merchant gave a FULLY EXPLICIT prompt on this or a prior turn, OR
  • Merchant confirmed ("yes", "correct") or pasted their complete intended prompt

If unsure about ANYTHING material → clarify first with a specific question.
Do NOT use confirm_intent when real ambiguity remains — ask the ambiguous field.

MANDATORY promotion_family GATE (before confirm_intent, pass, or ANY other question):
  For EVERY supported promotion that is NOT tiered_discount, ask promotion_family
  FIRST with TWO suggested drafts (one free_gift, one buy_x_get_y).
  This applies to ALL non-tiered cases — including "Spend $100 get a free gift",
  "Buy 2 shirts get 1 cap free", "Buy three pants and get a pant", etc.
  The merchant MUST choose free_gift or buy_x_get_y before you proceed.
  Skip promotion_family ONLY when:
    • tiered_discount is detected (multiple tiers, spend/quantity ladders, or
      merchant/LLM clearly describes a tiered deal — handle tiered as today)
    • Merchant already answered promotion_family in this conversation
  FORBIDDEN: Skipping to confirm_intent, pass, or sub-questions before
  promotion_family is resolved on non-tiered promotions.

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
  2. When merchant confirms promotion_family answer:
     → Free gift option → set family free_gift; use free_gift suggested draft
     → Buy X Get Y option → set family buy_x_get_y; use buy_x_get_y suggested draft
     → THEN ask confirm_intent or pass — never pass before promotion_family is resolved
  3. When merchant confirms a detail ("yes" to "is 200 dollars?"):
     → IMMEDIATELY bake the answer in (e.g. "200" → "$200 USD")
     → Do NOT ask that same question again
     → Either ask the NEXT unresolved detail OR return pass if complete
  4. suggested_promotions[0].prompt MUST equal proposed_prompt exactly.
  5. Multi-condition promotions: resolve one ambiguity at a time.
     Example: cart subtotal + cart quantity → draft both once confirmed:
     "Spend over $200 with at least 2 items in the cart and get a free gift"
  6. NEVER return the same question twice in a row if merchant answered it.
  7. Bare numbers without currency → ask once; after "yes" to dollar amount,
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
  Explicit BUY X → GET Y with distinct qualifying products (X) and reward
  product (Y), even when Y is free → buy_x_get_y (free_y), NOT free_gift
  "Buy 3 shirts get a free pant", "Buy 2 from list A get product B free"

EXAMPLES:
  ✓ "Spend $100 and get a free gift"
  ✓ "VIP customers spend $150 and receive a complimentary tote bag"
  ✓ "Orders over $75 qualify for a free sample from our gift selection"

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
  "Buy 3 shirts from selected list and get a free pant"
  "Buy 3 from Collection A and get 50% off Collection B"
  "Spend $100 and get 10% off"          ← ONE spend threshold, ONE discount
  "Buy 2 get 1 free"                      ← ONE quantity threshold, ONE reward
  "Purchase 2 hoodies, get 50% off a cap"

STRUCTURE TEST:
  If the prompt describes BUY X → GET Y where X (qualifying purchase) and
  Y (reward: free product, % off, or $ off) are a single paired deal → buy_x_get_y.
  There is no second tier, no "spend more save more" escalation.

NOT buy_x_get_y:
  Multiple escalating thresholds in the same prompt → tiered_discount
  Spend/cart threshold + vague "free gift" with NO explicit X→Y product pairing
  (gift chosen from gift catalog later) → free_gift, NOT buy_x_get_y

EXAMPLES:
  ✓ "Buy 2 shirts and get 1 cap free"
  ✓ "Buy 3 shirts from selected list and get a free pant"
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
  Single tier, spend/cart threshold + vague free gift (no X→Y pairing)  →  free_gift
  Single tier, explicit buy X get Y (qualifying ≠ reward), free or discounted Y  →  buy_x_get_y
  Single tier, percentage or fixed $ off on Y after buying X  →  buy_x_get_y
  2+ tiers (same dimension), any reward type
    (% off, $ off, OR free product/gift card per tier)  →  tiered_discount
  If reward is a free product but X→Y vs gift-list free_gift is unclear → CLARIFY
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
  Do NOT label "Spend $100 and get a free gift" as buy_x_get_y (gift-list reward).
  Do NOT label "Buy 3 shirts get a free pant" as free_gift (explicit X→Y deal).
  Do NOT label multi-tier gift card promos as free_gift — 2+ tiers = tiered_discount.

═══════════════════════════════════════════════════════════
FAMILY DISAMBIGUATION — free_gift vs buy_x_get_y
(Do NOT confuse these — apply when classifying AND suggested_promotions)
═══════════════════════════════════════════════════════════

THE CORE DIFFERENCE:
  free_gift  →  customer hits a THRESHOLD → receives a promotional FREE GIFT
               (often vague "a free gift", or from gift-item catalog; admin may
               pick exact SKU later). NOT structured as "buy product A, get product B".
  buy_x_get_y  →  ONE paired deal: BUY X (qualifying) → GET Y (reward).
               X and Y are distinct: quantity/spend on qualifying items → discount
               or free item on Y (specific product, collection, cheapest, same item).

DECISION RULES:
  1. Explicit "Buy N [product/collection X] … get [product Y] free/discounted"
     with distinct qualifying (X) and reward (Y) → buy_x_get_y
     (even when Y is 100% free — that is free_y inside buy_x_get_y)
  2. "Spend $X / cart subtotal / buy N items … get a free gift / sample"
     with NO named qualifying-vs-reward product pairing → free_gift
  3. "Get [specific product] free" after buying DIFFERENT qualifying products
     → buy_x_get_y (NOT free_gift just because reward is free)
  4. If merchant could mean EITHER free_gift (threshold → promotional gift) OR
     buy_x_get_y (buy X → get Y free/discounted) → CLARIFY with promotion_family
     question (see below). Do NOT ask sub-questions about the reward item first.

WHEN TO CLARIFY (free_gift vs buy_x_get_y) — MANDATORY promotion_family FIRST:
  ALWAYS ask promotion_family for EVERY non-tiered promotion on the FIRST clarify
  turn — no exceptions. Examples (all require promotion_family before anything else):
    • "Spend $100 and get a free gift"
    • "Buy 2 shirts get 1 cap free"
    • "Buy 3 shirts and get a free tote"
    • "Buy three pants and get a pant"
  Skip promotion_family ONLY when:
    • tiered_discount: multiple tiers, spend/quantity ladders, or merchant describes
      a tiered deal (handle tiered flow as today — LLM can identify automatically)
    • Merchant already answered promotion_family in this conversation
  Question id: promotion_family
  Question (plain English — NO internal family names):
    "Which type of promotion do you want? Choose between a free gift reward
    and a Buy X Get Y deal. (Tiered discount does not apply here.)"
  Options:
    • "Free gift — customer qualifies by buying/spending and receives a promotional free gift (samples, welcome gifts, etc.)"
    • "Buy X Get Y — customer buys qualifying items and gets a specific product free or discounted (e.g. buy 3 pants, get 1 pant free)"
  Map answers internally:
    • Free gift option → free_gift family; use free_gift suggested draft
    • Buy X Get Y option → buy_x_get_y family; use buy_x_get_y suggested draft
  suggested_promotions MUST include BOTH drafts — [0] matches proposed_prompt (default
  buy_x_get_y draft), [1] is the free_gift draft. UI shows both side by side.
  FORBIDDEN: Skipping promotion_family and asking confirm_intent or sub-questions
  like "What should happen to the extra pant?" when family is still ambiguous.
  If reward could be a discount (% or $ off) rather than free → use reward_type
  question instead (separate from promotion_family).

MERCHANT-FACING LANGUAGE for promotion_family clarify:
  • NEVER say free_gift, buy_x_get_y, tiered_discount as internal labels in questions
  • Ask which promotion TYPE the merchant wants — free gift vs Buy X Get Y
  • Show two suggested drafts so they can see each interpretation

SIDE-BY-SIDE:
  free_gift                              buy_x_get_y
  ─────────────────────────────────────  ─────────────────────────────────────
  "Spend $100 get a free gift"           "Buy 2 shirts get 1 cap free"
  "Buy 3 items, receive a free sample"   "Buy 3 shirts from list, get free pant"
  (threshold → gift, no X→Y pairing)     (qualifying X → reward Y, one deal)

WHEN LABELING suggested_promotions:
  "Buy 3 shirts get a free pant" → buy_x_get_y (NOT free_gift)
  "Spend $100 get free Sample Gift" → free_gift (threshold → named gift)
  "Buy 2 get 1 free on shirts" → buy_x_get_y (same_item or quantity deal)

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
  • ANY non-tiered promotion → MUST ask promotion_family FIRST with two suggested
    drafts. Never skip — even when family seems obvious from wording.
  • Single tier detected for tiered_discount ("buy 2 get 10%" — only one tier;
    that is buy_x_get_y, not tiered_discount)
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
  ✗ {"prompt": "Buy 3 shirts get a free pant", "family": "free_gift"}  ← X→Y deal = buy_x_get_y
  ✗ {"prompt": "Spend $100 and get a free sample", "family": "buy_x_get_y"}  ← threshold gift = free_gift

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
[BUY X GET Y vs FREE GIFT — promotion_family required FIRST]
User: "Buy three shirts in selected list and get a free tote"
Reasoning:
  Could be free_gift OR buy_x_get_y. MUST ask promotion_family with TWO drafts.
Output:
{"verdict":"clarify","understood_so_far":"Buy 3 shirts from a selected list and receive a free tote.","inferred_family":"buy_x_get_y","proposed_prompt":"Buy 3 shirts from selected list and get a free tote","questions":[{"id":"promotion_family","question":"Which type of promotion do you want? Choose between a free gift reward and a Buy X Get Y deal.","options":["Free gift — customer qualifies by buying/spending and receives a promotional free gift (samples, welcome gifts, etc.)","Buy X Get Y — customer buys qualifying items and gets a specific product free or discounted (e.g. buy 3 shirts, get a free tote)"]}],"suggested_promotions":[{"prompt":"Buy 3 shirts from selected list and get a free tote","family":"buy_x_get_y"},{"prompt":"Buy 3 shirts from selected list and receive a free gift","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[Buy pants get pant — promotion_family required, NOT sub-questions]
User: "Buy three pants and get a pant"
Reasoning:
  Ambiguous free_gift vs buy_x_get_y. FORBIDDEN to ask "what should happen to the extra pant"
  before promotion_family. Show both drafts.
Output:
{"verdict":"clarify","understood_so_far":"Buy three pants and receive an additional pant.","inferred_family":"buy_x_get_y","proposed_prompt":"Buy 3 pants and get 1 pant free","questions":[{"id":"promotion_family","question":"Which type of promotion do you want? Choose between a free gift reward and a Buy X Get Y deal.","options":["Free gift — customer qualifies by buying/spending and receives a promotional free gift (samples, welcome gifts, etc.)","Buy X Get Y — customer buys qualifying items and gets a specific product free or discounted (e.g. buy 3 pants, get 1 pant free)"]}],"suggested_promotions":[{"prompt":"Buy 3 pants and get 1 pant free","family":"buy_x_get_y"},{"prompt":"Buy 3 pants and receive a free gift","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[After promotion_family answered — free_gift → then confirm_intent]
User: (chose Free gift option)
Output:
{"verdict":"clarify","understood_so_far":"Buy 3 pants; customer receives a promotional free gift when they qualify.","inferred_family":"free_gift","proposed_prompt":"Buy 3 pants and receive a free gift","questions":[{"id":"confirm_intent","question":"Is this what you want to set up?","options":["Yes, that's correct","No — I'll clarify further"]}],"suggested_promotions":[{"prompt":"Buy 3 pants and receive a free gift","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[BUY X GET Y — named free item, promotion_family required]
User: "Buy three shirts in selected list and get a free pant"
Reasoning:
  MUST ask promotion_family before confirm_intent — show both drafts.
Output:
{"verdict":"clarify","understood_so_far":"Buy 3 shirts from a selected list and receive a free pant.","inferred_family":"buy_x_get_y","proposed_prompt":"Buy 3 shirts from selected list and get a free pant","questions":[{"id":"promotion_family","question":"Which type of promotion do you want? Choose between a free gift reward and a Buy X Get Y deal.","options":["Free gift — customer qualifies by buying/spending and receives a promotional free gift (samples, welcome gifts, etc.)","Buy X Get Y — customer buys qualifying items and gets a specific product free or discounted (e.g. buy 3 shirts, get a free pant)"]}],"suggested_promotions":[{"prompt":"Buy 3 shirts from selected list and get a free pant","family":"buy_x_get_y"},{"prompt":"Buy 3 shirts from selected list and receive a free gift","family":"free_gift"}]}

────────────────────────────────────────────────────────────
[AMBIGUOUS — promotion_family question with two suggested drafts]
User: "Buy 3 items from my selected list and get something free"
Reasoning:
  Qualifying purchase clear; family unknown. Ask promotion_family only.
Output:
{"verdict":"clarify","understood_so_far":"Buy 3 items from a selected list and receive something free.","inferred_family":"buy_x_get_y","proposed_prompt":"Buy 3 items from selected list and get 1 free item","questions":[{"id":"promotion_family","question":"Which type of promotion do you want? Choose between a free gift reward and a Buy X Get Y deal.","options":["Free gift — customer qualifies by buying/spending and receives a promotional free gift (samples, welcome gifts, etc.)","Buy X Get Y — customer buys qualifying items and gets a specific product free or discounted"]}],"suggested_promotions":[{"prompt":"Buy 3 items from selected list and get 1 free item","family":"buy_x_get_y"},{"prompt":"Buy 3 items from selected list and receive a free gift","family":"free_gift"}]}

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
{"verdict":"clarify","understood_so_far":"You want to create a promotion for your customers.","questions":[{"id":"reward_type","question":"What should customers receive when they qualify?","options":["A free gift product added to their cart","A percentage or fixed discount on items they buy","Different discount levels the more they buy (tiered deal)"]}],"suggested_promotions":[{"prompt":"Spend $75 and get a free gift","family":"free_gift"},{"prompt":"Buy 2 items and get 10% off","family":"buy_x_get_y"},{"prompt":"Spend $100 get 10% off, spend $200 get 20% off","family":"tiered_discount"},{"prompt":"Buy 2 shirts and get 1 cap free","family":"buy_x_get_y"},{"prompt":"Orders over $50 qualify for a free sample","family":"free_gift"}]}

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


_GIFT_ITEMS_CATALOG_NAMES = (
    "sample gift", "mini sample pack", "mystery gift box", "free trial kit",
    "complimentary gift", "welcome gift", "free tote bag", "gift hamper",
)

_PROMOTION_FAMILY_RESOLVED_MARKERS = (
    "free gift — customer qualifies",
    "buy x get y — customer buys",
    "promotional free gift",
    "buy x get y deal",
    "free_gift",
    "buy_x_get_y",
)

_PROMOTION_FAMILY_FREE_GIFT_OPTION = (
    "Free gift — customer qualifies by buying/spending and receives a "
    "promotional free gift (samples, welcome gifts, etc.)"
)
_PROMOTION_FAMILY_BXGY_OPTION = (
    "Buy X Get Y — customer buys qualifying items and gets a specific "
    "product free or discounted (e.g. buy 3 pants, get 1 pant free)"
)

_WORD_TO_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
}


def conversation_text(session: "Stage1Session") -> str:
    return " ".join(session.user_turns)


def is_likely_tiered_discount(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "tiered" in t or ("tier" in t and "discount" in t):
        return True
    if re.search(r"\bspend\s+more\s+get\s+more\b", t):
        return True
    if len(re.findall(r"\d+\s*%\s*off", t)) >= 2:
        return True
    if len(re.findall(r"\bspend\s+\$?\d+", t)) >= 2:
        return True
    if len(re.findall(r"\bbuy\s+\d+.*?(?:get|receive).*?(?:\d+\s*%\s*off|\d+\s*off)", t)) >= 2:
        return True
    return False


_PROMOTION_SIGNAL_RE = re.compile(
    r"\b(?:buy|purchase|spend|subtotal|cart|order|get|receive|give|free|gift|"
    r"discount|off|percent|sample|qualif|promo|deal|reward)\b",
    re.I,
)


def is_tiered_promotion(text: str, session: Optional["Stage1Session"] = None) -> bool:
    """True when the conversation is a tiered discount (skip promotion_family gate)."""
    if is_likely_tiered_discount(text):
        return True
    if session:
        last = session.last_result or {}
        fam = (last.get("inferred_family") or last.get("family") or "").strip()
        if fam == "tiered_discount":
            return True
    return False


def is_non_tiered_promotion(
    text: str,
    session: Optional["Stage1Session"] = None,
) -> bool:
    """True for any supported non-tiered promotion — requires promotion_family first."""
    t = (text or "").strip()
    if not t:
        return False
    if any(m in t.lower() for m in _PROMOTION_FAMILY_RESOLVED_MARKERS):
        return False
    if is_tiered_promotion(t, session):
        return False
    return bool(_PROMOTION_SIGNAL_RE.search(t))


def is_free_gift_vs_bxgy_ambiguous(text: str) -> bool:
    """Backward-compatible alias — now True for all non-tiered promotions."""
    return is_non_tiered_promotion(text)


def build_family_draft_variants(text: str) -> tuple[str, str]:
    """Return (free_gift_draft, buy_x_get_y_draft) for any non-tiered promotion."""
    raw = (text or "").strip()
    t = raw.lower()
    if not raw:
        return (
            "Spend $100 and receive a free gift",
            "Buy 2 items and get 1 item free",
        )

    spend_m = re.search(
        r"(?:spend|subtotal|cart\s+(?:total|reaches|exceeds)|orders?\s+over)"
        r"\s*(?:over|at least|reaches?|exceeds)?\s*\$?\s*(\d+)",
        t,
    )
    if spend_m:
        amount = spend_m.group(1)
        free_gift = raw
        if not re.search(r"\bfree\s+gift\b", t, re.I):
            free_gift = re.sub(
                r"(?:get|receive|give)\s+[^.]+$",
                "receive a free gift",
                raw,
                flags=re.I,
            ).strip()
            if not re.search(r"\bfree\s+gift\b", free_gift, re.I):
                free_gift = f"Spend ${amount} and receive a free gift"
        bxgy = f"Spend ${amount} and get 10% off qualifying items"
        return (
            re.sub(r"\s+", " ", free_gift).strip(),
            re.sub(r"\s+", " ", bxgy).strip(),
        )

    if not re.search(r"\b(?:buy|spend|cart|subtotal|order|get|receive)\b", t, re.I):
        return (
            "Spend $100 and receive a free gift",
            "Buy 2 items and get 1 item free",
        )

    free_gift = re.sub(
        r"\b(?:and\s+)?(?:get|receive|give)\s+(?:\d+\s+)?(?:a\s+|an\s+|the\s+)?(?:(?:free\s+)?[\w\s]+?)\s*$",
        " and receive a free gift",
        raw,
        flags=re.I,
    ).strip()
    free_gift = re.sub(r"\s+", " ", free_gift)
    if not re.search(r"\bfree\s+gift\b", free_gift, re.I):
        free_gift = re.sub(r"\s+", " ", f"{raw.rstrip('.')} and receive a free gift")

    bxgy = raw
    m = re.search(
        r"\b(?:buy|purchase)\s+(one|two|three|four|five|six|\d+)\s+([\w\s]+?)"
        r"(?:\s+from|\s+in|\s+on|\s+and|\s+get|\s+receive|\s+give|$)",
        raw,
        re.I,
    )
    m2 = re.search(
        r"\b(?:get|receive|give)\s+(?:\d+\s+)?(?:a\s+|an\s+|the\s+)?(?:(?:free\s+)?(\w+))",
        raw,
        re.I,
    )
    if m and m2:
        qty = _WORD_TO_NUM.get(m.group(1).lower(), m.group(1))
        qual = m.group(2).strip().rstrip(",")
        reward = m2.group(1)
        qual_words = qual.split()
        product = qual_words[-1] if qual_words else "items"
        if (
            reward.rstrip("s").lower() == product.rstrip("s").lower()
            or ("pant" in reward.lower() and "pant" in product.lower())
        ):
            bxgy = f"Buy {qty} {qual} and get 1 {reward} free"
        else:
            bxgy = f"Buy {qty} {qual} and get a free {reward}"
    elif re.search(r"\bbuy\s+\d+\s+get\s+\d+\s+free\b", t):
        bxgy = raw
    elif "free" not in t and m2:
        bxgy = re.sub(
            r"\b(get|receive|give)\s+(a\s+|an\s+|the\s+)?(\w+)\s*$",
            r"get a free \3",
            raw,
            count=1,
            flags=re.I,
        )
    elif not re.search(r"\bbuy\b", t) or not re.search(r"\bget\b", t):
        bxgy = re.sub(r"\s+", " ", f"{raw.rstrip('.')} — Buy X Get Y deal")

    return re.sub(r"\s+", " ", free_gift).strip(), re.sub(r"\s+", " ", bxgy).strip()


def parse_promotion_family_answer(text: str) -> Optional[str]:
    """Returns 'free_gift' | 'buy_x_get_y' when the merchant picks a family."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if t == _PROMOTION_FAMILY_FREE_GIFT_OPTION.lower() or t.startswith("free gift —"):
        return "free_gift"
    if t == _PROMOTION_FAMILY_BXGY_OPTION.lower() or t.startswith("buy x get y"):
        return "buy_x_get_y"
    if any(m in t for m in ("promotional free gift", "free gift promotion", "free gift reward")):
        return "free_gift"
    if any(m in t for m in ("buy x get y", "buy x get y deal", "specific product free")):
        return "buy_x_get_y"
    return None


def promotion_family_resolved(session: "Stage1Session") -> bool:
    if session.promotion_family_resolved:
        return True
    for turn in session.user_turns:
        if parse_promotion_family_answer(turn):
            return True
    return False


def needs_promotion_family_gate(session: "Stage1Session") -> bool:
    if promotion_family_resolved(session):
        return False
    return is_non_tiered_promotion(conversation_text(session), session)


def try_apply_promotion_family_answer(session: "Stage1Session", user_input: str) -> None:
    choice = parse_promotion_family_answer(user_input)
    if not choice:
        return
    session.promotion_family_resolved = True
    session.promotion_family_choice = choice
    free_draft, bxgy_draft = build_family_draft_variants(conversation_text(session))
    draft = free_draft if choice == "free_gift" else bxgy_draft
    label = "Free gift (free_gift)" if choice == "free_gift" else "Buy X Get Y (buy_x_get_y)"
    session.history.append({
        "role": "user",
        "content": (
            f"[SYSTEM NOTE: Merchant answered promotion_family: {label}. "
            f"Set family={choice!r} and proposed_prompt={draft!r}. "
            "Do NOT ask promotion_family again or ask sub-questions like "
            "'what should happen to the extra pant'. Proceed with confirm_intent or pass.]"
        ),
    })


def build_promotion_family_clarify(session: "Stage1Session", llm_result: dict) -> dict:
    """Deterministic promotion_family clarify with two suggested drafts."""
    conv = conversation_text(session)
    free_gift_draft, bxgy_draft = build_family_draft_variants(conv)
    understood = (llm_result.get("understood_so_far") or "").strip()
    if not understood:
        understood = "Your promotion involves qualifying purchases and a reward item."
    clarified = {
        "verdict": "clarify",
        "understood_so_far": understood,
        "inferred_family": "buy_x_get_y",
        "proposed_prompt": bxgy_draft,
        "questions": [{
            "id": "promotion_family",
            "question": (
                "Which type of promotion do you want? Choose between a free gift "
                "reward and a Buy X Get Y deal. (Tiered discount does not apply here.)"
            ),
            "options": [_PROMOTION_FAMILY_FREE_GIFT_OPTION, _PROMOTION_FAMILY_BXGY_OPTION],
        }],
        "suggested_promotions": [
            {"prompt": bxgy_draft, "family": "buy_x_get_y"},
            {"prompt": free_gift_draft, "family": "free_gift"},
        ],
    }
    validate_result(clarified)
    return clarified


def enforce_promotion_family_gate(session: "Stage1Session", result: dict) -> dict:
    """Block pass/confirm_intent/sub-questions until promotion_family is resolved."""
    fam = (result.get("inferred_family") or result.get("family") or "").strip()
    if fam == "tiered_discount":
        return result
    if is_tiered_promotion(conversation_text(session), session):
        return result
    if not needs_promotion_family_gate(session):
        return result
    if result.get("verdict") == "unsupported":
        return result
    questions = result.get("questions") or []
    qids = [q.get("id") for q in questions]
    if result.get("verdict") == "clarify" and qids == ["promotion_family"]:
        return result
    return build_promotion_family_clarify(session, result)


# Backward-compatible aliases
mentions_named_free_reward = is_free_gift_vs_bxgy_ambiguous
gift_reward_resolved = promotion_family_resolved
needs_gift_reward_gate = needs_promotion_family_gate
try_apply_gift_reward_answer = try_apply_promotion_family_answer
build_gift_reward_clarify = build_promotion_family_clarify
enforce_gift_reward_gate = enforce_promotion_family_gate
parse_gift_reward_answer = parse_promotion_family_answer


def is_fully_explicit_prompt(text: str) -> bool:
    """Heuristic: merchant supplied enough detail to pass without confirm_intent."""
    t = (text or "").strip().lower()
    if len(t.split()) < 18:
        return False
    if is_non_tiered_promotion(text) and not any(
        m in t for m in _PROMOTION_FAMILY_RESOLVED_MARKERS
    ):
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
    gift_reward_resolved: bool = False
    gift_reward_family: Optional[str] = None
    promotion_family_resolved: bool = False
    promotion_family_choice: Optional[str] = None


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
        for i, opt in enumerate(q.get("options") or [], 1):
            if opt.strip():
                parts.append(f"{i}. {opt.strip()}")
    return "\n\n".join(parts).strip()


def get_promotion_family_question(result: dict) -> Optional[dict]:
    """Return the promotion_family question dict if present in a clarify result."""
    for q in result.get("questions") or []:
        if q.get("id") in ("promotion_family", "gift_reward"):
            return q
    return None


def get_gift_reward_question(result: dict) -> Optional[dict]:
    """Backward-compatible alias for get_promotion_family_question."""
    return get_promotion_family_question(result)


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

        try_apply_promotion_family_answer(session, user_input)

        cleaned_input = strip_confirmation_suffix(user_input)

        if needs_promotion_family_gate(session):
            session.history.append({
                "role": "user",
                "content": (
                    "[SYSTEM NOTE: This is a non-tiered promotion. You MUST ask "
                    "promotion_family FIRST with TWO suggested drafts (free_gift and "
                    "buy_x_get_y). Applies to ALL non-tiered cases — do NOT skip even "
                    "if family seems obvious. FORBIDDEN to go to confirm_intent, pass, "
                    "or sub-questions first. Tiered discount only: skip family question. "
                    f"Merchant said: {cleaned_input[:120]}…]"
                ),
            })

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
            and not needs_promotion_family_gate(session)
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

        result = enforce_promotion_family_gate(session, result)
        family_choice = session.promotion_family_choice or session.gift_reward_family
        if family_choice and result.get("verdict") == "pass":
            result = dict(result)
            result["family"] = family_choice

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
        "gift_reward_resolved": session.promotion_family_resolved,
        "gift_reward_family": session.promotion_family_choice,
        "promotion_family_resolved": session.promotion_family_resolved,
        "promotion_family_choice": session.promotion_family_choice,
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
        gift_reward_resolved=bool(
            data.get("promotion_family_resolved") or data.get("gift_reward_resolved")
        ),
        gift_reward_family=data.get("promotion_family_choice") or data.get("gift_reward_family"),
        promotion_family_resolved=bool(
            data.get("promotion_family_resolved") or data.get("gift_reward_resolved")
        ),
        promotion_family_choice=(
            data.get("promotion_family_choice") or data.get("gift_reward_family")
        ),
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