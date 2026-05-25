#!/usr/bin/env python3
"""
Store Context RAG
=================
Direct pandas NL query (no ReAct agent loop).

  1. Route question → products / variants / collections tool (one LLM call)
  2. Generate & execute pandas code using your exact instruction prompts
  3. Format the database result as structured bullets (one LLM call)

Requirements:
    pip install llama-index-core llama-index-llms-ollama pandas tabulate streamlit

Uses Ollama Cloud gpt-oss:120b (credentials in this file).

Usage:
    python stage_rag.py                   # CLI (Ollama Cloud gpt-oss:120b)
    python stage_rag.py --verbose
    streamlit run stage_rag.py            # Streamlit Q&A UI
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from io import StringIO
from typing import Any, Literal

import pandas as pd

from llama_index.core import Settings
from llama_index.core.llms.llm import LLM
from llama_index.llms.ollama import Ollama
from ollama import AsyncClient, Client


# ── Ollama credentials (hard-coded, self-contained) ─────────────────────────
OLLAMA_API_KEY      = "fc30369802484bb5a1fd3a8e08a7cf08.Ej-syunN81aqQTD95ojud8w3"
OLLAMA_CLOUD_HOST   = "https://ollama.com"
OLLAMA_LOCAL_HOST   = "http://127.0.0.1:11434"
DEFAULT_MODEL       = "gpt-oss:120b"

# Aliases kept for internal use throughout the file
OLLAMA_MODEL            = DEFAULT_MODEL
OLLAMA_TEMPERATURE_ANSWER = 0.6   # polished markdown replies
OLLAMA_TEMPERATURE_CODE   = 0.1   # routing + pandas code generation (accuracy)

# Bump when session/auth logic changes so Streamlit cache_resource rebuilds
_AGENT_CACHE_VERSION = 13  # bumped: hardening pass

ToolName = Literal[
    "products_tool", "variants_tool", "collections_tool", "catalog_overview",
]

# Human labels — never show internal tool ids to the user
TOOL_LABELS: dict[str, str] = {
    "products_tool": "Products",
    "variants_tool": "Variants & options",
    "collections_tool": "Collections",
    "catalog_overview": "Full catalog",
}


def get_ollama_clients(request_timeout: float = 300.0) -> tuple[Client, AsyncClient]:
    """Authenticated Ollama Cloud clients (same pattern as stage1.py)."""
    host = OLLAMA_CLOUD_HOST
    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return (
        Client(host=host, headers=headers, timeout=request_timeout),
        AsyncClient(host=host, headers=headers, timeout=request_timeout),
    )


def verify_ollama_cloud(model: str | None = None) -> None:
    """Fail fast if API key or cloud access is invalid."""
    model = model or OLLAMA_MODEL
    client, _ = get_ollama_clients(request_timeout=60.0)
    client.chat(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    )

# ═══════════════════════════════════════════════════════════════════════════
# Store data — full product catalog
# ═══════════════════════════════════════════════════════════════════════════

STORE_DATA: dict = {
  "store_context_schema": {
    "collections": [

      {
        "collection_name": "Summer Collection",
        "collection_tags": ["summer", "trending", "seasonal"],
        "products": [
          {
            "product_name": "Premium T-Shirt",
            "product_type": "Apparel",
            "subscription_product": False,
            "variants": {"color": ["Black","White","Blue","Red","Green"],
                         "size": ["XS","S","M","L","XL","XXL"],
                         "fit": ["Slim Fit","Regular Fit","Oversized"]},
            "attributes": ["cotton","breathable","lightweight","unisex"],
            "product_tags": ["best_seller","trending","budget_friendly","summer"]
          },
          {
            "product_name": "Polo Shirt",
            "product_type": "Apparel",
            "subscription_product": False,
            "variants": {"color": ["Navy","White","Grey"],
                         "size": ["S","M","L","XL"]},
            "attributes": ["formal_casual","cotton","stretchable"],
            "product_tags": ["office_wear","premium"]
          },
          {
            "product_name": "Sunglasses",
            "product_type": "Accessories",
            "subscription_product": False,
            "variants": {"frame_color": ["Black","Gold","Tortoise","Clear"],
                         "lens_type": ["UV400","Polarised","Mirrored"]},
            "attributes": ["uv_protection","lightweight","unisex"],
            "product_tags": ["summer","trending","best_seller"]
          },
          {
            "product_name": "Sneakers",
            "product_type": "Footwear",
            "subscription_product": False,
            "variants": {"color": ["White","Black","Grey","Navy"],
                         "size": ["UK6","UK7","UK8","UK9","UK10","UK11"]},
            "attributes": ["cushioned","lightweight","breathable"],
            "product_tags": ["summer","best_seller","trending"]
          },
          {
            "product_name": "Water Bottle",
            "product_type": "Fitness",
            "subscription_product": False,
            "variants": {"capacity": ["500ml","750ml","1L"],
                         "color": ["Black","Silver","Blue","Pink"]},
            "attributes": ["stainless_steel","insulated","bpa_free","leak_proof"],
            "product_tags": ["summer","budget_friendly","eco_friendly"]
          }
        ]
      },

      {
        "collection_name": "Skincare",
        "collection_tags": ["skincare","beauty","wellness"],
        "products": [
          {
            "product_name": "Face Wash",
            "product_type": "Skincare",
            "subscription_product": True,
            "variants": {"size": ["100ml","200ml"],
                         "skin_type": ["Oily","Dry","Normal","Combination"]},
            "attributes": ["hydrating","paraben_free","fragrance_free","gentle"],
            "product_tags": ["daily_use","best_seller","dermatologist_tested"]
          },
          {
            "product_name": "Vitamin C Serum",
            "product_type": "Skincare",
            "subscription_product": True,
            "variants": {"size": ["15ml","30ml"],
                         "concentration": ["10%","15%","20%"]},
            "attributes": ["brightening","anti_ageing","vegan","vitamin_c"],
            "product_tags": ["premium","best_seller","trending"]
          },
          {
            "product_name": "Night Cream",
            "product_type": "Skincare",
            "subscription_product": True,
            "variants": {"size": ["50ml","100ml"],
                         "skin_type": ["Normal","Dry","Sensitive"]},
            "attributes": ["nourishing","anti_ageing","hydrating","retinol"],
            "product_tags": ["premium","night_care","dermatologist_tested"]
          },
          {
            "product_name": "Sunscreen SPF50",
            "product_type": "Skincare",
            "subscription_product": True,
            "variants": {"size": ["50ml","100ml"],
                         "finish": ["Matte","Dewy","Invisible"]},
            "attributes": ["spf50","pa4","water_resistant","non_greasy","lightweight"],
            "product_tags": ["daily_use","summer","best_seller"]
          },
          {
            "product_name": "Moisturizer",
            "product_type": "Skincare",
            "subscription_product": True,
            "variants": {"size": ["50ml","150ml"],
                         "skin_type": ["Oily","Dry","Normal","Combination","Sensitive"]},
            "attributes": ["hydrating","oil_free","fragrance_free","vegan"],
            "product_tags": ["daily_use","budget_friendly","dermatologist_tested"]
          },
          {
            "product_name": "Eye Cream",
            "product_type": "Skincare",
            "subscription_product": False,
            "variants": {"size": ["15ml","30ml"]},
            "attributes": ["anti_ageing","hydrating","caffeine","depuffing"],
            "product_tags": ["premium","anti_ageing","trending"]
          }
        ]
      },

      {
        "collection_name": "Fitness",
        "collection_tags": ["fitness","gym","wellness","health"],
        "products": [
          {
            "product_name": "Yoga Mat",
            "product_type": "Fitness",
            "subscription_product": False,
            "variants": {"color": ["Purple","Blue","Black","Green"],
                         "thickness": ["4mm","6mm","8mm"]},
            "attributes": ["non_slip","eco_friendly","extra_thick","lightweight"],
            "product_tags": ["best_seller","eco_friendly","trending"]
          },
          {
            "product_name": "Resistance Bands",
            "product_type": "Fitness",
            "subscription_product": False,
            "variants": {"resistance_level": ["Light","Medium","Heavy","Extra Heavy"],
                         "material": ["Latex","Fabric"]},
            "attributes": ["portable","durable","multi_use","beginner_friendly"],
            "product_tags": ["best_seller","budget_friendly","trending"]
          },
          {
            "product_name": "Protein Powder",
            "product_type": "Nutrition",
            "subscription_product": True,
            "variants": {"flavor": ["Chocolate","Vanilla","Strawberry","Unflavored"],
                         "size": ["500g","1kg","2kg"]},
            "attributes": ["high_protein","whey","low_carb","gluten_free"],
            "product_tags": ["premium","best_seller","subscription_popular"]
          },
          {
            "product_name": "Gym Gloves",
            "product_type": "Fitness",
            "subscription_product": False,
            "variants": {"size": ["S","M","L","XL"],
                         "color": ["Black","Grey","Blue"]},
            "attributes": ["anti_slip","breathable","wrist_support","durable"],
            "product_tags": ["budget_friendly","gym_essential"]
          },
          {
            "product_name": "Foam Roller",
            "product_type": "Fitness",
            "subscription_product": False,
            "variants": {"size": ["30cm","45cm","60cm"],
                         "density": ["Soft","Medium","Firm"]},
            "attributes": ["deep_tissue","portable","durable","grid_texture"],
            "product_tags": ["recovery","trending","premium"]
          }
        ]
      },

      {
        "collection_name": "Electronics",
        "collection_tags": ["electronics","tech","gadgets"],
        "products": [
          {
            "product_name": "Wireless Earbuds",
            "product_type": "Electronics",
            "subscription_product": False,
            "variants": {"color": ["Black","White","Navy"],
                         "connectivity": ["Bluetooth 5.0","Bluetooth 5.3"]},
            "attributes": ["anc","wireless","waterproof","long_battery","ergonomic"],
            "product_tags": ["premium","best_seller","trending"]
          },
          {
            "product_name": "Bluetooth Speaker",
            "product_type": "Electronics",
            "subscription_product": False,
            "variants": {"color": ["Black","Grey","Blue","Red"],
                         "size": ["Compact","Standard","Large"]},
            "attributes": ["portable","waterproof","360_sound","long_battery"],
            "product_tags": ["best_seller","summer","trending"]
          },
          {
            "product_name": "Smart Watch",
            "product_type": "Electronics",
            "subscription_product": False,
            "variants": {"color": ["Black","Silver","Gold","Rose Gold"],
                         "strap_material": ["Silicone","Metal","Leather"],
                         "size": ["40mm","44mm"]},
            "attributes": ["fitness_tracking","gps","heart_rate","waterproof","amoled"],
            "product_tags": ["premium","best_seller","trending"]
          },
          {
            "product_name": "Portable Charger",
            "product_type": "Electronics",
            "subscription_product": False,
            "variants": {"capacity": ["10000mAh","20000mAh","30000mAh"],
                         "color": ["Black","White","Blue"]},
            "attributes": ["fast_charging","lightweight","compact","multi_port"],
            "product_tags": ["best_seller","budget_friendly","travel_essential"]
          },
          {
            "product_name": "Gaming Mouse",
            "product_type": "Electronics",
            "subscription_product": False,
            "variants": {"color": ["Black","White"],
                         "dpi": ["800","1600","3200","6400","12000"],
                         "connectivity": ["Wired","Wireless"]},
            "attributes": ["ergonomic","rgb","precision","lightweight"],
            "product_tags": ["gaming","premium","trending"]
          },
          {
            "product_name": "Mechanical Keyboard",
            "product_type": "Electronics",
            "subscription_product": False,
            "variants": {"switch_type": ["Red","Blue","Brown"],
                         "layout": ["Full","TKL","65%"],
                         "color": ["Black","White","Grey"]},
            "attributes": ["rgb","mechanical","tactile","compact_option"],
            "product_tags": ["gaming","premium","trending"]
          }
        ]
      },

      {
        "collection_name": "Accessories",
        "collection_tags": ["accessories","fashion","lifestyle"],
        "products": [
          {
            "product_name": "Cap",
            "product_type": "Accessories",
            "subscription_product": False,
            "variants": {"color": ["Black","White","Navy","Grey","Olive"],
                         "style": ["Snapback","Dad Cap","Fitted"]},
            "attributes": ["unisex","adjustable","breathable","cotton"],
            "product_tags": ["best_seller","trending","summer","budget_friendly"]
          },
          {
            "product_name": "Travel Tote Bag",
            "product_type": "Accessories",
            "subscription_product": False,
            "variants": {"color": ["Black","Beige","Navy","Olive"],
                         "size": ["Small","Medium","Large"]},
            "attributes": ["canvas","durable","spacious","eco_friendly","lightweight"],
            "product_tags": ["eco_friendly","summer","trending","travel_essential"]
          },
          {
            "product_name": "Laptop Sleeve",
            "product_type": "Accessories",
            "subscription_product": False,
            "variants": {"size": ["13inch","14inch","15inch","16inch"],
                         "color": ["Black","Grey","Navy"]},
            "attributes": ["water_resistant","padded","slim","neoprene"],
            "product_tags": ["office_wear","budget_friendly","best_seller"]
          },
          {
            "product_name": "Leather Wallet",
            "product_type": "Accessories",
            "subscription_product": False,
            "variants": {"color": ["Brown","Black","Tan"],
                         "style": ["Bifold","Trifold","Cardholder"]},
            "attributes": ["genuine_leather","rfid_blocking","slim","durable"],
            "product_tags": ["premium","gift_worthy","office_wear"]
          }
        ]
      },

      {
        "collection_name": "Gift Items",
        "collection_tags": ["gifts","gifting","occasion"],
        "products": [
          {
            "product_name": "Sample Gift",
            "product_type": "Gift Item",
            "subscription_product": False,
            "variants": {},
            "attributes": ["curated","multi_product","trial_size"],
            "product_tags": ["gift_worthy","budget_friendly","new_customer"]
          },
          {
            "product_name": "Mystery Gift Box",
            "product_type": "Gift Item",
            "subscription_product": True,
            "variants": {"tier": ["Essential","Premium","Luxury"]},
            "attributes": ["curated","surprise","multi_product","seasonal"],
            "product_tags": ["gift_worthy","premium","trending","subscription_popular"]
          },
          {
            "product_name": "Welcome Gift",
            "product_type": "Gift Item",
            "subscription_product": False,
            "variants": {"category": ["Skincare","Fitness","Apparel"]},
            "attributes": ["onboarding","multi_product","branded"],
            "product_tags": ["new_customer","gift_worthy","best_seller"]
          }
        ]
      },

      {
        "collection_name": "Apparel",
        "collection_tags": ["clothing","fashion","basics"],
        "products": [
          {
            "product_name": "Minimal Hoodie",
            "product_type": "Apparel",
            "subscription_product": False,
            "variants": {"color": ["Black","Grey","White","Olive","Navy"],
                         "size": ["XS","S","M","L","XL","XXL"],
                         "material": ["Cotton Fleece","French Terry"]},
            "attributes": ["cotton","warm","heavyweight","unisex","streetwear"],
            "product_tags": ["best_seller","trending","winter","premium"]
          },
          {
            "product_name": "Denim Jacket",
            "product_type": "Apparel",
            "subscription_product": False,
            "variants": {"color": ["Light Wash","Dark Wash","Black"],
                         "size": ["XS","S","M","L","XL"]},
            "attributes": ["denim","durable","classic","layering"],
            "product_tags": ["premium","trending","seasonal"]
          },
          {
            "product_name": "Joggers",
            "product_type": "Apparel",
            "subscription_product": False,
            "variants": {"color": ["Black","Grey","Navy","Olive"],
                         "size": ["XS","S","M","L","XL","XXL"],
                         "material": ["Cotton","Polyester Blend"]},
            "attributes": ["comfortable","elastic_waist","tapered","lightweight"],
            "product_tags": ["best_seller","budget_friendly","gym_essential"]
          },
          {
            "product_name": "Tank Top",
            "product_type": "Apparel",
            "subscription_product": False,
            "variants": {"color": ["Black","White","Grey","Navy","Red"],
                         "size": ["XS","S","M","L","XL"]},
            "attributes": ["cotton","breathable","lightweight","gym_ready"],
            "product_tags": ["summer","budget_friendly","gym_essential"]
          }
        ]
      },

      {
        "collection_name": "Beauty",
        "collection_tags": ["beauty","cosmetics","makeup"],
        "products": [
          {
            "product_name": "Luxury Perfume",
            "product_type": "Beauty",
            "subscription_product": False,
            "variants": {"size": ["30ml","50ml","100ml"],
                         "scent_family": ["Floral","Woody","Fresh","Oriental"]},
            "attributes": ["long_lasting","premium","imported","unisex_option"],
            "product_tags": ["premium","gift_worthy","luxury","trending"]
          },
          {
            "product_name": "Lipstick Set",
            "product_type": "Beauty",
            "subscription_product": False,
            "variants": {"shade_range": ["Nudes","Reds","Pinks","Berries"],
                         "finish": ["Matte","Glossy","Satin"]},
            "attributes": ["long_lasting","moisturising","vegan","cruelty_free"],
            "product_tags": ["premium","gift_worthy","trending"]
          },
          {
            "product_name": "Makeup Brush Set",
            "product_type": "Beauty",
            "subscription_product": False,
            "variants": {"piece_count": ["8 Piece","12 Piece","20 Piece"],
                         "bristle_type": ["Synthetic","Natural"]},
            "attributes": ["professional","vegan","durable","ergonomic"],
            "product_tags": ["premium","gift_worthy","best_seller"]
          },
          {
            "product_name": "Hair Serum",
            "product_type": "Beauty",
            "subscription_product": True,
            "variants": {"size": ["50ml","100ml"],
                         "hair_type": ["All Types","Frizzy","Damaged","Fine"]},
            "attributes": ["frizz_control","heat_protection","lightweight","shine"],
            "product_tags": ["premium","best_seller","trending"]
          }
        ]
      }

    ]
  }
}


# ═══════════════════════════════════════════════════════════════════════════
# DataFrame builders
# ═══════════════════════════════════════════════════════════════════════════

def build_dataframes(store_data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert the nested store JSON into three flat DataFrames.

    products_df  — one row per product
    variants_df  — one row per variant type per product (exploded)
    collections_df — one row per collection summary
    """
    products_rows = []
    variants_rows = []
    collections_rows = []

    for col in store_data["store_context_schema"]["collections"]:
        cname = col["collection_name"]
        ctags = col.get("collection_tags", [])

        collections_rows.append({
            "collection_name":  cname,
            "collection_tags":  " | ".join(ctags),
            "product_count":    len(col["products"]),
            "product_names":    " | ".join(p["product_name"] for p in col["products"]),
        })

        for product in col["products"]:
            pname   = product["product_name"]
            ptype   = product["product_type"]
            sub     = product["subscription_product"]
            attrs   = product.get("attributes", [])
            ptags   = product.get("product_tags", [])
            variants = product.get("variants", {})

            # all_tags = product_tags UNION collection_tags (deduplicated, product first).
            # Use this for ANY broad tag search so collection-inherited tags are always found.
            all_tags = list(dict.fromkeys(ptags + ctags))
            products_rows.append({
                "collection_name":      cname,
                "collection_tags":      " | ".join(ctags),
                "product_name":         pname,
                "product_type":         ptype,
                "subscription_product": sub,
                "attributes":           " | ".join(attrs),
                "product_tags":         " | ".join(ptags),
                "all_tags":             " | ".join(all_tags),
                "variant_types":        " | ".join(variants.keys()),
                "has_color_variant":    "color" in variants,
                "has_size_variant":     "size" in variants,
            })

            for variant_type, values in variants.items():
                variants_rows.append({
                    "product_name":    pname,
                    "product_type":    ptype,
                    "collection_name": cname,
                    "variant_type":    variant_type,
                    "variant_values":  " | ".join(str(v) for v in values),
                    "value_count":     len(values),
                })

    products_df    = pd.DataFrame(products_rows)
    variants_df    = pd.DataFrame(variants_rows) if variants_rows else pd.DataFrame()
    collections_df = pd.DataFrame(collections_rows)

    return products_df, variants_df, collections_df


# ═══════════════════════════════════════════════════════════════════════════
# system_prompts.py — inline (exact prompts for query build + bulleted answers)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTS_INSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

PRODUCTS_INSTRUCTION = """\
You are a pandas code generator for a Shopify store product database.
Your ONLY job: write Python pandas code that answers the user's question.
The code runs against a DataFrame named `df`.
Do NOT explain. Do NOT add prose. Output ONLY executable Python.

══════════════════════════════════════════════════════════════
DATAFRAME: df  (products_df)   37 rows · one row = one product
══════════════════════════════════════════════════════════════

EXACT COLUMN NAMES — copy-paste, never guess:
  df['collection_name']       str   — which collection this product belongs to
  df['collection_tags']       str   — pipe-separated tags of the PARENT COLLECTION
                                      (inherited from the collection, not set on this product)
  df['product_name']          str   — canonical product title
  df['product_type']          str   — normalized product category
  df['subscription_product']  bool  — True if subscription-eligible
  df['attributes']            str   — pipe-separated semantic / physical attributes
  df['product_tags']          str   — pipe-separated PRODUCT-LEVEL marketing tags
  df['all_tags']              str   — product_tags UNION collection_tags (deduplicated)
                                      USE THIS for any general "tagged X" query
  df['variant_types']         str   — pipe-separated variant dimension names
  df['has_color_variant']     bool  — True if product has a color variant
  df['has_size_variant']      bool  — True if product has a size variant

══════════════════════════════════════════════════════════════
TAG SYSTEM — THREE LEVELS (critical — read before any tag query)
══════════════════════════════════════════════════════════════

COLUMN TO USE FOR EACH TAG QUERY TYPE:
  "tagged X" / "what about X" / "X products" (no level specified)
      → df['all_tags'].str.contains('X', case=False)      ← USE THIS BY DEFAULT

  "product tag X" / "marketing tag X" / "product_tags contains X"
      → df['product_tags'].str.contains('X', case=False)

  "collection tag X" / "in X-tagged collections"
      → df['collection_tags'].str.contains('X', case=False)

  "attribute X" / "has attribute X"
      → df['attributes'].str.contains('X', case=False)

TAGS ONLY IN product_tags (NOT in collection_tags):
  luxury, winter, night_care, daily_use, dermatologist_tested,
  subscription_popular, gaming, travel_essential, gift_worthy,
  new_customer, office_wear, recovery, gym_essential, anti_ageing

TAGS ONLY IN collection_tags (NOT in product_tags) — products inherit these:
  wellness  → 11 products (all Skincare + all Fitness)
  gym       → 5 products  (all Fitness)
  health    → 5 products  (all Fitness)
  skincare  → 6 products  (all Skincare)
  tech      → 6 products  (all Electronics)
  gadgets   → 6 products  (all Electronics)
  electronics → 6 products (all Electronics)
  accessories → 4 products (all Accessories)
  lifestyle → 4 products  (all Accessories)
  gifts     → 3 products  (all Gift Items)
  gifting   → 3 products  (all Gift Items)
  occasion  → 3 products  (all Gift Items)
  clothing  → 4 products  (all Apparel)
  basics    → 4 products  (all Apparel)
  cosmetics → 4 products  (all Beauty)
  makeup    → 4 products  (all Beauty)

TAGS IN BOTH product_tags AND collection_tags:
  summer    → 7 product-tagged + 5 collection-tagged (use all_tags → may overlap)
  trending  → 19 product-tagged + 5 collection-tagged
  seasonal  → 1 product-tagged (Denim Jacket) + 5 collection-tagged (Summer Collection)
  eco_friendly → 3 product-tagged
  beauty    → collection_tag for Beauty collection (4 products)
  fashion   → collection_tag for Accessories + Apparel (8 products)
  premium   → 16 product-tagged products

══════════════════════════════════════════════════════════════
ALL 37 PRODUCTS (product_name → collection → product_type)
══════════════════════════════════════════════════════════════
Premium T-Shirt      → Summer Collection → Apparel
Polo Shirt           → Summer Collection → Apparel
Sunglasses           → Summer Collection → Accessories
Sneakers             → Summer Collection → Footwear
Water Bottle         → Summer Collection → Fitness
Face Wash            → Skincare          → Skincare   (subscription=True)
Vitamin C Serum      → Skincare          → Skincare   (subscription=True)
Night Cream          → Skincare          → Skincare   (subscription=True)
Sunscreen SPF50      → Skincare          → Skincare   (subscription=True)
Moisturizer          → Skincare          → Skincare   (subscription=True)
Eye Cream            → Skincare          → Skincare
Yoga Mat             → Fitness           → Fitness
Resistance Bands     → Fitness           → Fitness
Protein Powder       → Fitness           → Nutrition  (subscription=True)
Gym Gloves           → Fitness           → Fitness
Foam Roller          → Fitness           → Fitness
Wireless Earbuds     → Electronics       → Electronics
Bluetooth Speaker    → Electronics       → Electronics
Smart Watch          → Electronics       → Electronics
Portable Charger     → Electronics       → Electronics
Gaming Mouse         → Electronics       → Electronics
Mechanical Keyboard  → Electronics       → Electronics
Cap                  → Accessories       → Accessories
Travel Tote Bag      → Accessories       → Accessories
Laptop Sleeve        → Accessories       → Accessories
Leather Wallet       → Accessories       → Accessories
Sample Gift          → Gift Items        → Gift Item
Mystery Gift Box     → Gift Items        → Gift Item  (subscription=True)
Welcome Gift         → Gift Items        → Gift Item
Minimal Hoodie       → Apparel           → Apparel
Denim Jacket         → Apparel           → Apparel
Joggers              → Apparel           → Apparel
Tank Top             → Apparel           → Apparel
Luxury Perfume       → Beauty            → Beauty
Lipstick Set         → Beauty            → Beauty
Makeup Brush Set     → Beauty            → Beauty
Hair Serum           → Beauty            → Beauty     (subscription=True)

══════════════════════════════════════════════════════════════
PANDAS PATTERNS — copy these exactly
══════════════════════════════════════════════════════════════

# ── DEFAULT TAG SEARCH — use all_tags for any "tagged X" / "X products" query ──
df[df['all_tags'].str.contains('wellness', case=False)]   # → 11 products (Skincare+Fitness)
df[df['all_tags'].str.contains('seasonal', case=False)]   # → 6 products (Denim Jacket + 5 Summer)
df[df['all_tags'].str.contains('luxury', case=False)]     # → 1 product (Luxury Perfume)
df[df['all_tags'].str.contains('gym', case=False)]        # → 5 Fitness products
df[df['all_tags'].str.contains('tech', case=False)]       # → 6 Electronics products
df[df['all_tags'].str.contains('eco_friendly', case=False)] # → 3 products

# ── PRODUCT TAG ONLY (strict product-level marketing tags) ──────────────────
df[df['product_tags'].str.contains('best_seller', case=False)]
df[df['product_tags'].str.contains('premium', case=False)]
df[df['product_tags'].str.contains('seasonal', case=False)]  # → Denim Jacket ONLY
df[df['product_tags'].str.contains('winter', case=False)]    # → Minimal Hoodie ONLY
df[df['product_tags'].str.contains('luxury', case=False)]    # → Luxury Perfume ONLY

# ── COLLECTION TAG SEARCH (all products in collections with that tag) ────────
df[df['collection_tags'].str.contains('wellness', case=False)]  # → 11 products
df[df['collection_tags'].str.contains('seasonal', case=False)]  # → 5 Summer Collection products
df[df['collection_tags'].str.contains('gym', case=False)]       # → 5 Fitness products

# ── ATTRIBUTE SEARCH (physical/semantic traits) ──────────────────────────────
df[df['attributes'].str.contains('vegan', case=False)]
df[df['attributes'].str.contains('waterproof', case=False)]
df[df['attributes'].str.contains('hydrating', case=False)]

# ── COLLECTION / TYPE FILTER ─────────────────────────────────────────────────
df[df['collection_name'] == 'Skincare']
df[df['product_type'] == 'Apparel']
df[df['collection_name'].isin(['Skincare','Fitness'])]

# ── SUBSCRIPTION ─────────────────────────────────────────────────────────────
df[df['subscription_product'] == True]

# ── MULTI-CONDITION ──────────────────────────────────────────────────────────
df[(df['all_tags'].str.contains('wellness')) & (df['subscription_product'] == True)]
df[(df['product_type'] == 'Apparel') & (df['product_tags'].str.contains('premium'))]

# ── COUNT / AGGREGATE ────────────────────────────────────────────────────────
len(df[df['all_tags'].str.contains('wellness', case=False)])
df.groupby('collection_name')['product_name'].apply(list)

# ── SHOW TAG LEVEL BREAKDOWN ─────────────────────────────────────────────────
# When user asks about a tag and you want to show which level it comes from:
term = 'seasonal'
prod_match = df[df['product_tags'].str.contains(term, case=False)]
coll_match  = df[df['collection_tags'].str.contains(term, case=False) &
                 ~df['product_tags'].str.contains(term, case=False)]
print(f"product_tags match ({len(prod_match)}): {list(prod_match['product_name'])}")
print(f"collection_tags match ({len(coll_match)}): {list(coll_match['product_name'])}")

══════════════════════════════════════════════════════════════
OUTPUT RULE
══════════════════════════════════════════════════════════════
Always end with print(). For product lists include product_name + collection_name + product_type.
Example:
  result = df[df['all_tags'].str.contains('wellness', case=False)][['product_name','collection_name','product_type']]
  print(result.to_string(index=False))
"""


# ═══════════════════════════════════════════════════════════════════════════
# VARIANTS_INSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

VARIANTS_INSTRUCTION = """\
You are a pandas code generator for a store product variants database.
Your ONLY job: write Python pandas code that answers the user's question.
The code runs against a DataFrame named `df`.
Do NOT explain. Do NOT add prose. Output ONLY executable Python.

══════════════════════════════════════════════════════════════
DATAFRAME: df  (variants_df)
One row = one variant dimension for one product.  75 rows total.
══════════════════════════════════════════════════════════════

EXACT COLUMN NAMES:
  df['product_name']     str  — product this variant belongs to
  df['product_type']     str  — normalized product category
  df['collection_name']  str  — collection the product belongs to
  df['variant_type']     str  — the variant dimension (color, size, flavor, etc.)
  df['variant_values']   str  — pipe-separated allowed values for this variant
  df['value_count']      int  — number of options in this variant

══════════════════════════════════════════════════════════════
ALL UNIQUE variant_type VALUES (28 types):
══════════════════════════════════════════════════════════════
color | size | fit | frame_color | lens_type | capacity | skin_type
concentration | finish | resistance_level | material | flavor | density
connectivity | strap_material | dpi | switch_type | layout | style
tier | category | thickness | hair_type | piece_count | bristle_type
scent_family | shade_range

══════════════════════════════════════════════════════════════
COMPLETE VARIANT MAP (product → variant_type → values)
══════════════════════════════════════════════════════════════
Premium T-Shirt      color=[Black,White,Blue,Red,Green]  size=[XS,S,M,L,XL,XXL]  fit=[Slim Fit,Regular Fit,Oversized]
Polo Shirt           color=[Navy,White,Grey]  size=[S,M,L,XL]
Sunglasses           frame_color=[Black,Gold,Tortoise,Clear]  lens_type=[UV400,Polarised,Mirrored]
Sneakers             color=[White,Black,Grey,Navy]  size=[UK6,UK7,UK8,UK9,UK10,UK11]
Water Bottle         capacity=[500ml,750ml,1L]  color=[Black,Silver,Blue,Pink]
Face Wash            size=[100ml,200ml]  skin_type=[Oily,Dry,Normal,Combination]
Vitamin C Serum      size=[15ml,30ml]  concentration=[10%,15%,20%]
Night Cream          size=[50ml,100ml]  skin_type=[Normal,Dry,Sensitive]
Sunscreen SPF50      size=[50ml,100ml]  finish=[Matte,Dewy,Invisible]
Moisturizer          size=[50ml,150ml]  skin_type=[Oily,Dry,Normal,Combination,Sensitive]
Eye Cream            size=[15ml,30ml]
Yoga Mat             color=[Purple,Blue,Black,Green]  thickness=[4mm,6mm,8mm]
Resistance Bands     resistance_level=[Light,Medium,Heavy,Extra Heavy]  material=[Latex,Fabric]
Protein Powder       flavor=[Chocolate,Vanilla,Strawberry,Unflavored]  size=[500g,1kg,2kg]
Gym Gloves           size=[S,M,L,XL]  color=[Black,Grey,Blue]
Foam Roller          size=[30cm,45cm,60cm]  density=[Soft,Medium,Firm]
Wireless Earbuds     color=[Black,White,Navy]  connectivity=[Bluetooth 5.0,Bluetooth 5.3]
Bluetooth Speaker    color=[Black,Grey,Blue,Red]  size=[Compact,Standard,Large]
Smart Watch          color=[Black,Silver,Gold,Rose Gold]  strap_material=[Silicone,Metal,Leather]  size=[40mm,44mm]
Portable Charger     capacity=[10000mAh,20000mAh,30000mAh]  color=[Black,White,Blue]
Gaming Mouse         color=[Black,White]  dpi=[800,1600,3200,6400,12000]  connectivity=[Wired,Wireless]
Mechanical Keyboard  switch_type=[Red,Blue,Brown]  layout=[Full,TKL,65%]  color=[Black,White,Grey]
Cap                  color=[Black,White,Navy,Grey,Olive]  style=[Snapback,Dad Cap,Fitted]
Travel Tote Bag      color=[Black,Beige,Navy,Olive]  size=[Small,Medium,Large]
Laptop Sleeve        size=[13inch,14inch,15inch,16inch]  color=[Black,Grey,Navy]
Leather Wallet       color=[Brown,Black,Tan]  style=[Bifold,Trifold,Cardholder]
Sample Gift          (no variants)
Mystery Gift Box     tier=[Essential,Premium,Luxury]
Welcome Gift         category=[Skincare,Fitness,Apparel]
Minimal Hoodie       color=[Black,Grey,White,Olive,Navy]  size=[XS,S,M,L,XL,XXL]  material=[Cotton Fleece,French Terry]
Denim Jacket         color=[Light Wash,Dark Wash,Black]  size=[XS,S,M,L,XL]
Joggers              color=[Black,Grey,Navy,Olive]  size=[XS,S,M,L,XL,XXL]  material=[Cotton,Polyester Blend]
Tank Top             color=[Black,White,Grey,Navy,Red]  size=[XS,S,M,L,XL]
Luxury Perfume       size=[30ml,50ml,100ml]  scent_family=[Floral,Woody,Fresh,Oriental]
Lipstick Set         shade_range=[Nudes,Reds,Pinks,Berries]  finish=[Matte,Glossy,Satin]
Makeup Brush Set     piece_count=[8 Piece,12 Piece,20 Piece]  bristle_type=[Synthetic,Natural]
Hair Serum           size=[50ml,100ml]  hair_type=[All Types,Frizzy,Damaged,Fine]

══════════════════════════════════════════════════════════════
PANDAS PATTERNS
══════════════════════════════════════════════════════════════

# ── Get all variants for one product ─────────────────────────────────
df[df['product_name'] == 'Smart Watch']

# ── Get specific variant type for one product ─────────────────────────
df[(df['product_name'] == 'Smart Watch') & (df['variant_type'] == 'color')]

# ── Get ALL products that have a specific variant type ─────────────────
df[df['variant_type'] == 'color']                   # all products with color
df[df['variant_type'] == 'size']                    # all products with size
df[df['variant_type'] == 'flavor']                  # products with flavor option

# ── Search inside variant_values (pipe-separated) ──────────────────────
df[df['variant_values'].str.contains('Black', case=False)]   # all Black variants
df[df['variant_values'].str.contains('XL', case=False)]      # all XL options
df[(df['variant_type'] == 'color') & (df['variant_values'].str.contains('Black'))]

# ── Products available in a specific color ─────────────────────────────
df[(df['variant_type'] == 'color') & (df['variant_values'].str.contains('Navy'))]

# ── Products available in a specific size ─────────────────────────────
df[(df['variant_type'] == 'size') & (df['variant_values'].str.contains('XL'))]

# ── Products with the most variant options ────────────────────────────
df.sort_values('value_count', ascending=False).head(10)

# ── How many color options does a product have ────────────────────────
df[(df['product_name'] == 'Premium T-Shirt') & (df['variant_type'] == 'color')]['value_count']

# ── All products from a collection with their variants ────────────────
df[df['collection_name'] == 'Electronics']

# ── Get variant count per product ─────────────────────────────────────
df.groupby('product_name')['variant_type'].count().sort_values(ascending=False)

# ── Products with specific skin_type option ───────────────────────────
df[(df['variant_type'] == 'skin_type') & (df['variant_values'].str.contains('Sensitive'))]

# ── All skincare products and their variant types ─────────────────────
df[df['product_type'] == 'Skincare'][['product_name','variant_type','variant_values']]

══════════════════════════════════════════════════════════════
OUTPUT RULE
══════════════════════════════════════════════════════════════
For variant value questions: extract and split the pipe-separated string.
Example to show clean values:
  row = df[(df['product_name']=='Smart Watch') & (df['variant_type']=='color')].iloc[0]
  values = row['variant_values'].split(' | ')
  print(f"Smart Watch colors: {values}")

For tabular results:
  print(result[['product_name','variant_type','variant_values']].to_string(index=False))
"""


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS_INSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

COLLECTIONS_INSTRUCTION = """\
You are a pandas code generator for a store collections database.
Your ONLY job: write Python pandas code that answers the user's question.
The code runs against a DataFrame named `df`.
Do NOT explain. Do NOT add prose. Output ONLY executable Python.

══════════════════════════════════════════════════════════════
DATAFRAME: df  (collections_df)
One row = one collection.  8 rows total.
══════════════════════════════════════════════════════════════

EXACT COLUMN NAMES:
  df['collection_name']  str  — collection title
  df['collection_tags']  str  — pipe-separated semantic tags
  df['product_count']    int  — number of products in collection
  df['product_names']    str  — pipe-separated list of product names

══════════════════════════════════════════════════════════════
COMPLETE COLLECTIONS DATA (memorize this):
══════════════════════════════════════════════════════════════
Summer Collection  tags=[summer,trending,seasonal]     5 products
                   products=[Premium T-Shirt, Polo Shirt, Sunglasses, Sneakers, Water Bottle]

Skincare           tags=[skincare,beauty,wellness]     6 products
                   products=[Face Wash, Vitamin C Serum, Night Cream, Sunscreen SPF50, Moisturizer, Eye Cream]

Fitness            tags=[fitness,gym,wellness,health]  5 products
                   products=[Yoga Mat, Resistance Bands, Protein Powder, Gym Gloves, Foam Roller]

Electronics        tags=[electronics,tech,gadgets]     6 products
                   products=[Wireless Earbuds, Bluetooth Speaker, Smart Watch, Portable Charger, Gaming Mouse, Mechanical Keyboard]

Accessories        tags=[accessories,fashion,lifestyle] 4 products
                   products=[Cap, Travel Tote Bag, Laptop Sleeve, Leather Wallet]

Gift Items         tags=[gifts,gifting,occasion]       3 products
                   products=[Sample Gift, Mystery Gift Box, Welcome Gift]

Apparel            tags=[clothing,fashion,basics]      4 products
                   products=[Minimal Hoodie, Denim Jacket, Joggers, Tank Top]

Beauty             tags=[beauty,cosmetics,makeup]      4 products
                   products=[Luxury Perfume, Lipstick Set, Makeup Brush Set, Hair Serum]

══════════════════════════════════════════════════════════════
PANDAS PATTERNS
══════════════════════════════════════════════════════════════

# ── Get one collection ────────────────────────────────────────────────
df[df['collection_name'] == 'Electronics']

# ── Get products in a collection (returns pipe-separated string) ──────
df[df['collection_name'] == 'Skincare']['product_names'].values[0]
df[df['collection_name'] == 'Skincare']['product_names'].values[0].split(' | ')

# ── Sort by product count ─────────────────────────────────────────────
df.sort_values('product_count', ascending=False)
df.sort_values('product_count', ascending=True)

# ── Collection with most products ─────────────────────────────────────
df.loc[df['product_count'].idxmax()]

# ── Filter by tag ─────────────────────────────────────────────────────
df[df['collection_tags'].str.contains('wellness', case=False)]
df[df['collection_tags'].str.contains('fashion', case=False)]
df[df['collection_tags'].str.contains('summer', case=False)]

# ── Which collection contains a specific product ──────────────────────
df[df['product_names'].str.contains('Smart Watch', case=False)]['collection_name']
df[df['product_names'].str.contains('Protein Powder', case=False)]['collection_name']

# ── List all collections ──────────────────────────────────────────────
df[['collection_name','product_count','collection_tags']]

# ── Total products across all collections ────────────────────────────
df['product_count'].sum()

══════════════════════════════════════════════════════════════
OUTPUT RULE
══════════════════════════════════════════════════════════════
For product lists, split the pipe-separated string:
  products = df[df['collection_name']=='Skincare']['product_names'].values[0].split(' | ')
  print(f"Skincare products: {products}")

For collection summaries:
  print(df[['collection_name','product_count']].to_string(index=False))
"""


# ═══════════════════════════════════════════════════════════════════════════
# AGENT_SYSTEM_PROMPT
# ═══════════════════════════════════════════════════════════════════════════

AGENT_SYSTEM_PROMPT = """\
You are a precise store data assistant with access to three query tools.
Your job: answer any question about store products, collections, and variants
with 100% accuracy by querying the correct tool(s).

══════════════════════════════════════════════════════════════
THREE TOOLS — use EXACTLY the right one
══════════════════════════════════════════════════════════════

products_tool
  Data: 37 products with name, type, collection, attributes, tags, subscription flag
  Use for:
    → Which products have tag X?          ("best_seller", "premium", "eco_friendly"…)
    → Which products have attribute X?    ("vegan", "waterproof", "hydrating"…)
    → Which products are subscription eligible?
    → What product type is X?
    → Find all Apparel / Skincare / Electronics products
    → Is product X in collection Y?
    → How many products in the store?

variants_tool
  Data: 75 rows — one per variant dimension per product (color, size, flavor…)
  Use for:
    → What colors does product X come in?
    → What sizes are available for X?
    → What flavors does Protein Powder have?
    → What DPI options does the Gaming Mouse have?
    → What capacity options does the Portable Charger have?
    → Which products come in Black?
    → Which products have more than N options?
    → What strap options does the Smart Watch have?
    → Does product X come in size XL?

collections_tool
  Data: 8 collections with name, tags, product count, product list
  Use for:
    → What products are in collection X?
    → Which collection has the most products?
    → Which collections are tagged with Y?
    → What is in the Gift Items / Electronics / Beauty collection?
    → List all collections
    → How many collections are there?

══════════════════════════════════════════════════════════════
CHAINING RULES — when to use multiple tools
══════════════════════════════════════════════════════════════

Use BOTH products_tool AND variants_tool for:
  "Which vegan products come in Black?"
  → products_tool: find vegan products
  → variants_tool: filter those products with color=Black
  → Intersect the product names

  "Show me all subscription products and their size options"
  → products_tool: get subscription=True products
  → variants_tool: get size variants for those products

  "Which premium Electronics come in Black?"
  → products_tool: filter product_type=Electronics + tag=premium
  → variants_tool: check color variants for those products

Use BOTH collections_tool AND products_tool for:
  "Which Fitness collection products can I subscribe to?"
  → collections_tool: get products in Fitness collection
  → products_tool: filter those for subscription_product=True

══════════════════════════════════════════════════════════════
TAG LEVELS — COMPLETE REFERENCE (never return 0 for collection-level tags)
══════════════════════════════════════════════════════════════

products_df has THREE tag columns:
  df['product_tags']   — tags set directly on a product
  df['collection_tags'] — tags inherited from the parent collection
  df['all_tags']       — UNION of both (USE THIS for general "tagged X" queries)

COLLECTION-ONLY TAGS (zero product_tags hits — must search all_tags or collection_tags):
  wellness  → 11 products in Skincare (6) + Fitness (5) collections
  gym       → 5 products in Fitness collection
  health    → 5 products in Fitness collection
  skincare  → 6 products in Skincare collection
  tech      → 6 products in Electronics collection
  gadgets   → 6 products in Electronics collection
  electronics → 6 products in Electronics collection
  accessories → 4 products in Accessories collection
  lifestyle → 4 products in Accessories collection
  gifts / gifting / occasion → 3 products in Gift Items collection
  clothing / basics → 4 products in Apparel collection
  cosmetics / makeup → 4 products in Beauty collection

PRODUCT-ONLY TAGS (only in product_tags, not collection_tags):
  luxury → Luxury Perfume (1 product)
  winter → Minimal Hoodie (1 product)
  seasonal → Denim Jacket (1 product via product_tag)
  gaming, travel_essential, gift_worthy, new_customer, office_wear,
  recovery, gym_essential, anti_ageing, night_care, daily_use,
  dermatologist_tested, subscription_popular

RULE: When user says "tagged X" or "what about X" or "X products" —
  → ALWAYS use products_tool with df['all_tags'].str.contains('X')
  → NEVER return 0 results when the tag exists at collection level
  → If all_tags returns results, explain whether each product has the tag
    directly (product_tag) or via its collection (collection_tag)

══════════════════════════════════════════════════════════════
COMPLETE STORE KNOWLEDGE (memorize — use for instant answers)
══════════════════════════════════════════════════════════════

8 COLLECTIONS (37 total products):
  Summer Collection  → 5 products  (Apparel, Accessories, Footwear, Fitness)
  Skincare           → 6 products  (all Skincare type)
  Fitness            → 5 products  (Fitness + Nutrition)
  Electronics        → 6 products  (all Electronics type)
  Accessories        → 4 products  (all Accessories type)
  Gift Items         → 3 products  (all Gift Item type)
  Apparel            → 4 products  (all Apparel type)
  Beauty             → 4 products  (all Beauty type)

SUBSCRIPTION PRODUCTS (8 total):
  Face Wash, Vitamin C Serum, Night Cream, Sunscreen SPF50, Moisturizer
  (all Skincare), Protein Powder (Fitness), Mystery Gift Box (Gift Items),
  Hair Serum (Beauty)

PRODUCT TYPES (9 types):
  Apparel → Premium T-Shirt, Polo Shirt, Minimal Hoodie, Denim Jacket, Joggers, Tank Top
  Skincare → Face Wash, Vitamin C Serum, Night Cream, Sunscreen SPF50, Moisturizer, Eye Cream
  Fitness → Yoga Mat, Resistance Bands, Gym Gloves, Foam Roller, Water Bottle
  Electronics → Wireless Earbuds, Bluetooth Speaker, Smart Watch, Portable Charger, Gaming Mouse, Mechanical Keyboard
  Accessories → Cap, Travel Tote Bag, Laptop Sleeve, Leather Wallet, Sunglasses
  Footwear → Sneakers
  Nutrition → Protein Powder
  Gift Item → Sample Gift, Mystery Gift Box, Welcome Gift
  Beauty → Luxury Perfume, Lipstick Set, Makeup Brush Set, Hair Serum

PRODUCTS BY KEY TAG:
  best_seller: Premium T-Shirt, Sunglasses, Sneakers, Face Wash, Vitamin C Serum,
               Sunscreen SPF50, Yoga Mat, Resistance Bands, Protein Powder,
               Wireless Earbuds, Bluetooth Speaker, Smart Watch, Portable Charger,
               Cap, Laptop Sleeve, Welcome Gift, Minimal Hoodie, Joggers,
               Makeup Brush Set, Hair Serum
  premium:     Polo Shirt, Vitamin C Serum, Night Cream, Foam Roller, Wireless Earbuds,
               Smart Watch, Gaming Mouse, Mechanical Keyboard, Leather Wallet,
               Mystery Gift Box, Minimal Hoodie, Denim Jacket, Luxury Perfume,
               Lipstick Set, Makeup Brush Set, Hair Serum
  eco_friendly: Yoga Mat, Travel Tote Bag, Water Bottle
  gaming:       Gaming Mouse, Mechanical Keyboard
  gift_worthy:  Leather Wallet, Sample Gift, Mystery Gift Box, Welcome Gift,
                Luxury Perfume, Lipstick Set, Makeup Brush Set
  vegan:        Vitamin C Serum, Moisturizer, Lipstick Set, Makeup Brush Set
  budget_friendly: Premium T-Shirt, Water Bottle, Resistance Bands, Gym Gloves,
                   Cap, Laptop Sleeve, Joggers, Tank Top, Sample Gift, Moisturizer
  trending:     Premium T-Shirt, Sunglasses, Sneakers, Vitamin C Serum, Eye Cream,
                Yoga Mat, Resistance Bands, Foam Roller, Wireless Earbuds,
                Bluetooth Speaker, Smart Watch, Cap, Travel Tote Bag, Minimal Hoodie,
                Denim Jacket, Luxury Perfume, Lipstick Set, Hair Serum

══════════════════════════════════════════════════════════════
RESPONSE FORMAT — always structured and complete
══════════════════════════════════════════════════════════════

For product lists:   Name each product with its collection and type.
For variant queries: List values cleanly, e.g. "Black | White | Navy | Grey"
For counts:          Give the number first, then list items.
For comparisons:     Use a structured breakdown.
Never say "I don't know" — query the tool and return the result.
Never return raw DataFrames or code — always interpret and present clearly.

GOOD RESPONSE EXAMPLES:

Q: "Which products are subscription eligible?"
A: "8 products offer subscription:
    Skincare (5): Face Wash, Vitamin C Serum, Night Cream, Sunscreen SPF50, Moisturizer
    Fitness (1): Protein Powder
    Gift Items (1): Mystery Gift Box
    Beauty (1): Hair Serum"

Q: "What colors does the Smart Watch come in?"
A: "The Smart Watch is available in 4 colors: Black, Silver, Gold, Rose Gold.
    It also has 3 strap materials (Silicone, Metal, Leather) and 2 sizes (40mm, 44mm)."

Q: "Which eco-friendly products exist?"
A: "3 products have the eco_friendly attribute or tag:
    • Yoga Mat (Fitness) — attributes: non_slip, eco_friendly, extra_thick, lightweight
    • Travel Tote Bag (Accessories) — attributes: canvas, durable, spacious, eco_friendly
    • Water Bottle (Summer Collection, Fitness) — tags: eco_friendly"
"""


# ═══════════════════════════════════════════════════════════════════════════
# Pandas NL query tools (LlamaIndex-style, without llama-index-experimental)
# ═══════════════════════════════════════════════════════════════════════════

_SAFE_BUILTINS: dict[str, Any] = {
    "str": str, "int": int, "float": float, "bool": bool,
    "len": len, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "min": min, "max": max, "sum": sum, "sorted": sorted,
    "enumerate": enumerate, "zip": zip, "range": range,
    "round": round, "abs": abs, "any": any, "all": all,
    "isinstance": isinstance, "type": type,
    "print": print,
    "True": True, "False": False, "None": None,
}


def _extract_python_code(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    return text


def _llm_complete(llm: Ollama, prompt: str, *, temperature: float) -> str:
    """Complete with explicit temperature (Ollama uses instance temperature in options)."""
    prev = llm.temperature
    llm.temperature = temperature
    try:
        return (llm.complete(prompt).text or "").strip()
    finally:
        llm.temperature = prev


MARKDOWN_ANSWER_PROMPT = """\
You are a polished store assistant in a chat UI. Turn the database lookup below into a \
beautiful, scannable **Markdown** answer.

══════════════════════════════════════════════════════════════
MARKDOWN RULES (follow exactly)
══════════════════════════════════════════════════════════════
• Open with one short summary line (counts or direct answer).
• Use `###` section headings when there are multiple groups (collections, types, etc.).
• Use **bold** for product names and collection names.
• Use `-` bullet lists; indent sub-bullets with two spaces.
• Separate sections with a blank line — never one long run-on paragraph.
• Tags / attributes: show as comma-separated words (e.g. summer, trending, seasonal).
• Variant values: show inline or as sub-bullets (e.g. Black · White · Navy).
• Counts first, then items (e.g. "**8 products** offer subscription:").
• Tone: helpful, concise, like ChatGPT / Claude — not robotic.
• Never mention tools, pandas, routers, JSON, or "database".
• Never wrap the entire answer in a code fence.
• **Counts must exactly match the raw data** — never invent or reuse a count from chat history.
• Answer **only** what was asked — do not dump the full catalog unless the lookup is a full catalog.
• For follow-ups ("which are those?"), list only the items from the raw data — not every collection.

══════════════════════════════════════════════════════════════
TAG LEVEL RULE — CRITICAL (read before formatting any tag answer)
══════════════════════════════════════════════════════════════
Products have TWO tag sources:
  product_tags     — tags set directly on the product (marketing tags)
  collection_tags  — tags inherited from the collection the product belongs to

When the raw data contains products found via all_tags (which combines both):
  • ALWAYS list the products found — never say "0 products match"
  • For each product, clarify whether the tag is a direct product tag or collection-inherited
  • Use wording like: "via the **Skincare** collection (tagged: wellness, beauty)"

Collection-level tags and their products (memorize):
  wellness   → **Skincare** collection: Face Wash, Vitamin C Serum, Night Cream,
               Sunscreen SPF50, Moisturizer, Eye Cream
               **Fitness** collection: Yoga Mat, Resistance Bands, Protein Powder,
               Gym Gloves, Foam Roller  (11 products total)
  gym/health → **Fitness** collection: same 5 Fitness products
  tech/gadgets → **Electronics** collection: Wireless Earbuds, Bluetooth Speaker,
               Smart Watch, Portable Charger, Gaming Mouse, Mechanical Keyboard
  seasonal   → **Apparel**: Denim Jacket (product tag)
               **Summer Collection**: Premium T-Shirt, Polo Shirt, Sunglasses,
               Sneakers, Water Bottle (collection tag) — 6 total
  luxury     → **Luxury Perfume** only (product tag) — 1 product, this is correct

NEVER SAY "0 products match the wellness/gym/health/seasonal/tech/gadgets tag."
  → wellness, gym, health, tech, gadgets are collection-level tags — always return products.

══════════════════════════════════════════════════════════════
EXAMPLE — wellness tag question
══════════════════════════════════════════════════════════════
**11 products** are in wellness-tagged collections:

### Skincare collection *(tagged: skincare, beauty, wellness)*
- **Face Wash**
- **Vitamin C Serum**
- **Night Cream**
- **Sunscreen SPF50**
- **Moisturizer**
- **Eye Cream**

### Fitness collection *(tagged: fitness, gym, wellness, health)*
- **Yoga Mat**
- **Resistance Bands**
- **Protein Powder**
- **Gym Gloves**
- **Foam Roller**

*Note: "wellness" is a collection-level tag — none of these products have it as a direct product tag.*

══════════════════════════════════════════════════════════════
EXAMPLE — seasonal tag question
══════════════════════════════════════════════════════════════
**6 products** are associated with the **seasonal** tag:

**Direct product tag (1):**
- **Denim Jacket** (Apparel · Apparel collection)

**Via Summer Collection tag (5):**
- **Premium T-Shirt**, **Polo Shirt**, **Sunglasses**, **Sneakers**, **Water Bottle**
  *(Summer Collection is tagged: summer, trending, seasonal)*

══════════════════════════════════════════════════════════════
EXAMPLE — luxury tag question
══════════════════════════════════════════════════════════════
**1 product** has the **luxury** marketing tag:

- **Luxury Perfume** (Beauty · Beauty collection)

*"luxury" is a direct product tag — no other products or collections carry it.*

══════════════════════════════════════════════════════════════
EXAMPLE — subscription question
══════════════════════════════════════════════════════════════
**8 products** are subscription eligible:

**Skincare (5)**
- Face Wash · Vitamin C Serum · Night Cream · Sunscreen SPF50 · Moisturizer

**Fitness (1)**
- Protein Powder

**Gift Items (1)**
- Mystery Gift Box

**Beauty (1)**
- Hair Serum
"""


def _format_markdown_answer(
    llm: LLM,
    user_question: str,
    raw_chunk: str,
    tool_name: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Turn executed pandas / catalog data into polished Markdown for the chat UI."""
    raw = (raw_chunk or "").strip()
    _EMPTY_SIGNALS = ("no matching records found", "exec failed", "exec_failed", "query error")
    if not raw or any(sig in raw.lower() for sig in _EMPTY_SIGNALS):
        return "I couldn't find anything in the catalog that matches that question."

    source = TOOL_LABELS.get(tool_name, "Store data")
    if tool_name == "combined":
        source = "Products, variants, and/or collections"

    hist_block = _format_history(history or [], max_turns=3)
    prompt = f"""{MARKDOWN_ANSWER_PROMPT}

Conversation (for context only — do not repeat prior answers as the full reply):
{hist_block}

Customer question: {user_question}
Lookup area: {source}

Raw catalog data (format this — counts and names must match exactly):
{raw}

Output ONLY the Markdown answer."""
    formatted = _llm_complete(llm, prompt, temperature=OLLAMA_TEMPERATURE_ANSWER)
    return formatted or raw


MARKDOWN_ANSWER_JUSTIFICATION_ADDENDUM = """\
══════════════════════════════════════════════════════════════
JUSTIFICATION & ACCURACY RULES (CRITICAL — follow always)
══════════════════════════════════════════════════════════════
• ALWAYS start with a count: "Found X products / collections / variants"
• For SEMANTIC queries (color values, seasonal suitability, use-case):
    → Explain WHY each item qualifies (which tag / attribute / variant value)
    → Separate EXACT matches from INFERRED matches
    → If searching for a color/value not in DB: say "No exact match for '{value}'.
       Closest available: ..." and list alternatives
• For WINTER / SEASONAL queries:
    → Category 1 — Explicitly tagged 'winter': list products
    → Category 2 — Warm/heavy attributes inferred as winter-suitable: list with reason
    → Category 3 — Seasonal collection items: note they're summer-tagged
    → End with: "Recommendation for winter: ..." 
• For MULTI-TOOL results: show how data from each source was combined
• NEVER skip the justification — users need to understand WHY results match
• Counts must match raw data exactly
"""


def _format_markdown_answer_with_justification(
    llm: "Ollama",
    user_question: str,
    raw_chunk: str,
    tool_name: str,
    history: list[dict[str, str]] | None = None,
    justification_ctx: str = "",
) -> str:
    """
    Enhanced formatter: turns raw data into polished Markdown WITH justification.
    Uses the ReAct plan's semantic_expansion and justification_instruction to
    produce 100%-accurate, fully-reasoned answers.

    Hardened:
    - Blocks hallucination when raw data is empty/exec-failed
    - Strips exec-failure diagnostics from the user-facing answer
    """
    raw = (raw_chunk or "").strip()

    # ── Guard: never let the formatter hallucinate on empty/failed data ───────
    _EMPTY_SIGNALS = (
        "no matching records found",
        "exec failed",
        "exec_failed",
        "query error",
    )
    raw_lower = raw.lower()
    is_empty = not raw or any(sig in raw_lower for sig in _EMPTY_SIGNALS)

    if is_empty:
        # Surface a helpful, honest "not found" message — no hallucination
        q_lower = user_question.lower()
        # Try to give a useful hint based on query type
        if any(w in q_lower for w in ("color", "colour", "size", "variant", "come in")):
            return (
                "I couldn't find variant information for that query in the catalog. "
                "Could you check the product name? For example: "
                "*\"What colors does the Smart Watch come in?\"*"
            )
        if any(w in q_lower for w in ("t-shirt", "tshirt", "shirt", "apparel", "hoodie",
                                       "jacket", "jogger", "tank")):
            return (
                "I searched the catalog but the query returned no results. "
                "Apparel products available include: **Premium T-Shirt**, **Polo Shirt**, "
                "**Minimal Hoodie**, **Denim Jacket**, **Joggers**, and **Tank Top**. "
                "Try asking about one of these specifically."
            )
        return (
            "I couldn't find a match for that in the catalog. "
            "Try rephrasing with a specific product name or collection — "
            "e.g. *\"What variants does the Minimal Hoodie have?\"*"
        )

    source = TOOL_LABELS.get(tool_name, "Store data")
    if tool_name == "combined":
        source = "Products, variants, and/or collections"

    hist_block = _format_history(history or [], max_turns=3)
    prompt = f"""{MARKDOWN_ANSWER_PROMPT}
{MARKDOWN_ANSWER_JUSTIFICATION_ADDENDUM}

Conversation (for context only):
{hist_block}

Customer question: {user_question}
Lookup area: {source}
{justification_ctx}

Raw catalog data (format this — counts and names must match exactly):
{raw}

Output ONLY the Markdown answer."""
    formatted = _llm_complete(llm, prompt, temperature=OLLAMA_TEMPERATURE_ANSWER)
    return formatted or raw


@dataclass
class PandasNLTool:
    """Natural-language → pandas code → raw database result (one DataFrame)."""
    name: str
    df: pd.DataFrame
    instruction: str
    llm: LLM
    verbose: bool = False

    def execute_raw(self, natural_language_query: str) -> str:
        """
        Generate pandas from instruction prompt, run on df, return raw output.

        Hardened:
        - exec errors are surfaced (not swallowed) so formatter never hallucinates
        - retries once with a simpler fallback prompt if generated code fails
        - always prints verbose info for both attempts
        - empty result after exec → explicit "no records" rather than silent empty
        """
        if self.df.empty:
            return f"No data in {self.name}."

        def _build_prompt(query: str, simplified: bool = False) -> str:
            cols = ", ".join(self.df.columns)
            extra = (
                "\nKEEP IT SIMPLE: use only df[mask] filtering and print(). "
                "No groupby, no apply, no lambda. Just filter + print.\n"
                if simplified else ""
            )
            return (
                f"{self.instruction}\n\n"
                f"User question: {query}\n\n"
                f"Write Python pandas code against `df` ({len(self.df)} rows).\n"
                f"Columns: {cols}\n"
                f"Use only `df` and `pd` (pandas already imported). No files or network.\n"
                f"{extra}"
                f"Output ONLY executable Python code."
            )

        def _exec_code(code: str) -> tuple[str, Exception | None]:
            """Returns (output_string, error_or_None)."""
            local: dict[str, Any] = {"df": self.df.copy(), "pd": pd, "answer": ""}
            buf = StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(code, {"__builtins__": _SAFE_BUILTINS}, local)  # noqa: S102
            except Exception as exc:
                return "", exc
            printed = buf.getvalue().strip()
            if printed:
                return printed, None
            # Check local variables set by code
            for key in ("answer", "result", "output"):
                val = local.get(key)
                if val is not None and str(val).strip() not in ("", "None"):
                    return str(val).strip(), None
            return "", None

        # ── Attempt 1: normal prompt ──────────────────────────────────────────
        raw_llm = _llm_complete(self.llm, _build_prompt(natural_language_query),
                                temperature=OLLAMA_TEMPERATURE_CODE)
        code = _extract_python_code(raw_llm)
        if self.verbose:
            print(f"\n[{self.name}] attempt-1 code:\n{code}\n")

        result, err1 = _exec_code(code)
        if result:
            if self.verbose:
                print(f"[{self.name}] attempt-1 result:\n{result}\n")
            return result

        if self.verbose:
            if err1:
                print(f"[{self.name}] attempt-1 EXEC ERROR: {err1}\n")
            else:
                print(f"[{self.name}] attempt-1 returned empty output — retrying\n")

        # ── Attempt 2: simplified fallback prompt ─────────────────────────────
        raw_llm2 = _llm_complete(self.llm,
                                 _build_prompt(natural_language_query, simplified=True),
                                 temperature=OLLAMA_TEMPERATURE_CODE)
        code2 = _extract_python_code(raw_llm2)
        if self.verbose:
            print(f"[{self.name}] attempt-2 code:\n{code2}\n")

        result2, err2 = _exec_code(code2)
        if result2:
            if self.verbose:
                print(f"[{self.name}] attempt-2 result:\n{result2}\n")
            return result2

        # ── Both attempts failed — return diagnostic, never empty ─────────────
        errors = "; ".join(str(e) for e in [err1, err2] if e)
        diag = (
            f"EXEC_FAILED tool={self.name} | "
            f"query={natural_language_query[:120]} | "
            f"errors=[{errors}] | "
            f"code1={code[:200]} | code2={code2[:200]}"
        )
        if self.verbose:
            print(f"[{self.name}] BOTH ATTEMPTS FAILED:\n{diag}\n")
        return f"No matching records found. (exec failed: {errors or 'empty output'})"


def _catalog_overview_text(
    products_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    collections_df: pd.DataFrame,
) -> str:
    """Structured plain text for the markdown formatter (not shown to users as-is)."""
    lines = [
        "LOOKUP: full store catalog",
        f"SUMMARY: {len(collections_df)} collections, "
        f"{len(products_df)} products, {len(variants_df)} variant rows.",
        "",
    ]
    for _, row in collections_df.iterrows():
        tags = ", ".join(str(row["collection_tags"]).split(" | "))
        products = ", ".join(str(row["product_names"]).split(" | "))
        lines.extend([
            f"COLLECTION: {row['collection_name']}",
            f"  product_count: {row['product_count']}",
            f"  tags: {tags}",
            f"  products: {products}",
            "",
        ])
    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════════════════
# REACT INTENT PLANNER — understand intent → plan query → justify answer
# ═══════════════════════════════════════════════════════════════════════════

REACT_INTENT_PLANNER_PROMPT = """\
You are an expert store assistant reasoning engine.
Your job is to deeply understand WHAT the user REALLY wants, then plan the
most accurate query execution strategy BEFORE touching any data.

══════════════════════════════════════════════════════════════
STEP 1 — UNDERSTAND USER INTENT (read carefully)
══════════════════════════════════════════════════════════════
Before planning, classify the user's intent:

A) DIRECT_LOOKUP   — user names a specific product/collection/tag
   e.g. "What colors does the Smart Watch come in?"

B) SEMANTIC_FILTER — user describes a property or use-case, not exact data values
   e.g. "Which products are available in beige?" → beige is NOT a direct DB value
        "Show me winter products" → 'winter' needs tag + attribute + semantic reasoning
        "What's good for an outdoor workout?" → needs attribute reasoning
        "Any gift ideas under a budget?" → needs tag + type reasoning

C) INFERENCE_QUERY — user asks for reasoning across multiple dimensions
   e.g. "Which subscription skincare products also come in sensitive?"
        "What's the most versatile product for travel?"

D) COMPARATIVE     — user wants comparison or ranking
   e.g. "Which collection has the most variants?"

E) AMBIGUOUS       — not enough info to answer accurately → MUST ASK for clarification

══════════════════════════════════════════════════════════════
STEP 2 — SEMANTIC EXPANSION RULES (critical for accuracy)
══════════════════════════════════════════════════════════════

COLOR / VARIANT VALUE QUERIES:
When user asks for a color/value like "beige":
  1. Check variant_values in variants_df for exact match (case-insensitive)
  2. Check product names/attributes for partial semantic match
  3. If no exact match → look for nearest variants (Cream, Tan, Ivory, Sand)
  4. ALWAYS report: "Found [N] products with [color] as variant value"
     and separately "These products have similar neutral tones: ..."

SEASON / USE-CASE QUERIES:
When user asks "winter collection" or "winter products":
  1. Check product_tags for 'winter' tag → Minimal Hoodie
  2. Check attributes for warmth indicators: 'warm', 'heavyweight', 'thermal',
     'cotton', 'fleece' → infer suitability
  3. Check collection_tags for 'seasonal' → Denim Jacket + Summer Collection
  4. Check product types: Apparel items with layering/warm attributes
  5. JUSTIFY each product: "Minimal Hoodie — tagged 'winter', warm + heavyweight"
  6. Report total: "Found X products. Of these, Y are genuinely winter-suitable
     based on: tags, attributes, and product type."

CONTEXTUAL / USE-CASE QUERIES (gym, travel, office, gifting, etc.):
  → Check product_tags, attributes, AND collection_tags
  → Report how each product qualifies (tag vs attribute vs collection)

══════════════════════════════════════════════════════════════
STEP 3 — QUERY PLAN OUTPUT FORMAT
══════════════════════════════════════════════════════════════
Output ONLY a JSON object with this exact structure:

{
  "intent_type": "DIRECT_LOOKUP|SEMANTIC_FILTER|INFERENCE_QUERY|COMPARATIVE|AMBIGUOUS",
  "intent_summary": "<one sentence: what user really wants>",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool|variants_tool|collections_tool|catalog_overview",
      "query": "<precise sub-query for this tool>",
      "reasoning": "<why this tool and this exact query>"
    }
  ],
  "semantic_expansion": "<null or description of any semantic expansion applied>",
  "justification_instruction": "<how to justify the final answer to the user>"
}

If needs_clarification is true: set clarification_question and leave steps as [].
══════════════════════════════════════════════════════════════
30 COMPLEX FEW-SHOT EXAMPLES (memorize all patterns)
══════════════════════════════════════════════════════════════

--- Example 1 ---
User: "Which products are available in beige?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products that have 'Beige' as a variant color value",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Find all products where variant_values contains 'Beige' (case-insensitive search across all variant types)",
      "reasoning": "Beige is a variant value — must search variant_values column, not product names"
    }
  ],
  "semantic_expansion": "Beige may appear as a variant color value. Also check for near-neutral tones like Tan, Ivory, Cream as secondary results.",
  "justification_instruction": "Report: found X products where 'Beige' appears as a variant value. List each product, its variant type (e.g. color), and confirm the value. If none found exactly, suggest closest neutral color variants available."
}

--- Example 2 ---
User: "Show me winter products"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products suitable for winter use — via tags, attributes, and warmth indicators",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find all products where product_tags contains 'winter' OR attributes contains any of: warm, heavyweight, cotton, fleece, thermal, layering. Also check collection_tags for seasonal.",
      "reasoning": "Winter suitability is multi-signal: explicit winter tag, warmth attributes, layering capability"
    }
  ],
  "semantic_expansion": "winter tag → Minimal Hoodie. Warmth attributes → Minimal Hoodie (warm, heavyweight), Denim Jacket (layering), Joggers (comfortable, elastic). Also Protein Powder for winter nutrition. Show ALL candidates with justification per product.",
  "justification_instruction": "Report: found X total products. Of these, Y are directly tagged winter, Z are inferred as winter-suitable via attributes. List each with reason."
}

--- Example 3 ---
User: "What's good for an outdoor workout?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products suited for outdoor exercise — checking fitness tags, breathable/lightweight attributes",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where attributes contains breathable OR lightweight OR gym_ready OR collection is Fitness. Also check tags for gym_essential, summer.",
      "reasoning": "Outdoor workout spans gear (Yoga Mat, Resistance Bands), apparel (Tank Top, Joggers), accessories (Water Bottle, Sneakers)"
    },
    {
      "tool": "variants_tool",
      "query": "Get variant options for the top outdoor workout products to show size/color availability",
      "reasoning": "User may want to buy, so showing options adds value"
    }
  ],
  "semantic_expansion": "outdoor workout = fitness gear + breathable apparel + hydration. Cross Fitness collection + Summer Collection + Apparel collection.",
  "justification_instruction": "Group results by use: Gear (Yoga Mat, Resistance Bands), Apparel (Tank Top, Joggers, Sneakers), Hydration (Water Bottle). Explain why each qualifies."
}

--- Example 4 ---
User: "Which skincare products work for sensitive skin?"
Output: {
  "intent_type": "INFERENCE_QUERY",
  "intent_summary": "Find skincare products with a Sensitive skin_type variant or gentle/fragrance_free attributes",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Find products where variant_type is skin_type and variant_values contains Sensitive",
      "reasoning": "skin_type variant directly captures sensitive skin suitability"
    },
    {
      "tool": "products_tool",
      "query": "Find skincare products (product_type=Skincare) where attributes contains fragrance_free OR gentle OR paraben_free",
      "reasoning": "Attributes like fragrance_free and gentle are strong indicators for sensitive skin"
    }
  ],
  "semantic_expansion": "Sensitive skin = skin_type variant 'Sensitive' + fragrance_free + gentle + paraben_free attributes",
  "justification_instruction": "Report two groups: 1) Products with explicit Sensitive skin_type variant, 2) Products with sensitive-skin-friendly attributes. Merge and deduplicate with reason for each."
}

--- Example 5 ---
User: "I want to buy a gift for someone who likes tech"
Output: {
  "intent_type": "INFERENCE_QUERY",
  "intent_summary": "Find gift-worthy tech/electronics products",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where collection_name is Electronics AND (product_tags contains gift_worthy OR premium)",
      "reasoning": "Gift for tech lover = Electronics collection filtered by gift-worthy or premium tags"
    }
  ],
  "semantic_expansion": "Tech gift = Electronics collection + gift_worthy/premium tags. Smart Watch, Wireless Earbuds are top candidates.",
  "justification_instruction": "Present as gift recommendations with why each qualifies: premium quality, tech category, gift-worthy tag."
}

--- Example 6 ---
User: "Which subscription products can I get for dry skin?"
Output: {
  "intent_type": "INFERENCE_QUERY",
  "intent_summary": "Find products that are subscription-eligible AND have Dry skin_type variant",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find all products where subscription_product is True",
      "reasoning": "First get all subscription products"
    },
    {
      "tool": "variants_tool",
      "query": "Find products where variant_type is skin_type and variant_values contains Dry",
      "reasoning": "Then filter those that have Dry skin_type variant — intersect with subscription list"
    }
  ],
  "semantic_expansion": "Subscription + Dry skin = Face Wash (Dry), Night Cream (Dry), Moisturizer (Dry) all have subscription=True",
  "justification_instruction": "Show intersection: X subscription products exist, Y of them have Dry skin_type variant. List each with its sizes and confirm subscription status."
}

--- Example 7 ---
User: "What eco-friendly options do I have?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products tagged eco_friendly OR with eco-friendly attributes",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where product_tags contains eco_friendly OR all_tags contains eco_friendly OR attributes contains eco_friendly",
      "reasoning": "eco_friendly appears in both product_tags and attributes — search both"
    }
  ],
  "semantic_expansion": "eco_friendly tag: Yoga Mat, Travel Tote Bag, Water Bottle. Attribute eco_friendly: Yoga Mat, Travel Tote Bag. Show all with source of eco claim.",
  "justification_instruction": "For each product, clarify: eco_friendly as product tag, as attribute, or both. Report total found."
}

--- Example 8 ---
User: "Show me everything vegan"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find all vegan products — via attributes and product tags",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where attributes contains vegan OR product_tags contains vegan",
      "reasoning": "vegan is listed as attribute on Vitamin C Serum, Moisturizer, Lipstick Set, Makeup Brush Set"
    }
  ],
  "semantic_expansion": "vegan attribute products: Vitamin C Serum, Moisturizer, Lipstick Set, Makeup Brush Set. Also check cruelty_free as related.",
  "justification_instruction": "List each vegan product with its category and which field confirms vegan status (attribute vs tag)."
}

--- Example 9 ---
User: "I need something for frizzy hair"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products targeting frizzy hair — via hair_type variant or frizz_control attribute",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Find products where variant_type is hair_type and variant_values contains Frizzy",
      "reasoning": "Hair Serum has hair_type variant including Frizzy"
    },
    {
      "tool": "products_tool",
      "query": "Find products where attributes contains frizz_control",
      "reasoning": "frizz_control attribute on Hair Serum directly targets this need"
    }
  ],
  "semantic_expansion": "Frizzy hair → Hair Serum (hair_type: Frizzy, attribute: frizz_control, heat_protection)",
  "justification_instruction": "Report products found via hair_type variant AND frizz_control attribute. Merge, deduplicate, explain why each works for frizzy hair."
}

--- Example 10 ---
User: "Which products come in the most color options?"
Output: {
  "intent_type": "COMPARATIVE",
  "intent_summary": "Rank products by number of color variant options",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Filter rows where variant_type is color, then sort by value_count descending to find products with most color options",
      "reasoning": "value_count column in variants_df holds the count of options per variant dimension"
    }
  ],
  "semantic_expansion": null,
  "justification_instruction": "Show top 5 products by color count. Report as ranked list: #1 Product — N colors: value1, value2, ..."
}

--- Example 11 ---
User: "Are there any products for office use?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products tagged office_wear or suitable for professional use",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where product_tags contains office_wear OR attributes contains formal_casual",
      "reasoning": "office_wear tag: Polo Shirt, Laptop Sleeve, Leather Wallet. formal_casual attribute: Polo Shirt"
    }
  ],
  "semantic_expansion": "office = office_wear tag (Polo Shirt, Laptop Sleeve, Leather Wallet) + formal_casual attribute",
  "justification_instruction": "Group by use: Apparel (Polo Shirt), Accessories (Laptop Sleeve, Leather Wallet). Note why each is office-suitable."
}

--- Example 12 ---
User: "What gaming setup products do you have?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find gaming-tagged or gaming-relevant products",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where product_tags contains gaming OR all_tags contains gaming",
      "reasoning": "gaming tag: Gaming Mouse, Mechanical Keyboard"
    }
  ],
  "semantic_expansion": "gaming tag products: Gaming Mouse, Mechanical Keyboard. Also check Wireless Earbuds (anc, audio) as gaming-adjacent.",
  "justification_instruction": "List gaming-tagged products with their variants (DPI options for mouse, switch types for keyboard). Mention gaming-adjacent products."
}

--- Example 13 ---
User: "Which collections are good for wellness?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find collections tagged or associated with wellness",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "collections_tool",
      "query": "Find collections where collection_tags contains wellness",
      "reasoning": "wellness is a collection-level tag on Skincare and Fitness collections"
    }
  ],
  "semantic_expansion": "wellness collection_tag: Skincare (6 products) + Fitness (5 products) = 11 total products",
  "justification_instruction": "Show 2 wellness collections with their product lists and explain what makes each wellness-oriented."
}

--- Example 14 ---
User: "I want a protein shake — what flavors do you have?"
Output: {
  "intent_type": "DIRECT_LOOKUP",
  "intent_summary": "Get flavor variants for Protein Powder",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Get all variant options for Protein Powder, especially the flavor variant_type",
      "reasoning": "Protein Powder has flavor variant: Chocolate, Vanilla, Strawberry, Unflavored"
    }
  ],
  "semantic_expansion": null,
  "justification_instruction": "List all Protein Powder flavors clearly. Also mention available sizes (500g, 1kg, 2kg) and subscription availability."
}

--- Example 15 ---
User: "Do you have anything waterproof for the beach?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find waterproof products suitable for beach/summer use",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where attributes contains waterproof AND (all_tags contains summer OR collection_name = Summer Collection)",
      "reasoning": "waterproof + summer tag → beach-relevant products"
    }
  ],
  "semantic_expansion": "waterproof attribute: Wireless Earbuds, Smart Watch, Bluetooth Speaker. Summer tagged: Sneakers, Premium T-Shirt, Sunglasses. Intersection and union both useful.",
  "justification_instruction": "Group: waterproof electronics (Earbuds, Smart Watch, Speaker), summer apparel/accessories (Sunglasses, Sneakers). Explain beach suitability of each."
}

--- Example 16 ---
User: "Which products have the most variants overall?"
Output: {
  "intent_type": "COMPARATIVE",
  "intent_summary": "Rank products by total number of variant dimensions",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Group by product_name, count distinct variant_type per product, sort descending",
      "reasoning": "Count of variant dimensions (color+size+fit = 3) per product from variants_df"
    }
  ],
  "semantic_expansion": null,
  "justification_instruction": "Show top 5 most variant-rich products with count and list of variant types."
}

--- Example 17 ---
User: "I'm looking for anti-aging skincare"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find skincare products with anti-ageing attributes or tags",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where (product_type = Skincare OR collection_name = Skincare) AND (attributes contains anti_ageing OR product_tags contains anti_ageing)",
      "reasoning": "anti_ageing is an attribute on Night Cream, Eye Cream; also a product_tag on Eye Cream"
    }
  ],
  "semantic_expansion": "anti_ageing attribute: Night Cream (retinol), Eye Cream (caffeine, depuffing). Vitamin C Serum (brightening) is anti-ageing adjacent.",
  "justification_instruction": "List anti-ageing products with specific anti-ageing ingredients/attributes. Explain why each qualifies."
}

--- Example 18 ---
User: "What's available in Navy?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find all products that have Navy as a variant color value",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Find all products where variant_type is color (or frame_color) AND variant_values contains Navy",
      "reasoning": "Navy is a color variant value — search variants_df variant_values"
    }
  ],
  "semantic_expansion": "Navy appears in: Polo Shirt, Sneakers, Wireless Earbuds, Cap, Travel Tote Bag, Laptop Sleeve, Minimal Hoodie, Joggers, Tank Top",
  "justification_instruction": "List each product available in Navy with its full color range for comparison. Report total count."
}

--- Example 19 ---
User: "Give me travel-friendly products"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products tagged travel_essential or with portable/compact/lightweight attributes",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where product_tags contains travel_essential OR attributes contains portable OR attributes contains compact OR attributes contains lightweight",
      "reasoning": "travel = explicit travel_essential tag + portable/compact/lightweight attributes"
    }
  ],
  "semantic_expansion": "travel_essential tag: Portable Charger, Travel Tote Bag. portable attribute: Resistance Bands, Foam Roller, Bluetooth Speaker. compact: Portable Charger.",
  "justification_instruction": "Group by why travel-friendly: tagged vs attribute. Report total. Suggest best combo (e.g. Tote Bag + Portable Charger + Earbuds)."
}

--- Example 20 ---
User: "Show me premium products under the beauty category"
Output: {
  "intent_type": "INFERENCE_QUERY",
  "intent_summary": "Find premium-tagged products in the Beauty collection",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where collection_name = Beauty AND product_tags contains premium",
      "reasoning": "Beauty collection + premium tag intersection"
    }
  ],
  "semantic_expansion": "Beauty collection: Luxury Perfume, Lipstick Set, Makeup Brush Set, Hair Serum. All 4 have premium tag.",
  "justification_instruction": "All 4 Beauty products are premium — list them with their variants and what makes each premium."
}

--- Example 21 ---
User: "Do you have anything for a beginner at the gym?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find beginner-friendly fitness products via attributes and tags",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where attributes contains beginner_friendly OR collection_name = Fitness AND product_tags contains budget_friendly",
      "reasoning": "beginner_friendly attribute: Resistance Bands. Budget-friendly fitness = good beginner entry point"
    }
  ],
  "semantic_expansion": "beginner_friendly attribute: Resistance Bands. Budget-friendly Fitness: Gym Gloves, Resistance Bands. Yoga Mat is also beginner-accessible.",
  "justification_instruction": "Show beginner fitness products with justification: explicitly beginner_friendly attribute, budget-friendly, and low barrier to start."
}

--- Example 22 ---
User: "What sizes does the Minimal Hoodie come in and what colors?"
Output: {
  "intent_type": "DIRECT_LOOKUP",
  "intent_summary": "Get size and color variants for Minimal Hoodie",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Get all variant rows for product_name = Minimal Hoodie, specifically size and color variant types",
      "reasoning": "Direct product variant lookup — Minimal Hoodie has color (5), size (6), material (2)"
    }
  ],
  "semantic_expansion": null,
  "justification_instruction": "List sizes: XS, S, M, L, XL, XXL. Colors: Black, Grey, White, Olive, Navy. Materials: Cotton Fleece, French Terry. Clear formatted response."
}

--- Example 23 ---
User: "I want something brightening for my skin"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find skincare products with brightening attributes",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where attributes contains brightening OR (product_type = Skincare AND attributes contains vitamin_c)",
      "reasoning": "brightening attribute on Vitamin C Serum; vitamin_c is the active ingredient for brightening"
    }
  ],
  "semantic_expansion": "brightening = Vitamin C Serum (brightening, anti_ageing, vitamin_c). Also Sunscreen SPF50 protects against darkening.",
  "justification_instruction": "Recommend Vitamin C Serum as primary brightening product. Mention concentrations available (10%, 15%, 20%). Secondary: Sunscreen for prevention."
}

--- Example 24 ---
User: "Which products are both trending and premium?"
Output: {
  "intent_type": "INFERENCE_QUERY",
  "intent_summary": "Find products tagged both trending AND premium",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find products where product_tags contains trending AND product_tags contains premium",
      "reasoning": "Both tags must be present in product_tags simultaneously"
    }
  ],
  "semantic_expansion": "trending ∩ premium: Vitamin C Serum, Foam Roller, Wireless Earbuds, Smart Watch, Minimal Hoodie, Denim Jacket, Luxury Perfume, Lipstick Set, Hair Serum",
  "justification_instruction": "List all trending+premium products grouped by collection. Show count first."
}

--- Example 25 ---
User: "What collections exist and how many products are in each?"
Output: {
  "intent_type": "DIRECT_LOOKUP",
  "intent_summary": "Get all collections with their product counts",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "collections_tool",
      "query": "List all collections with collection_name and product_count, sorted by product_count descending",
      "reasoning": "collections_df has exactly this data"
    }
  ],
  "semantic_expansion": null,
  "justification_instruction": "Show ranked table of all 8 collections by product count. Total = 37 products."
}

--- Example 26 ---
User: "Any blue products available?"
Output: {
  "intent_type": "SEMANTIC_FILTER",
  "intent_summary": "Find products with Blue as a variant color value",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Find all products where variant_values contains 'Blue' (search all variant types including color, frame_color)",
      "reasoning": "Blue appears as a color variant value across multiple products"
    }
  ],
  "semantic_expansion": "Blue in variant_values: Premium T-Shirt, Water Bottle, Yoga Mat, Gym Gloves, Wireless Earbuds, Bluetooth Speaker, Portable Charger, Resistance Bands (Fabric?), Mechanical Keyboard",
  "justification_instruction": "List all products available in Blue, grouped by category. Show which variant dimension has Blue (color vs frame_color etc)."
}

--- Example 27 ---
User: "Which products can I subscribe to and get in bulk?"
Output: {
  "intent_type": "INFERENCE_QUERY",
  "intent_summary": "Find subscription products that also come in large/bulk size variants",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find all products where subscription_product is True",
      "reasoning": "Get subscription products first"
    },
    {
      "tool": "variants_tool",
      "query": "For subscription products, find those with size variants that include large options like 2kg, 200ml, 150ml, 100ml",
      "reasoning": "Bulk = large size variants. Protein Powder (2kg), Moisturizer (150ml), Face Wash (200ml) qualify"
    }
  ],
  "semantic_expansion": "subscription=True: 8 products. Bulk sizes available: Protein Powder (2kg), Moisturizer (150ml), Face Wash (200ml), Night Cream (100ml).",
  "justification_instruction": "Show subscription products, then highlight those with large size options. Note: Protein Powder is the best bulk-subscription option."
}

--- Example 28 ---
User: "I'm building a morning skincare routine — what do I need?"
Output: {
  "intent_type": "INFERENCE_QUERY",
  "intent_summary": "Recommend a complete morning skincare routine from available products",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "products_tool",
      "query": "Find all Skincare products tagged daily_use, and include Sunscreen SPF50 specifically. Get their attributes.",
      "reasoning": "Morning routine = cleanser + moisturizer + SPF. daily_use tag identifies AM/PM products."
    }
  ],
  "semantic_expansion": "Morning routine steps: 1) Cleanse → Face Wash (daily_use). 2) Treat → Vitamin C Serum (brightening). 3) Moisturize → Moisturizer (daily_use). 4) Protect → Sunscreen SPF50 (daily_use, water_resistant). Night Cream is PM only.",
  "justification_instruction": "Present as a step-by-step morning routine. For each step, name the product, explain why it fits (attributes/tags), and show size options."
}

--- Example 29 ---
User: "What's the difference between the gift items?"
Output: {
  "intent_type": "COMPARATIVE",
  "intent_summary": "Compare all products in the Gift Items collection",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "collections_tool",
      "query": "Get products in Gift Items collection",
      "reasoning": "Get the 3 gift products first"
    },
    {
      "tool": "products_tool",
      "query": "Get attributes, tags, and subscription status for Sample Gift, Mystery Gift Box, Welcome Gift",
      "reasoning": "Need full product details for comparison"
    }
  ],
  "semantic_expansion": null,
  "justification_instruction": "Side-by-side comparison: Sample Gift (trial, budget, new customer), Mystery Gift Box (premium, subscription, surprise), Welcome Gift (onboarding, category choice). Highlight when each is appropriate."
}

--- Example 30 ---
User: "How many products have more than 3 color choices?"
Output: {
  "intent_type": "COMPARATIVE",
  "intent_summary": "Count products with 4+ color variant options",
  "needs_clarification": false,
  "clarification_question": null,
  "steps": [
    {
      "tool": "variants_tool",
      "query": "Filter rows where variant_type is 'color' and value_count > 3, then count and list product_names",
      "reasoning": "value_count column holds exact count of options per variant type"
    }
  ],
  "semantic_expansion": "color variant_type + value_count > 3: Premium T-Shirt (5), Cap (5), Minimal Hoodie (5), Joggers (4), Tank Top (5), Smart Watch (4), Bluetooth Speaker (4), Water Bottle (4)",
  "justification_instruction": "Report total count found, then list each product with exact color count and the color names."
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLARIFICATION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

CLARIFICATION_CASES = {
    "color_not_found": (
        "I couldn't find any products with '{value}' as an exact variant value. "
        "Did you mean one of these similar options: {alternatives}? "
        "Or were you looking for something else?"
    ),
    "ambiguous_category": (
        "Could you clarify — are you looking for products in the **{a}** category "
        "or the **{b}** category? Both have relevant items."
    ),
    "use_case_unclear": (
        "To give you the best recommendations, could you tell me more about "
        "the occasion or use case? For example: everyday use, gifting, fitness, travel?"
    ),
}


def _parse_react_plan(raw: str) -> dict:
    """Parse the ReAct planner JSON output safely."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _run_react_planner(
    llm: "Ollama",
    query: str,
    history: list[dict[str, str]],
    verbose: bool = False,
) -> dict | None:
    """
    Run the ReAct intent planner. Returns parsed plan dict or None on failure.
    The plan contains intent_type, steps with enriched sub-queries, semantic_expansion,
    needs_clarification, and justification_instruction.
    """
    hist_block = _format_history(history, max_turns=4)
    prompt = (
        f"{REACT_INTENT_PLANNER_PROMPT}\n\n"
        f"Conversation history:\n{hist_block}\n\n"
        f"Current user query: {query}\n\n"
        "Output ONLY the JSON plan — no other text."
    )
    raw = _llm_complete(llm, prompt, temperature=OLLAMA_TEMPERATURE_CODE)
    if verbose:
        print(f"\n[react_planner] raw:\n{raw}\n")
    try:
        plan = _parse_react_plan(raw)
        if verbose:
            print(f"[react_planner] intent={plan.get('intent_type')} "
                  f"steps={len(plan.get('steps', []))} "
                  f"clarify={plan.get('needs_clarification')}\n")
        return plan
    except (json.JSONDecodeError, ValueError) as exc:
        if verbose:
            print(f"[react_planner] parse failed: {exc}\n")
        return None


QUERY_ROUTER_PROMPT = """\
You route user questions to the correct store database tool(s).
Use the conversation history for follow-ups like "which are those?".

""" + AGENT_SYSTEM_PROMPT + """

══════════════════════════════════════════════════════════════
ROUTING RULES (critical)
══════════════════════════════════════════════════════════════
• Follow-ups ("which are those?", "name them", "list them"): resolve from history.
  Never use catalog_overview for follow-ups. Never dump the full store.
• ANY tag query ("tagged X", "what about X", "X products", "wellness", "gym",
  "seasonal", "luxury", "tech", etc.) → products_tool ONLY.
  Sub-query MUST say: "find products where all_tags contains X".
  products_df has an all_tags column that combines product_tags + collection_tags.
  NEVER say the sub-query is "product_tags contains" for collection-level tags.
• "collection tagged X" / "which collections have tag X" → collections_tool ONLY.
• catalog_overview ONLY for: "what do you sell", "full catalog", "list everything",
  "store overview" — never for tag/subscription/attribute questions.
• Variant/color/size/flavor/options → variants_tool.
• Collection contents ("what is in X collection") → collections_tool.

TAG ROUTING QUICK REFERENCE:
  wellness / gym / health / tech / gadgets / skincare as tag
    → products_tool, sub-query: "find products where all_tags contains wellness"
  seasonal as tag (no "collection" in question)
    → products_tool, sub-query: "find products where all_tags contains seasonal"
  luxury → products_tool, sub-query: "find products where all_tags contains luxury"
  winter  → products_tool, sub-query: "find products where all_tags contains winter"
  best_seller / premium / eco_friendly / vegan / gaming / gift_worthy
    → products_tool, sub-query uses all_tags or product_tags (either works)

══════════════════════════════════════════════════════════════
ROUTER OUTPUT (follow exactly)
══════════════════════════════════════════════════════════════
Reply with ONLY a JSON object — no markdown, no prose:
{"steps": [{"tool": "products_tool"|"variants_tool"|"collections_tool"|"catalog_overview", "query": "<sub-question>"}]}

Rules:
- Use the minimum steps (usually 1).
- Use 2+ steps only when CHAINING RULES above require it.
- For catalog_overview set query to "".
- Each sub-query must be fully self-contained (include tag/collection name from history if follow-up).
"""


def _format_history(history: list[dict[str, str]], max_turns: int = 6) -> str:
    """Recent chat turns for router / follow-up resolution."""
    if not history:
        return "(no prior messages)"
    recent = history[-max_turns * 2 :]
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:800]}")
    return "\n".join(lines) or "(no prior messages)"


_FOLLOWUP_PHRASES = (
    "which are those", "what are those", "which ones", "what ones",
    "name them", "list them", "show them", "what are they", "which are they",
    "those?", "them?", "and those", "what are the",
)


def _is_follow_up(query: str) -> bool:
    ql = query.lower().strip().rstrip("?").strip()
    if len(ql.split()) > 8:
        return False
    return any(p.rstrip("?") in ql or ql.startswith(p.rstrip("?")) for p in _FOLLOWUP_PHRASES)


def _resolve_follow_up_heuristic(query: str, history: list[dict[str, str]]) -> str | None:
    """Fast follow-up rewrite without LLM when topic is obvious."""
    if not history or not _is_follow_up(query):
        return None
    context = " ".join(
        (m.get("content") or "").lower() for m in history[-6:]
    )
    # collection-level tags — must use all_tags
    if "wellness" in context:
        return (
            "Find all products where all_tags contains 'wellness'. "
            "List product_name, collection_name, and product_type."
        )
    if "gym" in context and "gym_essential" not in context:
        return (
            "Find all products where all_tags contains 'gym'. "
            "List product_name, collection_name, and product_type."
        )
    if "health" in context and "collection" not in context:
        return (
            "Find all products where all_tags contains 'health'. "
            "List product_name, collection_name, and product_type."
        )
    if "tech" in context or "gadgets" in context:
        return (
            "Find all products where all_tags contains 'tech'. "
            "List product_name, collection_name, and product_type."
        )
    # product-level tags
    if "seasonal" in context:
        if "collection" in context and "product tag" not in context:
            return "Which collections have seasonal in collection_tags?"
        return (
            "Find all products where all_tags contains 'seasonal'. "
            "List product_name, collection_name, and product_type."
        )
    if "winter" in context:
        return (
            "List every product where product_tags contains winter, "
            "with product_name, collection_name, and product_type"
        )
    if "luxury" in context:
        return (
            "Find all products where all_tags contains 'luxury'. "
            "List product_name, collection_name, and product_type."
        )
    if "subscription" in context:
        return "List all products where subscription_product is True"
    if "premium" in context and "tag" in context:
        return "List all products where product_tags contains premium"
    if "eco" in context:
        return (
            "List all products where product_tags contains eco_friendly "
            "or attributes contains eco_friendly"
        )
    if "vegan" in context:
        return "List all products where attributes contains vegan"
    return None


FOLLOWUP_RESOLVE_PROMPT = """\
Rewrite the follow-up into ONE standalone store catalog question using all_tags.

TAG LEVELS — critical:
- all_tags column = product_tags UNION collection_tags (USE THIS by default)
- wellness/gym/health/tech/gadgets/cosmetics/lifestyle/basics/clothing/gifting
    → collection-level tags only → search all_tags, NOT product_tags
    → wellness returns 11 products (Skincare 6 + Fitness 5)
- seasonal → product_tag: Denim Jacket (1); collection_tag: Summer Collection (5)
    → use all_tags → 6 products total
- luxury / winter → product-level only → all_tags or product_tags both work
- NEVER say "0 products match" for collection-level tags

Conversation:
{history}

Follow-up: {query}

Output ONLY the rewritten question using "all_tags contains X" phrasing — no explanation."""


def _normalize_tag_query(query: str) -> str:
    """
    Rewrite ambiguous tag queries to use all_tags so collection-inherited
    tags (wellness, gym, health, tech, seasonal, etc.) are always found.
    """
    ql = query.lower()

    # Collection-only tags that users ask about as if they were product tags.
    # These must be searched in all_tags, not product_tags.
    _COLLECTION_TAGS = {
        "wellness": "wellness",
        "gym": "gym",
        "health": "health",
        "tech": "tech",
        "gadgets": "gadgets",
        "cosmetics": "cosmetics",
        "makeup": "makeup",
        "lifestyle": "lifestyle",
        "basics": "basics",
        "clothing": "clothing",
        "gifting": "gifting",
        "occasion": "occasion",
    }

    for keyword, canonical in _COLLECTION_TAGS.items():
        if keyword in ql and "collection" not in ql:
            return (
                f"Find all products where all_tags contains '{canonical}'. "
                f"List each with product_name, collection_name, and product_type."
            )

    # Seasonal: exists at BOTH levels — use all_tags for comprehensive result
    if "seasonal" in ql and "collection" not in ql:
        return (
            "Find all products where all_tags contains 'seasonal'. "
            "Show product_name, collection_name, product_type, and note whether "
            "'seasonal' is in product_tags or only in collection_tags."
        )

    # Winter: product-level only
    if "winter" in ql and "collection" not in ql:
        return (
            "List all products where product_tags contains winter, "
            "with product_name, collection_name, and product_type."
        )

    # Luxury: product-level only
    if "luxury" in ql and "collection" not in ql and "perfume" not in ql:
        return (
            "Find all products where all_tags contains 'luxury'. "
            "List each with product_name, collection_name, and product_type."
        )

    return query


def _resolve_query(
    llm: Ollama,
    query: str,
    history: list[dict[str, str]],
    verbose: bool = False,
) -> str:
    """Turn vague follow-ups into a concrete catalog question."""
    query = _normalize_tag_query(query)
    heuristic = _resolve_follow_up_heuristic(query, history)
    if heuristic:
        if verbose:
            print(f"[follow-up] heuristic → {heuristic}\n")
        return heuristic
    if not history or not _is_follow_up(query):
        return query
    prompt = FOLLOWUP_RESOLVE_PROMPT.format(
        history=_format_history(history),
        query=query,
    )
    rewritten = _llm_complete(llm, prompt, temperature=OLLAMA_TEMPERATURE_CODE)
    if verbose and rewritten and rewritten != query:
        print(f"[follow-up] LLM → {rewritten}\n")
    return rewritten.strip() or query


def _parse_router_json(raw: str) -> list[tuple[ToolName, str]]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    steps = data.get("steps") or data.get("tools") or []
    if not isinstance(steps, list) or not steps:
        raise ValueError("router returned no steps")
    out: list[tuple[ToolName, str]] = []
    valid: set[str] = {
        "products_tool", "variants_tool", "collections_tool", "catalog_overview",
    }
    for step in steps:
        if isinstance(step, str):
            tool, sub_q = step, ""
        else:
            tool = step.get("tool") or step.get("name") or ""
            sub_q = step.get("query") or step.get("question") or ""
        if tool not in valid:
            raise ValueError(f"unknown tool: {tool}")
        out.append((tool, str(sub_q).strip()))  # type: ignore[arg-type]
    return out


def _route_query_heuristic(query: str) -> list[tuple[ToolName, str]]:
    """Fallback when router JSON fails."""
    q = query.lower()
    if _is_follow_up(q):
        # Never full catalog on a follow-up
        if "collection" in q and "tag" in q:
            return [("collections_tool", query)]
        return [("products_tool", query)]
    if any(
        p in q
        for p in (
            "what do you sell", "full catalog", "entire store", "all collections",
            "list everything", "store overview",
        )
    ) and not any(t in q for t in ("seasonal", "winter", "tag", "premium", "vegan")):
        return [("catalog_overview", "")]
    tag_kw = (
        # product-level tags
        "seasonal", "winter", "tag", "premium", "vegan", "eco", "subscription",
        "best_seller", "trending", "attribute", "product_tags", "luxury",
        "gaming", "gift_worthy", "office_wear", "budget_friendly", "anti_ageing",
        # collection-level tags (inherited by products via all_tags)
        "wellness", "gym", "health", "tech", "gadgets", "skincare",
        "cosmetics", "makeup", "lifestyle", "fashion", "basics", "clothing",
        "gifts", "gifting", "occasion", "accessories",
        # combined / general tag searches
        "all_tags", "collection_tags",
    )
    variant_kw = (
        "color", "size", "flavor", "variant", "dpi", "capacity", "strap",
        "shade", "scent", "come in", "options does", "available in",
    )
    collection_kw = (
        "collection", "in the electronics", "in skincare", "in fitness",
        "in beauty", "in apparel", "gift items", "how many collections",
    )
    steps: list[tuple[ToolName, str]] = []
    if any(k in q for k in tag_kw):
        steps.append(("products_tool", query))
    if any(k in q for k in variant_kw):
        steps.append(("variants_tool", query))
    if any(k in q for k in collection_kw):
        steps.append(("collections_tool", query))
    if not steps:
        if any(k in q for k in variant_kw):
            steps = [("variants_tool", query)]
        elif any(k in q for k in collection_kw):
            steps = [("collections_tool", query)]
        else:
            steps = [("products_tool", query)]
    return steps


# ═══════════════════════════════════════════════════════════════════════════
# Query session (direct — no ReAct)
# ═══════════════════════════════════════════════════════════════════════════

def build_ollama_llm(
    model: str | None = None,
    request_timeout: float = 300.0,
) -> Ollama:
    """Ollama Cloud LLM with authenticated sync + async clients."""
    client, async_client = get_ollama_clients(request_timeout)
    return Ollama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_CLOUD_HOST,
        request_timeout=request_timeout,
        client=client,
        async_client=async_client,
        is_function_calling_model=False,
        temperature=OLLAMA_TEMPERATURE_ANSWER,
    )


def _refresh_ollama_clients(llm: Ollama, request_timeout: float = 300.0) -> None:
    """New httpx clients — required after asyncio.run() in another thread (Streamlit)."""
    client, async_client = get_ollama_clients(request_timeout)
    llm._client = client
    llm._async_client = async_client


@dataclass
class RagSession:
    """Direct query session: route → pandas execute → bulleted answer."""
    llm: Ollama
    products: PandasNLTool
    variants: PandasNLTool
    collections: PandasNLTool
    catalog_overview: str
    verbose: bool = False

    def _route(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[tuple[ToolName, str]]:
        history = history or []
        hist_block = _format_history(history)
        prompt = (
            f"{QUERY_ROUTER_PROMPT}\n\n"
            f"Conversation history:\n{hist_block}\n\n"
            f"Current question: {query}\n"
        )
        raw = _llm_complete(self.llm, prompt, temperature=OLLAMA_TEMPERATURE_CODE)
        if self.verbose:
            print(f"\n[router] raw:\n{raw}\n")
        try:
            steps = _parse_router_json(raw)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            if self.verbose:
                print(f"[router] parse failed ({exc}), using heuristics\n")
            steps = _route_query_heuristic(query)
        # Block full-catalog dump on tag/follow-up questions
        if len(steps) == 1 and steps[0][0] == "catalog_overview":
            ql = query.lower()
            if _is_follow_up(query) or any(
                t in ql for t in ("seasonal", "winter", "tag", "which are", "those")
            ):
                steps = [("products_tool", query)]
        if self.verbose:
            print(f"[router] steps: {steps}\n")
        return steps

    def _direct_variant_lookup(self, product_name: str) -> str | None:
        """
        Pure-Python fallback: directly query variants_df for a product without
        relying on LLM-generated pandas code.  Called when the pandas exec fails.
        Returns formatted string or None if product not found.
        """
        df = self.variants.df
        mask = df["product_name"].str.lower() == product_name.lower()
        rows = df[mask]
        if rows.empty:
            # Try partial match
            mask2 = df["product_name"].str.lower().str.contains(
                product_name.lower(), regex=False
            )
            rows = df[mask2]
        if rows.empty:
            return None
        lines = [f"Variants for {rows.iloc[0]['product_name']}:"]
        for _, row in rows.iterrows():
            vals = str(row["variant_values"]).replace(" | ", ", ")
            lines.append(f"  {row['variant_type']}: {vals}  ({row['value_count']} options)")
        return "\n".join(lines)

    def _direct_product_lookup(self, product_name: str) -> str | None:
        """Pure-Python fallback: look up a product in products_df by name."""
        df = self.products.df
        mask = df["product_name"].str.lower() == product_name.lower()
        rows = df[mask]
        if rows.empty:
            mask2 = df["product_name"].str.lower().str.contains(
                product_name.lower(), regex=False
            )
            rows = df[mask2]
        if rows.empty:
            return None
        lines: list[str] = []
        for _, row in rows.iterrows():
            lines.append(
                f"product_name: {row['product_name']}\n"
                f"  collection: {row['collection_name']}\n"
                f"  type: {row['product_type']}\n"
                f"  subscription: {row['subscription_product']}\n"
                f"  attributes: {row['attributes']}\n"
                f"  product_tags: {row['product_tags']}\n"
                f"  variant_types: {row['variant_types']}"
            )
        return "\n".join(lines)

    def _run_tool(self, tool: ToolName, sub_query: str) -> str:
        if tool == "catalog_overview":
            return self.catalog_overview
        if tool == "products_tool":
            result = self.products.execute_raw(sub_query or "list all products")
            # Fallback: if exec failed and query looks like a product name lookup,
            # use direct Python lookup
            if "EXEC_FAILED" in result or "exec failed" in result.lower():
                fallback = self._direct_product_lookup(sub_query)
                if fallback:
                    if self.verbose:
                        print(f"[_run_tool] products fallback used for: {sub_query[:80]}\n")
                    return fallback
            return result
        if tool == "variants_tool":
            result = self.variants.execute_raw(sub_query)
            # Fallback: extract product name from query and do direct lookup
            if "EXEC_FAILED" in result or "exec failed" in result.lower() or \
               result.strip() == "No matching records found.":
                # Try to extract a product name from the sub_query
                for pname in self.variants.df["product_name"].unique():
                    if pname.lower() in sub_query.lower():
                        fallback = self._direct_variant_lookup(pname)
                        if fallback:
                            if self.verbose:
                                print(f"[_run_tool] variants fallback: {pname}\n")
                            return fallback
            return result
        if tool == "collections_tool":
            return self.collections.execute_raw(sub_query)
        return "No matching records found."

    def chat(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        _refresh_ollama_clients(self.llm)
        history = history or []

        # ── PHASE 1: Resolve follow-ups & normalize tag queries ──────────────
        resolved = _resolve_query(self.llm, query, history, verbose=self.verbose)

        # ── PHASE 2: ReAct Intent Planner ────────────────────────────────────
        plan = _run_react_planner(self.llm, resolved, history, verbose=self.verbose)

        # ── PHASE 3: Clarification gate ──────────────────────────────────────
        if plan and plan.get("needs_clarification") and plan.get("clarification_question"):
            return plan["clarification_question"]

        # ── PHASE 4: Extract steps — use plan if valid, else fall back ────────
        steps: list[tuple[ToolName, str]] = []
        if plan and plan.get("steps"):
            try:
                for step in plan["steps"]:
                    tool = step.get("tool", "")
                    sub_q = step.get("query", "")
                    valid_tools: set[str] = {
                        "products_tool", "variants_tool",
                        "collections_tool", "catalog_overview",
                    }
                    if tool in valid_tools:
                        steps.append((tool, str(sub_q).strip()))  # type: ignore[arg-type]
                if not steps:
                    raise ValueError("plan had no valid steps")
            except Exception as exc:
                if self.verbose:
                    print(f"[react->fallback] ({exc}), using router\n")
                steps = self._route(resolved, history)
        else:
            steps = self._route(resolved, history)

        # ── PHASE 5: Execute tools ────────────────────────────────────────────
        chunks: list[tuple[str, str]] = []
        for tool_name, sub_query in steps:
            chunks.append((tool_name, self._run_tool(tool_name, sub_query)))

        # ── PHASE 6: Build justification context from plan ────────────────────
        justification_ctx = ""
        if plan:
            semantic_exp = plan.get("semantic_expansion") or ""
            just_instr = plan.get("justification_instruction") or ""
            intent_sum = plan.get("intent_summary") or ""
            if semantic_exp or just_instr:
                justification_ctx = (
                    f"\n\nINTENT: {intent_sum}"
                    f"\nSEMANTIC EXPANSION: {semantic_exp}"
                    f"\nJUSTIFICATION INSTRUCTION: {just_instr}"
                )

        # ── PHASE 7: Format final answer with justification ───────────────────
        if len(chunks) == 1:
            name, raw = chunks[0]
            return _format_markdown_answer_with_justification(
                self.llm, query, raw, name,
                history=history,
                justification_ctx=justification_ctx,
            )

        combined = "\n\n".join(
            f"--- {TOOL_LABELS.get(name, name)} ---\n{raw}"
            for name, raw in chunks
        )
        return _format_markdown_answer_with_justification(
            self.llm, query, combined, "combined",
            history=history,
            justification_ctx=justification_ctx,
        )


def build_agent(
    products_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    collections_df: pd.DataFrame,
    model: str | None = None,
    verbose: bool = False,
) -> RagSession:
    llm = build_ollama_llm(model=model)
    Settings.llm = llm
    return RagSession(
        llm=llm,
        products=PandasNLTool(
            name="products_tool",
            df=products_df,
            instruction=PRODUCTS_INSTRUCTION,
            llm=llm,
            verbose=verbose,
        ),
        variants=PandasNLTool(
            name="variants_tool",
            df=variants_df,
            instruction=VARIANTS_INSTRUCTION,
            llm=llm,
            verbose=verbose,
        ),
        collections=PandasNLTool(
            name="collections_tool",
            df=collections_df,
            instruction=COLLECTIONS_INSTRUCTION,
            llm=llm,
            verbose=verbose,
        ),
        catalog_overview=_catalog_overview_text(
            products_df, variants_df, collections_df,
        ),
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

LINE   = "─" * 62
DLINE  = "═" * 62

BANNER = f"""
{DLINE}
  STORE ASSISTANT  ·  ReAct Intent Engine
  Catalog Q&A · Ollama Cloud
{DLINE}
  Pipeline: Understand → Plan → Execute → Justify
  Ask semantic questions: "beige products", "winter collection",
  "frizzy hair", "outdoor workout", "morning skincare routine"
  Type 'schema' to see the store layout.
  Type 'quit' or Ctrl-C to exit.
{LINE}"""

SCHEMA_SUMMARY = """
Store schema summary:
  8 Collections → 39 Products → 200+ Variant options

  Collections and their product types:
  ┌─────────────────────┬────────────────────────────────────────────────┐
  │ Collection          │ Products                                       │
  ├─────────────────────┼────────────────────────────────────────────────┤
  │ Summer Collection   │ Premium T-Shirt, Polo Shirt, Sunglasses,      │
  │                     │ Sneakers, Water Bottle                         │
  │ Skincare            │ Face Wash, Vitamin C Serum, Night Cream,      │
  │                     │ Sunscreen SPF50, Moisturizer, Eye Cream        │
  │ Fitness             │ Yoga Mat, Resistance Bands, Protein Powder,   │
  │                     │ Gym Gloves, Foam Roller                        │
  │ Electronics         │ Wireless Earbuds, Bluetooth Speaker,          │
  │                     │ Smart Watch, Portable Charger,                 │
  │                     │ Gaming Mouse, Mechanical Keyboard              │
  │ Accessories         │ Cap, Travel Tote Bag, Laptop Sleeve,          │
  │                     │ Leather Wallet                                  │
  │ Gift Items          │ Sample Gift, Mystery Gift Box, Welcome Gift   │
  │ Apparel             │ Minimal Hoodie, Denim Jacket, Joggers,        │
  │                     │ Tank Top                                        │
  │ Beauty              │ Luxury Perfume, Lipstick Set,                 │
  │                     │ Makeup Brush Set, Hair Serum                   │
  └─────────────────────┴────────────────────────────────────────────────┘

  Query examples you can try:
    "Show me all best-selling products"
    "What colors does the Smart Watch come in?"
    "Which products are subscription eligible?"
    "Find all products with the premium tag"
    "What sizes does the Minimal Hoodie come in?"
    "Which skincare products are vegan?"
    "Show me all eco-friendly products"
    "What is in the Electronics collection?"
    "How many products are in each collection?"
    "Which products have color and size variants?"
"""


def run_cli(session: RagSession) -> None:
    print(BANNER)
    history: list[dict[str, str]] = []

    while True:
        try:
            sys.stdout.write("\n  Query: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            sys.exit(0)

        if user_input.lower() == "schema":
            print(SCHEMA_SUMMARY)
            continue

        print(f"\n{LINE}")
        try:
            answer = session.chat(user_input, history=history)

            # Wrap long lines for clean terminal output
            wrapped = "\n".join(
                textwrap.fill(line, width=60, subsequent_indent="    ")
                if len(line) > 62 else line
                for line in answer.splitlines()
            )
            print(f"\n  {wrapped}\n")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"\n  Error: {e}\n")

        print(LINE)


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════

EXAMPLE_QUERIES = [
    "Which products are available in beige?",
    "Show me winter products",
    "Do you have anything waterproof for the beach?",
    "What's good for an outdoor workout?",
    "I need something for frizzy hair",
    "Which subscription products work for dry skin?",
    "Any eco-friendly options?",
    "Give me travel-friendly products",
    "Show me all best-selling products",
    "Which products are both trending and premium?",
    "What colors does the Smart Watch come in?",
    "Which products are subscription eligible?",
    "I'm building a morning skincare routine — what do I need?",
    "What's the difference between the gift items?",
    "How many products have more than 3 color choices?",
]


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Store Assistant",
        page_icon="🛍️",
        layout="wide",
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    st.title("🛍️ Store Assistant")
    st.caption(
        "Ask anything about products, collections, sizes, colors, subscriptions, and tags. "
        "Powered by **ReAct intent planning** — understands what you *mean*, not just what you type."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        verbose = st.checkbox("Show debug logs in terminal", value=False)
        st.caption(f"Powered by **{OLLAMA_MODEL}**")

        with st.expander("🧠 How it works", expanded=False):
            st.markdown("""
**ReAct Intent Pipeline:**
1. **Understand** — classifies your query (direct lookup, semantic filter, inference, comparative)
2. **Plan** — picks the right tools and sub-queries with semantic expansion
3. **Execute** — runs pandas queries against live DataFrames
4. **Justify** — explains *why* each result matches your intent

**Semantic examples:**
- *"beige"* → searches variant_values for exact match + nearby neutrals
- *"winter"* → checks tags, warmth attributes, and layering properties
- *"frizzy hair"* → finds hair_type variant + frizz_control attribute
""")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.pending_question = None
            st.rerun()

        if st.button("Reload catalog & connection", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        st.subheader("💡 Example questions")
        for i, q in enumerate(EXAMPLE_QUERIES):
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        st.divider()
        with st.expander("Store catalog (JSON)"):
            st.json(STORE_DATA)
        with st.expander("Store schema"):
            st.markdown(SCHEMA_SUMMARY)

    @st.cache_resource(show_spinner="Loading store data and query engines…")
    def _cached_setup(_cache_version: int, _verbose: bool) -> RagSession:
        verify_ollama_cloud()
        products_df, variants_df, collections_df = build_dataframes(STORE_DATA)
        session = build_agent(
            products_df, variants_df, collections_df,
            verbose=_verbose,
        )
        return session, products_df, variants_df, collections_df

    try:
        session, products_df, variants_df, collections_df = _cached_setup(
            _AGENT_CACHE_VERSION, verbose,
        )
    except Exception as exc:
        import traceback
        st.error(
            f"Could not connect to Ollama Cloud (`{OLLAMA_MODEL}`). "
            f"Check API key and network.\n\n"
            f"```\n{exc}\n\n{traceback.format_exc()}\n```\n\n"
            "If you just updated the code, click **Reload catalog & connection** in the sidebar."
        )
        st.stop()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Products", len(products_df))
    m2.metric("Variants", len(variants_df))
    m3.metric("Collections", len(collections_df))
    m4.metric("Model", OLLAMA_MODEL)
    m5.metric("Pipeline", "ReAct")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.session_state.pending_question
    if query is None:
        query = st.chat_input("e.g. Which products are subscription eligible?")

    if query:
        st.session_state.pending_question = None
        st.session_state.chat_messages.append({"role": "user", "content": query})
        prior = st.session_state.chat_messages[:-1]
        with st.spinner("🧠 Planning query → executing → justifying…"):
            try:
                answer = session.chat(query, history=prior)
                if not answer:
                    answer = "I couldn't find an answer for that. Try naming a product or collection."
            except Exception as exc:
                import traceback
                answer = (
                    f"Something went wrong while searching the catalog.\n\n"
                    f"**Error:** `{exc}`\n\n"
                    f"<details><summary>Technical details</summary>\n\n"
                    f"```\n{traceback.format_exc()}\n```\n</details>"
                )
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Store Context Catalog RAG")
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help=f"Ollama Cloud model (default: {OLLAMA_MODEL})",
    )
    parser.add_argument("--verbose", action="store_true", help="Show tool calls and pandas code")
    args = parser.parse_args()

    print(f"  Loading store data and building DataFrames...")
    products_df, variants_df, collections_df = build_dataframes(STORE_DATA)
    print(f"  ✓ products_df   : {len(products_df)} rows × {len(products_df.columns)} columns")
    print(f"  ✓ variants_df   : {len(variants_df)} rows × {len(variants_df.columns)} columns")
    print(f"  ✓ collections_df: {len(collections_df)} rows × {len(collections_df.columns)} columns")

    print(f"\n  Connecting to Ollama Cloud ({args.model} @ {OLLAMA_CLOUD_HOST})...")
    session = build_agent(
        products_df, variants_df, collections_df,
        model=args.model,
        verbose=args.verbose,
    )
    print(f"  ✓ Query session ready (overview · products · variants · collections)")

    run_cli(session)


if __name__ == "__main__":
    if _is_streamlit_runtime():
        run_streamlit_app()
    else:
        main()