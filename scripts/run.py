#!/usr/bin/env python3
"""Wrapper script: sets up imports and registers custom agents, then runs tau2 CLI.

Usage:
    python scripts/run.py run --domain banking_knowledge --agent declarative_agent \
        --agent-llm dashscope/qwen3.5-flash --user-llm gpt-4o-mini

This avoids modifying any tau2-bench source code.
"""

import os
import sys

# Add tau2-bench/src so tau2 package is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU2_SRC = os.path.join(PROJECT_ROOT, "tau2-bench", "src")
OUR_SRC = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, TAU2_SRC)
sys.path.insert(0, OUR_SRC)

# Set working directory to tau2-bench so data paths resolve correctly
os.chdir(os.path.join(PROJECT_ROOT, "tau2-bench"))

# Load tau2-bench .env if it exists
from dotenv import load_dotenv

env_path = os.path.join(PROJECT_ROOT, "tau2-bench", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# Register our custom agents before CLI parses args
from agents.register import register_all

register_all()

# Register DeepSeek embedder as a retrieval option (no tau2-bench modification needed).
# Injects into both EMBEDDER_REGISTRY dicts used by EmbeddingIndexer / EmbeddingEncoder,
# then adds a named variant to RETRIEVAL_VARIANTS so --retrieval-config deepseek_embeddings works.
from agents.local_embedder import LocalEmbedder
from tau2.knowledge.document_preprocessors.embedding_indexer import (
    EMBEDDER_REGISTRY as _doc_emb_reg,
)
from tau2.knowledge.input_preprocessors.embedding_encoder import (
    EMBEDDER_REGISTRY as _query_emb_reg,
)

_doc_emb_reg["local"] = LocalEmbedder
_query_emb_reg["local"] = LocalEmbedder

from tau2.domains.banking_knowledge.retrieval import (
    PROMPTS_DIR,
    RETRIEVAL_VARIANTS,
    PipelineSpec,
    RetrievalVariant,
    standard_prompt,
)

RETRIEVAL_VARIANTS["deepseek_embeddings"] = RetrievalVariant(
    name="deepseek_embeddings",
    prompt_template=PROMPTS_DIR / "classic_rag_deepseek_no_grep.md",
    build_prompt=standard_prompt,
    kb_search=PipelineSpec(
        type="embedding",
        embedder_type="local",
        embedder_model="all-MiniLM-L6-v2",
    ),
    supports_top_k=True,
)

# Register pricing for models not in LiteLLM's registry
import litellm

litellm.model_cost["deepseek/deepseek-v4-pro"] = {
    "input_cost_per_token": 0.00000027,     # $0.27/M
    "output_cost_per_token": 0.0000011,     # $1.10/M
    "max_tokens": 8192,
    "max_input_tokens": 65536,
    "litellm_provider": "deepseek",
    "mode": "chat",
    "supports_function_calling": True,
}

litellm.model_cost["dashscope/qwen3.5-flash"] = {
    "input_cost_per_token": 0.0000001,     # $0.10/M (Alibaba Cloud intl pricing)
    "output_cost_per_token": 0.0000004,    # $0.40/M
    "max_tokens": 65536,
    "max_input_tokens": 991000,
    "litellm_provider": "dashscope",
    "mode": "chat",
    "supports_function_calling": True,
}

# Run the tau2 CLI
from tau2.cli import main

main()
