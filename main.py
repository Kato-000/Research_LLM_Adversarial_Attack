"""
main.py — Entry point

Responsibilities of this module:
  - AttackConfig setup (models, parameters, vulnerability categories)
  - MemoryStore initialization (one Memory per VulnCategory)
  - Benchmark dataset loading and attack loop execution

Configuration:
    Set models and parameters via config = AttackConfig(...).
    See field comments for details.

Usage:
    python main.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, cast

import datasets
from loguru import logger

from src.attack import main as run_attack
from src.memory import MemoryStore
from src.models import AttackConfig, AnthropicModel, HFModel, LocalModel, Model
from src.vuln import VulnCategory

logger.add(sink="logs.txt", level=30)
 
 
# =============================================================================
# Configuration
# =============================================================================
 
# =========================================================
# Attacker / Target model configuration examples
# =========================================================
# [A] Local LM Studio / OpenAI-compatible endpoint
#   target_remote=Model(
#       api_base="http://192.168.2.104:1234/v1",
#       api_key="lm-studio",
#       model_name="openai/llama-3.2-1b-instruct",
#       temperature=0.0,
#   ),
#
# [B] OpenAI API
#   target_remote=Model(
#       api_base="https://api.openai.com/v1",
#       api_key="sk-...",           # or OPENAI_API_KEY environment variable
#       model_name="openai/gpt-4o",
#       temperature=0.0,
#   ),
#
# [C] Anthropic Claude API
#   target_claude=AnthropicModel(
#       model_name="claude-sonnet-4-5",
#       api_key="sk-ant-...",       # or ANTHROPIC_API_KEY environment variable
#       temperature=0.0,
#       max_tokens=1024,
#   ),
#
# [D] HuggingFace Inference API (free-tier models)
#   target_hf=HFModel(
#       model_name="meta-llama/Llama-3.3-70B-Instruct",
#       api_key="hf_...",           # or HF_TOKEN environment variable
#       provider="hf-inference",    # Use only HF servers (avoids SambaNova 402)
#       temperature=0.0,
#       max_tokens=1024,
#   ),
#
# [E] HuggingFace Inference Endpoints (custom URL)
#   target_hf=HFModel(
#       model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
#       api_key="hf_...",
#       api_base="https://<endpoint-id>.us-east-1.aws.endpoints.huggingface.cloud/v1",
#       temperature=0.0,
#       max_tokens=1024,
#   ),
#
# [F] Local ablated model (white-box)
#   target_local=LocalModel(
#       model_path="/path/to/model",
#       temperature=0.0,
#       max_tokens=512,
#   ),
#
# Vulnerability category filtering examples:
#   vuln_categories=[VulnCategory.PERSONA_ROLEPLAY, VulnCategory.INTENT_CONCEAL],
#   vuln_categories=None  # rotate through all categories (default)
# =========================================================
 
config = AttackConfig(
    # Tree search parameters
    branching_factor=3,
    root_width=5,
    depth=10,
    stop_score=4,
    width=10,
 
    # Memory
    memory_dir="data/memory",

    # Ablation: set True + point memory_dir at a fresh directory (e.g.
    # "data/memory_global_ablation") to run the "memory isolation OFF"
    # condition of the memory-separation-vs-prompt-design ablation.
    # Category-specific attacker prompts/evaluation are unaffected; only
    # memory sharing across categories is toggled. Default False =
    # current per-category MCTAP behavior.
    memory_global_mode=False,
 
    # Attacker model
    attacker_remote=Model(
        api_base="http://133.5.17.104:1234/v1",
        api_key="lm-studio",
        model_name="openai/mistralai/mistral-nemo-instruct-2407",
        temperature=0.7,
    ),
 
    # Target model (see comments above)
    target_remote=Model(
        api_base="http://133.5.17.104:1234/v1",
        api_key="lm-studio",
        model_name="openai/llama-3.2-1b-instruct",
        temperature=0.0,
    ),
 
    # Embedding model (for Memory)
    embedding_model=Model(
        api_base="http://133.5.17.104:1234/v1",
        api_key="lm-studio",
        model_name="text-embedding-nomic-embed-text-v1.5",
    ),
 
    # Output paths
    output_name="memory",
    output_base="data/jailbreaks/C1",
    trace_base="data/traces/memory/",
    benchmark_path="Dataset/Original_Prompt/C1_Physical_Harm_Facilitation.jsonl",
 
    # Vulnerability categories (None = rotate through all)
    vuln_categories=[VulnCategory.PROGRESSIVE_MANIP],
 
    # Multi-turn parameters (PROGRESSIVE_MANIP)
    multiturn_depth=3,
    exploit_threshold=3,
    patience=2,
)

# VulnCategory.ENC_EVASION         エンコーディング/難読化でフィルタ回避
# VulnCategory.DIRECT_OVERRIDE     安全指示の直接上書き
# VulnCategory.PERSONA_ROLEPLAY    ペルソナでアイデンティティ置換
# VulnCategory.INTENT_CONCEAL      正当な文脈で有害意図を隠蔽
# VulnCategory.PROGRESSIVE_MANIP   段階的な文脈の積み上げ

# VulnCategory.CONTEXT_INJECTION   外部ソースへの悪意ある注入
 
 
# =============================================================================
# Execution
# =============================================================================
 
if __name__ == "__main__":
    Path(config.output_base).mkdir(parents=True, exist_ok=True)
    Path(config.trace_base).mkdir(parents=True, exist_ok=True)
 
    summary_file = Path(config.output_base) / f"{config.output_name}_summary.json"
 
    # Initialize MemoryStore (per-category, or global if memory_global_mode)
    if config.memory_dir:
        memory = MemoryStore.load(
            config.memory_dir,
            embedding_model=config.embedding_model,
            global_mode=config.memory_global_mode,
        )
        stats = memory.stats()
        mode_label = "GLOBAL (ablation)" if config.memory_global_mode else "per-category"
        logger.info(
            f"[+] MemoryStore loaded from '{config.memory_dir}' "
            f"[{mode_label}]: {stats}"
        )
    else:
        memory = None
        logger.info("[+] Memory disabled (memory_dir=None)")
 
    # Load benchmark dataset
    goals = cast(
        List[str],
        datasets.Dataset.from_json(path_or_paths=config.benchmark_path)["harmful"],
    )
    logger.info(f"[+] Loaded {len(goals)} goals from {config.benchmark_path}")
 
    # Attack loop
    for goal in goals:
        run_attack(goal, config, memory=memory, summary_file=summary_file)