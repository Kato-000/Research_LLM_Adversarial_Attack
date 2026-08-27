"""
memory.py — RAG store for memorizing and retrieving successful attacks

Responsibilities of this module:
  - Store past successful attacks as embedding vectors (.pt files)
  - Retrieve top-k results for a new goal using cosine similarity, or via
    two ablation modes ("random" / "curated") that isolate whether the
    benefit of memory comes from goal-relevance retrieval specifically,
    or merely from having concrete example prompts present at all.

Storage format — one file per VulnCategory:
    data/memory/ENC_EVASION.pt
    data/memory/PERSONA_ROLEPLAY.pt
    ...

Each .pt file contains:
    {
        "embeddings": Tensor (N, 768)  <- embedding of goal text
        "data":       List[dict]       <- corresponding attack records
    }

Contents of data[i]:
    {
        "adversarial_prompt": str   <- successful adversarial prompt
        "goal":               str   <- attack goal
        "score":              int   <- evaluation score (1-10)
        "vuln_category":      str   <- vulnerability category used
    }
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional
from typing_extensions import Self

import torch as t
from jaxtyping import Float
from loguru import logger
from openai import OpenAI
from torch import Tensor

from src.models import Model
from src.vuln import VulnCategory


class Memory:
    """RAG store for a single VulnCategory.

    Stores past successful attacks as embedding vectors and retrieves
    the top-k most semantically similar examples for a given goal.

    Usage:
        # Create new
        mem = Memory.new("data/memory/PERSONA_ROLEPLAY.pt", embedding_model)

        # Load from existing file
        mem = Memory.from_file("data/memory/PERSONA_ROLEPLAY.pt", embedding_model)

        # Add a successful attack (saved to .pt immediately)
        mem.add(key=goal, new_data={...})

        # Retrieve similar successful examples for a goal
        examples = mem.retrieve(key=goal, k=3)

        # Ablation: ignore goal-similarity, sample randomly
        examples = mem.retrieve_random(k=3)

        # Ablation: highest-scoring examples, regardless of goal
        examples = mem.top_by_score(k=3)
    """

    def __init__(
        self,
        embeddings: Float[Tensor, "n_records d_model"],
        data: List[dict],
        memory_file: str,
        embedding_model: Model,
    ) -> None:
        self.embeddings    = embeddings
        self.data          = data
        self.memory_file   = memory_file
        self.embedding_model = embedding_model

    @classmethod
    def from_file(cls, memory_file: str, embedding_model: Model) -> Self:
        """Restore a Memory instance from a saved .pt file."""
        checkpoint = t.load(memory_file)
        return cls(
            embeddings=checkpoint["embeddings"],
            data=checkpoint["data"],
            memory_file=memory_file,
            embedding_model=embedding_model,
        )

    @classmethod
    def new(cls, memory_file: str, embedding_model: Model, d_model: int = 768) -> "Memory":
        """Create a new empty Memory instance."""
        return cls(
            embeddings=t.zeros(0, d_model),
            data=[],
            memory_file=memory_file,
            embedding_model=embedding_model,
        )

    def save(self) -> None:
        """Write the current state to a .pt file."""
        t.save({"embeddings": self.embeddings, "data": self.data}, self.memory_file)

    def embed(self, sentence: str) -> Optional[Float[Tensor, "d_model"]]:
        """Vectorize text using the embedding model. Returns None on failure."""
        try:
            client = OpenAI(
                base_url=self.embedding_model.api_base,
                api_key=self.embedding_model.api_key,
            )
            return t.tensor(
                client.embeddings.create(
                    input=[sentence], model=self.embedding_model.model_name
                ).data[0].embedding,
                device=self.embeddings.device,
                dtype=self.embeddings.dtype if self.embeddings.numel() > 0 else t.float32,
            )
        except Exception as e:
            logger.warning(f"[Memory] Embedding failed: {e}")
            return None

    def add(self, key: str, new_data: dict) -> None:
        """Add a successful attack and immediately save to file.

        Args:
            key:      Text used as the embedding key (typically the goal)
            new_data: Attack record dict to store
        """
        embed = self.embed(key)
        if embed is None:
            logger.warning("[Memory] Skipping add due to embedding failure")
            return
        if self.embeddings.numel() == 0:
            self.embeddings = embed.unsqueeze(0)
        else:
            self.embeddings = t.cat([self.embeddings, embed.unsqueeze(0)], dim=0)
        self.data.append(new_data)
        self.save()

    def retrieve(self, key: str, k: int = 3) -> List[dict]:
        """Return the top-k past attack records most semantically similar to key.

        Searches by cosine similarity. Returns [] if memory is empty or embedding fails.
        """
        if self.embeddings.numel() == 0 or len(self.data) == 0:
            return []
        k = min(k, len(self.data))
        key_embed = self.embed(key)
        if key_embed is None:
            logger.warning("[Memory] Skipping retrieve due to embedding failure")
            return []
        best_indices = (
            t.cosine_similarity(self.embeddings, key_embed.unsqueeze(0), dim=-1)
            .topk(k=k)
            .indices.tolist()
        )
        return [self.data[idx] for idx in best_indices]

    def retrieve_random(self, k: int = 3) -> List[dict]:
        """Ablation: ignore goal-similarity entirely, sample k records at random.

        Re-sampled on every call (does not depend on the goal), so it
        isolates the effect of "having some concrete example present" from
        the effect of "the example being relevant to this specific goal".
        """
        if len(self.data) == 0:
            return []
        k = min(k, len(self.data))
        idx = random.sample(range(len(self.data)), k)
        return [self.data[i] for i in idx]

    def top_by_score(self, k: int = 3) -> List[dict]:
        """Ablation: deterministic top-k by score, ignoring the goal entirely.

        Used for the "curated" static condition — the caller is expected to
        compute this once (e.g. at MemoryStore.load time) and cache the
        result so the same examples are reused for every goal in the run,
        isolating the effect of goal-adaptive retrieval.
        """
        if len(self.data) == 0:
            return []
        ranked = sorted(self.data, key=lambda d: d.get("score", 0), reverse=True)
        return ranked[:k]

    def __len__(self) -> int:
        return len(self.data)


class MemoryStore:
    """Per-category (or optionally global) Memory manager.

    Normal mode: maintains one Memory instance per VulnCategory, each backed
    by its own .pt file. This allows the system to retrieve examples that
    were generated using the same attack strategy as the current attempt.

    Global mode (``global_mode=True``): all VulnCategory keys resolve to a
    single, shared Memory instance backed by one .pt file. This is an
    ablation configuration used to isolate the contribution of
    category-separated memory from the contribution of the
    category-specific attacker prompts / evaluation logic, which are
    unaffected by this flag: ``retrieve()`` / ``add()`` keep the exact same
    signature and are called identically by src/attack.py regardless of
    mode, since every VulnCategory simply maps to the same underlying
    Memory object.

    File layout (normal mode):
        {memory_dir}/ENC_EVASION.pt
        {memory_dir}/DIRECT_OVERRIDE.pt
        {memory_dir}/PERSONA_ROLEPLAY.pt
        {memory_dir}/INTENT_CONCEAL.pt
        {memory_dir}/PROGRESSIVE_MANIP.pt
        {memory_dir}/CONTEXT_INJECTION.pt

    File layout (global mode):
        {memory_dir}/GLOBAL.pt

    Usage:
        store = MemoryStore.load("data/memory", embedding_model)
        store = MemoryStore.load("data/memory_ablation", embedding_model,
                                  global_mode=True)

        # Retrieve examples for a specific category
        examples = store.retrieve(goal, category=VulnCategory.PERSONA_ROLEPLAY, k=3)

        # Save a successful attack under its category
        store.add(goal, data={...}, category=VulnCategory.PERSONA_ROLEPLAY)
    """

    def __init__(self, memories: Dict[VulnCategory, Memory], global_mode: bool = False) -> None:
        self._memories = memories
        self.global_mode = global_mode
        # "curated"モード用キャッシュ：カテゴリ（globalモードでは"GLOBAL"固定）
        # ごとに実行内で一度だけ選定し、以降固定する。
        self._curated_cache: Dict[object, List[dict]] = {}

    @classmethod
    def load(
        cls,
        memory_dir: str,
        embedding_model: Model,
        global_mode: bool = False,
    ) -> "MemoryStore":
        """Load (or create) Memory instance(s) from memory_dir.

        If global_mode is False (default): one Memory per VulnCategory, as
        before. If a category's .pt file already exists it is loaded;
        otherwise a new empty Memory is created (and written on first add).

        If global_mode is True: a single shared Memory backed by
        ``{memory_dir}/GLOBAL.pt`` is created (or loaded), and every
        VulnCategory member maps to this same instance. Use a distinct
        memory_dir for global-mode runs so they do not mix with, or
        overwrite, an existing per-category memory directory.
        """
        dirpath = Path(memory_dir)
        dirpath.mkdir(parents=True, exist_ok=True)

        if global_mode:
            fpath = dirpath / "GLOBAL.pt"
            if fpath.exists():
                logger.info(f"[MemoryStore] Loading GLOBAL memory from {fpath}")
                shared = Memory.from_file(str(fpath), embedding_model)
            else:
                logger.info(f"[MemoryStore] Creating new GLOBAL memory at {fpath}")
                shared = Memory.new(str(fpath), embedding_model)
            # Every category resolves to the SAME Memory object, so
            # self._memories[category] in retrieve()/add() below reads and
            # writes one shared store no matter which category is passed.
            memories: Dict[VulnCategory, Memory] = {cat: shared for cat in VulnCategory}
            return cls(memories, global_mode=True)

        memories: Dict[VulnCategory, Memory] = {}
        for cat in VulnCategory:
            fpath = dirpath / f"{cat.value}.pt"
            if fpath.exists():
                logger.info(f"[MemoryStore] Loading {fpath}")
                memories[cat] = Memory.from_file(str(fpath), embedding_model)
            else:
                logger.info(f"[MemoryStore] Creating new memory for {cat.value} at {fpath}")
                memories[cat] = Memory.new(str(fpath), embedding_model)

        return cls(memories, global_mode=False)

    @staticmethod
    def _format_records(records: List[dict]) -> List[str]:
        """Convert raw stored records into hint strings shown to the attacker.

        Records store a unified ``conversation`` list regardless of category:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

        The returned hint joins all user turns with " ||| " so the attacker LM
        sees the full multi-turn structure (or the single prompt for other categories).
        Falls back to ``adversarial_prompt`` for older records without ``conversation``.
        """
        results = []
        for r in records:
            if r.get("conversation"):
                user_turns = [
                    msg["content"]
                    for msg in r["conversation"]
                    if msg.get("role") == "user"
                ]
                results.append(" ||| ".join(user_turns))
            else:
                results.append(r.get("adversarial_prompt", ""))
        return results

    def retrieve(
        self,
        goal: str,
        category: VulnCategory,
        k: int = 3,
    ) -> List[str]:
        """Return up to k prompt hints for the given category, chosen by
        semantic similarity to ``goal`` (memory_mode="retrieved").

        In global mode, ``category`` is still required for API
        compatibility with src/attack.py, but is ignored for indexing
        purposes: every category maps to the same shared Memory, so hints
        may originate from goals attempted under a different category.
        """
        mem = self._memories[category]
        records = mem.retrieve(key=goal, k=k)
        return self._format_records(records)

    def retrieve_random(
        self,
        category: VulnCategory,
        k: int = 3,
    ) -> List[str]:
        """Ablation: return k hints sampled at random, ignoring goal
        similarity entirely (memory_mode="random"). Re-sampled on every
        call. Isolates "having some concrete example" from "the example
        being relevant to this goal".
        """
        mem = self._memories[category]
        records = mem.retrieve_random(k=k)
        return self._format_records(records)

    def retrieve_curated(
        self,
        category: VulnCategory,
        k: int = 3,
    ) -> List[str]:
        """Ablation: return the same k highest-scoring hints for every goal
        in the run (memory_mode="curated"). The selection is computed once
        per category (or once globally, in global_mode) and cached, so it
        stays fixed across the whole run — isolating "goal-adaptive
        retrieval" from "having good, but static, examples available".
        No new content is generated here; the examples are drawn entirely
        from records already present in the store.
        """
        cache_key = "GLOBAL" if self.global_mode else category
        if cache_key not in self._curated_cache:
            mem = self._memories[category]
            self._curated_cache[cache_key] = mem.top_by_score(k=k)
        return self._format_records(self._curated_cache[cache_key])

    def add(self, goal: str, data: dict, category: VulnCategory) -> None:
        """Save a successful attack record under the given category.

        In global mode this appends to the single shared Memory
        regardless of ``category`` (the record's own ``vuln_category``
        field, set by the caller in src/attack.py, still preserves which
        category actually produced it for later analysis).

        Args:
            goal:     The attack goal used as the embedding key
            data:     Full attack record dict (adversarial_prompt, score, etc.)
            category: The VulnCategory this attack belongs to
        """
        self._memories[category].add(key=goal, new_data=data)
        if self.global_mode:
            logger.info(
                f"[MemoryStore] Saved to GLOBAL memory "
                f"(originating category: {category.value}, "
                f"total: {len(self._memories[category])})"
            )
        else:
            logger.info(
                f"[MemoryStore] Saved to {category.value} "
                f"(total: {len(self._memories[category])})"
            )

    def stats(self) -> Dict[str, int]:
        """Return the number of stored records per category.

        In global mode all categories share one Memory, so this returns a
        single "GLOBAL" entry instead of repeating the same count under
        every category name (which would otherwise look misleadingly like
        6 separate, equally-sized stores).
        """
        if self.global_mode:
            any_cat = next(iter(self._memories))
            return {"GLOBAL": len(self._memories[any_cat])}
        return {cat.value: len(mem) for cat, mem in self._memories.items()}
