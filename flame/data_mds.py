"""Pre-tokenized MDS dataset reader for flame training.

Wraps mosaicml-streaming's StreamingDataset so it yields
`{'input_ids': torch.LongTensor}` dicts that flame's
DataCollatorForLanguageModeling consumes directly.

Designed for `princeton-nlp/prolong-data-64K` (and any other MDS dataset
with the same schema):

    {'domain': str,
     'indices': (n_docs, 2) uint32  -- per-doc boundaries, unused here,
     'input_ids': (seq_len,) uint32 -- packed tokens, exactly seq_len long,
     'length': uint64}

Each sample is already exactly the target sequence length (e.g. 65,536),
so we skip the buffer-shuffle/repack logic in flame's
BufferShuffledIterableDataset and just emit one chunk per MDS sample.

Usage in flame.train via the `mds:` URI scheme on `--training.dataset`:

    mds:/data/prolong-data-64k                    # all of it
    mds:/data/prolong-data-64k?proc=proc0[0-4]-64 # 5B subset (procs 00-04 / 64)
    mds:/data/prolong-data-64k?proc=proc0[0-4]-64&domain=arxiv,book-65536
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


def parse_mds_uri(uri: str) -> tuple[str, dict]:
    """Parse an `mds:/path[?key=val&...]` URI into (path, params)."""
    assert uri.startswith("mds:"), uri
    body = uri[4:]
    if "?" in body:
        path, query = body.split("?", 1)
        params = dict(pair.split("=", 1) for pair in query.split("&") if pair)
    else:
        path, params = body, {}
    return path, params


def discover_proc_dirs(
    root: str,
    proc_glob: Optional[str] = None,
    domain_filter: Optional[set[str]] = None,
) -> list[str]:
    """Walk root/<domain>/<procXX-64>/ and return matching proc dir paths."""
    proc_glob = proc_glob or "proc*-64"
    out = []
    for domain in sorted(os.listdir(root)):
        if domain.startswith(".") or not os.path.isdir(os.path.join(root, domain)):
            continue
        if domain_filter is not None and domain not in domain_filter:
            continue
        for proc in sorted(glob.glob(os.path.join(root, domain, proc_glob))):
            if os.path.isdir(proc):
                out.append(proc)
    return out


class MDSPretokenizedDataset(IterableDataset):
    """Iterates over pre-tokenized MDS samples and yields flame-compatible dicts.

    Each MDS sample is *already* the target seq_len, so we just emit it as-is.

    Multi-stream over many `procXX-64/` dirs is handled via mosaicml-streaming's
    `Stream` API; each Stream becomes one shard source, and StreamingDataset
    interleaves them. mosaic-streaming auto-shards across torch.distributed
    ranks if `dist.is_initialized()` at dataset construction.
    """
    def __init__(
        self,
        root: str,
        seq_len: int = 65536,
        proc_glob: Optional[str] = "proc0[0-9]-64",   # default ≈ 5.5B subset (10/64 procs)
        domain_filter: Optional[set[str]] = None,
        shuffle: bool = True,
        seed: int = 42,
        batch_size: int = 1,
    ):
        from streaming import Stream, StreamingDataset

        self.seq_len = seq_len

        proc_dirs = discover_proc_dirs(root, proc_glob=proc_glob,
                                       domain_filter=domain_filter)
        if not proc_dirs:
            raise FileNotFoundError(
                f"No MDS proc dirs under {root} matching glob={proc_glob!r}, "
                f"domain_filter={domain_filter}"
            )
        logger.info(f"MDSPretokenizedDataset: {len(proc_dirs)} streams over "
                    f"{root} (proc_glob={proc_glob!r})")
        for d in proc_dirs[:5]:
            logger.info(f"  e.g. {d}")
        if len(proc_dirs) > 5:
            logger.info(f"  ... and {len(proc_dirs) - 5} more")

        streams = [Stream(local=d) for d in proc_dirs]
        self.ds = StreamingDataset(
            streams=streams,
            shuffle=shuffle,
            shuffle_seed=seed,
            batch_size=batch_size,
            # mosaic-streaming auto-detects torch.distributed; nothing else
            # needed for DP sharding.
        )

    def __iter__(self):
        for sample in self.ds:
            ids = sample["input_ids"]
            # Sanity: ProLong samples are exactly seq_len long, but tolerate
            # the tail edge cases by truncating.
            if len(ids) > self.seq_len:
                ids = ids[: self.seq_len]
            yield {"input_ids": torch.from_numpy(np.ascontiguousarray(ids)).long()}


def build_mds_dataset(uri: str, seq_len: int, seed: int = 42) -> MDSPretokenizedDataset:
    """Entry point used by flame.data.build_dataset on an `mds:` URI."""
    path, params = parse_mds_uri(uri)
    domain_filter = set(params["domain"].split(",")) if "domain" in params else None
    proc_glob = params.get("proc", "proc0[0-9]-64")
    seq_len = int(params.get("seq_len", seq_len))
    return MDSPretokenizedDataset(
        root=path,
        seq_len=seq_len,
        proc_glob=proc_glob,
        domain_filter=domain_filter,
        seed=seed,
    )


__all__ = ["MDSPretokenizedDataset", "build_mds_dataset", "parse_mds_uri"]
