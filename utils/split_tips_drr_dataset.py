#!/usr/bin/env python3
"""Scan tips_drr_newdataset, report prefix-duplicate stats, and write 8:1 train/val lists."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def identifier_from_drr(fname: str) -> str:
    if not fname.endswith("-DRR.mhd"):
        raise ValueError(fname)
    return fname[: -len("-DRR.mhd")]


def patient_prefix(identifier: str) -> str:
    return identifier.split("-")[0]


def collect_identifiers(data_dir: Path) -> list[str]:
    return sorted(
        identifier_from_drr(f.name)
        for f in data_dir.iterdir()
        if f.name.endswith("-DRR.mhd")
    )


def split_patient_level(
    by_patient: dict[str, list[str]], seed: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    patients = sorted(by_patient.keys())
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n_val_pat = max(1, round(len(shuffled) / 9))
    val_patients = sorted(shuffled[:n_val_pat])
    train_patients = sorted(shuffled[n_val_pat:])
    train_ids = sorted(i for p in train_patients for i in by_patient[p])
    val_ids = sorted(i for p in val_patients for i in by_patient[p])
    return train_ids, val_ids, train_patients, val_patients


def split_sample_level(identifiers: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = identifiers[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) / 9))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/zzx/data/tips_drr_newdataset"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-level",
        choices=("patient", "sample"),
        default="patient",
        help="patient: no same-patient leakage; sample: exact ~8:1 count",
    )
    args = parser.parse_args()

    identifiers = collect_identifiers(args.data_dir)
    by_patient: dict[str, list[str]] = defaultdict(list)
    for ident in identifiers:
        by_patient[patient_prefix(ident)].append(ident)

    prefix_stats = {
        p: {"count": len(v), "identifiers": sorted(v)}
        for p, v in sorted(by_patient.items())
    }

    train_ids_p, val_ids_p, train_pat, val_pat = split_patient_level(by_patient, args.seed)
    train_ids_s, val_ids_s = split_sample_level(identifiers, args.seed)

    use_train = train_ids_p if args.split_level == "patient" else train_ids_s
    use_val = val_ids_p if args.split_level == "patient" else val_ids_s

    manifest = {
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "split_ratio": "8:1",
        "split_level": args.split_level,
        "num_samples": len(identifiers),
        "num_patients": len(by_patient),
        "num_train_samples": len(use_train),
        "num_val_samples": len(use_val),
        "prefix_duplicate_stats": prefix_stats,
        "all_identifiers": identifiers,
        "train_identifiers": use_train,
        "val_identifiers": use_val,
        "patient_level_split": {
            "num_train_samples": len(train_ids_p),
            "num_val_samples": len(val_ids_p),
            "train_patients": train_pat,
            "val_patients": val_pat,
            "train_identifiers": train_ids_p,
            "val_identifiers": val_ids_p,
        },
        "sample_level_split": {
            "num_train_samples": len(train_ids_s),
            "num_val_samples": len(val_ids_s),
            "train_identifiers": train_ids_s,
            "val_identifiers": val_ids_s,
        },
    }

    out = args.data_dir
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    (out / "train_list.json").write_text(
        json.dumps(use_train, indent=2, ensure_ascii=False)
    )
    (out / "val_list.json").write_text(json.dumps(use_val, indent=2, ensure_ascii=False))

    print(json.dumps(manifest, indent=2, ensure_ascii=False)[:2000])
    print(f"\nWrote {out}/dataset_manifest.json, train_list.json, val_list.json")


if __name__ == "__main__":
    main()
