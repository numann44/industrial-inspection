"""An immutable validation-normal-only bank with equally weighted families."""
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from inspection.checkpointing import atomic_json, config_digest
from inspection.data import InspectionDataset, load_manifest
from inspection.model import anomaly_score
from inspection.validation_bank import load_bank as load_v2_bank, harmonic_selection_score

from .protocol import SELECTION_RULE, sha256
from .synthesis import synthesize, changed_mask, withheld_translation

NEW_FAMILIES = ("v3/subtle_patch", "v3/boundary_warp", "withheld/hard_translation")


def validate_split_sources(manifest):
    identities = {}
    for name in ("train", "validation", "calibration"):
        records = manifest["splits"][name]
        if len(records) < 2 or any(r["label"] != 0 or r["mask"] is not None for r in records):
            raise ValueError("At least two normal unmasked images are required in every development split")
        identities[name] = {r["image_sha256"] for r in records}
    if any(identities[a] & identities[b] for a, b in
           (("train", "validation"), ("train", "calibration"), ("validation", "calibration"))):
        raise ValueError("Donor/source identities cross development splits")


def load_bank(path, verify=True):
    path = Path(path)
    bank = json.loads(path.read_text())
    if bank.get("bank_digest") != config_digest({k: v for k, v in bank.items() if k != "bank_digest"}):
        raise ValueError("V3 bank digest mismatch")
    if bank.get("schema_version") != "study-v3-bank-1" or bank["source_split"] != "validation":
        raise ValueError("Only a validation-normal V3 bank can select these models")
    if verify:
        seen = set()
        for row in bank["records"]:
            for key in ("image", "mask"):
                relative = Path(row[key])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Bank arrays must remain within the immutable bank")
                if row[key] not in seen:
                    if sha256(path.parent / row[key]) != row[f"{key}_sha256"]:
                        raise ValueError("V3 bank content mismatch")
                    seen.add(row[key])
    return bank


def create_bank(manifest_path, old_bank_path, output, seed=3_000_042):
    """Copies old NPY examples exactly, then adds three fixed challenge families.

    Donors are exclusively validation normals here. Training has its own strictly
    train-only donor pool. No train pixels, calibration pixels or test pixels are
    used in this function.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("V3 bank directory already exists; banks are immutable")
    manifest = load_manifest(manifest_path, verify_splits=("validation",))
    validate_split_sources(manifest)
    old = load_v2_bank(old_bank_path)
    if old["split_digest"] != manifest["split_digest"]:
        raise ValueError("Original bank and audited split differ")
    size = old["canonical_image_size"]
    sources = manifest["splits"]["validation"]
    source_hashes = {r["image_sha256"] for r in sources}
    if any(r["source_sha256"] not in source_hashes for r in old["records"]):
        raise ValueError("Original bank has sources outside this validation split")
    # Publish bank directory only after every array and metadata file exists.
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError("Incomplete bank staging exists; preserve/inspect it before retrying")
    temporary.mkdir(parents=True)
    (temporary / "arrays").mkdir()
    rows = []
    old_root = Path(old_bank_path).parent
    for index, record in enumerate(old["records"]):
        row = {"family": "clean" if record["label"] == 0 else f'{record["strategy"]}/{record["mode"]}',
               "label": record["label"], "source_sha256": record["source_sha256"],
               "donor_sha256": None, "origin": "unchanged_v2"}
        for key in ("image", "mask"):
            relative = f"arrays/old_{index:04d}_{key}.npy"
            shutil.copyfile(old_root / record[key], temporary / relative)
            row[key], row[f"{key}_sha256"] = relative, sha256(temporary / relative)
            if row[f"{key}_sha256"] != record[f"{key}_sha256"]:
                raise ValueError("Original bank array changed during copy")
        rows.append(row)
    dataset = InspectionDataset(sources, size, include_masks=False)
    clean_images = [dataset[i]["image"] for i in range(len(dataset))]
    for index, clean in enumerate(clean_images):
        donor_index = (index + 1) % len(clean_images)
        donor = clean_images[donor_index]
        for family_index, family in enumerate(NEW_FAMILIES):
            for attempt in range(64):
                selected_seed = seed + index * 10_000 + family_index * 100 + attempt
                generator = torch.Generator().manual_seed(selected_seed)
                if family == NEW_FAMILIES[2]:
                    changed = withheld_translation(clean, donor, generator)
                    mask = changed_mask(clean, changed)
                else:
                    changed, mask = synthesize(clean, donor, generator, "A" if family_index == 0 else "B",
                                               mode="texture" if family_index == 0 else "warp")
                if mask.any():
                    break
            else:
                raise ValueError(f"No nonempty {family} challenge from validation source {index}")
            row = {"family": family, "label": 1, "origin": "v3",
                   "source_sha256": sources[index]["image_sha256"],
                   "donor_sha256": sources[donor_index]["image_sha256"], "seed": selected_seed}
            for key, array in (("image", changed.numpy().astype(np.float32)),
                               ("mask", mask.numpy().astype(np.uint8))):
                relative = f"arrays/new_{index:04d}_{family_index}_{key}.npy"
                np.save(temporary / relative, array, allow_pickle=False)
                row[key], row[f"{key}_sha256"] = relative, sha256(temporary / relative)
            rows.append(row)
    counts = {family: sum(r["family"] == family for r in rows) for family in sorted({r["family"] for r in rows})}
    bank = {"schema_version": "study-v3-bank-1", "source_split": "validation", "seed": seed,
            "split_digest": manifest["split_digest"], "source_images": len(sources),
            "image_size": size, "old_bank_digest": old["bank_digest"],
            "old_bank_file_sha256": sha256(old_bank_path), "family_counts": counts,
            "selection_rule": SELECTION_RULE, "records": rows,
            "warning": "Additional synthetic operators are proxy validation, not independently acquired real defects."}
    bank["bank_digest"] = config_digest(bank)
    atomic_json(temporary / "bank.json", bank)
    temporary.rename(output)
    return bank


class BankDataset:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.bank = load_bank(path, verify)
        self.records = self.bank["records"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {"image": torch.from_numpy(np.load(self.path.parent / row["image"], allow_pickle=False).copy()).float(),
                "mask": torch.from_numpy(np.load(self.path.parent / row["mask"], allow_pickle=False).copy()).float(),
                "family": row["family"], "label": row["label"]}


def family_metrics(records, bank):
    clean = records["clean"]
    result = {}
    for family, anomalous in records.items():
        if family == "clean":
            continue
        combined = clean + anomalous
        auroc = float(roc_auc_score([r[0] for r in combined], [r[1] for r in combined]))
        pixel_ap = float(average_precision_score(np.concatenate([r[2].ravel() for r in combined]),
                                                  np.concatenate([r[3].ravel() for r in combined])))
        result[family] = {"image_auroc": auroc, "pixel_ap": pixel_ap,
                          "harmonic_mean": harmonic_selection_score(auroc, pixel_ap),
                          "defective_images": len(anomalous), "clean_controls": len(clean)}
    if set(result) != set(bank["family_counts"]) - {"clean"}:
        raise ValueError("Missing validation family results")
    old = [value["harmonic_mean"] for family, value in result.items() if family not in NEW_FAMILIES]
    new = [value["harmonic_mean"] for family, value in result.items() if family in NEW_FAMILIES]
    return {"macro_h": float(np.mean([v["harmonic_mean"] for v in result.values()])),
            "old_macro_h": float(np.mean(old)), "new_macro_h": float(np.mean(new)),
            "families": result, "bank_digest": bank["bank_digest"],
            "split_digest": bank["split_digest"], "selection_rule": SELECTION_RULE,
            "real_test_images_read": 0}


@torch.inference_mode()
def evaluate(model, bank_path, device, batch_size=8):
    dataset = BankDataset(bank_path, verify=False)  # trainer/runner verified at entry
    records = defaultdict(list)
    model.eval()
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        _, logits = model(batch["image"].to(device))
        maps, scores = torch.sigmoid(logits).cpu().numpy(), anomaly_score(logits).cpu().tolist()
        if not np.isfinite(maps).all() or not np.isfinite(scores).all():
            raise ValueError("Nonfinite validation predictions")
        for i, family in enumerate(batch["family"]):
            records[family].append((int(batch["label"][i]), scores[i], batch["mask"][i].numpy(), maps[i]))
    return family_metrics(records, dataset.bank)
