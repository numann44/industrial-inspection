"""Audited-split adapter around Amazon Science's PatchCore runtime.

Uses ImageNet WideResNet50-2 features, unlike our randomly initialized models.
The official implementation is vendored with its Apache-2.0 notices. An exact
PyTorch L2 search avoids a duplicate FAISS/OpenMP runtime on macOS.
Our 256-square preprocessing and audited 70/15/15 split differ from the paper.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .checkpointing import atomic_json, atomic_torch_save, collect_provenance, prepare_output
from .data import InspectionDataset, load_manifest

MODEL_KIND = "patchcore_imagenet"
SCORE_DEFINITION = "maximum nearest-neighbor squared patch-feature distance (official PatchCore); not a probability"


def capture_reference_provenance(config, manifest):
    """Snapshot adapter/runtime sources before model initialization or fitting."""
    provenance = collect_provenance(config, manifest, extra_sources=("patchcore_baseline.py",))
    vendor_root = Path(__file__).parent.parent / "patchcore"
    vendor_files = sorted(path for path in vendor_root.rglob("*") if path.is_file() and
                          (path.suffix == ".py" or path.name in {"UPSTREAM.json", "LICENSE", "NOTICE"}))
    provenance["reference"] = json.loads((vendor_root / "UPSTREAM.json").read_text())
    provenance["vendor_source_sha256"] = {
        path.relative_to(vendor_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in vendor_files}
    provenance["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for package in ("torchvision", "timm", "tqdm"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = None
    provenance["capture_stage"] = "before model initialization, normal-feature fitting and calibration"
    provenance["search_backend"] = "exact squared L2 using PyTorch CPU; FAISS avoided to prevent duplicate macOS OpenMP runtime"
    return provenance


def validate_reference_provenance(expected, actual):
    """Fail closed if scientific implementation/environment drifted during fit."""
    fields = ("python", "platform", "packages", "source_files", "config_digest", "split_digest",
              "data_seed", "training_seed", "cpu_threads", "reference", "vendor_source_sha256", "adapter_sha256")
    changed = [field for field in fields if expected.get(field) != actual.get(field)]
    if changed:
        raise ValueError(f"PatchCore implementation/environment changed during fitting: {changed}; checkpoint not published")


def _weights_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ExactTorchNN:
    """Exact squared Euclidean search with the upstream FAISS adapter interface."""
    def fit(self, features):
        self.features = torch.as_tensor(np.asarray(features, dtype=np.float32)).contiguous()

    def run(self, n_nearest_neighbours, query_features, index_features=None):
        bank = self.features if index_features is None else torch.as_tensor(index_features, dtype=torch.float32)
        if n_nearest_neighbours < 1 or n_nearest_neighbours > len(bank):
            raise ValueError("Invalid number of nearest neighbors")
        norm = bank.square().sum(1)[None]
        distances, indices = [], []
        for query in torch.as_tensor(query_features, dtype=torch.float32).split(256):
            squared = (query.square().sum(1)[:, None] + norm - 2 * query @ bank.T).clamp_min_(0)
            values, positions = squared.topk(n_nearest_neighbours, largest=False, dim=1)
            distances.append(values.numpy())
            indices.append(positions.numpy())
        return np.concatenate(distances), np.concatenate(indices)


class PatchCoreReference(nn.Module):
    def __init__(self, image_size=256, pretrained=False, device="cpu", coreset_fraction=0.01):
        super().__init__()
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
        from patchcore.patchcore import PatchCore
        from patchcore.sampler import ApproximateGreedyCoresetSampler
        self.device = torch.device(device)
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.name = "wideresnet50"
        self.core = PatchCore(self.device)
        self.core.load(backbone=self.backbone, layers_to_extract_from=["layer2", "layer3"],
                       device=self.device, input_shape=(3, image_size, image_size),
                       pretrain_embed_dimension=1024, target_embed_dimension=1024,
                       patchsize=3, anomaly_score_num_nn=1,
                       featuresampler=ApproximateGreedyCoresetSampler(coreset_fraction, self.device),
                       nn_method=ExactTorchNN())
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.to(self.device).eval()

    def normalized(self, image):
        return (image.to(self.device) - self.mean) / self.std

    @torch.inference_mode()
    def fit(self, loader):
        self.core.fit(({"image": self.normalized(batch["image"])} for batch in loader))

    @torch.inference_mode()
    def predict_batch(self, image):
        scores, maps = self.core.predict(self.normalized(image))
        return torch.from_numpy(np.asarray(maps, dtype=np.float32))[:, None].to(image.device), torch.as_tensor(np.asarray(scores), dtype=torch.float32, device=image.device)

    def forward(self, image):
        maps, _ = self.predict_batch(image)
        return image, maps

    def export_state(self):
        state = {f"backbone.{name}": value.detach().cpu().clone() for name, value in self.backbone.state_dict().items()}
        state["memory_bank"] = torch.from_numpy(self.core.anomaly_scorer.detection_features.copy())
        return state

    def load_export_state(self, state):
        self.backbone.load_state_dict({key[len("backbone."):]: value for key, value in state.items() if key.startswith("backbone.")})
        self.core.anomaly_scorer.fit([state["memory_bank"].numpy()])


def fit_baseline(manifest_path, output, image_size=256, seed=42):
    manifest = load_manifest(manifest_path, verify_splits=("train", "calibration"))
    records = manifest["splits"]["train"]
    if not records or not manifest["splits"]["calibration"]:
        raise ValueError("PatchCore requires nonempty normal training and calibration splits")
    category = records[0]["category"]
    if any(record["label"] or record["category"] != category for split in ("train", "calibration") for record in manifest["splits"][split]):
        raise ValueError("PatchCore fitting and calibration require one category of normal images")
    output = prepare_output(output)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = {"category": category, "image_size": image_size, "training_seed": seed,
              "backbone": "WideResNet50_2.IMAGENET1K_V1", "coreset_fraction": .01,
              "threshold_quantile": .90, "model_kind": MODEL_KIND, "device": "cpu",
              "feature_layers": ["layer2", "layer3"], "patch_size": 3,
              "pretrain_embed_dimension": 1024, "target_embed_dimension": 1024,
              "nearest_neighbors": 1, "image_score": "maximum unsmoothed patch distance",
              "map_smoothing_sigma": 4, "normalization_mean": [.485, .456, .406],
              "normalization_std": [.229, .224, .225]}
    started = time.perf_counter()
    provenance = capture_reference_provenance(config, manifest)
    atomic_json(output / "provenance-start.json", provenance)
    atomic_json(output / "config.json", config)
    model = PatchCoreReference(image_size=image_size, pretrained=True)
    from torchvision.models import Wide_ResNet50_2_Weights
    backbone_provenance = {"weights": "WideResNet50_2.IMAGENET1K_V1",
                           "weights_url": Wide_ResNet50_2_Weights.IMAGENET1K_V1.url,
                           "state_sha256": _weights_digest(model.backbone.state_dict()),
                           "pretrained": True, "frozen": True}
    atomic_json(output / "backbone-provenance.json", backbone_provenance)
    model.fit(DataLoader(InspectionDataset(records, image_size, False), batch_size=4))
    scores = []
    for batch in DataLoader(InspectionDataset(manifest["splits"]["calibration"], image_size, False), batch_size=4):
        _, batch_scores = model.predict_batch(batch["image"])
        scores.extend(batch_scores.tolist())
    threshold = float(np.quantile(scores, .90, method="linear"))
    end_provenance = capture_reference_provenance(config, manifest)
    validate_reference_provenance(provenance, end_provenance)
    if _weights_digest(model.backbone.state_dict()) != backbone_provenance["state_sha256"]:
        raise ValueError("Frozen ImageNet backbone weights changed during fitting")
    provenance["verified_unchanged_at_utc"] = end_provenance["created_at_utc"]
    provenance["backbone"] = backbone_provenance
    provenance["status"] = "captured before fitting; adapter, vendor, packages and backbone verified unchanged before checkpoint save"
    atomic_json(output / "provenance-end.json", end_provenance)
    load_manifest(manifest_path, verify_splits=("train", "calibration"))
    checkpoint = {"schema_version": 2, "model_kind": MODEL_KIND, "model_state": model.export_state(),
                  "model_config": {"image_size": image_size, "coreset_fraction": .01}, "config": config,
                  "image_size": image_size, "preprocess": {"mode": "square", "height": image_size, "width": image_size},
                  "threshold": threshold, "threshold_source": "90th percentile of separate normal calibration scores",
                  "score_definition": SCORE_DEFINITION, "split_digest": manifest["split_digest"],
                  "provenance": provenance, "calibration": {"quantile": .90, "quantile_method": "linear", "scores": scores, "count": len(scores),
                                                            "calibration_exceedances": int(sum(score >= threshold for score in scores))}}
    atomic_torch_save(output / "checkpoint.pt", checkpoint)
    summary = {"model_kind": MODEL_KIND, "category": category, "fit_images": len(records),
               "config": config, "image_size": image_size, "calibration": checkpoint["calibration"],
               "memory_bank_patches": len(checkpoint["model_state"]["memory_bank"]),
               "threshold": threshold, "seconds": time.perf_counter() - started,
               "pretrained": True, "selection": "fixed reference; not a from-scratch candidate",
               "split_digest": manifest["split_digest"], "provenance": provenance}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", config)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit_baseline(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
