"""Release fixtures are generated locally; no benchmark or network access."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import package_study as release
from scripts.run_study import select_by_validation


def test_direct_cli_help_runs_without_loading_datasets(tmp_path):
    result = subprocess.run([sys.executable, str(Path(release.__file__).resolve()), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--study" in result.stdout and "--output" in result.stdout


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_study(tmp_path, monkeypatch, supervised=False):
    study = tmp_path / "outputs/study-v2"
    dump(study / "status.json", {"state": "complete", "protocol": "study-v2"})
    dump(study / "declaration.json", {"protocol": "study-v2"})
    deployments, metrics, candidates, manifests = {}, {}, [], {}
    categories = (*release.CATEGORIES, "kolektor_surface") if supervised else release.CATEGORIES
    for index, category in enumerate(categories):
        run = tmp_path / "runs" / category
        run.mkdir(parents=True)
        kind = "supervised_segmentation" if category == "kolektor_surface" else "joint_reconstruction_segmentation"
        config = {"category": category, "training_seed": 43, "threshold_quantile": .9}
        checkpoint = {"model_state": {"fixture": torch.tensor([float(index)])}, "resumable": False,
                      "model_kind": kind, "split_digest": f"split-{category}", "threshold": .5,
                      "best_metric": .9, "config": config, "image_size": 32, "best_epoch": 10,
                      "calibration": {"quantile": .9, "normal_count": 3}}
        checkpoint['preprocess'] = ({'mode': 'letterbox', 'height': 640, 'width': 256}
                                    if category == 'kolektor_surface' else {'mode': 'square', 'height': 32, 'width': 32})
        torch.save(checkpoint, run / "checkpoint.pt")
        item = {"checkpoint": (run / "checkpoint.pt").relative_to(tmp_path).as_posix(),
                "run": run.relative_to(tmp_path).as_posix(), "checkpoint_sha256": release.sha256(run / "checkpoint.pt"),
                "model_kind": kind, "seed": 43, "split_digest": f"split-{category}", "threshold": .5,
                "validation_score": .9, "bank_digest": None if category == "kolektor_surface" else f"bank-{category}",
                "config": config, "best_epoch": 10, "manifest": f"data/{category}.json"}
        deployments[category] = item
        candidates.append(item)
        # A seed chosen by validation can fail the dataset target; preserve it.
        metric = {"checkpoint_sha256": item["checkpoint_sha256"], "split_digest": item["split_digest"],
                  "threshold": .5, "defect_recall": .5 if supervised else 1.0,
                  "normal_false_alarm_rate": .2 if supervised else 0.0,
                  "image_auroc": .7, "pixel_average_precision": .4}
        metrics[item["checkpoint_sha256"]] = metric
        dump(study / "evaluations" / f"{category}.json", metric)
        name = "deployment-ksdd2" if category == "kolektor_surface" else f"deployment-{category}"
        dump(study / "selections" / f"{name}.json", select_by_validation([item]))
        records = []
        data_root = tmp_path / "fake-data" / category
        for defect, label in (("good", 0), ("scratch", 1)):
            for filename in ("001.png", "000.png"):
                image = data_root / "test" / defect / filename
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"fixture-{category}-{defect}-{filename}".encode())
                records.append({"image": str(image), "category": category, "defect": defect, "label": label,
                                "image_sha256": release.sha256(image)})
        manifests[str(tmp_path / item["manifest"])] = {"root": str(data_root), "split_digest": item["split_digest"],
                                                     "splits": {"test": records}}
    mvtec = {key: item for key, item in deployments.items() if key != "kolektor_surface"}
    mvtec_metrics = {item["checkpoint_sha256"]: metrics[item["checkpoint_sha256"]] for item in mvtec.values()}
    dump(study / "mvtec-pretest-freeze.json", {"test_selection": False, "deployments": mvtec,
                                             "checkpoints": candidates[:3]})
    fallback = {"triggered": supervised}
    if supervised:
        selected = deployments["kolektor_surface"]
        fallback.update(selected=selected, evaluations={selected["checkpoint_sha256"]: metrics[selected["checkpoint_sha256"]]})
        dump(study / "ksdd2-pretest-freeze.json", {"test_selection": False,
                                                 "deployments": {"kolektor_surface": selected}, "checkpoints": [selected]})
    results = {"protocol": "study-v2", "model_and_seed_selection_used_test_metrics": False,
               "pretest_selections": mvtec, "all_mvtec_evaluations": mvtec_metrics,
               "conditional_supervised": fallback, "any_validation_selected_model_met_dataset_target": not supervised}
    dump(study / "study-results.json", results)
    gallery_sources, portable_categories = {}, {}
    for category, item in deployments.items():
        folder = study / "galleries" / category
        dump(folder / "index.json", {"threshold": .5, "items": [{"file": "panel.png"}]})
        (folder / "panel.png").write_bytes(f"fixture-rendered-gallery-{category}".encode())
        (folder / "ATTRIBUTION.md").write_text("Fixture rendered derivative; CC BY-NC-SA 4.0.\n")
        gallery_sources[category] = folder
        portable_categories[category] = {"selected": {"parameters": 1, "checkpoint_sha256": item["checkpoint_sha256"]},
                                         "gallery": f"controlled-study-galleries/{category}/index.json"}
    gate_calls = []
    def gate(root, study):
        gate_calls.append((root, study))
        path = Path(study) / "status.json"
        evidence = {path.as_posix(): release.sha256(root / path)}
        evidence.update({path.relative_to(root).as_posix(): release.sha256(path)
                         for folder in gallery_sources.values() for path in folder.iterdir()})
        return {"evidence_sha256": evidence, "gallery_sources": gallery_sources,
                "portable": {"categories": portable_categories}}
    monkeypatch.setattr(release, "verify_completed_study", gate)
    monkeypatch.setattr("inspection.evaluate.load_evaluation_manifest", lambda path, verify_splits=(): manifests[str(path)])
    # Existing active registry must remain untouched by every packaging path.
    registry = tmp_path / "artifacts/models.json"
    dump(registry, {"schema_version": 1, "models": [{"id": "active-pilot"}]})
    return study, deployments, metrics, manifests, gate_calls, registry


def test_release_preserves_pretest_selected_weights_and_attributed_examples_without_registry_changes(tmp_path, monkeypatch):
    study, deployments, metrics, manifests, calls, active = fixture_study(tmp_path, monkeypatch)
    original_registry = active.read_bytes()
    output = release.package_study(tmp_path)
    assert calls == [(tmp_path.resolve(), "outputs/study-v2")]
    assert active.read_bytes() == original_registry
    candidate = release.read_json(output / "artifacts/models.json")
    assert candidate["published"] is False
    assert candidate["path_base"] == "release bundle root"
    assert len(candidate["models"]) == 3
    for model in candidate["models"]:
        item = deployments[model["category"]]
        assert model["training_seed"] == 43
        assert release.sha256(output / model["local_path"]) == item["checkpoint_sha256"]
        from inspection.artifacts import resolve_checkpoint
        assert resolve_checkpoint(model, output) == output / model["local_path"]
        assert model["target_met"] is True
        assert 'normal-only' in model['label']
        assert 'normal MVTec AD images and procedural corruption masks' in model['description']
        assert 'square preparation' in model['preparation_note']
        assert model['preprocess']['mode'] == 'square'
        expected_scope = 'exploratory' if model['category'] == 'metal_nut' else 'frozen before held-out testing'
        assert expected_scope in model['evaluation_note']
        checkpoint = torch.load(tmp_path / item["checkpoint"], map_location="cpu", weights_only=True)
        assert "parameters" not in checkpoint  # Matches real main-training outputs.
        assert release.read_json(output / "models" / f"{model['category']}.json")["parameters"] == 1
        assert "url" not in model  # No fictional published download location.
        assert len(model["examples"]) == 2
        for example in model["examples"]:
            assert example["source_path"].endswith("000.png")
            assert release.sha256(output / example["path"]) == example["sha256"]
            assert example["license"] == "CC BY-NC-SA 4.0"
            assert example["changes"] == "none; byte-identical copy"
    portable = release.read_json(output / "controlled-study.json")
    for category, group in portable["categories"].items():
        index_path = output / group["gallery"]
        assert index_path.is_file()
        assert (index_path.parent / "ATTRIBUTION.md").is_file()
        for panel in release.read_json(index_path)["items"]:
            assert (index_path.parent / panel["file"]).read_bytes() == f"fixture-rendered-gallery-{category}".encode()
    notes = (output / "RELEASE_NOTES.md").read_text()
    assert "exploratory" in notes and "not production approval" in notes
    assert "PatchCore comparison remains separate" in notes
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert release.sha256(output / name) == digest
    with pytest.raises(FileExistsError, match="never replaced"):
        release.package_study(tmp_path)


def test_conditional_supervised_bundle_keeps_unmet_targets_and_separate_supervision(tmp_path, monkeypatch):
    fixture_study(tmp_path, monkeypatch, supervised=True)
    output = release.package_study(tmp_path)
    models = release.read_json(output / "artifacts/models.json")["models"]
    assert len(models) == 4
    assert all(item["target_met"] is False and "NOT MET" in item["status_label"] for item in models)
    summary = release.read_json(output / "models/kolektor_surface.json")
    assert summary["supervision"] == "real defect images and masks"
    assert summary["experimental_status"] == "frozen-held-out"
    assert "Kolektor" in models[-1]["examples"][0]["attribution"]
    surface = next(model for model in models if model['category'] == 'kolektor_surface')
    assert 'real-defect supervised' in surface['label']
    assert 'REAL-DEFECT SUPERVISED' in surface['status_label']
    assert 'real normal and defective' in surface['description']
    assert 'annotated defect masks' in surface['description']
    assert 'letterbox padding to 256 × 640' in surface['preparation_note']
    assert surface['preprocess'] == {'mode': 'letterbox', 'height': 640, 'width': 256}
    assert 'real labeled defects and masks' in surface['evaluation_note']
    assert 'not interchangeable with normal-only MVTec results' in surface['evaluation_note']


def test_incomplete_study_is_rejected_before_any_weights_or_images_are_opened(tmp_path, monkeypatch):
    study = tmp_path / "outputs/study-v2"
    dump(study / "status.json", {"state": "running", "protocol": "study-v2"})
    monkeypatch.setattr(release, "verify_completed_study", lambda *args: pytest.fail("No evidence read for incomplete study"))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("No checkpoint read for incomplete study"))
    with pytest.raises(ValueError, match="complete study-v2"):
        release.package_study(tmp_path)
    assert not (tmp_path / "outputs/release-v0.1.0").exists()


@pytest.mark.parametrize("corruption", ["weight", "evaluation-split", "evaluation-threshold", "selection"])
def test_inconsistent_evidence_refuses_bundle(tmp_path, monkeypatch, corruption):
    study, deployments, _, _, _, _ = fixture_study(tmp_path, monkeypatch)
    first = deployments["metal_nut"]
    if corruption == "weight":
        (tmp_path / first["checkpoint"]).write_bytes(b"changed")
    elif corruption == "selection":
        path = study / "selections/deployment-metal_nut.json"
        value = release.read_json(path)
        value["selected"]["seed"] = 44
        dump(path, value)
    else:
        path = study / "study-results.json"
        value = release.read_json(path)
        key = "split_digest" if corruption == "evaluation-split" else "threshold"
        value["all_mvtec_evaluations"][first["checkpoint_sha256"]][key] = "wrong" if key == "split_digest" else .7
        dump(path, value)
    with pytest.raises(ValueError):
        release.package_study(tmp_path)
    assert not (tmp_path / "outputs/release-v0.1.0").exists()


def test_changed_example_refuses_atomic_bundle_and_cleans_only_own_staging(tmp_path, monkeypatch):
    _, _, _, manifests, _, active = fixture_study(tmp_path, monkeypatch)
    manifest = next(iter(manifests.values()))
    chosen = release.choose_examples(manifest, "metal_nut")[0][1]
    Path(chosen["image"]).write_bytes(b"changed after audit")
    original = active.read_bytes()
    with pytest.raises(ValueError, match="example bytes changed"):
        release.package_study(tmp_path)
    assert active.read_bytes() == original
    assert not list((tmp_path / "outputs").glob(".release-v0.1.0-*"))
    assert not (tmp_path / "outputs/release-v0.1.0").exists()


def test_shared_verification_failure_stops_packaging_before_dataset_access(tmp_path, monkeypatch):
    fixture_study(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError("incomplete seed-variability evidence")
    monkeypatch.setattr(release, "verify_completed_study", fail)
    monkeypatch.setattr("inspection.evaluate.load_evaluation_manifest", lambda *args, **kwargs: pytest.fail("No raw dataset access"))
    with pytest.raises(ValueError, match="seed-variability"):
        release.package_study(tmp_path)
    assert not (tmp_path / "outputs/release-v0.1.0").exists()


def test_evidence_mutation_during_copy_refuses_release_publication(tmp_path, monkeypatch):
    study, _, _, _, _, _ = fixture_study(tmp_path, monkeypatch)
    original = release._copy_examples
    def changing_copy(*args, **kwargs):
        examples = original(*args, **kwargs)
        dump(study / "status.json", {"state": "failed", "protocol": "study-v2"})
        return examples
    monkeypatch.setattr(release, "_copy_examples", changing_copy)
    with pytest.raises(ValueError, match="evidence changed during"):
        release.package_study(tmp_path)
    assert not (tmp_path / "outputs/release-v0.1.0").exists()
    assert not list((tmp_path / "outputs").glob(".release-v0.1.0-*"))


@pytest.mark.parametrize("corruption", ["null-parameters", "different-deployment", "missing-gallery-reference"])
def test_verified_portable_deployment_details_must_resolve_in_the_bundle(tmp_path, monkeypatch, corruption):
    fixture_study(tmp_path, monkeypatch)
    original = release.verify_completed_study
    def wrong_details(*args, **kwargs):
        value = original(*args, **kwargs)
        group = value["portable"]["categories"]["metal_nut"]
        if corruption == "null-parameters":
            group["selected"]["parameters"] = None
        elif corruption == "different-deployment":
            group["selected"]["checkpoint_sha256"] = "another-checkpoint"
        else:
            group["gallery"] = "controlled-study-galleries/missing/index.json"
        return value
    monkeypatch.setattr(release, "verify_completed_study", wrong_details)
    with pytest.raises(ValueError, match="parameter count|gallery reference"):
        release.package_study(tmp_path)
    assert not (tmp_path / "outputs/release-v0.1.0").exists()
    assert not list((tmp_path / "outputs").glob(".release-v0.1.0-*"))
