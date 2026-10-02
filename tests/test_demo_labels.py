"""Registry-only UI checks: no weights, images, inference or network access."""
from pathlib import Path

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from scripts.package_study import _presentation

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('category,kind,preprocess', [
    ('metal_nut', 'joint_reconstruction_segmentation', {'mode': 'square', 'width': 256, 'height': 256}),
    ('screw', 'segmentation_only', {'mode': 'square', 'width': 128, 'height': 128}),
    ('kolektor_surface', 'supervised_segmentation', {'mode': 'letterbox', 'width': 256, 'height': 640}),
])
def test_selected_task_explains_supervision_and_checkpoint_preparation_without_inference(monkeypatch, category, kind, preprocess):
    summary = {'model_kind': kind, 'preprocess': preprocess, 'measured_quality_target_met': False}
    entry = {'id': 'metadata-only-fixture', 'category': category, 'model_kind': kind,
             **_presentation(category, summary), 'examples': [], 'metrics': {}}
    monkeypatch.setattr('inspection.artifacts.load_registry', lambda *args: {'schema_version': 1, 'models': [entry]})
    monkeypatch.setattr('inspection.artifacts.resolve_checkpoint', lambda *args: pytest.fail('Presentation must not load weights'))
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=15).run()
    assert not app.exception
    assert not app.json and not app.metric
    captions = '\n'.join(item.value for item in app.caption)
    prose = '\n'.join(item.value for item in app.markdown)
    if kind == 'supervised_segmentation':
        assert 'Real-defect supervised training' in captions
        assert 'Real normal/defective images and annotated defect masks' in captions
        assert 'letterbox padding to 256 × 640 pixels' in captions
        assert 'exclude padding from image scoring' in captions
        assert 'separate supervised surface-inspection task' in prose
        assert 'This selected MVTec model learns' not in prose
    else:
        assert 'Normal-only training' in captions
        assert 'Procedural synthetic defects' in captions
        assert 'square preparation' in captions
        assert 'This selected MVTec model learns' in prose
        assert 'This selected KolektorSDD2 model learns' not in prose
    assert category == app.selectbox[0].value['category']


def test_supervised_presentation_refuses_a_square_checkpoint_claim():
    summary = {'model_kind': 'supervised_segmentation', 'preprocess': {'mode': 'square', 'width': 256, 'height': 256},
               'measured_quality_target_met': False}
    with pytest.raises(ValueError, match='preserve its letterbox preparation'):
        _presentation('kolektor_surface', summary)


def test_legacy_pilot_registry_without_new_metadata_still_renders(monkeypatch):
    entry = {'id': 'legacy-pilot-fixture', 'label': 'Metal nut · exploratory pilot', 'category': 'metal_nut',
             'status_label': 'EXPLORATORY MODEL · TARGET NOT MET',
             'description': 'Compact reconstruction and segmentation model trained from random weights.',
             'conditions': 'Centered metal nuts matching MVTec AD.', 'examples': [], 'metrics': {}}
    monkeypatch.setattr('inspection.artifacts.load_registry', lambda *args: {'schema_version': 1, 'models': [entry]})
    monkeypatch.setattr('inspection.artifacts.resolve_checkpoint', lambda *args: pytest.fail('Presentation must not load weights'))
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=15).run()
    assert not app.exception
    captions = '\n'.join(item.value for item in app.caption)
    assert 'Normal-only training' in captions
    assert 'same orientation, color and resizing rules' in captions
    assert 'letterbox' not in captions
