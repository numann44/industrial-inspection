"""Complete temporary evidence fixtures; never train, infer or read datasets."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.checkpointing import config_digest
from inspection.engine import model_digest
from inspection.model import create_model
from inspection.reporting import image_uncertainty
from scripts.report_study import publish_study, verify_study
from scripts.run_study import BASELINE_KIND, MATRIX, freeze_selection_rule, plan, quality_target_met, seed_variability, select_by_validation, sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def finished_fixture(root, fallback=False):
    """Tiny actual checkpoint tensors and result JSON confined to tmp_path."""
    study = root / 'outputs/study-v2'
    runner = root / 'scripts/run_study.py'
    runner.parent.mkdir(parents=True)
    runner.write_text('fixture runner: no jobs')
    write(study / 'declaration.json', {**plan(), 'runner_sha256': sha256(runner)})
    status = {'protocol': 'study-v2', 'state': 'complete', 'plan': plan(), 'steps': {}}

    def candidate(name, category, seed, score, kind='joint_reconstruction_segmentation', size=256, scratch=True):
        config = {'model_kind': kind, 'category': category, 'training_seed': seed,
                  'image_size': size, 'base_channels': 2, 'threshold_quantile': .90,
                  'scratch_enabled': scratch, 'device': 'cpu'}
        torch.manual_seed(seed + len(name))
        model = create_model(kind, {'base_channels': 2})
        ck = {'config': config, 'model_kind': kind, 'model_config': {'base_channels': 2},
              'model_state': model.state_dict(), 'best_epoch': 5, 'best_metric': score,
              'threshold': .5, 'split_digest': f'split-{category}', 'bank_digest': f'bank-{category}',
              'resumable': False, 'smoke_run': False,
              'parameters': sum(p.numel() for p in model.parameters()),
              'calibration': {'count': 33, 'quantile': .90, 'quantile_method': 'linear', 'scores': [.5]*33},
              'provenance': {'config_digest': config_digest(config)}}
        folder = root / 'runs' / name
        folder.mkdir(parents=True)
        torch.save(ck, folder / 'checkpoint.pt')
        write(folder / 'summary.json', {k: v for k, v in ck.items() if k != 'model_state'})
        return {'run': f'runs/{name}', 'checkpoint': f'runs/{name}/checkpoint.pt',
                'checkpoint_sha256': sha256(folder / 'checkpoint.pt'), 'model_kind': kind,
                'best_epoch': 5, 'validation_score': score, 'split_digest': ck['split_digest'],
                'bank_digest': ck['bank_digest'], 'config_digest': config_digest(config),
                'config': config, 'seed': seed, 'threshold': .5, 'manifest': 'data/metadata-only.json'}

    def choice(name, group):
        selected = {**select_by_validation(group), 'frozen_at_utc': '2026-10-02T10:00:00+00:00'}
        write(study / 'selections' / f'{name}.json', selected)
        return selected['selected']

    initial = []
    for index, (name, _) in enumerate(MATRIX):
        kind = BASELINE_KIND if name == 'reconstruction-256' else 'segmentation_only' if name == 'segmentation-256' else 'joint_reconstruction_segmentation'
        score = .99 if kind == BASELINE_KIND else .95 if name == 'joint-256' else .70 - index * .01
        initial.append(candidate(f'protocol-v2-{name}-seed42', 'metal_nut', 42, score, kind,
                                 size=128 if name == 'joint-128' else 256, scratch=name != 'no-scratch-256'))
        status['steps'][f'initial-{name}'] = {'state': 'complete'}
    main = choice('main-variant', initial)
    groups = {'metal_nut': [main] + [candidate(f'study-v2-metal_nut-seed{s}', 'metal_nut', s, .97 if s == 43 else .90) for s in (43, 44)]}
    for category in ('screw', 'transistor'):
        groups[category] = [candidate(f'study-v2-{category}-seed{s}', category, s, .97 if s == 43 else .90) for s in (42, 43, 44)]
    for category in groups:
        for seed in (42, 43, 44):
            if category != 'metal_nut' or seed != 42:
                status['steps'][f'{category}-seed{seed}'] = {'state': 'complete'}
        status['steps'][f'analyze-{category}'] = {'state': 'complete'}
    selected = {category: choice(f'deployment-{category}', group) for category, group in groups.items()}
    frozen = initial + groups['metal_nut'][1:] + groups['screw'] + groups['transistor']
    write(study / 'mvtec-pretest-freeze.json', {'protocol': 'study-v2', 'checkpoints': frozen,
          'deployments': selected, 'selection_rule': plan()['selection'], 'test_selection': False})

    def evaluation(c):
        # Seed44 can beat selected43 on test; selection must never change.
        good = .1
        scores = [good, .9, .1 if fallback and c['seed'] != 44 else .8]
        labels = [0, 1, 1]
        tp = sum(s >= .5 for s in scores[1:])
        confusion = {'true_positive': tp, 'false_positive': 0, 'true_negative': 1, 'false_negative': 2-tp}
        from sklearn.metrics import average_precision_score, roc_auc_score
        category = c['config']['category']
        ck = torch.load(root / c['checkpoint'], weights_only=True)
        preds = [{'image': f'/reserved/{category}/{i}.png', 'label': label, 'category': category,
                  'defect': 'good' if not label else 'scratch', 'score': s, 'predicted_defective': s >= .5}
                 for i, (label, s) in enumerate(zip(labels, scores))]
        m = {'checkpoint_sha256': c['checkpoint_sha256'], 'model_sha256': model_digest(ck['model_state']),
             'split_digest': c['split_digest'], 'threshold': .5, 'model_kind': c['model_kind'],
             'pixel_metric_space': 'original', 'images': 3, 'image_auroc': roc_auc_score(labels, scores),
             'image_average_precision': average_precision_score(labels, scores), 'pixel_average_precision': .7,
             'confusion': confusion, 'defect_recall': tp/2, 'normal_false_alarm_rate': 0, 'precision': 1,
             'predictions': preds, 'uncertainty': image_uncertainty(labels, scores, .5, bootstrap_samples=10),
             'recall_by_defect': {f'{category}/scratch': {'images': 2, 'detected': tp, 'recall': tp/2}},
             'defect_area_groups': {'groups': {'small': {'images': 2, 'detected': tp, 'recall': tp/2}}},
             'experimental_status': 'exploratory' if category == 'metal_nut' else 'frozen-held-out'}
        m['measured_quality_target_met'] = quality_target_met(m)
        if c in selected.values():
            folder = study / 'galleries' / Path(c['run']).name
            folder.mkdir(parents=True)
            Image.new('RGB', (3, 2), 'white').save(folder / 'panel.png')
            gallery = {'threshold': .5, 'items': [{'file': 'panel.png', 'group': 'TP'}]}
            write(folder / 'index.json', gallery)
            (folder / 'ATTRIBUTION.md').write_text('Fixture derivative credit: CC BY-NC-SA 4.0')
            m['gallery'] = gallery
        write(study / 'evaluations' / f"{Path(c['run']).name}.json", m)
        status['steps'][f"evaluate-{Path(c['run']).name}"] = {'state': 'complete', 'started_at': '2026-10-02T11:00:00+00:00'}
        return m

    evaluations = {c['checkpoint_sha256']: evaluation(c) for c in frozen}
    selected_metrics = {category: evaluations[c['checkpoint_sha256']] for category, c in selected.items()}
    targets = {category: quality_target_met(m) for category, m in selected_metrics.items()}
    conditional = {'triggered': not any(targets.values())}
    if fallback:
        ksdd = [candidate(f'study-v2-ksdd2-seed{s}', 'kolektor_surface', s, .98 if s == 43 else .9, 'supervised_segmentation') for s in (42, 43, 44)]
        for seed in (42, 43, 44):
            status['steps'][f'ksdd2-seed{seed}'] = {'state': 'complete'}
        status['steps']['analyze-ksdd2'] = {'state': 'complete'}
        kchoice = choice('deployment-ksdd2', ksdd)
        selected['kolektor_surface'] = kchoice
        write(study / 'ksdd2-pretest-freeze.json', {'protocol': 'study-v2', 'checkpoints': ksdd,
              'deployments': {'kolektor_surface': kchoice}, 'test_selection': False,
              'selection_rule': freeze_selection_rule(supervised=True)})
        kevals = {c['checkpoint_sha256']: evaluation(c) for c in ksdd}
        conditional.update(selected=kchoice, evaluations=kevals,
            training_seed_variability=seed_variability(ksdd, kevals),
            measured_quality_target_met=quality_target_met(kevals[kchoice['checkpoint_sha256']]))
        evaluations.update(kevals)
    for category, c in selected.items():
        m = evaluations[c['checkpoint_sha256']]
        stress = {'checkpoint_sha256': c['checkpoint_sha256'], 'split_digest': c['split_digest'], 'threshold': .5,
                  'results': {name: {'confusion': m['confusion'], 'threshold': .5, 'images': 3,
                                    'decision_flips': 0, 'defect_recall': m['defect_recall'], 'normal_false_alarm_rate': 0}
                              for name in ('original', 'gaussian_blur_radius_1', 'brightness_0.8', 'brightness_1.2', 'jpeg_quality_60')}}
        timing = {'checkpoint_sha256': c['checkpoint_sha256'], 'model_sha256': m['model_sha256'],
                  'measurements': 32, 'warmup_forwards': 3, 'single_image_median_ms': 10,
                  'single_image_p95_ms': 15, 'device': 'cpu'}
        write(study / 'analyses' / f'{category}.json', {'checkpoint_sha256': c['checkpoint_sha256'],
              'stress': stress, 'latency': {'cpu': timing}, 'latency_system_state': 'caller must keep other compute idle'})
    mvtec_selected = {k: v for k, v in selected.items() if k != 'kolektor_surface'}
    write(study / 'study-results.json', {'protocol': 'study-v2', 'pretest_selections': mvtec_selected,
          'mvtec_target_met': targets, 'mvtec_selected_metrics': selected_metrics,
          'all_mvtec_evaluations': {c['checkpoint_sha256']: evaluations[c['checkpoint_sha256']] for c in frozen},
          'mvtec_training_seed_variability': {category: seed_variability(group, evaluations) for category, group in groups.items()},
          'conditional_supervised': conditional,
          'any_validation_selected_model_met_dataset_target': any(targets.values()) or conditional.get('measured_quality_target_met', False),
          'selection_used_test_metrics': False, 'model_and_seed_selection_used_test_metrics': False,
          'dataset_fallback_trigger_uses_frozen_mvtec_test_targets': True})
    write(study / 'status.json', status)
    return study


def test_finished_study_publishes_real_parameter_counts_and_validation_choices(tmp_path, monkeypatch):
    study = finished_fixture(tmp_path)
    monkeypatch.setattr(Image, 'open', lambda *args, **kwargs: pytest.fail('Reporting must not decode original dataset images'))
    existing = tmp_path / 'docs/results/EXPERIMENT_REPORT.md'
    existing.parent.mkdir(parents=True)
    existing.write_text('Existing pilot report stays intact')
    result = publish_study(tmp_path)
    assert Path(result['report']).is_file()
    assert existing.read_text() == 'Existing pilot report stays intact'
    data = json.loads((existing.parent / 'controlled-study.json').read_text())
    assert len(data['candidate_ablations']) == 6
    assert all(group['selected']['seed'] == 43 for group in data['categories'].values())
    assert all(len(group['all_seeds']) == 3 for group in data['categories'].values())
    kinds = {row['model_kind']: row['parameters'] for row in data['candidate_ablations']}
    assert kinds['joint_reconstruction_segmentation'] > kinds[BASELINE_KIND]
    assert '/reserved/' not in (existing.parent / 'controlled-study.json').read_text()
    assert not (tmp_path / 'reserved').exists()
    assert (existing.parent / 'controlled-study-galleries/screw/ATTRIBUTION.md').is_file()


def test_conditional_supervised_track_is_separate_and_keeps_test_winning_seed_unselected(tmp_path):
    finished_fixture(tmp_path, fallback=True)
    verified = verify_study(tmp_path)
    data = verified['portable']
    assert data['conditional_supervised_triggered']
    assert data['categories']['kolektor_surface']['selected']['seed'] == 43
    assert not data['categories']['kolektor_surface']['selected']['target_met']
    assert next(r for r in data['categories']['kolektor_surface']['all_seeds'] if r['seed'] == 44)['target_met']
    assert not any(data['categories'][c]['selected']['target_met'] for c in ('metal_nut', 'screw', 'transistor'))
    assert 'real labeled' in data['categories']['kolektor_surface']['validation_source']
    assert 'synthetic' not in data['categories']['kolektor_surface']['pretest_freeze_selection_rule']
    assert 'synthetic challenge' in data['categories']['screw']['validation_source']


@pytest.mark.parametrize('wrong_rule', [None, 'synthetic validation harmonic score', 'minimum validation loss'])
def test_supervised_freeze_rejects_synthetic_or_undeclared_validation_claim(tmp_path, wrong_rule):
    study = finished_fixture(tmp_path, fallback=True)
    path = study / 'ksdd2-pretest-freeze.json'
    value = json.loads(path.read_text())
    if wrong_rule is None:
        del value['selection_rule']
    else:
        value['selection_rule'] = wrong_rule
    write(path, value)
    with pytest.raises(ValueError, match='Supervised pretest freeze selection rule'):
        publish_study(tmp_path)
    assert not (tmp_path / 'docs/results/CONTROLLED_STUDY.md').exists()


@pytest.mark.parametrize('mutation,match', [
    ('incomplete', 'not complete'), ('selection', 'Validation-only selection'),
    ('evaluation', 'threshold mismatch'), ('checkpoint', 'checksum mismatch'),
    ('analysis', 'Stress threshold mismatch'), ('gallery', 'Missing or unsafe gallery'),
    ('seed', 'seed variability mismatch'), ('count', 'Confusion counts'),
    ('chronology', 'before model/seed selection freeze'), ('steps', 'Missing declared'),
    ('parameters', 'parameter count'), ('native', 'native-mask pixel AP'),
    ('calibration', 'Summary calibration evidence mismatch'),
])
def test_refuses_incomplete_or_mismatched_evidence_before_writing_public_results(tmp_path, mutation, match):
    study = finished_fixture(tmp_path)
    if mutation == 'incomplete':
        path = study / 'status.json'
        value = json.loads(path.read_text()); value['state'] = 'running'
    elif mutation == 'selection':
        path = study / 'selections/deployment-screw.json'
        value = json.loads(path.read_text()); value['selected'] = value['candidates'][0]
    elif mutation == 'chronology':
        path = study / 'status.json'
        value = json.loads(path.read_text())
        key = next(key for key in value['steps'] if key.startswith('evaluate-'))
        value['steps'][key]['started_at'] = '2026-10-02T09:00:00+00:00'
    elif mutation == 'steps':
        path = study / 'status.json'
        value = json.loads(path.read_text()); del value['steps']['analyze-transistor']
    elif mutation in {'parameters', 'calibration'}:
        path = next((tmp_path / 'runs').glob('*/summary.json'))
        value = json.loads(path.read_text())
        if mutation == 'parameters': value['parameters'] = 1
        else: value['calibration']['quantile'] = .95
    elif mutation == 'checkpoint':
        ck = next((tmp_path / 'runs').glob('*/checkpoint.pt'))
        ck.write_bytes(b'tampered')
        path = None
    elif mutation == 'gallery':
        next((study / 'galleries').glob('*/panel.png')).unlink()
        path = None
    elif mutation in {'evaluation', 'count', 'native'}:
        path = next((study / 'evaluations').glob('*.json'))
        value = json.loads(path.read_text())
        if mutation == 'evaluation': value['threshold'] = .1
        elif mutation == 'native': value['pixel_metric_space'] = 'model'
        else: value['confusion']['true_positive'] = 0
    elif mutation == 'analysis':
        path = study / 'analyses/screw.json'
        value = json.loads(path.read_text()); value['stress']['threshold'] = .1
    else:
        path = study / 'study-results.json'
        value = json.loads(path.read_text()); value['mvtec_training_seed_variability']['screw']['metrics']['image_auroc']['mean'] = .01
    if path is not None:
        write(path, value)
    with pytest.raises(ValueError, match=match):
        publish_study(tmp_path)
    assert not (tmp_path / 'docs/results/CONTROLLED_STUDY.md').exists()
    assert not (tmp_path / 'docs/results/controlled-study.json').exists()


def test_zero_flagged_parts_retains_undefined_precision_and_failed_recall():
    from scripts.report_study import _metric_check
    c = {'checkpoint_sha256': 'checkpoint', 'split_digest': 'split', 'threshold': .5,
         'model_kind': 'joint_reconstruction_segmentation', 'config': {'category': 'screw'}}
    labels, scores = [0, 1, 1], [.1, .1, .1]
    m = {**{k: c[k] for k in ('checkpoint_sha256', 'split_digest', 'threshold', 'model_kind')},
         'model_sha256': 'tensor-digest', 'pixel_metric_space': 'original', 'image_auroc': .5,
         'image_average_precision': 2/3, 'pixel_average_precision': .1, 'defect_recall': 0,
         'normal_false_alarm_rate': 0, 'precision': None, 'images': 3,
         'confusion': {'true_positive': 0, 'false_positive': 0, 'true_negative': 1, 'false_negative': 2},
         'predictions': [{'image': f'/never-read/{i}', 'category': 'screw', 'label': label,
                          'defect': 'scratch' if label else 'good', 'score': .1, 'predicted_defective': False}
                         for i, label in enumerate(labels)],
         'uncertainty': image_uncertainty(labels, scores, .5, bootstrap_samples=10),
         'recall_by_defect': {'screw/scratch': {'images': 2, 'detected': 0, 'recall': 0}},
         'defect_area_groups': {'groups': {'tiny': {'images': 2, 'detected': 0, 'recall': 0}}},
         'measured_quality_target_met': False, 'experimental_status': 'frozen-held-out'}
    _metric_check(m, c, 'tensor-digest')
    assert m['precision'] is None
