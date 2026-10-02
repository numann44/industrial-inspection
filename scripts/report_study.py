"""Publish verified, completed study evidence without inference or dataset access.

Only frozen output artifacts and already-rendered gallery panels are read. No
report is written while training, evaluation or conditional fallback is incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_study import BASELINE_KIND, CATEGORIES, MATRIX, SEEDS, freeze_selection_rule, plan, quality_target_met, seed_variability, select_by_validation


def _category(candidate):
    return candidate['config'].get('category', 'kolektor_surface')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root, value):
    path = (root / value).resolve()
    _require(path.is_relative_to(root), 'Evidence path escapes project root')
    return path


def _same(actual, expected, message):
    _require(actual == expected, message)


def _number(value, name, lower=0, upper=1):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and lower <= value <= upper, f'Invalid {name}')


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    _require(parsed.tzinfo is not None, 'Selection/status timestamps need timezone information')
    return parsed


def _metric_check(evaluation, candidate, model_sha):
    from sklearn.metrics import average_precision_score, roc_auc_score
    for key in ('checkpoint_sha256', 'split_digest', 'threshold', 'model_kind'):
        _same(evaluation.get(key), candidate[key], f'Evaluation {key} mismatch')
    _same(evaluation.get('model_sha256'), model_sha, 'Evaluation model tensor digest mismatch')
    _same(evaluation.get('pixel_metric_space'), 'original', 'Publication requires native-mask pixel AP')
    for key in ('image_auroc', 'image_average_precision', 'pixel_average_precision', 'defect_recall', 'normal_false_alarm_rate'):
        _number(evaluation.get(key), key)
    if evaluation.get('precision') is not None:
        _number(evaluation['precision'], 'precision')
    predictions = evaluation.get('predictions', [])
    _require(predictions and len({p['image'] for p in predictions}) == len(predictions), 'Missing or duplicate prediction records')
    category = _category(candidate)
    _require(all(p['category'] == category and p['label'] in (0, 1) for p in predictions), 'Prediction category/label mismatch')
    counts = {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0}
    for prediction in predictions:
        _number(prediction['score'], 'prediction score', lower=-1e30, upper=1e30)
        decision = prediction['score'] >= candidate['threshold']
        _same(prediction['predicted_defective'], decision, 'Prediction decision violates frozen threshold')
        counts[('true_' if decision == bool(prediction['label']) else 'false_') + ('positive' if decision else 'negative')] += 1
    _same(evaluation.get('confusion'), counts, 'Confusion counts disagree with prediction records')
    _same(evaluation.get('images'), len(predictions), 'Image count mismatch')
    positive, negative = counts['true_positive'] + counts['false_negative'], counts['false_positive'] + counts['true_negative']
    _require(positive > 0 and negative > 0, 'Both original test classes must be represented')
    expected = {'defect_recall': counts['true_positive'] / positive,
                'normal_false_alarm_rate': counts['false_positive'] / negative,
                'precision': counts['true_positive'] / (counts['true_positive'] + counts['false_positive']) if counts['true_positive'] + counts['false_positive'] else None,
                'image_auroc': roc_auc_score([p['label'] for p in predictions], [p['score'] for p in predictions]),
                'image_average_precision': average_precision_score([p['label'] for p in predictions], [p['score'] for p in predictions])}
    for key, value in expected.items():
        if value is None:
            _same(evaluation[key], None, f'Inconsistent {key}')
        else:
            _require(math.isclose(evaluation[key], float(value), abs_tol=1e-12), f'Inconsistent {key}')
    uncertainty = evaluation.get('uncertainty', {})
    _same(uncertainty.get('defective_images'), positive, 'Uncertainty defective count mismatch')
    _same(uncertainty.get('normal_images'), negative, 'Uncertainty normal count mismatch')
    for key in ('image_auroc_interval', 'defect_recall_interval', 'normal_false_alarm_rate_interval'):
        interval = uncertainty.get(key)
        _require(isinstance(interval, list) and len(interval) == 2 and 0 <= interval[0] <= interval[1] <= 1, f'Missing/invalid {key}')
    _require(uncertainty.get('bootstrap_samples', 0) > 0, 'Missing bootstrap uncertainty measurement')
    defect_counts = {}
    for prediction in predictions:
        if prediction['label']:
            key = f"{category}/{prediction['defect']}"
            images, detected = defect_counts.get(key, (0, 0))
            defect_counts[key] = images + 1, detected + prediction['predicted_defective']
    _same(set(evaluation.get('recall_by_defect', {})), set(defect_counts), 'Missing defect-type analysis')
    for key, (images, detected) in defect_counts.items():
        _same(evaluation['recall_by_defect'][key], {'images': images, 'detected': detected, 'recall': detected / images}, 'Defect-type analysis mismatch')
    areas = evaluation.get('defect_area_groups', {})
    _require(areas, 'Missing native defect-size analysis')
    groups = areas.get('groups', areas)
    if isinstance(groups, dict):
        groups = [value for value in groups.values() if isinstance(value, dict) and 'images' in value]
    _require(isinstance(groups, list) and sum(group['images'] for group in groups) == positive, 'Defect-size groups do not cover defective images')
    _require(sum(group.get('detected', -1) for group in groups) == counts['true_positive'], 'Defect-size detected counts mismatch')
    for group in groups:
        _require(isinstance(group['images'], int) and 0 <= group.get('detected', -1) <= group['images'], 'Invalid defect-size counts')
        _same(group.get('recall'), group['detected'] / group['images'] if group['images'] else None, 'Defect-size recall mismatch')
    _same(evaluation.get('measured_quality_target_met'), quality_target_met(evaluation), 'Stored operating-target result is inconsistent')
    _same(evaluation.get('experimental_status'), 'exploratory' if category == 'metal_nut' else 'frozen-held-out', 'Held-out/exploratory label mismatch')


def verify_study(root, study='outputs/study-v2'):
    """Return verified artifacts; never read original dataset images or masks.

    Public keys: selected, selected_evaluations, results, evidence_sha256, portable.
    Private gallery_sources contains only pre-rendered derivative panel paths.
    """
    root = Path(root).resolve()
    folder = _inside(root, study)
    evidence = {}

    def read(path):
        path = Path(path)
        _require(path.is_file(), f'Incomplete study: missing {path.name}')
        data = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON {value}')))
        evidence[path.relative_to(root).as_posix()] = _hash(path)
        return data

    status = read(folder / 'status.json')
    _require(status.get('state') == 'complete' and status.get('protocol') == 'study-v2', 'Study is not complete')
    _require(status.get('steps') and all(s.get('state') == 'complete' for s in status['steps'].values()), 'Incomplete study steps')
    declaration = read(folder / 'declaration.json')
    for key, value in plan().items():
        _same(declaration.get(key), value, f'Declaration mismatch: {key}')
    _same(declaration.get('runner_sha256'), _hash(root / 'scripts/run_study.py'), 'Declared study runner changed')
    _same(status.get('plan'), plan(), 'Status plan differs from finalized declaration')
    results = read(folder / 'study-results.json')
    _same(results.get('protocol'), 'study-v2', 'Study result protocol mismatch')
    for key in ('selection_used_test_metrics', 'model_and_seed_selection_used_test_metrics'):
        _same(results.get(key), False, 'Test-based model/seed selection cannot be published')
    _same(results.get('dataset_fallback_trigger_uses_frozen_mvtec_test_targets'), True, 'Conditional dataset choice must be disclosed')

    candidates, checkpoint_details = {}, {}

    def verify_candidate(item):
        from inspection.model import create_model
        from inspection.checkpointing import config_digest
        from inspection.engine import model_digest
        import torch
        digest = item['checkpoint_sha256']
        if digest in candidates:
            _same(candidates[digest], item, 'Conflicting frozen candidate metadata')
            return
        path = _inside(root, item['checkpoint'])
        _require(path.is_file() and _hash(path) == digest, 'Frozen checkpoint checksum mismatch')
        evidence[item['checkpoint']] = digest
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        _same(checkpoint.get('resumable'), False, 'A resumable training artifact is not a finalized model')
        _require(not checkpoint.get('smoke_run', False), 'Smoke runs cannot enter the controlled study')
        summary = read(_inside(root, item['run']) / 'summary.json')
        for key in ('model_kind', 'best_epoch', 'split_digest', 'bank_digest', 'threshold', 'config'):
            _same(checkpoint.get(key), item.get(key), f'Checkpoint candidate {key} mismatch')
            _same(summary.get(key), checkpoint.get(key), f'Summary checkpoint {key} mismatch')
        _same(checkpoint.get('best_metric'), item['validation_score'], 'Validation selection score mismatch')
        _same(summary.get('best_metric'), item['validation_score'], 'Summary validation score mismatch')
        _same(checkpoint['config'].get('training_seed', 42), item['seed'], 'Training seed mismatch')
        _same(checkpoint['config'].get('threshold_quantile'), .90, 'Calibration quantile must stay predeclared at 90%')
        if item['model_kind'] != 'supervised_segmentation':
            _require(bool(item.get('bank_digest')), 'Missing common synthetic validation bank digest')
        _same(checkpoint['provenance']['config_digest'], item['config_digest'], 'Config provenance mismatch')
        _same(config_digest(checkpoint['config']), item['config_digest'], 'Config digest does not match configuration')
        model = create_model(item['model_kind'], checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state'], strict=True)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if 'parameters' in summary:
            _same(summary['parameters'], parameters, 'Declared parameter count differs from actual model')
        checkpoint_details[digest] = {'parameters': parameters, 'model_sha256': model_digest(checkpoint['model_state']),
                                     'provenance': checkpoint['provenance'], 'calibration_count': checkpoint['calibration'].get('count', checkpoint['calibration'].get('normal_count'))}
        _require(isinstance(checkpoint_details[digest]['calibration_count'], int) and checkpoint_details[digest]['calibration_count'] > 0, 'Missing normal calibration count')
        _number(item['threshold'], 'frozen threshold', upper=1e30)
        calibration = checkpoint['calibration']
        _same(summary.get('calibration'), calibration, 'Summary calibration evidence mismatch')
        _same(calibration.get('quantile'), .90, 'Checkpoint calibration quantile mismatch')
        _same(calibration.get('quantile_method'), 'linear', 'Checkpoint calibration interpolation mismatch')
        scores = calibration.get('scores', [])
        _same(len(scores), checkpoint_details[digest]['calibration_count'], 'Calibration scores/count mismatch')
        import numpy as np
        _require(np.isfinite(scores).all() and math.isclose(float(np.quantile(scores, .90, method='linear')), item['threshold'], abs_tol=1e-12), 'Frozen threshold differs from normal calibration quantile')
        candidates[digest] = item

    def selection(name, expected):
        choice = read(folder / 'selections' / f'{name}.json')
        eligible = [item for item in expected if item['model_kind'] != BASELINE_KIND]
        _same(choice.get('candidates'), eligible, 'Selection candidate set differs from frozen checkpoints')
        timings = {item['checkpoint_sha256']: item['cpu_p50_ms'] for item in choice.get('tie_measurements', [])}
        replay = select_by_validation(expected, lambda item: timings[item['checkpoint_sha256']])
        for key, value in replay.items():
            _same(choice.get(key), value, f'Validation-only selection mismatch: {key}')
        _timestamp(choice['frozen_at_utc'])
        return choice

    freeze = read(folder / 'mvtec-pretest-freeze.json')
    _same(freeze.get('protocol'), 'study-v2', 'Pretest freeze protocol mismatch')
    _same(freeze.get('test_selection'), False, 'Pretest freeze used test selection')
    _same(freeze.get('selection_rule'), freeze_selection_rule(), 'Pretest freeze selection rule mismatch')
    _same(freeze.get('selection_rule'), declaration['selection'], 'Pretest freeze differs from declared synthetic validation rule')
    frozen = freeze.get('checkpoints', [])
    _require(len(frozen) == 14 and len({c['checkpoint_sha256'] for c in frozen}) == 14, 'Expected all14 MVTec checkpoint candidates')
    for item in frozen:
        verify_candidate(item)
    initial = [c for c in frozen if Path(c['run']).name.startswith('protocol-v2-')]
    _same({Path(c['run']).name for c in initial}, {f'protocol-v2-{name}-seed42' for name, _ in MATRIX}, 'Incomplete six-variant ablation matrix')
    _require(all(c['seed'] == 42 and _category(c) == 'metal_nut' for c in initial), 'Ablation matrix category/seed mismatch')
    main = selection('main-variant', initial)
    selected, selected_evaluations, analyses, galleries, choices = {}, {}, {}, {}, {'main-variant': main}
    category_candidates = {}
    initial_hashes = {c['checkpoint_sha256'] for c in initial}
    for category in CATEGORIES:
        group = [c for c in frozen if _category(c) == category and
                 (c['checkpoint_sha256'] not in initial_hashes or c == main['selected'])]
        _same(sorted(c['seed'] for c in group), list(SEEDS), 'Missing or duplicate category training seeds')
        _require(len({c['model_kind'] for c in group}) == 1, 'Different methods mixed in seed variability')
        def method_config(c):
            return {k: v for k, v in c['config'].items() if k not in {'training_seed', 'category', 'device'}}
        _require(all(method_config(c) == method_config(main['selected']) for c in group), 'Category/seed method differs from validation-selected configuration')
        choice = selection(f'deployment-{category}', group)
        choices[category] = choice
        selected[category] = choice['selected']
        category_candidates[category] = group
    _same(freeze.get('deployments'), selected, 'Pretest deployments differ from validation selections')
    _same(results.get('pretest_selections'), selected, 'Final deployments changed after pretest freeze')

    def evaluation(item):
        value = read(folder / 'evaluations' / f"{Path(item['run']).name}.json")
        _metric_check(value, item, checkpoint_details[item['checkpoint_sha256']]['model_sha256'])
        step = status['steps'].get(f"evaluate-{Path(item['run']).name}", {})
        start = _timestamp(step['started_at'])
        selection_times = [choice['frozen_at_utc'] for name, choice in choices.items() if name != 'ksdd2']
        _require(all(start >= _timestamp(time) for time in selection_times), 'Test evaluation started before model/seed selection freeze')
        return value

    evaluations = {c['checkpoint_sha256']: evaluation(c) for c in frozen}
    _same(results.get('all_mvtec_evaluations'), evaluations, 'Final summary differs from evaluation artifacts')
    for category, group in category_candidates.items():
        record_ids = [{(p['image'], p['label'], p['defect']) for p in evaluations[c['checkpoint_sha256']]['predictions']} for c in group]
        _require(all(ids == record_ids[0] for ids in record_ids), 'Training seeds evaluated on different image records')
        _same(results['mvtec_training_seed_variability'][category], seed_variability(group, evaluations), 'Training-seed variability mismatch')
        value = evaluations[selected[category]['checkpoint_sha256']]
        selected_evaluations[category] = value
        _same(results['mvtec_selected_metrics'][category], value, 'Selected metrics differ from pretest-selected checkpoint')
        _same(results['mvtec_target_met'][category], quality_target_met(value), 'Selected category operating target mismatch')
    fallback = results.get('conditional_supervised', {})
    triggered = not any(results['mvtec_target_met'].values())
    _same(fallback.get('triggered'), triggered, 'Conditional supervised fallback trigger mismatch')
    if triggered:
        ksdd_freeze = read(folder / 'ksdd2-pretest-freeze.json')
        _same(ksdd_freeze.get('protocol'), 'study-v2', 'KSDD2 freeze protocol mismatch')
        _same(ksdd_freeze.get('test_selection'), False, 'KSDD2 test selection forbidden')
        _same(ksdd_freeze.get('selection_rule'), freeze_selection_rule(supervised=True), 'Supervised pretest freeze selection rule must describe real-defect validation')
        group = ksdd_freeze.get('checkpoints', [])
        _same(sorted(c['seed'] for c in group), list(SEEDS), 'Incomplete KSDD2 three-seed fallback')
        _require(all(c['model_kind'] == 'supervised_segmentation' and _category(c) == 'kolektor_surface' for c in group), 'KSDD2 model/category mismatch')
        for item in group:
            verify_candidate(item)
        choice = selection('deployment-ksdd2', group)
        choices['ksdd2'] = choice
        selected['kolektor_surface'] = choice['selected']
        _same(ksdd_freeze.get('deployments'), {'kolektor_surface': choice['selected']}, 'KSDD2 frozen deployment mismatch')
        _same(fallback.get('selected'), choice['selected'], 'KSDD2 final deployment mismatch')
        values = {}
        for item in group:
            value = evaluation(item)
            step = status['steps'][f"evaluate-{Path(item['run']).name}"]
            _require(_timestamp(step['started_at']) >= _timestamp(choice['frozen_at_utc']), 'KSDD2 tests started before selection freeze')
            values[item['checkpoint_sha256']] = value
        _same(fallback.get('evaluations'), values, 'KSDD2 final evaluations mismatch')
        _same(fallback.get('training_seed_variability'), seed_variability(group, values), 'KSDD2 seed variability mismatch')
        selected_evaluations['kolektor_surface'] = values[choice['selected']['checkpoint_sha256']]
        _same(fallback.get('measured_quality_target_met'), quality_target_met(selected_evaluations['kolektor_surface']), 'KSDD2 operating target mismatch')
        evaluations.update(values)
        category_candidates['kolektor_surface'] = group
    _same(results.get('any_validation_selected_model_met_dataset_target'), any(quality_target_met(v) for v in selected_evaluations.values()), 'Study target summary mismatch')
    required_steps = {f'initial-{name}' for name, _ in MATRIX}
    required_steps.update(f'{category}-seed{seed}' for category in CATEGORIES for seed in SEEDS
                          if category != 'metal_nut' or seed != 42)
    required_steps.update(f"evaluate-{Path(c['run']).name}" for c in candidates.values())
    required_steps.update(f'analyze-{category}' for category in CATEGORIES)
    if triggered:
        required_steps.update(f'ksdd2-seed{seed}' for seed in SEEDS)
        required_steps.add('analyze-ksdd2')
    _require(required_steps.issubset(status['steps']), 'Missing declared training/evaluation/analysis completion steps')

    for category, item in selected.items():
        analysis = read(folder / 'analyses' / f'{category}.json')
        _same(analysis.get('checkpoint_sha256'), item['checkpoint_sha256'], 'Analysis deployment checksum mismatch')
        stress = analysis['stress']
        for key in ('checkpoint_sha256', 'split_digest', 'threshold'):
            _same(stress.get(key), item[key], f'Stress {key} mismatch')
        _same(set(stress.get('results', {})), {'original', 'gaussian_blur_radius_1', 'brightness_0.8', 'brightness_1.2', 'jpeg_quality_60'}, 'Incomplete paired stress analysis')
        _same(stress['results']['original']['confusion'], selected_evaluations[category]['confusion'], 'Stress original decisions mismatch')
        for name, value in stress['results'].items():
            _same(value.get('threshold'), item['threshold'], 'Stress retuned the frozen threshold')
            _same(value.get('images'), selected_evaluations[category]['images'], 'Stress image count mismatch')
            _require(0 <= value.get('decision_flips', -1) <= value['images'], 'Invalid stress decision-flip count')
        _require('cpu' in analysis.get('latency', {}), 'Missing selected-model CPU timing')
        for device, value in analysis['latency'].items():
            _same(value.get('checkpoint_sha256'), item['checkpoint_sha256'], 'Timing checkpoint mismatch')
            _same(value.get('model_sha256'), checkpoint_details[item['checkpoint_sha256']]['model_sha256'], 'Timing model tensor mismatch')
            _require(value.get('measurements', 0) > 0 and value.get('warmup_forwards', 0) > 0, 'Timing warmup/measurements missing')
            _number(value['single_image_median_ms'], 'timing median', upper=1e9)
            _require(value['single_image_p95_ms'] >= value['single_image_median_ms'] > 0, 'Invalid timing percentiles')
        analyses[category] = analysis
        gallery_folder = folder / 'galleries' / Path(item['run']).name
        gallery = read(gallery_folder / 'index.json')
        _same(gallery.get('threshold'), item['threshold'], 'Gallery frozen threshold mismatch')
        _require(gallery.get('items'), 'Missing selected gallery panels')
        attribution = gallery_folder / 'ATTRIBUTION.md'
        _require(attribution.is_file() and 'CC BY-NC-SA' in attribution.read_text(), 'Missing derivative gallery attribution')
        evidence[attribution.relative_to(root).as_posix()] = _hash(attribution)
        _require(selected_evaluations[category].get('gallery') is not None, 'Selected evaluation lacks gallery evidence')
        _same(selected_evaluations[category]['gallery'], gallery, 'Selected gallery differs from evaluation evidence')
        for panel in gallery['items']:
            image = (gallery_folder / panel['file']).resolve()
            _require(image.is_relative_to(gallery_folder.resolve()) and image.suffix.lower() == '.png' and image.is_file(), 'Missing or unsafe gallery panel path')
            evidence[image.relative_to(root).as_posix()] = _hash(image)
        galleries[category] = gallery_folder

    def compact(item):
        value = evaluations[item['checkpoint_sha256']]
        detail = checkpoint_details[item['checkpoint_sha256']]
        return {key: item[key] for key in ('checkpoint_sha256', 'model_kind', 'validation_score', 'best_epoch', 'bank_digest', 'split_digest', 'seed', 'threshold')} | {
            'category': _category(item), 'run': Path(item['run']).name,
            'parameters': detail['parameters'], 'calibration_count': detail['calibration_count'],
            'training_provenance': {key: detail['provenance'][key] for key in ('source_files', 'supervised_source_files', 'packages', 'python', 'git_revision', 'git_dirty', 'config_digest') if key in detail['provenance']},
            'method_config': {k: v for k, v in item['config'].items() if k in {'image_size', 'base_channels', 'restrict_foreground', 'scratch_enabled', 'synthesis', 'noise_std', 'epochs', 'learning_rate'}},
            'metrics': {k: value[k] for k in ('image_auroc', 'image_average_precision', 'pixel_average_precision', 'confusion', 'defect_recall', 'normal_false_alarm_rate', 'precision', 'uncertainty', 'recall_by_defect', 'defect_area_groups', 'experimental_status')},
            'target_met': quality_target_met(value)}

    portable = {'protocol': 'study-v2', 'declaration': declaration,
                'candidate_ablations': [compact(c) for c in initial],
                'categories': {category: {'selected': compact(item), 'all_seeds': [compact(c) for c in category_candidates[category]],
                    'seed_variability': seed_variability(category_candidates[category], evaluations),
                    'selection_rule': choices['ksdd2' if category == 'kolektor_surface' else category]['rule'],
                    'pretest_freeze_selection_rule': freeze_selection_rule(supervised=category == 'kolektor_surface'),
                    'validation_source': 'real labeled normal/defective KSDD2 validation images and masks' if category == 'kolektor_surface' else 'common synthetic challenge generated exclusively from held-out normal validation images',
                    'tie_measurements': choices['ksdd2' if category == 'kolektor_surface' else category]['tie_measurements'],
                    'stress': analyses[category]['stress'], 'latency': analyses[category]['latency'],
                    'latency_system_state': analyses[category].get('latency_system_state'),
                    'gallery': f'controlled-study-galleries/{category}/index.json'} for category, item in selected.items()},
                'main_variant_selection': {'run': Path(main['selected']['run']).name, 'rule': main['rule'], 'validation_score': main['selected']['validation_score'], 'tie_measurements': main['tie_measurements']},
                'conditional_supervised_triggered': triggered, 'evidence_sha256': evidence,
                'limitations': ['Metal-nut tests were inspected in exploratory development.', 'Screw/transistor frozen-held-out evaluations are category-specific, not production guarantees.', 'KSDD2 is a separate real-defect supervised task conditionally chosen after MVTec target results.', 'Image intervals and three-seed variability describe different sources of uncertainty.', 'Native map interpolation cannot recover defect evidence lost at model input resolution.', 'Serialized timing cannot verify external system idleness; warm model-only time excludes preprocessing and serving.']}
    return {'selected': selected, 'selected_evaluations': selected_evaluations, 'results': results,
            'evidence_sha256': evidence, 'portable': portable, 'gallery_sources': galleries}


def _f(value, digits=3):
    return f'{value:.{digits}f}' if value is not None else 'n/a'


def _ci(value):
    return f'[{value[0]:.3f}, {value[1]:.3f}]'


def render_report(data):
    lines = ['# Controlled study: frozen selections and measured outcomes', '',
             'This report was generated only after every declared training, evaluation and conditional fallback step completed and its evidence passed consistency checks. Metal-nut measurements remain **exploratory**. Screw and transistor use frozen held-out tests after all MVTec weights and model/seed selections were fixed. No test-winning seed replaces a validation-selected seed.', '',
             '## Candidate ablations and validation-only selection', '',
             'Every candidate uses random initialization. The common canonical synthetic validation bank ranks candidates by the harmonic mean of image AUROC and pixel AP; the normal-only reconstruction baseline is reported but excluded from main-model eligibility. Ties within 1e-12 use predeclared CPU median latency, then checkpoint SHA-256. These are controlled protocol comparisons, with capacity differences disclosed by actual parameter counts; synthetic validation does not establish transfer to real defects.', '',
             '| Candidate | Method | Input | Actual parameters | Selected epoch | Validation H | Eligible | Exploratory test AUROC / native AP |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    p = data['portable']
    for row in p['candidate_ablations']:
        lines.append(f"| {row['run']} | {row['model_kind']} | {row['method_config'].get('image_size')} | {row['parameters']:,} | {row['best_epoch']} | {row['validation_score']:.6f} | {'no' if row['model_kind'] == BASELINE_KIND else 'yes'} | {row['metrics']['image_auroc']:.4f} / {row['metrics']['pixel_average_precision']:.4f} |")
    main = p['main_variant_selection']
    lines += ['', f"The main configuration was selected as `{main['run']}` using validation H={main['validation_score']:.6f}. Its later test outcomes did not choose this configuration. Deployment seeds use the same validation-only rule separately within each category. Full hashes, ties, size/type measurements and all-seed uncertainty records are in [controlled-study.json](controlled-study.json).", '',
              '## MVTec: all training seeds and frozen operating thresholds', '',
              '| Category | Seed | Pretest selected | Image AUROC [95% image CI] | Native pixel AP | Defects detected | Good parts falsely flagged | Recall / FPR | Target |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    def result_row(category, row, selected):
        m, u = row['metrics'], row['metrics']['uncertainty']
        c = m['confusion']
        return f"| {category} | {row['seed']} | {'yes' if selected else 'no'} | {m['image_auroc']:.4f} {_ci(u['image_auroc_interval'])} | {m['pixel_average_precision']:.4f} | {c['true_positive']}/{c['true_positive']+c['false_negative']} | {c['false_positive']}/{c['false_positive']+c['true_negative']} | {m['defect_recall']:.1%} {_ci(u['defect_recall_interval'])} / {m['normal_false_alarm_rate']:.1%} {_ci(u['normal_false_alarm_rate_interval'])} | {'met' if row['target_met'] else '**failed**'} |"
    for category in CATEGORIES:
        group = p['categories'][category]
        for row in group['all_seeds']:
            lines.append(result_row(category, row, row['checkpoint_sha256'] == group['selected']['checkpoint_sha256']))
    lines += ['', 'The target requires recall ≥90% **and** normal false alarms ≤10%, using each checkpoint’s unchanged 90th-percentile normal-calibration threshold. Nonselected seeds remain descriptive evidence; they cannot rescue a failed selected checkpoint. Test point estimates do not establish factory reliability. Defective prevalence and the small normal sample affect precision and false-alarm uncertainty.', '',
              '## Training-seed variation and image uncertainty', '',
              'Each row reports the mean ± sample standard deviation across seeds 42/43/44. This is descriptive training variability, not a confidence interval; image-bootstrap and Wilson rate intervals in the JSON instead describe finite-image sampling uncertainty.', '',
              '| Category | AUROC | Native pixel AP | Defect recall | Normal false-alarm rate |', '| --- | --- | --- | --- | --- |']
    for category, group in p['categories'].items():
        values = group['seed_variability']['metrics']
        lines.append('| ' + category + ' | ' + ' | '.join(f"{_f(values[key]['mean'])} ± {_f(values[key]['sample_standard_deviation'])}" for key in ('image_auroc', 'pixel_average_precision', 'defect_recall', 'normal_false_alarm_rate')) + ' |')
    lines += ['', '## Native defect size and type: validation-selected checkpoints', '',
              '| Category / annotated type | Detected / defective images | Recall |', '| --- | --- | --- |']
    for category, group in p['categories'].items():
        for name, value in group['selected']['metrics']['recall_by_defect'].items():
            lines.append(f"| {name} | {value['detected']}/{value['images']} | {value['recall']:.1%} |")
    lines += ['', '| Category / native-mask area bin | Detected / defective images | Recall [95% Wilson interval] |', '| --- | --- | --- |']
    for category, group in p['categories'].items():
        areas = group['selected']['metrics']['defect_area_groups']
        for name, value in areas.get('groups', areas).items():
            if isinstance(value, dict) and 'images' in value:
                interval = _ci(value['recall_interval']) if value.get('recall_interval') else 'n/a'
                lines.append(f"| {category} / {name} | {value['detected']}/{value['images']} | {_f(value['recall'])} {interval} |")
    lines += ['', 'Defect-size analyses use original annotation areas. The JSON retains every area bin and its recall interval; zero-image bins carry no recall estimate. A small defect can lose visual evidence during input resizing even though its original mask remains intact for evaluation.', '',
              '## Paired perturbations and timing', '',
              '| Category | Perturbation | Recall | False alarms | Decisions changed |', '| --- | --- | --- | --- | --- |']
    for category, group in p['categories'].items():
        for name, value in group['stress']['results'].items():
            lines.append(f"| {category} | {name} | {value['defect_recall']:.1%} | {value['normal_false_alarm_rate']:.1%} | {value['decision_flips']} |")
    lines += ['', 'The same test images receive predeclared blur, brightness and JPEG perturbations, with no threshold adjustment or retraining. These paired image-decision checks do not establish generalization to new cameras; pixel AP is not computed for the stress runs.', '',
              '| Category | Device | Warm median / p95 (ms) | Measurements |', '| --- | --- | --- | --- |']
    for category, group in p['categories'].items():
        for device, value in group['latency'].items():
            lines.append(f"| {category} | {device} | {value['single_image_median_ms']:.2f} / {value['single_image_p95_ms']:.2f} | {value['measurements']} |")
    lines += ['', 'Timing uses the selected frozen model and a normal calibration image after scheduler serialization. The benchmark cannot enforce external system idleness. These warm forward-and-score measurements exclude checkpoint loading, decoding, preprocessing, transfers, rendering and service overhead.', '', '## Conditional KSDD2 track', '']
    if p['conditional_supervised_triggered']:
        lines += ['The predeclared fallback ran because no validation-selected MVTec category met both targets. KSDD2 trains on **real labeled defects** and selects a model/seed by real validation metrics. Its results belong to a separate supervised task; they are not an anomaly-detection ablation or evidence of MVTec transfer. Dataset choice was conditional on the frozen MVTec test target results, while KSDD2 model and seed selection remained validation-only.', '', '| Category | Seed | Pretest selected | Image AUROC [95% image CI] | Native pixel AP | Defects detected | Good parts falsely flagged | Recall / FPR | Target |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        group = p['categories']['kolektor_surface']
        for row in group['all_seeds']:
            lines.append(result_row('KSDD2', row, row['checkpoint_sha256'] == group['selected']['checkpoint_sha256']))
    else:
        lines += ['The conditional KSDD2 fallback was not triggered. No supervised fallback results are invented or implied.']
    lines += ['', '## Selected galleries, attribution and provenance', '', 'Galleries retain existing deterministic FP/FN/TP/TN selections. Original images, fixed-scale predicted activation and ground truth occupy separate panels; activations are not calibrated probabilities or pixel decisions. No original dataset images were reopened to build this report.', '']
    for category in p['categories']:
        lines.append(f'- [{category} gallery index](controlled-study-galleries/{category}/index.json) and [CC BY-NC-SA 4.0 attribution](controlled-study-galleries/{category}/ATTRIBUTION.md).')
    lines += ['', 'Public evidence uses relative paths and SHA-256 checksums. Checkpoints and raw datasets stay outside the source repository. All limitations, failed targets and nonselected seeds remain in the record; this report does not change the released demo model or claim production readiness.', '']
    return '\n'.join(lines)


def publish_study(root, study='outputs/study-v2', output='docs/results'):
    root = Path(root).resolve()
    data = verify_study(root, study)  # Validate everything before creating public output.
    destination = _inside(root, output)
    public = json.dumps(data['portable'], indent=2, allow_nan=False) + '\n'
    _require(str(root) not in public and '/Users/' not in public, 'Public evidence contains local absolute paths')
    for relative, digest in data['evidence_sha256'].items():
        _same(_hash(_inside(root, relative)), digest, 'Study evidence changed after verification')
    destination.mkdir(parents=True, exist_ok=True)
    for category, source in data['gallery_sources'].items():
        target = destination / 'controlled-study-galleries' / category
        target.mkdir(parents=True, exist_ok=True)
        index = json.loads((source / 'index.json').read_text())
        for name in ['index.json', 'ATTRIBUTION.md'] + [panel['file'] for panel in index['items']]:
            shutil.copy2(source / name, target / name)
    for relative, digest in data['evidence_sha256'].items():
        _same(_hash(_inside(root, relative)), digest, 'Study evidence changed during publication')
    (destination / 'controlled-study.json').write_text(public)
    (destination / 'CONTROLLED_STUDY.md').write_text(render_report(data))
    return {'report': str(destination / 'CONTROLLED_STUDY.md'), 'categories': list(data['selected'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--study', default='outputs/study-v2')
    parser.add_argument('--output', default='docs/results')
    args = parser.parse_args()
    print(json.dumps(publish_study(args.root, args.study, args.output), indent=2))


if __name__ == '__main__':
    main()
