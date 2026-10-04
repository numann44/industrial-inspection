# Delivery checklist

The project target is a category-specific defect detector trained from random initialization, with measured ≥90% recall and ≤10% normal false alarms, useful spatial analysis, reproducible evidence and a public CPU demo. Quality and engineering completion are assessed separately.

| Stage | Current evidence | Remaining gate |
| --- | --- | --- |
| Data | Full MVTec AD and KolektorSDD2 decoded, hashed and audited | Preserve frozen partitions during all experiments |
| Legacy evidence | Three measured exploratory runs, curves, native galleries and confidence intervals | Retain results unchanged |
| Training reliability | All 17 study-v2 runs completed; interrupted training recovered from compatible atomic checkpoints | Preserve checkpoint/source provenance for separate follow-ups |
| Shared inference | Four published weights verified by SHA; hosted example categories/identities/decisions checked; real upload and four exports agree with local engine | Preserve the inference contract and expose display saturation limits |
| Controlled study | All 14 MVTec runs and frozen evaluations complete; five-way eligible selection and all category seed repeats reported | Selected models fail the recall target; further methods need a new declared experiment |
| Supervised fallback | Three KSDD2 runs complete; validation-selected seed 44 detects 105/110 defects with 89/894 false alarms | Point estimates pass; uncertainty crosses targets and brightness/JPEG robustness fails |
| Demo | All four v0.1.0 models observed; surface upload and four exports verified; malformed and oversized uploads rejected correctly | Separate physical-client check remains unperformed |
| Publication | Public v0.1.0 release at e76ff78; four remote weights match local checksums; main/tag Linux checks pass | Retain honest measured limits while quality work continues |
| Follow-up study v3 | Three 100-epoch runs complete; both candidates fail the validation gate; no model promoted and no conditional runs started | Closed; preserve negative evidence |
| Surface acquisition screen | Both runs and exploratory evaluation complete; validation gate passed; control 104/110 detections with 98/894 false alarms, augmentation 106/110 with 115/894 | Both original-test false-alarm targets fail; preserve mixed robustness evidence, no promotion |
| Learned image-decision experiment | Both runs, frozen evaluation and 24 attributed examples complete; head detects 104/110 with 102/894 false alarms; 198 local tests and Linux checks passed | Original-test FPR target fails; bounded architecture search closed, no promotion |

The [completed study](results/CONTROLLED_STUDY.md) records all candidates and seeds, including failed targets. The [model card](MODEL_CARD.md) separates the passing supervised surface task from the failing normal-only MVTec tasks. Source checks and hosted publication are engineering gates, not evidence of model quality.

## Ownership

- Data audit: sources, full integrity checks, duplicate handling, image-level partition limits and supervised adapters.
- Model/training: models, corruption controls, resumability, serialized runs and validation-only selection.
- Evaluation: shared bank, metrics, uncertainty, error groups, stress tests and comparable timing.
- Integration: inference contract, demo, code review, documentation, clean installation and publication.

Code and analysis can proceed in parallel. Heavy MPS jobs are serialized. Training source/configuration hashes are frozen while resumable experiments are running; a changed method requires a newly declared run rather than quietly continuing old weights.

## Order of execution

1. Preserve existing checkpoints/results and declare protocol v2.
2. Verify resume, data isolation, common scoring and image geometry.
3. Finish the initial bounded matrix; select using the frozen synthetic bank.
4. Repeat the selected configuration at seeds 42/43/44 and select deployment weights without test labels.
5. Train separate screw/transistor checkpoints with the fixed method; freeze selections and evaluate all declared seeds.
6. If the MVTec target is unmet across selected categories, run three supervised KSDD2 seeds and evaluate the frozen selection on its official test.
7. Publish real results, failure cases, model identity and a tested CPU demo. Keep an experimental label when quality gates remain unmet.

Detailed methods are in [EXPERIMENTS.md](EXPERIMENTS.md), [DATA.md](DATA.md) and [SUPERVISED_PROTOCOL.md](SUPERVISED_PROTOCOL.md). The result report is evidence; a planned or running experiment is not reported as completed.

## Completed bounded experiments and next direction

The frozen v2 study and [v0.1.0 release](https://github.com/numann44/industrial-inspection/releases/tag/v0.1.0) are complete and preserved. This does not mean every category meets its quality target. The [v3 screen](results/STUDY_V3_SCREEN.md) has also completed: fresh control/A/B each trained for 100 epochs, with source/protocol frozen at `b158f56`, identical budgets and one immutable synthetic validation bank.

Family-macro H was 0.8971445541893278 for control, 0.7626782874899303 for A and 0.7425017629318905 for B. Neither candidate achieved the required ≥0.01 gain and ≤0.02 old-family regression. The cycle stopped at its gate: **three new runs completed, zero of the six conditional continuation runs started, no candidate selected**. The published registry remains on v0.1.0. [Portable evidence](results/study-v3-screen.json) retains all outcomes and provenance; the screen read no real test images.

The separate [surface acquisition screen](KSDD2_ROBUSTNESS_SCREEN.md) has completed its two fresh seed-42 runs and fixed-threshold exploratory evaluation. Control stopped at epoch 45 (selected 30); augmentation stopped at epoch 50 (selected 35). Pooled validation H improved from 0.823098 to 0.848836 and passed the frozen gate. Only then were both thresholds set once from the same 313 clean normal calibration sources. This did not extend the failed v3 budget.

The [completed comparison](results/KSDD2_ROBUSTNESS_RESULTS.md) records **104/110 detections and 98/894 false alarms for control**, versus **106/110 and 115/894 for augmentation**, with native pixel AP 0.800701 and 0.811400. **Neither meets the original-test joint target.** Brightness ×1.2 false alarms improved from 207 to 131, but JPEG-quality-60 false alarms worsened from 153 to 185, each out of 894 normal images. The paired validation AUROC interval crosses zero. [Portable evidence](results/KSDD2_ROBUSTNESS_RESULTS.json) and [24 attributed examples](results/KSDD2_ROBUSTNESS_RESULTS.md#auditable-examples-and-errors) preserve the favorable and unfavorable outcomes. These reused KSDD2 results are exploratory. The [launch record](results/KSDD2_ROBUSTNESS_LAUNCH.json) remains a dated historical snapshot, not the current state.

The evaluation source at `05d6660487fe4891bd3432a35925b98c59c6c194` passed 164 local tests and [Linux CI](https://github.com/numann44/industrial-inspection/actions/runs/37185709663). No new checkpoint was promoted; the demo registry remains on v0.1.0.

The [exactly two-run learned image-decision comparison](results/KSDD2_DECISION_RESULTS.md) is complete. Its validation gate passed (pooled partial AUROC 0.855588 → 0.948094), followed by one-time clean-normal calibration and a separately frozen score-aware evaluation. Control selected epoch 5, stopped at 20 and detected 97/110 defects with 81/894 false alarms; the head selected epoch 25, stopped at 40 and detected **104/110 with 102/894 false alarms**. Native pixel AP was 0.649225/0.834704. Both miss the joint original-test target. The head meets point targets under blur and dimming but fails under original, brightened and JPEG conditions. Different selected epochs confound attribution of localization gains to the detached head.

[Portable evidence](results/KSDD2_DECISION_RESULTS.json) includes all scores, hashes, thresholds and uncertainty; the [verification record](results/KSDD2_DECISION_VERIFICATION.json) covers 198 passing local tests, Linux CI, native/stress parity and 24 attributed figures. Source `81306f4` implements the isolated adapter/evaluator without changing the historical inference engine. The [launch snapshot](results/KSDD2_DECISION_LAUNCH.json) remains historical. This bounded architecture search is now closed. Neither new weight is deployed; the original study, negative v3 screen and mixed acquisition evidence remain intact.

The original metal-nut, screw and transistor tests remain development-inspected/exploratory for v3. Repartitioning these images cannot restore independence. Cable remained prospective, with no continuation run or test evaluation; a future comparison would require a prior-test-exposure audit and frozen models/thresholds before its test is read. No new independent same-category holdout is available. Broader success claims require genuinely untouched holdouts and robustness evidence.

## Next data gate

Training and publication for the declared cycles are complete; the broader quality target remains unresolved. More epochs or random seeds on this inspected evidence do not supply independent reliability evidence. No further architecture, threshold or seed search is scheduled.

A new same-surface holdout must contain both normal and defective examples from genuinely separate acquisition sessions or production lots, with rights to use the images. Keep session/camera/lighting and physical-item identifiers where available, plus image-level labels and native defect masks. Split physical items and acquisition groups before modeling; resized copies, nearby video frames and repeated photos of the same item do not create independent samples. The unused 37 positive calibration images cannot measure normal false alarms.

Before inspecting new holdout outcomes, declare the deployment setting, sample-size/uncertainty plan, model, preprocessing, threshold and acceptance criteria. Preserve a sealed holdout if separate development data supports any future change. A different public dataset can support a separately labeled task; it does not retroactively establish reliability for these original parts or surfaces. No unseen same-category acquisition set is currently available locally. Resuming independent quality work therefore needs an actual new data source; existing results and the free v0.1.0 demo remain usable as a documented research portfolio.
