# Delivery checklist

The project target is a category-specific defect detector trained from random initialization, with measured ≥90% recall and ≤10% normal false alarms, useful spatial analysis, reproducible evidence and a public CPU demo. Quality and engineering completion are assessed separately.

| Stage | Current evidence | Remaining gate |
| --- | --- | --- |
| Data | Full MVTec AD and KolektorSDD2 decoded, hashed and audited | Preserve frozen partitions during all experiments |
| Legacy evidence | Three measured exploratory runs, curves, native galleries and confidence intervals | Retain results unchanged |
| Training reliability | All 17 declared runs completed; interrupted training recovered from compatible atomic checkpoints | Preserve checkpoint/source provenance |
| Shared inference | Four published weights verified by SHA; hosted example categories/identities/decisions checked; real upload and four exports agree with local engine | Preserve the inference contract and expose display saturation limits |
| Controlled study | All 14 MVTec runs and frozen evaluations complete; five-way eligible selection and all category seed repeats reported | Selected models fail the recall target; further methods need a new declared experiment |
| Supervised fallback | Three KSDD2 runs complete; validation-selected seed 44 detects 105/110 defects with 89/894 false alarms | Point estimates pass; uncertainty crosses targets and brightness/JPEG robustness fails |
| Demo | All four v0.1.0 models observed; surface upload and four exports verified; malformed and oversized uploads rejected correctly | Separate physical-client check remains unperformed |
| Publication | Public v0.1.0 release at e76ff78; four remote weights match local checksums; main/tag Linux checks pass | Retain honest measured limits while quality work continues |
| Follow-up study v3 | Source/protocol frozen at b158f56; fresh control/A/B screen active, sequential and local | Apply validation-only gate; at most six conditional continuation runs require an executor and review |

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

## Active bounded quality work

The frozen v2 study and [v0.1.0 release](https://github.com/numann44/industrial-inspection/releases/tag/v0.1.0) are complete and preserved. This is not a claim that every category meets its quality target. The [v3 declaration](../experiments/study_v3/README.md) tests subtler donor-based synthesis (A) and additional silhouette-crossing deformation (B) against a freshly trained unchanged control, with identical budgets and a shared immutable validation bank. Source and protocol were frozen at `b158f56` before execution.

The three screening runs are active and serialized on local MPS. Continue only when a candidate improves family-macro validation H by ≥0.01, with ≤0.02 regression on the old-family mean. A failed gate stops the cycle. A passed gate permits only the six listed continuation jobs, capping this cycle at nine new runs; their executor is not implemented yet and requires review. The published registry stays on v0.1.0 during screening.

The original metal-nut, screw and transistor tests are development-inspected/exploratory for v3. Repartitioning these images cannot restore independence. Cable is a prospective separate-category comparison requiring a prior-test-exposure audit and frozen models/thresholds before its test is read. No new independent same-category holdout is available. Broader success claims need genuinely untouched holdouts and robustness evidence, not more epochs or random-seed searches.
