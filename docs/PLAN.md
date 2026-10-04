# Delivery checklist

The project target is a category-specific defect detector trained from random initialization, with measured ≥90% recall and ≤10% normal false alarms, useful spatial analysis, reproducible evidence and a public CPU demo. Quality and engineering completion are assessed separately.

| Stage | Current evidence | Remaining gate |
| --- | --- | --- |
| Data | Full MVTec AD and KolektorSDD2 decoded, hashed and audited | Preserve frozen partitions during all experiments |
| Legacy evidence | Three measured exploratory runs, curves, native galleries and confidence intervals | Retain results unchanged |
| Training reliability | All 17 declared runs completed; interrupted training recovered from compatible atomic checkpoints | Preserve checkpoint/source provenance |
| Shared inference | CLI/evaluation/demo use the same engine; float maps and letterbox geometry tested | Published model artifacts load from clean installation |
| Controlled study | All 14 MVTec runs and frozen evaluations complete; five-way eligible selection and all category seed repeats reported | Selected models fail the recall target; further methods need a new declared experiment |
| Supervised fallback | Three KSDD2 runs complete; validation-selected seed 44 detects 105/110 defects with 89/894 false alarms | Point estimates pass; uncertainty crosses targets and brightness/JPEG robustness fails |
| Demo | Hosted CPU app verified in an independent unsigned-in browser, including upload and all exports | Final selected models after study; separate physical-client check |
| Publication | Completed controlled report, portable provenance, attributed galleries and verified local `v0.1.0` bundle | Publish selected assets and verify their actual hosted deployment |

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

## Further quality work

The frozen study is complete and its evidence is preserved. A later cycle must state its hypothesis, controlled changes, bounded run budget and validation selection rule before training. Merely adding epochs or searching random seeds is not an improvement rationale. Any test set used to guide changes becomes development-inspected for that later cycle; it cannot substantiate a new independent success claim. New independent holdouts and robustness evidence remain necessary for broader claims.
