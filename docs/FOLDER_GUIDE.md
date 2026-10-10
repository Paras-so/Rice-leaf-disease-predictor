# Project folder map

Start with [README](../README.md) for launching the application, or the
[segmentation and U-Net guide](SEGMENTATION_STEP_BY_STEP.md) for the full workflow.
Commands and inline paths in the project guides are relative to the project root.

| Folder/file | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `requirements.txt`, `.venv/`, `.streamlit/` | Dependencies, installed environment and app configuration |
| `rice_disease/` | Classification, candidate generation, mask data, U-Net and inference code |
| `scripts/` | Dataset checks, mask validation, review exports and notebook maintenance |
| `tests/` | Automated checks |
| `configs/` | Classifier, U-Net and application settings |
| `knowledge_base/` | Disease-management and product-reference catalog |
| `docs/` | Guides, project notes, reference documents and illustrations |
| `Dataset/` | Cleaned source photos used by the frozen manifest |
| `archive (7)/` | Original imported image dataset; retained as source material |
| `artifacts/` | Models, masks, metrics and traceable experiment outputs |
| `FinalYearProject.ipynb` | Original dataset-cleaning notebook |
| `RiceDiseaseBenchmark.ipynb` | Classifier benchmark notebook |
| `.git/` | Version history |

## Active outputs

| Location under `artifacts/` | Purpose |
|---|---|
| `selected_model.json` | Active classifier checkpoint path and hash |
| `final_model/hyperparameter_search_v2/` | Active classifier weights |
| `evaluations/hyperparameter_search_v2/` | Selected classifier evaluation |
| `split_manifest.csv`, `split_summary.json`, `dataset_audit.json` | Image/split identities and audit |
| `instance_segmentation/` | Current candidate instance export and searchable gallery |
| `segmentation_annotations/` | Editable semantic labels and review status |
| `boundary_review_20261010/` | Corrected-mask provenance, original/corrected gallery and backups |
| `pixel_mask_audit.json`, `unet_readiness.json` | Annotation integrity and current pilot status |
| `unet/` | Active U-Net weights, history, provenance and validation review |
| `SEGMENTATION_README.md` | Index explaining segmentation outputs |
| `PROJECT_REPORT.md` | Classifier experiment report |

`runs/`, `experiments/`, `selection_history/`, the previous `final_evaluation/`,
and `archive/` retain research history, comparisons and rollback data. Their
checkpoints and reports are referenced by model-selection records and project
documentation. They are not duplicate disposable caches. The semantic-mask folder
and instance-mask folder also use different encodings and serve different stages.

## What cleanup removed

On 10 October 2026, cleanup removed 2,353 generated/redundant files, reclaiming
approximately 261.81 MiB:

- Disposable `.cache/` contents, including pip downloads, browser checks and pilot previews.
- Project `__pycache__/` directories and bytecode.
- Rebuildable classifier image and feature caches in `artifacts/cache/`.
- Old console `.log` files directly under `artifacts/`.
- `rice_disease/tempCodeRunnerFile.py`, verified byte-for-byte identical to `models.py`.
- An unreferenced temporary `artifacts/segmentation_input_preview.jpg`.

Downloaded pretrained weights remain in `artifacts/cache/torch/` so classifier
training can run without downloading those weights again. Normal execution may
recreate small caches. Metrics, history JSON, correction records and the project
log are retained because they explain and reproduce results.

`PROJECT_LOG.md`, `PROJECT_OVERVIEW.md` and `PROJECT_STEP_BY_STEP.md` now live in
`docs/`. Historical log entries describe files that existed at the time; disposable
cache screenshots and console logs mentioned there may have been cleaned up.

To continue training, follow the guide and choose a fresh output directory.
Do not remove source photos, reviewed annotations or manifests to resolve a missing
file error; restore the actual file or correct its documented path.

## Verification after cleanup

All 45 automated tests passed, and the installed environment has no broken
requirements. The semantic audit checked all 1,941 masks with zero missing
annotations or integrity errors. The instance validator also passed all 1,941
images, covering 2,033 leaf components and 27,735 lesion candidates. Both active model hashes match their recorded
identities. The active galleries' 8,103 file references resolve, as do the checked
Markdown links. A Streamlit display check confirmed all six class percentages,
the model score, one masked image and the affected-area metric.
