# How I Built My Rice Disease Detection Project

Paths below are relative to the project root. For the complete segmentation workflow,
see [instance segmentation and U-Net training](SEGMENTATION_STEP_BY_STEP.md).

1. I started by loading the rice leaf image folders in Google Colab and importing NumPy, Pandas, Matplotlib, Seaborn, and Pillow for the initial dataset checks. **File:** `FinalYearProject.ipynb`.

2. I checked the original training folder and found **2,100 images**, with 350 images in each of six classes: bacterial leaf blight, brown spot, healthy, leaf blast, leaf scald, and narrow brown spot. **File:** `FinalYearProject.ipynb`.

3. I checked image sizes, colour modes, file formats, and unreadable files. All 2,100 images opened successfully and were RGB JPEGs; two were 1600 x 1548 and the rest were 1600 x 1600. **File:** `FinalYearProject.ipynb`.

4. I compared image hashes and found **159 duplicate copies** inside the original training folder. I also found **83 shared images** between the original training and validation folders. **File:** `FinalYearProject.ipynb`.

5. I kept one copy of each unique image from the original training folder, leaving **1,941 images**. I created a fresh split from these images instead of using the original overlapping validation folder. **File:** `FinalYearProject.ipynb`.

6. I split the cleaned images into **1,552 training images and 389 test images**, using a stratified 80/20 split with seed 42. I copied them into new class folders and checked their counts, keeping the original files unchanged. **File:** `FinalYearProject.ipynb`.

7. I continued locally with the cleaned dataset in `Dataset/clean_dataset-20261001T114830Z-1-001/clean_dataset`. I set up the Python environment and imported PyTorch and Torchvision for model training. Training ran on the CPU because CUDA was unavailable. **Files:** `requirements.txt`, `rice_disease/models.py`, `rice_disease/benchmark.py`.

8. I audited all 1,941 cleaned images again and confirmed that none were unreadable or had identical decoded pixels. The class counts ranged from 297 to 348 images. **File:** `rice_disease/data.py`. **Saved result:** `artifacts/dataset_audit.json`.

9. I kept the existing 389 test images separate and took 20% of the 1,552-image training pool for validation. My final split was **1,241 training, 311 validation, and 389 test images**, using seed 42. **File:** `rice_disease/data.py`.

10. I saved each image's path, class, hash, and split so every model would use the same data. **File:** `rice_disease/data.py`. **Saved files:** `artifacts/split_manifest.csv`, `artifacts/split_summary.json`.

11. I converted each image to RGB, resized it to **224 x 224**, converted it into a PyTorch tensor, and applied ImageNet normalization. **File:** `rice_disease/preprocessing.py`.

12. I gave each training image one original view and one augmented view using flips, 90-degree rotations, and small brightness and contrast changes. I used no augmentation for validation or testing. **Files:** `rice_disease/preprocessing.py`, `configs/benchmark.json`.

13. I downloaded pretrained ImageNet weights for **MobileNetV3-Small, EfficientNet-B0, ResNet18, and ResNet50**. **File:** `rice_disease/download_weights.py`.

14. I used the Torchvision model builders, replaced each model's final layer with a six-class output layer, and froze the other layers. I trained only the final layer of each model. **File:** `rice_disease/models.py`.

15. I set the shared **baseline** training settings: **AdamW**, cross-entropy loss, learning rate **0.001**, weight decay **0.0001**, batch size **32**, and a maximum of **40 epochs**. I used early stopping after eight epochs without improved validation macro F1. These remain the original base parameters for all later comparisons. **Files:** `configs/benchmark.json`, `rice_disease/benchmark.py`.

16. I loaded training and validation images in batches, extracted features using the frozen models, and cached those features for reuse during training and tuning. **Files:** `rice_disease/preprocessing.py`, `rice_disease/benchmark.py`.

17. I trained **MobileNetV3-Small** first. Its best checkpoint was from epoch **16**, with **97.50% training accuracy**, **94.21% validation accuracy**, and **0.9431 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/mobilenet_v3_small_metrics.json`.

18. I trained **EfficientNet-B0** with the same settings. Its best checkpoint was from epoch **12**, with **98.39% training accuracy**, **95.82% validation accuracy**, and **0.9591 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/efficientnet_b0_metrics.json`.

19. I trained **ResNet18** next. Its best checkpoint was from epoch **29**, with **97.74% training accuracy**, **94.53% validation accuracy**, and **0.9458 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/resnet18_metrics.json`.

20. I trained **ResNet50** next. Its best checkpoint was from epoch **27**, with **99.03% training accuracy**, **97.11% validation accuracy**, and **0.9718 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/resnet50_metrics.json`.

21. I compared validation scores, model sizes, training times, and prediction speeds. ResNet50 had the highest validation macro F1, followed by EfficientNet-B0, ResNet18, and MobileNetV3-Small. **Files:** `rice_disease/benchmark.py`, `rice_disease/report.py`. **Saved result:** `artifacts/runs/frozen_baseline/comparison.csv`.

22. I checked for overfitting using training and validation results and learning curves. The accuracy gaps were **3.29 percentage points for MobileNetV3-Small, 2.57 for EfficientNet-B0, 3.21 for ResNet18, and 1.93 for ResNet50**. None exceeded the code's five-point warning threshold, so I recorded no large accuracy gap; this does not rule out overfitting. **Files:** `rice_disease/benchmark.py`, `rice_disease/report.py`.

23. I tried learning rates **0.0003 and 0.003** on the two best architectures, ResNet50 and EfficientNet-B0. ResNet50 reached **95.50% and 97.11% validation accuracy**, respectively; EfficientNet-B0 reached **96.46% and 95.82%**. **File:** `rice_disease/tune.py`. **Saved results:** `artifacts/runs/head_lr_0.0003/comparison.json`, `artifacts/runs/head_lr_0.003/comparison.json`.

24. In the **historical v1 study**, I selected the original **ResNet50 at learning rate 0.001** because its validation macro F1 remained the highest across all eight trials. I saved the selected model details before evaluating the test set. The previous selection is archived in `artifacts/selection_history/` when the expanded study activates its winner; `artifacts/selected_model.json` always identifies the current final model. **Baseline checkpoint:** `artifacts/runs/frozen_baseline/resnet50.pt`.

25. I evaluated the selected ResNet50 once on the **389 reserved test images**. It correctly classified **367 images** and misclassified **22**, achieving **94.34% test accuracy** and **0.9443 macro F1**. **File:** `rice_disease/evaluate.py`. **Saved result:** `artifacts/final_evaluation/metrics.json`.

26. I examined the confusion matrix and incorrect predictions. Leaf blast had the lowest test recall at **83.82%**; six leaf-blast images were predicted as brown spot and three as healthy. I kept these findings as error analysis without tuning against the test results. **Files:** `rice_disease/evaluate.py`, `rice_disease/report.py`. **Saved errors:** `artifacts/final_evaluation/misclassified.json`.

27. I added a healthy-or-diseased result by mapping the healthy class to healthy and the five disease classes to diseased. This achieved **98.46% test accuracy**, or **383 correct out of 389**, using the existing predictions without training another model. **Files:** `rice_disease/binary.py`, `rice_disease/report.py`. **Saved result:** `artifacts/final_evaluation/binary_metrics.json`.

28. I added prediction code that loads the saved model, processes a new image, and returns the disease class, healthy-or-diseased status, and class scores. **Files:** `rice_disease/inference.py`, `rice_disease/models.py`.

29. I connected each disease to stored management information and source links. Automatic pesticide selection, spray timing, and dosage remain unavailable because verified regional product rules have not been added. **Files:** `knowledge_base/diseases.json`, `rice_disease/inference.py`, `rice_disease/quantity.py`.

30. I added an optional calculator for affected leaf area from a supplied labelled mask. I left automatic segmentation and severity prediction unfinished because the dataset has no labelled masks. **Files:** `rice_disease/segmentation.py`, `docs/SEGMENTATION_PLAN.md`.

31. I built a Streamlit app where I can upload a leaf image, click **Analyze leaf**, view predictions and management references, and download the result as JSON. The updated app uses only the final selected classifier, with **Analyze a leaf**, **Final model**, and **Dataset** tabs. There is no model dropdown or baseline fallback. **File:** `app.py`.

32. I handled image replacement, changes to the deployed checkpoint, clearing uploads, and invalid files so the app does not keep showing an old result for a new input. The app checks the selected model's checksum before loading and displays test metrics only when their checkpoint and split match the final selection. **Files:** `app.py`, `tests/test_app.py`.

33. I generated learning curves, confusion matrices, example error images, and the final project report. I also added a notebook that runs the local pipeline and displays its saved results. **Files:** `rice_disease/report.py`, `artifacts/PROJECT_REPORT.md`, `RiceDiseaseBenchmark.ipynb`.

34. I checked dataset separation, preprocessing, model outputs, saved-model loading, healthy-or-diseased calculations, mask arithmetic, and app uploads. The project log records **10 passing tests** and a successful browser upload and prediction check. **Files:** `tests/test_pipeline.py`, `tests/test_app.py`, `docs/PROJECT_LOG.md`.

35. I ran the finished local app using `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501` and opened `http://127.0.0.1:8501`. **Files:** `app.py`, `README.md`.


36. I expanded the study while keeping the original baseline configuration unchanged. I declared **52 candidates** before running the search: the four original baselines plus **12 new recipes on each of the four architectures**. All trials share seed 42, the same train/validation membership, image preprocessing, cached frozen features, maximum 40 epochs and patience 8. **Files:** `rice_disease/tune.py`, `artifacts/experiments/hyperparameter_search_v2/plan.json`.

37. I compared **AdamW, Adam and SGD with momentum 0.9**, each with ordinary cross-entropy, label-smoothed cross-entropy (smoothing **0.1**) and multiclass focal loss (gamma **2**). Adam and AdamW use learning rate 0.001; SGD uses 0.01. I also compared AdamW/cross-entropy variants changing one setting at a time: learning rate **0.0003** or **0.003**, batch size **64**, or weight decay **0.001**. The original AdamW/cross-entropy combination is represented by the preserved baselines. **Files:** `rice_disease/benchmark.py`, `rice_disease/tune.py`.

38. I enabled live epoch and batch progress with **verbose=1**. Each trial saves its configuration, best validation checkpoint, training history and metrics. Training uses the configured loss; reported cross-entropy remains a common comparison measure, and history also records the actual objective loss. Interrupted searches resume completed trials only after checking their configuration, split and checkpoint metrics. **Log:** `artifacts/hyperparameter_search.log`.

    To watch each epoch run live, use the run name **`live_training_demo`**. Run this command from the project directory:

    ```powershell
    .\.venv\Scripts\python.exe -u -m rice_disease.benchmark --run live_training_demo --verbose 1
    ```

    This starts a separate four-model baseline training run using `configs/benchmark.json`, with live batch progress, training/validation loss and accuracy, and validation macro F1. Results are saved under `artifacts/runs/live_training_demo/`: `<model>_history.json` contains every epoch's metrics, `<model>_metrics.json` contains the best checkpoint's results, and `comparison.csv` compares the models. This command does not activate a new deployed model.

    `verbose=1` is already enabled by default. Completed models are skipped even with verbose output enabled. Once `live_training_demo` finishes, use a new name such as `live_training_demo_02` to watch another fresh run. This is a command to run yourself, not a record that the demonstration has already been executed.

39. I completed all **52 candidates**. The winner was **ResNet50 + AdamW + label-smoothed cross-entropy (0.1)**, with **97.43% validation accuracy (303/311 correct)** and **0.9749 validation macro F1**. Its learning rate is **0.001**, weight decay **0.0001**, batch size **32**, and best checkpoint is from **epoch 22** (early stopping ended training at epoch 30). Training accuracy is **98.87%**. The original baseline achieved **97.11% accuracy (302/311)** and **0.9718 macro F1**. The gain is one validation image; it does not establish statistical superiority. Candidates rank by validation macro F1, then smaller parameter count; exact ties retain the earlier candidate. No test images were read during selection. **All results:** `artifacts/experiments/hyperparameter_search_v2/comparison.csv`.

40. The deployment step copies only the winning checkpoint and its configuration into `artifacts/final_model/hyperparameter_search_v2/`, archives the previous selection, and atomically updates `artifacts/selected_model.json`. Research checkpoints remain in `artifacts/runs/` for audit and reproducibility, but cannot be selected in Streamlit. Rerunning the same completed search returns its verified winner.

41. The original 389-image test holdout was already evaluated for v1 in steps 25–27. Any new winner evaluation is saved separately in `artifacts/evaluations/hyperparameter_search_v2/` and is explicitly a **reused holdout**, not fresh independent confirmation. No test metric participates in ranking, and no further search is driven by its errors. One split and one seed do not establish statistical superiority.


42. After freezing the new winner, I evaluated it on the previously used **389-image holdout**. It classified **368 correctly** and **21 incorrectly**, with **94.60% test accuracy**, **0.9472 macro F1**, **0.9503 macro precision** and **0.9477 macro recall**. The historical v1 model classified 367 correctly. This test comparison did not change the selected model or trigger additional tuning. The healthy/diseased mapping achieved **97.94% accuracy**. **Saved results:** `artifacts/evaluations/hyperparameter_search_v2/metrics.json`, `misclassified.json`, `binary_metrics.json`.

43. I generated the selected recipe's learning curves and the final model's confusion matrix, error examples and report. The app displays results only for the deployed checkpoint. **Files:** `artifacts/runs/hyperparameter_search_v2_adamw_smooth/figures/`, `artifacts/PROJECT_REPORT.md`, `app.py`.

44. I ran **20 tests successfully**, covering preprocessing and CLI commands, model loading, app upload/analyze/replace/clear behavior, absence of model selection options, loss values and gradients, optimizer updates, validation ranking, preservation of historical selection/test records, deployment checksums and completed-search resume. **Command:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. **Log:** `artifacts/verification.log`.

The current final recipe is:

| Setting | Final value |
|---|---|
| Architecture | ResNet50, frozen ImageNet backbone with trained six-class linear head |
| Optimizer | AdamW |
| Training loss | Label-smoothed cross-entropy |
| Label smoothing | 0.1 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Maximum epochs / patience | 40 / 8 |
| Selected epoch / epochs run | 22 / 30 |
| Seed | 42 |
| Validation accuracy / macro F1 | 97.43% / 0.9749 |
| Reused-holdout accuracy / macro F1 | 94.60% / 0.9472 |
| App checkpoint | `artifacts/final_model/hyperparameter_search_v2/model.pt` |

For reproducibility, the training APIs follow [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) and [PyTorch SGD documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html). Focal loss is implemented as the mean of `(1 - p_target)^gamma * cross_entropy`; gamma zero is tested against ordinary cross-entropy for both value and gradient.
