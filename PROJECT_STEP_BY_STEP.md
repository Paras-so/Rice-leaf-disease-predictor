# How I Built My Rice Disease Detection Project

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

15. I set the shared training settings: **AdamW**, cross-entropy loss, learning rate **0.001**, weight decay **0.0001**, batch size **32**, and a maximum of **40 epochs**. I used early stopping after eight epochs without improved validation macro F1. **Files:** `configs/benchmark.json`, `rice_disease/benchmark.py`.

16. I loaded training and validation images in batches, extracted features using the frozen models, and cached those features for reuse during training and tuning. **Files:** `rice_disease/preprocessing.py`, `rice_disease/benchmark.py`.

17. I trained **MobileNetV3-Small** first. Its best checkpoint was from epoch **16**, with **97.50% training accuracy**, **94.21% validation accuracy**, and **0.9431 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/mobilenet_v3_small_metrics.json`.

18. I trained **EfficientNet-B0** with the same settings. Its best checkpoint was from epoch **12**, with **98.39% training accuracy**, **95.82% validation accuracy**, and **0.9591 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/efficientnet_b0_metrics.json`.

19. I trained **ResNet18** next. Its best checkpoint was from epoch **29**, with **97.74% training accuracy**, **94.53% validation accuracy**, and **0.9458 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/resnet18_metrics.json`.

20. I trained **ResNet50** next. Its best checkpoint was from epoch **27**, with **99.03% training accuracy**, **97.11% validation accuracy**, and **0.9718 validation macro F1**. **Files:** `rice_disease/models.py`, `rice_disease/benchmark.py`. **Saved result:** `artifacts/runs/frozen_baseline/resnet50_metrics.json`.

21. I compared validation scores, model sizes, training times, and prediction speeds. ResNet50 had the highest validation macro F1, followed by EfficientNet-B0, ResNet18, and MobileNetV3-Small. **Files:** `rice_disease/benchmark.py`, `rice_disease/report.py`. **Saved result:** `artifacts/runs/frozen_baseline/comparison.csv`.

22. I checked for overfitting using training and validation results and learning curves. The accuracy gaps were **3.29 percentage points for MobileNetV3-Small, 2.57 for EfficientNet-B0, 3.21 for ResNet18, and 1.93 for ResNet50**. None exceeded the code's five-point warning threshold, so I recorded no large accuracy gap; this does not rule out overfitting. **Files:** `rice_disease/benchmark.py`, `rice_disease/report.py`.

23. I tried learning rates **0.0003 and 0.003** on the two best architectures, ResNet50 and EfficientNet-B0. ResNet50 reached **95.50% and 97.11% validation accuracy**, respectively; EfficientNet-B0 reached **96.46% and 95.82%**. **File:** `rice_disease/tune.py`. **Saved results:** `artifacts/runs/head_lr_0.0003/comparison.json`, `artifacts/runs/head_lr_0.003/comparison.json`.

24. I selected the original **ResNet50 at learning rate 0.001** because its validation macro F1 remained the highest across all eight trials. I saved the selected model details before evaluating the test set. **File:** `rice_disease/tune.py`. **Saved files:** `artifacts/selected_model.json`, `artifacts/runs/frozen_baseline/resnet50.pt`.

25. I evaluated the selected ResNet50 once on the **389 reserved test images**. It correctly classified **367 images** and misclassified **22**, achieving **94.34% test accuracy** and **0.9443 macro F1**. **File:** `rice_disease/evaluate.py`. **Saved result:** `artifacts/final_evaluation/metrics.json`.

26. I examined the confusion matrix and incorrect predictions. Leaf blast had the lowest test recall at **83.82%**; six leaf-blast images were predicted as brown spot and three as healthy. I kept these findings as error analysis without tuning against the test results. **Files:** `rice_disease/evaluate.py`, `rice_disease/report.py`. **Saved errors:** `artifacts/final_evaluation/misclassified.json`.

27. I added a healthy-or-diseased result by mapping the healthy class to healthy and the five disease classes to diseased. This achieved **98.46% test accuracy**, or **383 correct out of 389**, using the existing predictions without training another model. **Files:** `rice_disease/binary.py`, `rice_disease/report.py`. **Saved result:** `artifacts/final_evaluation/binary_metrics.json`.

28. I added prediction code that loads the saved model, processes a new image, and returns the disease class, healthy-or-diseased status, and class scores. **Files:** `rice_disease/inference.py`, `rice_disease/models.py`.

29. I connected each disease to stored management information and source links. Automatic pesticide selection, spray timing, and dosage remain unavailable because verified regional product rules have not been added. **Files:** `knowledge_base/diseases.json`, `rice_disease/inference.py`, `rice_disease/quantity.py`.

30. I added an optional calculator for affected leaf area from a supplied labelled mask. I left automatic segmentation and severity prediction unfinished because the dataset has no labelled masks. **Files:** `rice_disease/segmentation.py`, `docs/SEGMENTATION_PLAN.md`.

31. I built a Streamlit app where I can upload a leaf image, click **Analyze leaf**, view predictions and management references, and download the result as JSON. I also added model-comparison and dataset tabs. **File:** `app.py`.

32. I handled image replacement, model changes, clearing uploads, and invalid files so the app does not keep showing an old result for a new input. **Files:** `app.py`, `tests/test_app.py`.

33. I generated learning curves, confusion matrices, example error images, and the final project report. I also added a notebook that runs the local pipeline and displays its saved results. **Files:** `rice_disease/report.py`, `artifacts/PROJECT_REPORT.md`, `RiceDiseaseBenchmark.ipynb`.

34. I checked dataset separation, preprocessing, model outputs, saved-model loading, healthy-or-diseased calculations, mask arithmetic, and app uploads. The project log records **10 passing tests** and a successful browser upload and prediction check. **Files:** `tests/test_pipeline.py`, `tests/test_app.py`, `PROJECT_LOG.md`.

35. I ran the finished local app using `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501` and opened `http://127.0.0.1:8501`. **Files:** `app.py`, `README.md`.
