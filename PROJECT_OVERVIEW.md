# Project explained simply

**Project name:** Smart Rice Disease Detection and Pesticide Recommendation System

**Main goal:** Help a user spot possible rice leaf disease from a photo and understand basic crop-care steps.

## How it works

Upload a rice leaf photo → the AI checks it → the app shows healthy or diseased, the predicted disease name, and relevant crop-care information.

## AI model

- **Selected model: ResNet50.** It is an AI model that recognizes patterns in images.
- It already had general image-recognition knowledge. The project kept that part fixed and trained only its final decision layer to recognize rice leaf conditions.
- Four models were compared: MobileNetV3-Small, EfficientNet-B0, ResNet18, and ResNet50. ResNet50 had the best validation score across the six categories in these experiments.
- The six possible results are **healthy, bacterial leaf blight, brown spot, leaf blast, leaf scald, and narrow brown spot**.
- The healthy/diseased result comes from the same prediction: any of the five disease categories counts as diseased.

## Main features available now

- **Photo upload:** Analyze a rice leaf image through a simple browser app.
- **Disease prediction:** Show whether the leaf appears healthy or diseased and name the predicted category.
- **Model scores:** Show how strongly the model favors each category. These scores are not a guarantee that the prediction is correct.
- **Crop-care references:** Display basic disease-management information with source links.
- **Model comparison:** View the recorded accuracy and speed of the four models.
- **Dataset view:** See how the images were divided for training and checking results.
- **Download results:** Save the analysis as a JSON file.
- **Optional marked-image measurement:** If the user supplies a specially marked image showing healthy and diseased leaf areas, calculate the affected percentage from those markings.
- **Local use:** Run on a laptop through the browser app or a command-line command.

## Data and results

The dataset contains **1,941 rice leaf images**: 1,241 for training, 311 for comparing and choosing models, and 389 reserved for the final test.

| What was checked | Final test result |
|---|---|
| Choosing the correct one of six categories | **94.34%** — 367 of 389 images correct |
| Identifying healthy versus diseased | **98.46%** — 383 of 389 images correct |

These results come from the project's reserved test images. Performance on new farm photos has not been established.

## Work still needed

- **Pesticide recommendations, spray timing, and dosage:** Part of the broader project goal, but currently unavailable. These need verified product instructions and rules for the user's region.
- **Automatic affected-area measurement:** Needs training images with the diseased areas marked. Currently, the app only measures markings supplied by the user.
- **Real farm testing:** Needed to check performance across different lighting, backgrounds, and growing conditions. The current model always chooses from its six known categories.

## Technology used

**Python** runs the project, **PyTorch and TorchVision** handle the AI model, and **Streamlit** provides the browser interface. The current setup runs on the laptop's CPU.
