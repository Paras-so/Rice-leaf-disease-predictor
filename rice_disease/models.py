"""Explicit ImageNet checkpoints and six-class classification heads."""
import torch
from torch import nn
from torchvision import models

BUILDERS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
}


def create_model(name, pretrained=True, num_classes=6):
    builder, weights_enum = BUILDERS[name]
    model = builder(weights=weights_enum.IMAGENET1K_V1 if pretrained else None)
    if name.startswith("resnet"):
        features = model.fc.in_features
        model.fc = nn.Linear(features, num_classes)
        head = model.fc
    else:
        features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(features, num_classes)
        head = model.classifier[-1]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    return model, head


def remove_head(model, name):
    if name.startswith("resnet"):
        head = model.fc
        model.fc = nn.Identity()
    else:
        head = model.classifier[-1]
        model.classifier[-1] = nn.Identity()
    return head


def attach_head(model, name, head):
    if name.startswith("resnet"):
        model.fc = head
    else:
        model.classifier[-1] = head


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = create_model(checkpoint["architecture"], pretrained=False,
                            num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def parameter_counts(model):
    """Count scalar parameters according to the model's current training flags."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("Parameter counts for six-class classifiers (only the final head is trained):")
    print(f"{'Model':<22} {'Total':>14} {'Trainable':>14} {'Frozen':>14} {'FP32 MiB':>10}")
    for name in BUILDERS:
        # Weight values do not affect counts; avoid downloading ImageNet weights.
        model, _ = create_model(name, pretrained=False)
        counts = parameter_counts(model)
        print(f"{name:<22} {counts['total']:>14,} {counts['trainable']:>14,} "
              f"{counts['frozen']:>14,} {counts['total'] * 4 / (1024**2):>10.2f}")
    print("FP32 MiB estimates parameter storage only; excludes buffers, activations, and optimizer state.")
    print("Train/compare: python -m rice_disease.benchmark\n"
          "Predict: python -m rice_disease.inference path/to/leaf.jpg")
