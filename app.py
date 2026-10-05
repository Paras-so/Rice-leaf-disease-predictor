"""Laptop-only rice disease classification prototype."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch

from rice_disease.data import ROOT
from rice_disease.inference import predict, management
from rice_disease.models import load_checkpoint
from rice_disease.segmentation import affected_area
from rice_disease.binary import binary_metrics
from rice_disease.data import CLASSES

st.set_page_config(page_title="Rice Leaf Lab", page_icon="🌾", layout="wide")
st.title("Rice Leaf Lab")
st.caption("Rice disease classification · local research prototype")
run = ROOT / "artifacts/runs/frozen_baseline"
comparison_path = run / "comparison.json"
comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else []
selection_path = ROOT / "artifacts/selected_model.json"
selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
evaluation_path = ROOT / "artifacts/final_evaluation/metrics.json"
analyze_tab, benchmark_tab, dataset_tab = st.tabs(["Analyze a leaf", "Model comparison", "Dataset"])


@st.cache_resource
def cached_model(path, modified):
    torch.set_num_threads(4)
    return load_checkpoint(path)


def reset_analysis():
    st.session_state["analyze_leaf"] = False


with analyze_tab:
    st.write("Choose a clear rice leaf photo below, then click Analyze leaf to see the prediction.")
    choices = {}
    if selection:
        choices["Selected v1: " + selection["metrics"]["architecture"]] = ROOT / "artifacts" / selection["checkpoint"]
    choices.update({"Baseline: " + r["architecture"]: run / f"{r['architecture']}.pt"
                    for r in comparison if (run / f"{r['architecture']}.pt").exists()})
    available = list(choices)
    if not available:
        st.info("The first classifier is still being trained. Dataset details are available in the Dataset tab.")
    else:
        architecture = st.selectbox("Classifier", available, format_func=lambda x: x.replace("_", " "),
                                    on_change=reset_analysis)
        st.caption("Selection uses validation macro F1. These classifiers use frozen pretrained backbones.")
        uploaded = st.file_uploader("Choose a rice leaf photo", type=["jpg", "jpeg", "png", "webp"],
                                    help="Click Browse files or drag a JPG, JPEG, PNG, or WebP image here.",
                                    on_change=reset_analysis)
        if uploaded is not None:
            st.caption(f"Selected file: {uploaded.name}")
        if st.button("Analyze leaf", type="primary", disabled=uploaded is None):
            st.session_state["analyze_leaf"] = True
        if uploaded is not None and st.session_state.get("analyze_leaf", False):
            try:
                image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                image.load()
                left, right = st.columns(2)
                left.image(image, caption=uploaded.name, width="stretch")
                path = choices[architecture]
                model, checkpoint = cached_model(str(path), path.stat().st_mtime_ns)
                with st.spinner("Analyzing leaf…"):
                    prediction = predict(model, checkpoint, image)
                right.subheader(prediction["status"].title())
                right.write("Predicted class: " + prediction["disease"].replace("_", " ").title())
                right.metric("Model score", f"{prediction['softmax_score']:.1%}")
                right.caption(prediction["confidence_note"])
                right.bar_chart(pd.DataFrame({"Class": list(prediction["scores"]), "Score": list(prediction["scores"].values())}).set_index("Class"))
                info = management(prediction["disease"])
                st.subheader("Management reference")
                if info["management"]:
                    for item in info["management"]:
                        st.write("• " + item)
                    st.markdown(f"Source: [{info['source_title']}]({info['source_url']}) · {info['source_region']}")
                else:
                    st.write(info.get("note", "No management entry available."))
                st.info("Automatic spray advice and pesticide quantity are unavailable: no current, region-specific product-label rule has been verified. An image prediction alone does not establish a need to spray.")
                st.subheader("Affected leaf area")
                st.write("A segmentation model has not been trained because the dataset has no pixel masks. Classification scores do not measure disease severity.")
                with st.expander("Calculate area from an existing annotated mask"):
                    st.caption("Upload a grayscale PNG aligned to this image: 0 = background, 1 = unaffected leaf, 2 = diseased leaf. This reads your annotation; it does not predict a mask.")
                    mask_file = st.file_uploader("Annotated label mask", type=["png"], key="mask")
                    if mask_file:
                        with Image.open(mask_file) as mask_image:
                            if mask_image.size != image.size:
                                raise ValueError("Mask dimensions must match the displayed, orientation-corrected image.")
                            area = affected_area(np.array(mask_image))
                        st.metric("Affected leaf area in supplied annotation", f"{area['affected_area_percent']:.2f}%")
                        st.caption("Background is excluded. No low/moderate/high severity grade or treatment threshold has been assigned.")
                        prediction["annotated_area"] = area
                prediction["management_reference"] = info
                st.download_button("Download analysis JSON", json.dumps(prediction, indent=2), "rice_leaf_analysis.json", "application/json")
            except (OSError, ValueError) as exc:
                st.error(str(exc))

with benchmark_tab:
    st.subheader("Measured validation results")
    if comparison:
        table = [{"Model": r["architecture"], "Parameters": r["total_parameters"],
                  "Validation accuracy": r["validation"]["accuracy"],
                  "Macro F1": r["validation"]["macro_f1"],
                  "Training seconds": r["total_training_seconds"],
                  "Forward latency (ms)": r["inference_ms_batch1"]} for r in comparison]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Single seed, one stratified split. Architecture comparison uses validation data only. Parameter counts include the new six-class layer.")
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
            st.subheader("Final held-out evaluation of selected v1")
            a, b = st.columns(2)
            a.metric("Test accuracy", f"{evaluation['test']['accuracy']:.2%}")
            b.metric("Test macro F1", f"{evaluation['test']['macro_f1']:.4f}")
            st.caption(f"{evaluation['test_images']} reserved images; configuration frozen before this evaluation.")
            binary = binary_metrics(evaluation["confusion_matrix"], CLASSES)
            st.metric("Healthy / diseased test accuracy", f"{binary['accuracy']:.2%}")
            st.caption("Healthy when the predicted class is healthy; all five disease classes map to diseased.")
        for item in comparison:
            curve = run / "figures" / f"{item['architecture']}_learning_curves.png"
            if curve.exists():
                st.image(str(curve), caption=item["architecture"])
    else:
        st.info("Benchmark results will appear after the first model completes.")

with dataset_tab:
    audit_path = ROOT / "artifacts/dataset_audit.json"
    split_path = ROOT / "artifacts/split_summary.json"
    if audit_path.exists() and split_path.exists():
        audit = json.loads(audit_path.read_text())
        split = json.loads(split_path.read_text())
        st.metric("Verified images", audit["total_images"])
        st.dataframe(pd.DataFrame(split["split_counts"]), width="stretch")
        st.write(split["strategy"])
        st.caption("Source images remain unchanged. Split membership is stored in a hashed manifest.")
    else:
        st.info("Run dataset preparation to populate this view.")
