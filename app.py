"""Laptop-only rice disease classification prototype."""

import hashlib
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
from rice_disease.unet import load_unet, predict_mask, mask_overlay
from rice_disease.quantity import rice_tank_quantity, disease_product_reference

st.set_page_config(page_title="Rice Leaf Lab", page_icon="🌾", layout="wide")
st.title("Rice Leaf Lab")
st.caption("Rice disease classification · local research prototype")
selection_path = ROOT / "artifacts/selected_model.json"
selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
evaluation_path = ROOT / "artifacts" / (selection or {}).get("evaluation_dir", "final_evaluation") / "metrics.json"
analyze_tab, model_tab, dataset_tab = st.tabs(["Analyze a leaf", "Final model", "Dataset"])


@st.cache_resource
def cached_model(path, modified, expected_hash):
    torch.set_num_threads(4)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
        raise ValueError("The final model file has changed. Restore the selected checkpoint before analysis.")
    return load_checkpoint(path)


def reset_analysis():
    st.session_state["analyze_leaf"] = False


@st.cache_resource
def cached_segmentation(path, modified):
    torch.set_num_threads(4)
    return load_unet(path)


with analyze_tab:
    st.write("Choose a clear rice leaf photo below, then click Analyze leaf to see the prediction.")
    path = ROOT / "artifacts" / selection["checkpoint"] if selection else None
    if path is None or not path.is_file():
        st.info("The final classifier is not available yet. Complete model selection before analyzing a leaf.")
    else:
        if st.session_state.get("model_hash") != selection["checkpoint_sha256"]:
            reset_analysis()
            st.session_state["model_hash"] = selection["checkpoint_sha256"]
        st.caption("Using the final selected model: " + selection["metrics"]["architecture"].replace("_", " "))
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
                model, checkpoint = cached_model(str(path), path.stat().st_mtime_ns,
                                                 selection["checkpoint_sha256"])
                with st.spinner("Analyzing leaf…"):
                    prediction = predict(model, checkpoint, image)
                segmentation_path = ROOT / "artifacts/unet/model.pt"
                display_image = image.copy()
                segmentation_checkpoint = None
                if segmentation_path.is_file():
                    segmentation_model, segmentation_checkpoint = cached_segmentation(str(segmentation_path), segmentation_path.stat().st_mtime_ns)
                    with st.spinner("Segmenting leaf and affected tissue..."):
                        predicted_mask, area, rgb = predict_mask(segmentation_model, segmentation_checkpoint, image)
                    display_image = mask_overlay(rgb, predicted_mask)
                    prediction['segmentation'] = area
                    prediction['affected_area_percent'] = area['affected_area_percent']
                    prediction['segmentation_status'] = f"U-Net prediction; label quality: {segmentation_checkpoint['label_quality']}; not field-validated"

                # Bound both dimensions without changing the image used for inference.
                display_image.thumbnail((560, 340), Image.Resampling.LANCZOS)
                disease_name = prediction['disease'].replace('_', ' ').title()
                percentage = prediction.get('affected_area_percent')
                area_text = f"{percentage:.2f}%" if percentage is not None else "Unavailable"
                info = management(prediction["disease"])
                reference = disease_product_reference(prediction['disease'])
                rule = info.get("chemical_rule")
                application = json.loads((ROOT / "configs/application.json").read_text())
                verified_rule = rule and rule.get("label_verified") and rule.get("region") == application["region"]
                left, right = st.columns([3, 2])
                left.image(display_image, caption=f"{disease_name} | Affected leaf area: {area_text}", width=display_image.width)
                if segmentation_checkpoint is not None:
                    left.caption("Green = unaffected leaf · Red = predicted affected tissue")
                else:
                    left.caption("Original photo shown: the segmentation model is unavailable.")
                with right:
                    st.subheader(prediction["status"].title())
                    st.write(f"**Predicted disease: {disease_name}**")
                    st.metric("Estimated affected leaf area", area_text)
                    if percentage is None:
                        st.caption("No leaf pixels detected." if segmentation_checkpoint is not None else "Train the segmentation model to estimate affected area.")
                    st.markdown("**Pesticide suggestion**")
                    if verified_rule:
                        st.write(f"{rule['product']} ({rule['formulation']})")
                        st.markdown(f"[Product label]({rule['source_url']})")
                    elif reference['product']:
                        product = reference['product']
                        st.write(product['formulation'])
                        st.markdown(f"[{product['source_title']}]({product['source_url']})")
                        st.caption("Disease-matched label reference; confirm diagnosis and local label suitability before use.")
                    else:
                        st.write(reference['note'])

                st.subheader("Disease prediction scores")
                st.metric("Model score", f"{prediction['softmax_score']:.1%}")
                st.caption(prediction["confidence_note"])
                scores = pd.DataFrame({
                    "Disease": [name.replace('_', ' ').title() for name in prediction['scores']],
                    "Score (%)": [score * 100 for score in prediction['scores'].values()],
                }).set_index("Disease")
                chart, percentages = st.columns([3, 2])
                chart.bar_chart(scores, height=250)
                percentages.dataframe(scores, column_config={
                    "Score (%)": st.column_config.NumberColumn(format="%.2f%%"),
                }, width="stretch")

                with st.expander("Affected-area details"):
                    if segmentation_checkpoint is not None:
                        if segmentation_checkpoint.get('label_quality') == 'ai_visual_reviewed':
                            st.info(f"Experimental U-Net pilot: {segmentation_checkpoint['train_images']} training images and {segmentation_checkpoint['validation_images']} validation images with AI-reviewed masks. These are approximate annotations, not expert-validated ground truth.")
                        st.caption("Model estimate: affected pixels / total leaf pixels. Background is excluded. This is not field disease prevalence or a pesticide-dose multiplier.")

                with st.expander("Pesticide reference and management details"):
                    st.subheader("Management reference")
                    if info["management"]:
                        for item in info["management"]:
                            st.write("• " + item)
                        st.markdown(f"Source: [{info['source_title']}]({info['source_url']}) · {info['source_region']}")
                    else:
                        st.write(info.get("note", "No management entry available."))
                    st.caption(f"Crop: rice | Location: {application['state']}, {application['country']} | Tank: {application['spray_tank_ml']} mL ({application['spray_tank_ml']/1000:g} L)")
                    if verified_rule:
                        st.subheader("Product-label quantity calculator")
                        st.write(f"{rule['product']} ({rule['formulation']})")
                        st.markdown(f"[Product label]({rule['source_url']})")
                        for condition in rule.get('application_conditions', []):
                            st.write(condition)
                        volume = st.number_input("Spray mixture volume (mL)", min_value=1.0, value=float(application['spray_tank_ml']))
                        area_ha = st.number_input("Area treated by this mixture (hectares)", min_value=0.0, value=0.0) if rule.get('unit', '').endswith('/ha') else None
                        authorized = st.checkbox("I have confirmed the diagnosis and all product-label application conditions")
                        if authorized:
                            try:
                                quantity = rice_tank_quantity(rule, disease=prediction['disease'], region=application['region'],
                                                            tank_ml=volume, area_hectares=area_ha, application_authorized=True)
                                st.write(f"Product quantity: {quantity['quantity']:.6g} {quantity['unit']}")
                                prediction['product_label_quantity'] = quantity
                            except ValueError as exc:
                                st.warning(str(exc))
                    else:
                        st.subheader("Disease-matched product reference")
                        if reference['product']:
                            product = reference['product']
                            st.write(product['formulation'])
                            st.markdown(f"[{product['source_title']}]({product['source_url']})")
                            water_ml = st.number_input("Water for dilution reference (mL)", min_value=1.0,
                                                       value=float(application['spray_tank_ml']))
                            reference = disease_product_reference(prediction['disease'], water_ml=water_ml)
                            amount = reference['reference_quantity']
                            st.write(f"Label-ratio example: {amount['quantity']:.6g} mL product for {water_ml:g} mL water.")
                            st.caption(f"Derived from {product['formulation_ml_per_hectare']:g} mL product and {product['water_litres_per_hectare']:g} L water per hectare. This does not establish coverage for one plant.")
                            for condition in product['conditions']:
                                st.write(condition)
                        st.info(reference['note'])
                        prediction['pesticide_reference'] = reference
                with st.expander("Calculate area from an existing annotated mask"):
                    st.caption("Upload a grayscale PNG aligned to this image: 0 = background, 1 = unaffected leaf, 2 = diseased leaf. This reads your annotation; it does not predict a mask.")
                    mask_file = st.file_uploader("Annotated label mask", type=["png"], key="mask")
                    if mask_file:
                        with Image.open(mask_file) as mask_image:
                            if mask_image.size != image.size:
                                raise ValueError("Mask dimensions must match the original, orientation-corrected photo.")
                            area = affected_area(np.array(mask_image))
                        st.metric("Affected leaf area in supplied annotation", f"{area['affected_area_percent']:.2f}%")
                        st.caption("Background is excluded. No low/moderate/high severity grade or treatment threshold has been assigned.")
                        prediction["annotated_area"] = area
                prediction["management_reference"] = info
                st.download_button("Download analysis JSON", json.dumps(prediction, indent=2), "rice_leaf_analysis.json", "application/json")
            except (OSError, ValueError) as exc:
                st.error(str(exc))

with model_tab:
    st.subheader("Final selected model")
    if selection:
        result = selection["metrics"]
        st.write(result["architecture"].replace("_", " ").title())
        a, b = st.columns(2)
        a.metric("Validation accuracy", f"{result['validation']['accuracy']:.2%}")
        b.metric("Validation macro F1", f"{result['validation']['macro_f1']:.4f}")
        st.caption("Chosen by validation macro F1. Only this model is used for leaf analysis.")
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
            if (evaluation.get("checkpoint_sha256") == selection["checkpoint_sha256"]
                    and evaluation.get("split_sha256") == selection["split_sha256"]):
                st.subheader("Test evaluation")
                a, b = st.columns(2)
                a.metric("Test accuracy", f"{evaluation['test']['accuracy']:.2%}")
                b.metric("Test macro F1", f"{evaluation['test']['macro_f1']:.4f}")
                binary = binary_metrics(evaluation["confusion_matrix"], CLASSES)
                st.metric("Healthy / diseased test accuracy", f"{binary['accuracy']:.2%}")
                st.caption(f"{evaluation['test_images']} test images. " + evaluation.get("test_policy", ""))
        curve = ROOT / "artifacts/runs" / selection["run"] / "figures" / f"{result['architecture']}_learning_curves.png"
        if curve.exists():
            st.image(str(curve), caption="Final model learning curves")
    else:
        st.info("Final model details will appear after model selection.")

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
