"""Public CPU inspection demo. Run: streamlit run app.py."""
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from inspection.artifacts import load_registry, resolve_checkpoint
from inspection.engine import InspectionEngine

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Industrial Inspection", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
torch.set_num_threads(2)

st.markdown("""<style>
.block-container{max-width:1320px;padding-top:4.25rem;padding-bottom:3rem}
h1{letter-spacing:-.045em;font-weight:650!important} h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:#f3f5f6;border:1px solid #e4e8eb;border-radius:8px;padding:16px}
[data-testid="stSidebar"]{border-right:1px solid #e4e8eb}
.eyebrow{font:600 11px ui-monospace,monospace;letter-spacing:.15em;color:#58717a;margin-bottom:10px}
.intro{font-size:17px;line-height:1.6;color:#596872;max-width:720px}
.status{font:500 12px ui-monospace,monospace;color:#946326;background:#fff5df;padding:7px 11px;border-radius:4px;display:inline-block}
.decision{font-size:23px;font-weight:650;letter-spacing:-.03em;margin:4px 0 8px}
.caption{color:#667781;font-size:13px;line-height:1.6}
</style>""", unsafe_allow_html=True)


@st.cache_resource(max_entries=1, show_spinner="Loading verified model…")
def get_engine(entry_json):
    entry = json.loads(entry_json)
    return InspectionEngine(resolve_checkpoint(entry, ROOT), device="cpu")


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def model_context(entry):
    """Task-specific explanation, including compatibility with the legacy pilot."""
    supervised = entry.get('model_kind') == 'supervised_segmentation'
    if supervised:
        training = 'Real-defect supervised training · Real normal/defective images and annotated defect masks'
        explanation = ('This selected KolektorSDD2 model learns from real labeled defects and their pixel masks. '
                       'It is a separate supervised surface-inspection task; its results apply to that dataset’s acquisition conditions.')
    else:
        training = 'Normal-only training · Procedural synthetic defects and corruption masks'
        explanation = ('This selected MVTec model learns from normal training images and procedural corruption masks. '
                       'Original real-defect test images are used only for evaluation. Model weights apply to the selected part category.')
    preparation = entry.get('preparation_note')
    if not preparation:
        spec = entry.get('preprocess', {})
        dimensions = f" to {spec['width']} × {spec['height']} pixels (width × height)" if spec.get('width') and spec.get('height') else ''
        if spec.get('mode') == 'letterbox' or supervised:
            preparation = f'Preserve aspect ratio and add letterbox padding{dimensions}; exclude padding from image scoring.'
        elif spec.get('mode') == 'square':
            preparation = f'Use the checkpoint’s square resize{dimensions}.'
        else:
            preparation = 'Use the same orientation, color and resizing rules as this checkpoint’s evaluation.'
    return {'training': training, 'preparation': preparation, 'explanation': explanation}


def main():
    registry = load_registry(ROOT / "artifacts/models.json")
    with st.sidebar:
        st.markdown("### ◈ Industrial Inspection")
        st.caption("COMPUTER VISION · RESEARCH WORKBENCH")
        st.divider()
        entry = st.selectbox("Inspection model", registry["models"], format_func=lambda item: item["label"])
        context = model_context(entry)
        st.caption(entry["description"])
        st.markdown("**What this model knows**")
        st.caption(entry["conditions"])
        st.divider()
        st.markdown("**Training**")
        st.caption(f"Random initialization · PyTorch\n\n{context['training']}\n\nCategory-specific weights · Frozen decision threshold")
        st.markdown("**Image preparation**")
        st.caption(context['preparation'])
        st.markdown("[Source & experiments](https://github.com/numann44/industrial-inspection)")
        st.markdown("[Data & attribution](https://github.com/numann44/industrial-inspection/blob/main/docs/DATA.md)")

    st.markdown('<div class="eyebrow">VISUAL QUALITY INSPECTION / EXPERIMENTAL SYSTEM</div>', unsafe_allow_html=True)
    st.title("Visual inspection, with evidence.")
    st.markdown('<p class="intro">Inspect a part, locate unusual regions, and examine the evidence behind the decision. Built around measured results and visible failure cases.</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="status">{entry["status_label"]}</span>', unsafe_allow_html=True)
    st.write("")

    inspect_tab, evidence_tab, method_tab = st.tabs(["Inspect a part", "Measured performance", "How it works"])
    with inspect_tab:
        source = st.radio("Image source", ["Dataset examples", "Upload a photo"], horizontal=True)
        image = None
        example = None
        if source == "Dataset examples":
            examples = entry.get("examples", [])
            if examples:
                example = st.selectbox("Choose an example", examples, format_func=lambda item: item["label"])
                try:
                    image = (ROOT / example["path"]).read_bytes()
                except OSError:
                    st.error("This example image is unavailable. Choose another example or upload a photo.")
                st.caption("Examples include failures. Ground truth is shown for comparison after inspection.")
            else:
                st.info("Example images will accompany this model's release.")
        else:
            uploaded = st.file_uploader("PNG, JPEG or WebP · up to 10 MiB and 20 megapixels", type=["png", "jpg", "jpeg", "webp"])
            if uploaded:
                image = uploaded.getvalue()
            st.caption("Uploads are processed in memory and are not saved. Select the correct part model; arbitrary photos may produce unreliable results.")
        if image is not None:
            try:
                engine = get_engine(json.dumps(entry, sort_keys=True))
                with st.spinner("Inspecting image…"):
                    result = engine.inspect(image, entry["category"])
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(str(exc))
            else:
                summary = result["summary"]
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown("#### Original image")
                    st.image(result["image"], width="stretch")
                with right:
                    st.markdown("#### Anomaly overlay")
                    st.image(result["overlay"], width="stretch")
                st.markdown(f'<div class="decision">{"Flagged for review" if summary["predicted_defective"] else "Below the defect threshold"}</div>', unsafe_allow_html=True)
                score, threshold, duration = st.columns(3)
                score.metric("Anomaly score", f'{summary["score"]:.6g}')
                threshold.metric("Frozen decision threshold", f'{summary["threshold"]:.6g}')
                duration.metric("Model inference · CPU", f'{summary["inference_ms"]:.0f} ms')
                st.caption("The score is not a defect probability. Red regions show model activations, not confirmed defect boundaries. Color scale is fixed for this checkpoint.")
                if example:
                    actual = "Defective" if example["defective"] else "Normal"
                    matches = example["defective"] == summary["predicted_defective"]
                    st.info(f'Dataset ground truth: {actual} · {"Correct decision" if matches else "Model error — retained as a failure example"}.')
                    st.caption(example["attribution"])
                a, b, c, d = st.columns(4)
                a.download_button("Download result JSON", json.dumps(summary, indent=2), "inspection.json", "application/json", use_container_width=True, on_click="ignore")
                b.download_button("Download overlay", png_bytes(result["overlay"]), "overlay.png", "image/png", use_container_width=True, on_click="ignore")
                raw = io.BytesIO()
                np.savez_compressed(raw, model=result["raw_map"], original=result["native_map"])
                c.download_button("Download raw map", raw.getvalue(), "anomaly-map.npz", "application/octet-stream", use_container_width=True, on_click="ignore")
                d.download_button("Download heatmap", png_bytes(result["heatmap"]), "heatmap.png", "image/png", use_container_width=True, on_click="ignore")
                with st.expander("Inspection record"):
                    st.json(summary)

    with evidence_tab:
        st.subheader("Performance with the threshold fixed in advance")
        metrics = entry.get("metrics", {})
        if metrics:
            a, b, c = st.columns(3)
            a.metric("Defect recall", f'{100 * metrics["defect_recall"]:.1f}%')
            b.metric("Normal false alarms", f'{100 * metrics["normal_false_alarm_rate"]:.1f}%')
            c.metric("Native pixel AP", f'{metrics["pixel_average_precision"]:.3f}')
            st.write(f'Detected **{metrics["confusion"]["true_positive"]}** defective images; missed **{metrics["confusion"]["false_negative"]}**. '
                     f'Flagged **{metrics["confusion"]["false_positive"]}** normal images; accepted **{metrics["confusion"]["true_negative"]}**.')
            st.caption(entry["evaluation_note"])
        st.markdown("**Project target:** detect at least 90% of defects with at most 10% false alarms on normal images. Each category is assessed separately.")
        st.write("Small datasets make rates uncertain. See the experiment report for confidence intervals, seed variability, failure groups and protocol details.")
        st.markdown("[Read the experiment evidence](https://github.com/numann44/industrial-inspection/tree/main/docs/results)")

    with method_tab:
        st.subheader("From a photograph to a review decision")
        st.markdown(f"1. **Prepare the image.** {context['preparation']}\n2. **Run the trained model.** Produce an anomaly map from category-specific weights learned from random initialization.\n3. **Compare with a frozen threshold.** Average the strongest 1% of valid pixel activations and compare with a threshold fitted on separate normal images.\n4. **Inspect the evidence.** Review the overlay and retain the model identity and raw values in an export.")
        st.write(context['explanation'])
        st.caption("This is a research portfolio demonstrator. Results depend on the selected category and imaging conditions; a new production line requires independent validation.")
    st.divider()
    st.caption("Industrial Inspection · Ahmet Numan Şahin · Source code: MIT · Dataset examples: CC BY-NC-SA 4.0 with source attribution")


if __name__ == "__main__":
    main()
