from __future__ import annotations
import os
import tempfile
import pandas as pd
import gradio as gr

from src.inference.pipeline import FeedingDetector, FeedingDetection

_FEATURE_MODEL = os.path.join("models", "efficientnet_features.onnx")
_CLASSIFIER_MODEL = os.path.join("models", "temporal_classifier.onnx")


def _format_seconds(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def _detections_to_df(detections: list[FeedingDetection]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["#", "Start", "End", "Confidence"])
    rows = [
        {
            "#": i + 1,
            "Start": _format_seconds(d.start_s),
            "End": _format_seconds(d.end_s),
            "Confidence": d.confidence,
        }
        for i, d in enumerate(detections)
    ]
    return pd.DataFrame(rows)


def build_app(detector: FeedingDetector) -> gr.Blocks:
    with gr.Blocks(title="Tern Feeding Event Detector") as app:
        gr.Markdown("## Tern Feeding Event Detector")
        gr.Markdown("Upload a field video to detect feeding events.")

        with gr.Row():
            video_input = gr.File(label="Upload video (.mp4 or .mov)", file_types=[".mp4", ".mov"])
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.99, value=0.6, step=0.05,
                label="Confidence threshold (lower = more sensitive)",
            )

        run_btn = gr.Button("Run Detection", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        results_table = gr.Dataframe(
            headers=["#", "Start", "End", "Confidence"],
            label="Detected Feeding Events",
        )
        csv_output = gr.File(label="Download CSV", visible=False)

        def run_detection(video_file, threshold, progress=gr.Progress()):
            if video_file is None:
                return "Please upload a video first.", pd.DataFrame(), gr.update(visible=False)

            progress_values: list[float] = []

            def _cb(v: float):
                progress_values.append(v)
                progress(v, desc=f"Analysing window {len(progress_values)}…")

            detections = detector.detect(video_file.name, threshold=threshold, progress_cb=_cb)
            df = _detections_to_df(detections)

            if df.empty:
                return "No feeding events detected above threshold.", df, gr.update(visible=False)

            # Write CSV to temp file for download
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            df.to_csv(tmp.name, index=False)
            return f"Found {len(detections)} feeding event(s).", df, gr.update(value=tmp.name, visible=True)

        run_btn.click(
            fn=run_detection,
            inputs=[video_input, threshold_slider],
            outputs=[status, results_table, csv_output],
        )
    return app


def launch():
    for path in (_FEATURE_MODEL, _CLASSIFIER_MODEL):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ONNX model not found at {path}. "
                "Run training on Colab and download the model files to models/."
            )
    detector = FeedingDetector(_FEATURE_MODEL, _CLASSIFIER_MODEL)
    app = build_app(detector)
    app.launch()


if __name__ == "__main__":
    launch()
