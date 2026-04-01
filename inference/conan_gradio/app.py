#!/usr/bin/env python3
"""Gradio demo for the canonical Conan single-reference strong-style mainline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.Conan import StreamingVoiceConversion
from modules.Conan.style_profiles import available_mainline_style_profiles
from utils.commons.hparams import hparams, set_hparams


CANONICAL_CONFIG = "egs/conan_mainline_infer.yaml"
CANONICAL_EXP_NAME = "Conan"
SETTINGS_PATH = Path(__file__).with_name("gradio_settings.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=CANONICAL_CONFIG)
    parser.add_argument("--exp_name", type=str, default=CANONICAL_EXP_NAME)
    parser.add_argument("-hp", "--hparams", type=str, default="")
    parser.add_argument("--server_name", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_settings():
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_demo(engine, settings, *, config_path, exp_name):
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Gradio is not installed. Install it with `pip install gradio` "
            "or use the batch demo runners under inference/."
        ) from e

    title = settings.get("title", "Conan Mainline Demo")
    description = settings.get("description", "")
    article = settings.get("article", "")
    default_style_profile = settings.get("default_style_profile", "strong_style")
    default_style_strength = float(settings.get("default_style_strength", 1.0))
    style_profiles = available_mainline_style_profiles()
    if default_style_profile not in style_profiles:
        default_style_profile = style_profiles[0]

    def run_inference(src_wav, ref_wav, style_profile, style_strength):
        if not src_wav:
            raise gr.Error("Please provide src_wav.")
        if not ref_wav:
            raise gr.Error("Please provide ref_wav.")
        infer_input = {
            "src_wav": str(src_wav),
            "ref_wav": str(ref_wav),
            "style_profile": str(style_profile),
            "style_strength": float(style_strength),
        }
        wav_pred, _ = engine.infer_once(infer_input)
        wav_pred = np.asarray(wav_pred, dtype=np.float32)
        metadata = dict(engine.last_infer_metadata or {})
        metadata.update(
            {
                "canonical_config": str(config_path),
                "canonical_exp_name": str(exp_name),
                "style_profiles": style_profiles,
            }
        )
        pretty_metadata = json.loads(json.dumps(metadata, ensure_ascii=False, indent=2))
        return (int(hparams["audio_sample_rate"]), wav_pred), pretty_metadata

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if description:
            gr.Markdown(description)
        with gr.Row():
            src_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="src_wav (content / source)",
            )
            ref_wav = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="ref_wav (single reference)",
            )
        with gr.Row():
            style_profile = gr.Dropdown(
                choices=style_profiles,
                value=default_style_profile,
                label="style_profile",
            )
            style_strength = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                step=0.05,
                value=default_style_strength,
                label="style_strength",
            )
        run_button = gr.Button("Run Conan mainline inference", variant="primary")
        audio_out = gr.Audio(label="Converted audio")
        metadata_out = gr.JSON(label="Inference metadata")
        run_button.click(
            fn=run_inference,
            inputs=[src_wav, ref_wav, style_profile, style_strength],
            outputs=[audio_out, metadata_out],
        )
        if article:
            gr.Markdown(article)
    return demo


def main():
    args = parse_args()
    set_hparams(config=args.config, exp_name=args.exp_name, hparams_str=args.hparams)
    settings = load_settings()
    engine = StreamingVoiceConversion(hparams)
    demo = build_demo(
        engine,
        settings,
        config_path=args.config,
        exp_name=args.exp_name,
    )
    launch_kwargs = {"share": bool(args.share)}
    if args.server_name is not None:
        launch_kwargs["server_name"] = args.server_name
    if args.server_port is not None:
        launch_kwargs["server_port"] = int(args.server_port)
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
