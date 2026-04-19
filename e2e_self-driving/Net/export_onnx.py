#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export regression checkpoint to ONNX using checkpoint preprocess metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import build_model_for_checkpoint
from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, preprocess_config_from_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a regression checkpoint to ONNX.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state = checkpoint.get("model", checkpoint)
    preprocess = preprocess_config_from_dict(checkpoint.get("preprocess") if isinstance(checkpoint, dict) else None, fallback=DEFAULT_PREPROCESS_CONFIG)

    model = build_model_for_checkpoint(state).to(device)
    model.eval()

    height, width = preprocess.input_size
    dummy = torch.randn(1, 3, height, width, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ckpt: {ckpt_path}")
    print(f"out: {out_path}")
    print(f"preprocess: color_space={preprocess.color_space} input_size={preprocess.input_size} use_roi={preprocess.use_roi}")


if __name__ == "__main__":
    main()
