#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run ONNX inference for a regression model using explicit preprocess settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, preprocess_bgr_to_chw_float, preprocess_config_from_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX inference for a steering regression model.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--color-space", default=None)
    args = parser.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()

    preprocess = DEFAULT_PREPROCESS_CONFIG
    if args.height and args.width:
        preprocess = preprocess_config_from_dict(
            {
                "colorSpace": args.color_space or preprocess.color_space,
                "inputSize": [args.height, args.width],
                "useRoi": False,
            },
            fallback=preprocess,
        )
    elif args.color_space:
        preprocess = preprocess_config_from_dict({"colorSpace": args.color_space}, fallback=preprocess)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    chw = preprocess_bgr_to_chw_float(img, config=preprocess)
    batch = np.expand_dims(chw.astype(np.float32), axis=0)

    session = ort.InferenceSession(str(onnx_path))
    output = session.run(None, {session.get_inputs()[0].name: batch})
    print(f"onnx: {onnx_path}")
    print(f"image: {image_path}")
    print(f"preprocess: color_space={preprocess.color_space} input_size={preprocess.input_size} use_roi={preprocess.use_roi}")
    print(f"prediction: {float(np.asarray(output[0]).reshape(-1)[0]):.6f}")


if __name__ == "__main__":
    main()
