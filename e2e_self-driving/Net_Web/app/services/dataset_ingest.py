"""ZIP 解压、文件名解析、划分 train/val/test 列表。"""

from __future__ import annotations

import os
import random
import re
import zipfile
from pathlib import Path

from fastapi import UploadFile

_JPG = re.compile(r"\.jpe?g$", re.IGNORECASE)


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for member in zf.infolist():
        if member.is_dir():
            continue
        out = (dest / member.filename).resolve()
        if not str(out).startswith(str(dest.resolve())):
            raise ValueError("非法压缩包路径（zip slip）")
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member, "r") as src, open(out, "wb") as dst:
            dst.write(src.read())


def _parse_steering_from_name(path: Path) -> float | None:
    stem = path.stem
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1]
    try:
        return float(tail)
    except ValueError:
        return None


def _collect_images(root: Path) -> list[tuple[Path, float]]:
    found: list[tuple[Path, float]] = []
    for p in root.rglob("*"):
        if not p.is_file() or not _JPG.search(p.name):
            continue
        ang = _parse_steering_from_name(p)
        if ang is None:
            continue
        found.append((p.resolve(), ang))
    return found


def _split_three_way(pairs: list[tuple[Path, float]], train_ratio: float, val_ratio: float):
    n_total = len(pairs)
    if n_total < 3:
        raise ValueError("ZIP 内至少需要 3 张符合「序号_转向角.jpg」的图像，才能生成 train/val/test")

    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    if n_test < 1:
        deficit = 1 - n_test
        take_from_train = min(deficit, max(0, n_train - 1))
        n_train -= take_from_train
        deficit -= take_from_train
        take_from_val = min(deficit, max(0, n_val - 1))
        n_val -= take_from_val
        deficit -= take_from_val
        if deficit > 0:
            raise ValueError("数据量不足，无法同时保留 train/val/test 三个非空划分")
        n_test = n_total - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    return train_pairs, val_pairs, test_pairs


async def ingest_zip_to_folder(
    base: Path,
    upload: UploadFile,
    display_name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 256,
) -> dict:
    """
    在 base 目录下写入 extracted/、train.txt、val.txt、test.txt、upload.zip。
    返回 { name, image_count, root_dir }。
    """
    base.mkdir(parents=True, exist_ok=True)
    extracted = base / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)

    raw_name = upload.filename or "dataset.zip"
    zip_path = base / "upload.zip"
    zip_path.write_bytes(await upload.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extract(zf, extracted)

    pairs = _collect_images(extracted)
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    train_pairs, val_pairs, test_pairs = _split_three_way(pairs, train_ratio=train_ratio, val_ratio=val_ratio)

    def write_list(path: Path, items: list[tuple[Path, float]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for img_path, angle in items:
                f.write(f"{os.fspath(img_path)} {angle}\n")

    write_list(base / "train.txt", train_pairs)
    write_list(base / "val.txt", val_pairs)
    write_list(base / "test.txt", test_pairs)

    name = display_name.strip() or raw_name.rsplit(".", 1)[0]
    return {
        "name": name,
        "image_count": len(pairs),
        "root_dir": str(base.resolve()),
    }