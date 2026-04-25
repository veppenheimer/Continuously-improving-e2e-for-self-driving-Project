"""Convert raw progress dictionaries into TaskProgress models."""

from __future__ import annotations

from typing import Any

from app.schemas import LossPoint, LossSeriesBundle, TaskProgress


def _lp(xs: list[dict[str, Any]]) -> list[LossPoint]:
    out: list[LossPoint] = []
    for item in xs:
        out.append(
            LossPoint(
                epoch=int(item["epoch"]),
                train_loss=float(item["trainLoss"]),
                val_loss=float(item["valLoss"]),
            )
        )
    return out


def _bundle(raw: dict[str, Any] | None) -> LossSeriesBundle:
    raw = raw or {}
    return LossSeriesBundle(
        train_loss_series=_lp(raw.get("trainLossSeries") or []),
        val_loss_series=_lp(raw.get("valLossSeries") or []),
    )


def build_task_progress(raw: dict[str, Any], fallback: dict[str, Any] | None = None) -> TaskProgress:
    if fallback and (not raw or "baseline" not in raw):
        raw = {
            "status": fallback.get("status", "pending"),
            "currentEpoch": 0,
            "totalEpochs": fallback.get("totalEpochs", 0),
            "baseline": {"trainLossSeries": [], "valLossSeries": []},
            "augmented": (
                {"trainLossSeries": [], "valLossSeries": []} if fallback.get("domain_augmentation") else None
            ),
            "baselineProgress": fallback.get("baselineProgress", 0.0),
            "domainAugmentationProgress": fallback.get("domainAugmentationProgress"),
            "domainAugmentationText": fallback.get("domainAugmentationText"),
            "augmentedProgress": fallback.get("augmentedProgress"),
            "message": fallback.get("message"),
        }
        if fallback.get("status") == "completed":
            raw["currentEpoch"] = raw["totalEpochs"]

    augmented_raw = raw.get("augmented")
    augmented_bundle = _bundle(augmented_raw) if isinstance(augmented_raw, dict) else None

    total = int(raw.get("totalEpochs") or 0)
    if fallback and total == 0:
        total = int(fallback.get("totalEpochs") or 0)

    return TaskProgress(
        status=str(raw.get("status", "pending")),
        current_epoch=int(raw.get("currentEpoch") or 0),
        total_epochs=total,
        baseline=_bundle(raw.get("baseline")),
        augmented=augmented_bundle,
        baseline_progress=float(raw.get("baselineProgress") or 0.0),
        domain_augmentation_progress=(
            float(raw["domainAugmentationProgress"])
            if raw.get("domainAugmentationProgress") is not None
            else None
        ),
        domain_augmentation_text=(
            str(raw["domainAugmentationText"]) if raw.get("domainAugmentationText") else None
        ),
        augmented_progress=(float(raw["augmentedProgress"]) if raw.get("augmentedProgress") is not None else None),
        message=(str(raw["message"]) if raw.get("message") is not None else None),
    )
