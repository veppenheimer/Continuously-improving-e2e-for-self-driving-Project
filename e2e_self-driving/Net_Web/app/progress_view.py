"""内存进度 dict → Pydantic TaskProgress。"""

from __future__ import annotations

from typing import Any

from app.schemas import LossPoint, LossSeriesBundle, TaskProgress


def _lp(xs: list[dict[str, Any]]) -> list[LossPoint]:
    out: list[LossPoint] = []
    for x in xs:
        out.append(
            LossPoint(
                epoch=int(x["epoch"]),
                train_loss=float(x["trainLoss"]),
                val_loss=float(x["valLoss"]),
            )
        )
    return out


def _bundle(raw: dict[str, Any] | None) -> LossSeriesBundle:
    if not raw:
        raw = {}
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
            "competitionClassProgress": fallback.get("competitionClassProgress"),
            "competitionClassText": fallback.get("competitionClassText"),
            "competitionLiteProgress": fallback.get("competitionLiteProgress"),
            "competitionLiteText": fallback.get("competitionLiteText"),
            "message": fallback.get("message"),
        }
        if fallback.get("status") == "completed":
            raw["currentEpoch"] = raw["totalEpochs"]
    elif fallback:
        # 兼容旧进度结构：补齐新增分支，避免前端完全不显示
        if raw.get("competitionClass") is None and fallback.get("competitionClassProgress") is not None:
            raw["competitionClass"] = {"trainLossSeries": [], "valLossSeries": []}
        if raw.get("competitionLite") is None and fallback.get("competitionLiteProgress") is not None:
            raw["competitionLite"] = {"trainLossSeries": [], "valLossSeries": []}

    aug_raw = raw.get("augmented")
    aug_bundle = None
    if isinstance(aug_raw, dict):
        aug_bundle = _bundle(aug_raw)
    comp_class_raw = raw.get("competitionClass")
    comp_class_bundle = _bundle(comp_class_raw) if isinstance(comp_class_raw, dict) else None
    comp_lite_raw = raw.get("competitionLite")
    comp_lite_bundle = _bundle(comp_lite_raw) if isinstance(comp_lite_raw, dict) else None

    total = int(raw.get("totalEpochs") or 0)
    if fallback and total == 0:
        total = int(fallback.get("totalEpochs") or 0)
    cur = int(raw.get("currentEpoch") or 0)

    return TaskProgress(
        status=str(raw.get("status", "pending")),
        current_epoch=cur,
        total_epochs=total,
        baseline=_bundle(raw.get("baseline")),
        augmented=aug_bundle,
        competition_class=comp_class_bundle,
        competition_lite=comp_lite_bundle,
        baseline_progress=float(raw.get("baselineProgress") or 0.0),
        domain_augmentation_progress=(
            float(raw["domainAugmentationProgress"])
            if raw.get("domainAugmentationProgress") is not None
            else None
        ),
        domain_augmentation_text=(str(raw["domainAugmentationText"]) if raw.get("domainAugmentationText") else None),
        augmented_progress=(float(raw["augmentedProgress"]) if raw.get("augmentedProgress") is not None else None),
        competition_class_progress=(
            float(raw["competitionClassProgress"]) if raw.get("competitionClassProgress") is not None else None
        ),
        competition_class_text=(str(raw["competitionClassText"]) if raw.get("competitionClassText") else None),
        competition_lite_progress=(
            float(raw["competitionLiteProgress"]) if raw.get("competitionLiteProgress") is not None else None
        ),
        competition_lite_text=(str(raw["competitionLiteText"]) if raw.get("competitionLiteText") else None),
        message=raw.get("message"),
    )
