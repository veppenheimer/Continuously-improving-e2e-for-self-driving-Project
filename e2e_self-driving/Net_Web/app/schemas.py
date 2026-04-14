"""与前端约定的 Pydantic 模型（camelCase 序列化）。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class UserOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    username: str
    email: Optional[str] = None


class TokenUserResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    token: str
    user: UserOut


class RegisterBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    email: Optional[str] = None


class LoginBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str
    password: str


class LossPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    epoch: int
    train_loss: float = Field(serialization_alias="trainLoss")
    val_loss: float = Field(serialization_alias="valLoss")


class LossSeriesBundle(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    train_loss_series: list[LossPoint] = Field(serialization_alias="trainLossSeries")
    val_loss_series: list[LossPoint] = Field(serialization_alias="valLossSeries")


class TaskProgress(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    status: str
    current_epoch: int = Field(serialization_alias="currentEpoch")
    total_epochs: int = Field(serialization_alias="totalEpochs")
    baseline: LossSeriesBundle
    augmented: Optional[LossSeriesBundle] = None
    competition_class: Optional[LossSeriesBundle] = Field(default=None, serialization_alias="competitionClass")
    competition_lite: Optional[LossSeriesBundle] = Field(default=None, serialization_alias="competitionLite")
    baseline_progress: float = Field(default=0.0, serialization_alias="baselineProgress")
    domain_augmentation_progress: Optional[float] = Field(
        default=None, serialization_alias="domainAugmentationProgress"
    )
    domain_augmentation_text: Optional[str] = Field(default=None, serialization_alias="domainAugmentationText")
    augmented_progress: Optional[float] = Field(default=None, serialization_alias="augmentedProgress")
    competition_class_progress: Optional[float] = Field(default=None, serialization_alias="competitionClassProgress")
    competition_class_text: Optional[str] = Field(default=None, serialization_alias="competitionClassText")
    competition_lite_progress: Optional[float] = Field(default=None, serialization_alias="competitionLiteProgress")
    competition_lite_text: Optional[str] = Field(default=None, serialization_alias="competitionLiteText")
    message: Optional[str] = None


class DomainAugPairOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    index: int
    a_name: str = Field(serialization_alias="aName")
    c_name: str = Field(serialization_alias="cName")


class ModelMetrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    final_train_loss: float = Field(serialization_alias="finalTrainLoss")
    final_val_loss: float = Field(serialization_alias="finalValLoss")
    steering_error: float = Field(serialization_alias="steeringError")
    final_train_acc: Optional[float] = Field(default=None, serialization_alias="finalTrainAcc")
    final_val_acc: Optional[float] = Field(default=None, serialization_alias="finalValAcc")
    note: Optional[str] = None


class TaskResultSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    baseline: ModelMetrics
    augmented: Optional[ModelMetrics] = None
    competition_class: Optional[ModelMetrics] = Field(default=None, serialization_alias="competitionClass")
    competition_lite: Optional[ModelMetrics] = Field(default=None, serialization_alias="competitionLite")


class CreateTaskBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: str = Field(alias="datasetId")
    learning_rate: float = Field(alias="learningRate", gt=0)
    batch_size: int = Field(alias="batchSize", ge=1)
    epochs: int = Field(ge=1, le=5000)
    domain_augmentation: bool = Field(alias="domainAugmentation")
    domain_b_dataset_id: Optional[str] = Field(default=None, alias="domainBDatasetId")
    cyclegan_epochs: int = Field(default=20, alias="cycleGanEpochs", ge=1, le=1000)
    cyclegan_decay_epochs: int = Field(default=20, alias="cycleGanDecayEpochs", ge=0, le=1000)
    cyclegan_batch_size: int = Field(default=1, alias="cycleGanBatchSize", ge=1, le=64)
    cyclegan_save_epoch_freq: int = Field(default=5, alias="cycleGanSaveEpochFreq", ge=1, le=5000)
    cyclegan_save_latest_freq: int = Field(default=5000, alias="cycleGanSaveLatestFreq", ge=1, le=1000000)
    cyclegan_load_size: int = Field(default=286, alias="cycleGanLoadSize", ge=64, le=2048)
    cyclegan_crop_size: int = Field(default=256, alias="cycleGanCropSize", ge=64, le=2048)
    cyclegan_lambda_identity: float = Field(default=0.5, alias="cycleGanLambdaIdentity", ge=0, le=10)
    use_competition_class_model: bool = Field(default=False, alias="useCompetitionClassModel")
    use_competition_lite_model: bool = Field(default=False, alias="useCompetitionLiteModel")
    name: Optional[str] = Field(default=None, max_length=128, description="任务显示名称，可空则后端生成默认名")


class DatasetItemOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    name: str
    image_count: Optional[int] = Field(default=None, serialization_alias="imageCount")
    created_at: str = Field(serialization_alias="createdAt")


class TrainingTaskSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    name: str
    status: str
    created_at: str = Field(serialization_alias="createdAt")
    domain_augmentation: bool = Field(serialization_alias="domainAugmentation")
    params: dict[str, Any]
    result_summary: Optional[dict[str, Any]] = Field(default=None, serialization_alias="resultSummary")


class CompareInferOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    baseline_steering: float = Field(serialization_alias="baselineSteering")
    augmented_steering: Optional[float] = Field(default=None, serialization_alias="augmentedSteering")
    competition_class_steering: Optional[float] = Field(default=None, serialization_alias="competitionClassSteering")
    competition_lite_steering: Optional[float] = Field(default=None, serialization_alias="competitionLiteSteering")
