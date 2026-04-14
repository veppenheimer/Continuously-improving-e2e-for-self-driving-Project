/** 与 Python 后端约定的一致类型；字段名可按后端实际 JSON 在 adapter 中映射 */

export interface User {
  id: string;
  username: string;
  email?: string;
}

export type TaskStatus =
  | "pending"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "stopped";

export interface TrainingTaskSummary {
  id: string;
  /** 用户自定义或后端默认的显示名称 */
  name: string;
  status: TaskStatus;
  createdAt: string;
  /** 是否开启域增强（双模型对比） */
  domainAugmentation: boolean;
  params: TrainingParamsSnapshot;
  /** 简要结果，完成后由后端填充 */
  resultSummary?: TaskResultSummary;
}

export interface TrainingParamsSnapshot {
  learningRate: number;
  batchSize: number;
  epochs: number;
  datasetId: string;
  datasetName?: string;
  domainAugmentation?: boolean;
  domainBDatasetId?: string;
  domainBDatasetName?: string;
  cycleGanEpochs?: number;
  cycleGanDecayEpochs?: number;
  cycleGanBatchSize?: number;
  cycleGanSaveEpochFreq?: number;
  cycleGanSaveLatestFreq?: number;
  cycleGanLoadSize?: number;
  cycleGanCropSize?: number;
  cycleGanLambdaIdentity?: number;
  useCompetitionClassModel?: boolean;
  useCompetitionLiteModel?: boolean;
}

export interface LossPoint {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

export interface TaskProgress {
  status: TaskStatus;
  currentEpoch: number;
  totalEpochs: number;
  baseline: {
    trainLossSeries: LossPoint[];
    valLossSeries: LossPoint[];
  };
  augmented?: {
    trainLossSeries: LossPoint[];
    valLossSeries: LossPoint[];
  };
  competitionClass?: {
    trainLossSeries: LossPoint[];
    valLossSeries: LossPoint[];
  };
  competitionLite?: {
    trainLossSeries: LossPoint[];
    valLossSeries: LossPoint[];
  };
  baselineProgress: number;
  domainAugmentationProgress?: number;
  domainAugmentationText?: string;
  augmentedProgress?: number;
  competitionClassProgress?: number;
  competitionClassText?: string;
  competitionLiteProgress?: number;
  competitionLiteText?: string;
  message?: string;
}

export interface DomainAugPair {
  index: number;
  aName: string;
  cName: string;
}

export interface ModelMetrics {
  finalTrainLoss: number;
  finalValLoss: number;
  /** 转向角预测误差（如 MAE，单位与数据集一致） */
  steeringError: number;
  finalTrainAcc?: number;
  finalValAcc?: number;
  note?: string;
}

export interface TaskResultSummary {
  baseline: ModelMetrics;
  augmented?: ModelMetrics;
  competitionClass?: ModelMetrics;
  competitionLite?: ModelMetrics;
}

export interface DatasetItem {
  id: string;
  name: string;
  imageCount?: number;
  createdAt: string;
}

export interface CompareInferenceResult {
  baselineSteering: number;
  augmentedSteering?: number;
  competitionClassSteering?: number;
  competitionLiteSteering?: number;
}

/** 后端统一错误体（可调整） */
export interface ApiErrorBody {
  detail?: string;
  message?: string;
  code?: string;
}
