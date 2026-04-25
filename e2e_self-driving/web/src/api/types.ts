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

export type TrainModelVariant = "legacy" | "mobilenet_v2" | "temporal3";

export interface TrainingTaskSummary {
  id: string;
  projectId: string;
  name: string;
  status: TaskStatus;
  createdAt: string;
  domainAugmentation: boolean;
  params: TrainingParamsSnapshot;
  resultSummary?: TaskResultSummary;
}

export interface TrainingParamsSnapshot {
  learningRate: number;
  batchSize: number;
  epochs: number;
  datasetId: string;
  datasetName?: string;
  projectId?: string;
  projectName?: string;
  modelVariant?: TrainModelVariant;
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
  baselineProgress: number;
  domainAugmentationProgress?: number;
  domainAugmentationText?: string;
  augmentedProgress?: number;
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
  steeringError: number;
  requestedEpochs?: number;
  completedEpochs?: number;
  bestEpoch?: number;
  stoppedEpoch?: number;
  earlyStopped?: boolean;
  bestValLoss?: number;
  finalTestLoss?: number;
  usedDedicatedTestSplit?: boolean;
  valStressMAE?: number;
  modelVariant?: TrainModelVariant;
  numFrames?: number;
  frameStride?: number;
  note?: string;
}

export interface TaskResultSummary {
  baseline: ModelMetrics;
  augmented?: ModelMetrics;
}

export interface DatasetItem {
  id: string;
  projectId: string;
  name: string;
  imageCount?: number;
  createdAt: string;
}

export interface CompareInferenceResult {
  baselineSteering: number;
  augmentedSteering?: number;
}

export interface ProjectItem {
  id: string;
  name: string;
  createdAt: string;
}

export interface ApiErrorBody {
  detail?: string;
  message?: string;
  code?: string;
}
