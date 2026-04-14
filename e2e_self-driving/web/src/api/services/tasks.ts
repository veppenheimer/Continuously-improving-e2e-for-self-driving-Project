import { getApiClient } from "@/api/client";
import { paths } from "@/api/endpoints";
import type {
  CompareInferenceResult,
  DomainAugPair,
  TaskProgress,
  TaskResultSummary,
  TrainingTaskSummary,
} from "@/api/types";

export interface CreateTaskPayload {
  datasetId: string;
  learningRate: number;
  batchSize: number;
  epochs: number;
  domainAugmentation: boolean;
  domainBDatasetId?: string;
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
  /** 可选；留空时后端生成「训练 xxxxxxxx」 */
  name?: string;
}

export async function listTasks(): Promise<TrainingTaskSummary[]> {
  const { data } = await getApiClient().get<TrainingTaskSummary[]>(paths.tasks);
  return data;
}

export async function getTask(id: string): Promise<TrainingTaskSummary> {
  const { data } = await getApiClient().get<TrainingTaskSummary>(paths.task(id));
  return data;
}

export async function createTask(body: CreateTaskPayload): Promise<TrainingTaskSummary> {
  const { data } = await getApiClient().post<TrainingTaskSummary>(paths.tasks, body);
  return data;
}

export async function deleteTask(id: string): Promise<void> {
  await getApiClient().delete(paths.task(id));
}

export async function fetchTaskProgress(id: string): Promise<TaskProgress> {
  const { data } = await getApiClient().get<TaskProgress>(paths.taskProgress(id));
  return data;
}

export async function pauseTask(id: string): Promise<void> {
  await getApiClient().post(paths.taskPause(id));
}

export async function stopTask(id: string): Promise<void> {
  await getApiClient().post(paths.taskStop(id));
}

export async function fetchTaskResults(id: string): Promise<TaskResultSummary> {
  const { data } = await getApiClient().get<TaskResultSummary>(paths.taskResults(id));
  return data;
}

export async function inferCompare(id: string, file: File): Promise<CompareInferenceResult> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await getApiClient().post<CompareInferenceResult>(
    paths.taskInferCompare(id),
    form,
    { headers: { "Content-Type": "multipart/form-data" } },
  );
  return data;
}

export async function listDomainAugPairs(id: string): Promise<DomainAugPair[]> {
  const { data } = await getApiClient().get<DomainAugPair[]>(paths.taskDomainAugPairs(id));
  return data;
}

export async function fetchDomainAugImageBlob(
  id: string,
  index: number,
  kind: "a" | "c",
): Promise<Blob> {
  const { data } = await getApiClient().get(paths.taskDomainAugImage(id), {
    params: { index, kind },
    responseType: "blob",
  });
  return data as Blob;
}

export type ModelVariant = "baseline" | "augmented";

/** 带 Authorization 下载，避免 <a href> 无法带头的问题 */
export async function downloadModelFile(
  taskId: string,
  model: ModelVariant,
  filename: string,
): Promise<void> {
  const res = await getApiClient().get(paths.taskDownload(taskId), {
    params: { model },
    responseType: "blob",
  });
  const url = URL.createObjectURL(res.data);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
