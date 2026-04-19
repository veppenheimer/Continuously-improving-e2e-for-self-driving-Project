import { getApiClient } from "@/api/client";
import { paths } from "@/api/endpoints";
import type { DatasetItem } from "@/api/types";

export async function listDatasets(projectId?: string): Promise<DatasetItem[]> {
  const { data } = await getApiClient().get<DatasetItem[]>(paths.datasets, {
    params: projectId ? { projectId } : undefined,
  });
  return data;
}

export async function uploadDatasetZip(projectId: string, file: File, name?: string): Promise<DatasetItem> {
  const form = new FormData();
  form.append("file", file);
  form.append("projectId", projectId);
  if (name) form.append("name", name);
  const { data } = await getApiClient().post<DatasetItem>(paths.datasetUpload, form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 600_000,
  });
  return data;
}

export async function deleteDataset(datasetId: string): Promise<void> {
  await getApiClient().delete(paths.dataset(datasetId));
}
