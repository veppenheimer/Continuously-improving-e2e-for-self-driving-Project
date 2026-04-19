import { getApiClient } from "@/api/client";
import { paths } from "@/api/endpoints";
import type { ProjectItem } from "@/api/types";

export async function listProjects(): Promise<ProjectItem[]> {
  const { data } = await getApiClient().get<ProjectItem[]>(paths.projects);
  return data;
}

export async function createProject(name: string): Promise<ProjectItem> {
  const { data } = await getApiClient().post<ProjectItem>(paths.projects, { name });
  return data;
}

export async function renameProject(projectId: string, name: string): Promise<ProjectItem> {
  const { data } = await getApiClient().patch<ProjectItem>(paths.project(projectId), { name });
  return data;
}

export async function deleteProject(projectId: string): Promise<void> {
  await getApiClient().delete(paths.project(projectId));
}

