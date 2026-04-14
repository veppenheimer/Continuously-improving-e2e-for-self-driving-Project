import type { TaskStatus } from "@/api/types";

export function taskStatusLabel(s: TaskStatus): string {
  const map: Record<TaskStatus, string> = {
    pending: "排队中",
    running: "训练中",
    paused: "已暂停",
    completed: "已完成",
    failed: "失败",
    stopped: "已终止",
  };
  return map[s] ?? s;
}

export function taskStatusVariant(
  s: TaskStatus,
):
  | "default"
  | "secondary"
  | "success"
  | "warning"
  | "muted"
  | "outline"
  | "destructive" {
  switch (s) {
    case "running":
      return "default";
    case "completed":
      return "success";
    case "paused":
      return "warning";
    case "failed":
    case "stopped":
      return "destructive";
    default:
      return "muted";
  }
}
