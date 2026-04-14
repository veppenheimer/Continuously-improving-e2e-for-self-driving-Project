import { useCallback, useEffect, useRef, useState } from "react";
import { fetchTaskProgress } from "@/api/services/tasks";
import type { TaskProgress } from "@/api/types";
import { WS_BASE_URL } from "@/config/env";
import { paths } from "@/api/endpoints";
import { authTokenRef } from "@/store/authTokenRef";

const POLL_MS = 2500;

export function useTaskLiveProgress(taskId: string | undefined) {
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const load = useCallback(async () => {
    if (!taskId) return;
    try {
      const p = await fetchTaskProgress(taskId);
      setProgress(p);
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    if (!taskId) return;

    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | undefined;

    const applyMessage = (raw: string) => {
      try {
        const p = JSON.parse(raw) as TaskProgress;
        if (!cancelled) setProgress(p);
      } catch {
        /* ignore */
      }
    };

    if (WS_BASE_URL) {
      void load();
      const token = authTokenRef.getToken();
      const q = token ? `?token=${encodeURIComponent(token)}` : "";
      const url = `${WS_BASE_URL}${paths.taskStream(taskId)}${q}`;
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => {
        if (!cancelled) setWsConnected(true);
      };
      ws.onclose = () => {
        if (!cancelled) setWsConnected(false);
      };
      ws.onmessage = (ev) => applyMessage(String(ev.data));
      ws.onerror = () => {
        if (!cancelled) setWsConnected(false);
      };
    } else {
      void load();
      interval = setInterval(() => void load(), POLL_MS);
    }

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
      wsRef.current?.close();
      wsRef.current = null;
      setWsConnected(false);
    };
  }, [taskId, load]);

  return { progress, loading, wsConnected, refetch: load };
}
