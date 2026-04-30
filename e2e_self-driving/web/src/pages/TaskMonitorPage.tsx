import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getTask, pauseTask, stopTask } from "@/api/services/tasks";
import type { TrainingTaskSummary } from "@/api/types";
import { showApiError } from "@/api/client";
import { LossChart } from "@/components/charts/LossChart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useTaskLiveProgress } from "@/hooks/useTaskLiveProgress";
import { modelVariantLabel } from "@/lib/modelVariant";
import { taskStatusLabel, taskStatusVariant } from "@/lib/taskStatus";
import { Activity, BarChart3, Loader2, Pause, RefreshCw, Square, Trophy, Wifi, WifiOff } from "lucide-react";
import { toast } from "sonner";

function ProgressLine({
  label,
  value,
  barClassName,
  description,
}: {
  label: string;
  value: number;
  barClassName: string;
  description?: string;
}) {
  const safeValue = Math.max(0, Math.min(100, value));
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-xs text-muted-foreground">{safeValue.toFixed(1)}%</span>
      </div>
      <div className="ag-progress-track">
        <div className={`ag-progress-bar ${barClassName}`} style={{ width: `${safeValue}%` }} />
      </div>
      {description ? <p className="text-xs text-muted-foreground">{description}</p> : null}
    </div>
  );
}

export function TaskMonitorPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const { progress, loading, wsConnected, refetch } = useTaskLiveProgress(taskId);
  const [summary, setSummary] = useState<TrainingTaskSummary | null>(null);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    (async () => {
      try {
        const task = await getTask(taskId);
        if (!cancelled) setSummary(task);
      } catch {
        if (!cancelled) setSummary(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  const series = useMemo(() => {
    if (!progress) return [];
    const items = [
      {
        name: "基准阶段",
        train: progress.baseline.trainLossSeries,
        val: progress.baseline.valLossSeries,
        color: "#38bdf8",
      },
    ];
    if (progress.augmented) {
      items.push({
        name: "增强阶段",
        train: progress.augmented.trainLossSeries,
        val: progress.augmented.valLossSeries,
        color: "#a78bfa",
      });
    }
    return items;
  }, [progress]);

  async function onPause() {
    if (!taskId) return;
    try {
      await pauseTask(taskId);
      toast.success("已请求暂停/继续");
      await refetch();
    } catch (error) {
      showApiError(error);
    }
  }

  async function onStop() {
    if (!taskId) return;
    try {
      await stopTask(taskId);
      toast.success("已请求终止");
      await refetch();
    } catch (error) {
      showApiError(error);
    }
  }

  if (!taskId) return null;

  return (
    <div className="space-y-6">
      <section className="ag-page-hero">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="ag-eyebrow">
              <Activity className="h-3.5 w-3.5" />
              Training Monitor
            </div>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">训练监控</h1>
            <p className="mt-2 text-sm text-muted-foreground">
            {summary ? (
              <>
                <span className="font-medium text-foreground">{summary.name}</span>
                {" · "}
                {modelVariantLabel(summary.params.modelVariant)}
                {" · "}
              </>
            ) : null}
            ID <code className="text-primary">{taskId}</code>
            </p>
            <div className="mt-3 inline-flex items-center gap-2 rounded-md border border-white/10 bg-background/45 px-2.5 py-1 text-xs text-muted-foreground">
              {wsConnected ? <Wifi className="h-3.5 w-3.5 text-primary" /> : <WifiOff className="h-3.5 w-3.5 text-violet-300" />}
              {wsConnected ? "WebSocket 已连接" : "轮询刷新"}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
          {progress ? <Badge variant={taskStatusVariant(progress.status)}>{taskStatusLabel(progress.status)}</Badge> : null}
          <Button variant="outline" size="sm" onClick={() => void refetch()}>
            <RefreshCw className="h-4 w-4" />刷新
          </Button>
          <Button variant="secondary" size="sm" onClick={() => void onPause()} disabled={progress?.status !== "running"}>
            <Pause className="h-4 w-4" />暂停
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => void onStop()}
            disabled={!progress || progress.status === "completed" || progress.status === "stopped" || progress.status === "failed"}
          >
            <Square className="h-4 w-4" />终止
          </Button>
          {progress?.status === "completed" ? (
            <>
              <Button size="sm" asChild>
                <Link to={`/tasks/${taskId}/results`}>
                  <Trophy className="h-4 w-4" />查看结果
                </Link>
              </Button>
              {summary?.domainAugmentation ? (
                <Button size="sm" variant="secondary" asChild>
                  <Link to={`/tasks/${taskId}/domain-compare`}>查看 A/C 图像对比</Link>
                </Button>
              ) : null}
            </>
          ) : null}
          </div>
        </div>
      </section>

      {loading && !progress ? (
        <div className="flex justify-center py-24">
          <Loader2 className="h-10 w-10 animate-spin text-muted-foreground" />
        </div>
      ) : !progress ? (
        <p className="text-muted-foreground">暂无进度数据</p>
      ) : (
        <>
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-primary/20 bg-primary/10 text-primary">
                  <Activity className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>阶段进度</CardTitle>
                  <CardDescription>
                    Epoch {progress.currentEpoch} / {progress.totalEpochs}
                    {progress.message ? ` · ${progress.message}` : ""}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <ProgressLine
                label="整体进度"
                value={progress.totalEpochs ? (progress.currentEpoch / progress.totalEpochs) * 100 : 0}
                barClassName="bg-primary"
              />

              <ProgressLine label="基准阶段（仅 A）" value={progress.baselineProgress} barClassName="bg-sky-400" />

              {summary?.domainAugmentation ? (
                <>
                  <ProgressLine
                    label="域增强（CycleGAN 生成 C）"
                    value={progress.domainAugmentationProgress ?? 0}
                    barClassName="bg-violet-400"
                    description={progress.domainAugmentationText ?? "等待执行域增强"}
                  />
                  <ProgressLine label="增强阶段（A + C）" value={progress.augmentedProgress ?? 0} barClassName="bg-fuchsia-400" />
                </>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-violet-300/20 bg-violet-400/10 text-violet-300">
                  <BarChart3 className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>Loss 曲线</CardTitle>
                  <CardDescription>实线为训练 Loss，虚线为验证 Loss</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {series.length > 0 && series.some((item) => item.train.length || item.val.length) ? (
                <LossChart series={series} />
              ) : (
                <p className="py-12 text-center text-sm text-muted-foreground">等待后端写入曲线数据…</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
