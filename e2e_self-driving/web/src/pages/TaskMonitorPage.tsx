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
import { taskStatusLabel, taskStatusVariant } from "@/lib/taskStatus";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

export function TaskMonitorPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const { progress, loading, wsConnected, refetch } = useTaskLiveProgress(taskId);
  const [summary, setSummary] = useState<TrainingTaskSummary | null>(null);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    (async () => {
      try {
        const s = await getTask(taskId);
        if (!cancelled) setSummary(s);
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
    const base = [
      {
        name: "基准模型",
        train: progress.baseline.trainLossSeries,
        val: progress.baseline.valLossSeries,
        color: "#38bdf8",
      },
    ];
    if (progress.augmented) {
      base.push({
        name: "域增强模型",
        train: progress.augmented.trainLossSeries,
        val: progress.augmented.valLossSeries,
        color: "#a78bfa",
      });
    }
    if (progress.competitionClass) {
      base.push({
        name: "竞赛分类模型",
        train: progress.competitionClass.trainLossSeries,
        val: progress.competitionClass.valLossSeries,
        color: "#10b981",
      });
    }
    if (progress.competitionLite) {
      base.push({
        name: "竞赛轻量模型",
        train: progress.competitionLite.trainLossSeries,
        val: progress.competitionLite.valLossSeries,
        color: "#22c55e",
      });
    }
    return base;
  }, [progress]);

  async function onPause() {
    if (!taskId) return;
    try {
      await pauseTask(taskId);
      toast.success("已请求暂停");
      await refetch();
    } catch (e) {
      showApiError(e);
    }
  }

  async function onStop() {
    if (!taskId) return;
    try {
      await stopTask(taskId);
      toast.success("已请求终止");
      await refetch();
    } catch (e) {
      showApiError(e);
    }
  }

  if (!taskId) return null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">训练监控</h1>
          <p className="text-muted-foreground">
            {summary ? (
              <>
                <span className="font-medium text-foreground">{summary.name}</span>
                {" · "}
              </>
            ) : null}
            ID <code className="text-primary">{taskId}</code>
            {wsConnected ? " · WebSocket 已连接" : " · 轮询刷新"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {progress && (
            <Badge variant={taskStatusVariant(progress.status)}>{taskStatusLabel(progress.status)}</Badge>
          )}
          <Button variant="outline" size="sm" onClick={() => void refetch()}>
            刷新
          </Button>
          <Button variant="secondary" size="sm" onClick={() => void onPause()} disabled={progress?.status !== "running"}>
            暂停
          </Button>
          <Button variant="destructive" size="sm" onClick={() => void onStop()} disabled={!progress || progress.status === "completed" || progress.status === "stopped" || progress.status === "failed"}>
            终止
          </Button>
          {progress?.status === "completed" && (
            <>
              <Button size="sm" asChild>
                <Link to={`/tasks/${taskId}/results`}>查看结果</Link>
              </Button>
              {summary?.domainAugmentation ? (
                <Button size="sm" variant="secondary" asChild>
                  <Link to={`/tasks/${taskId}/domain-compare`}>查看A/C图像对比</Link>
                </Button>
              ) : null}
            </>
          )}
        </div>
      </div>

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
              <CardTitle>阶段进度</CardTitle>
              <CardDescription>
                Epoch {progress.currentEpoch} / {progress.totalEpochs}
                {progress.message ? ` · ${progress.message}` : ""}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <span>整体进度</span>
                  <span>{progress.totalEpochs ? ((progress.currentEpoch / progress.totalEpochs) * 100).toFixed(1) : "0.0"}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{
                      width: `${progress.totalEpochs ? Math.min(100, (progress.currentEpoch / progress.totalEpochs) * 100) : 0}%`,
                    }}
                  />
                </div>
              </div>
              <div>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <span>模型1（仅A）</span>
                  <span>{progress.baselineProgress.toFixed(1)}%</span>
                </div>
                <div
                  className="h-2 rounded-full bg-sky-500 transition-all"
                  style={{ width: `${Math.max(0, Math.min(100, progress.baselineProgress))}%` }}
                />
              </div>
              {summary?.domainAugmentation ? (
                <>
                  <div>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span>域增强（CycleGAN生成C）</span>
                      <span>{(progress.domainAugmentationProgress ?? 0).toFixed(1)}%</span>
                    </div>
                    <div
                      className="h-2 rounded-full bg-amber-500 transition-all"
                      style={{ width: `${Math.max(0, Math.min(100, progress.domainAugmentationProgress ?? 0))}%` }}
                    />
                    <p className="mt-1 text-xs text-muted-foreground">
                      {progress.domainAugmentationText ?? "等待执行域增强"}
                    </p>
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span>模型2（A + C）</span>
                      <span>{(progress.augmentedProgress ?? 0).toFixed(1)}%</span>
                    </div>
                    <div
                      className="h-2 rounded-full bg-violet-500 transition-all"
                      style={{ width: `${Math.max(0, Math.min(100, progress.augmentedProgress ?? 0))}%` }}
                    />
                  </div>
                </>
              ) : null}
              {summary?.params.useCompetitionClassModel ? (
                <div>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span>竞赛分类模型（Net_class）</span>
                    <span>{(progress.competitionClassProgress ?? 0).toFixed(1)}%</span>
                  </div>
                  <div
                    className="h-2 rounded-full bg-emerald-500 transition-all"
                    style={{ width: `${Math.max(0, Math.min(100, progress.competitionClassProgress ?? 0))}%` }}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    {progress.competitionClassText ?? "等待执行分类模型训练"}
                  </p>
                </div>
              ) : null}
              {summary?.params.useCompetitionLiteModel ? (
                <div>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span>竞赛轻量模型（Net_improve）</span>
                    <span>{(progress.competitionLiteProgress ?? 0).toFixed(1)}%</span>
                  </div>
                  <div
                    className="h-2 rounded-full bg-green-600 transition-all"
                    style={{ width: `${Math.max(0, Math.min(100, progress.competitionLiteProgress ?? 0))}%` }}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    {progress.competitionLiteText ?? "等待执行轻量模型训练"}
                  </p>
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Loss 曲线</CardTitle>
              <CardDescription>实线为训练 Loss，虚线为验证 Loss</CardDescription>
            </CardHeader>
            <CardContent>
              {series.length > 0 && series.some((s) => s.train.length || s.val.length) ? (
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
