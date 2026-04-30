import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { downloadModelFile, fetchTaskResults, getTask, inferCompare } from "@/api/services/tasks";
import type { TaskResultSummary, TrainingTaskSummary } from "@/api/types";
import { showApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { modelVariantLabel } from "@/lib/modelVariant";
import { ArrowLeft, BarChart3, BrainCircuit, Download, Gauge, ImagePlus, Loader2, UploadCloud } from "lucide-react";
import { toast } from "sonner";

function MetricTile({
  label,
  value,
  tone = "text-primary",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="ag-kpi">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`mt-2 font-mono text-lg font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

export function TaskResultsPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const [result, setResult] = useState<TaskResultSummary | null>(null);
  const [summary, setSummary] = useState<TrainingTaskSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [inferLoading, setInferLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [baselineAngle, setBaselineAngle] = useState<number | null>(null);
  const [augmentedAngle, setAugmentedAngle] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    (async () => {
      try {
        const [taskResult, taskSummary] = await Promise.all([fetchTaskResults(taskId), getTask(taskId)]);
        if (!cancelled) {
          setResult(taskResult);
          setSummary(taskSummary);
        }
      } catch (error) {
        showApiError(error);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  async function onInfer(event: React.FormEvent) {
    event.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!taskId || !file) {
      toast.error("请选择一张测试图像");
      return;
    }
    setInferLoading(true);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    try {
      const response = await inferCompare(taskId, file);
      setBaselineAngle(response.baselineSteering);
      setAugmentedAngle(response.augmentedSteering ?? null);
      toast.success("推理完成");
    } catch (error) {
      showApiError(error);
      setBaselineAngle(null);
      setAugmentedAngle(null);
    } finally {
      setInferLoading(false);
    }
  }

  async function onDownload(stage: "baseline" | "augmented") {
    if (!taskId) return;
    try {
      await downloadModelFile(taskId, stage, `${stage}_checkpoint.pth`);
      toast.success("开始下载");
    } catch (error) {
      showApiError(error);
    }
  }

  if (!taskId) return null;

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="h-10 w-10 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!result) {
    return <p className="text-muted-foreground">暂无结果，请确认任务已完成。</p>;
  }

  const activeModelVariant = summary?.params.modelVariant ?? result.baseline.modelVariant;

  return (
    <div className="space-y-8">
      <section className="ag-page-hero">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="ag-eyebrow">
              <BarChart3 className="h-3.5 w-3.5" />
              Result Lab
            </div>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">训练结果与对比</h1>
            <p className="mt-2 text-sm text-muted-foreground">
            {summary ? (
              <>
                <span className="font-medium text-foreground">{summary.name}</span>
                {" · "}
                {modelVariantLabel(activeModelVariant)}
                {" · "}
              </>
            ) : null}
            ID <code className="text-primary">{taskId}</code>
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
          {summary?.domainAugmentation ? (
            <Button variant="secondary" size="sm" asChild>
              <Link to={`/tasks/${taskId}/domain-compare`}>A/C 图像对比</Link>
            </Button>
          ) : null}
          <Button variant="outline" size="sm" asChild>
            <Link to={`/tasks/${taskId}/monitor`}>
              <ArrowLeft className="h-4 w-4" />返回监控
            </Link>
          </Button>
          </div>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        {summary ? (
          <Card className="md:col-span-2">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-primary/20 bg-primary/10 text-primary">
                  <BrainCircuit className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>本次训练参数快照</CardTitle>
                  <CardDescription>用于任务复盘与多次训练对比</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm md:grid-cols-3">
              <div className="ag-kpi">
                <p className="text-xs text-muted-foreground">模型架构</p>
                <p className="mt-1 font-medium">{modelVariantLabel(activeModelVariant)}</p>
              </div>
              <div className="ag-kpi">
                <p className="text-xs text-muted-foreground">数据集 A</p>
                <p className="mt-1 truncate font-medium">{summary.params.datasetName || summary.params.datasetId}</p>
              </div>
              <div className="ag-kpi">
                <p className="text-xs text-muted-foreground">域增强</p>
                <p className="mt-1 font-medium">{summary.domainAugmentation ? "开启" : "关闭"}</p>
              </div>
              <MetricTile label="学习率" value={String(summary.params.learningRate)} />
              <MetricTile label="Batch Size" value={String(summary.params.batchSize)} tone="text-violet-300" />
              <MetricTile label="Epochs" value={String(summary.params.epochs)} tone="text-sky-300" />
              {summary.domainAugmentation ? (
                <>
                  <div className="ag-kpi md:col-span-3">
                    <p className="text-xs text-muted-foreground">数据集 B</p>
                    <p className="mt-1 truncate font-medium">{summary.params.domainBDatasetName || summary.params.domainBDatasetId || "-"}</p>
                  </div>
                  <MetricTile label="CycleGAN n_epochs" value={String(summary.params.cycleGanEpochs ?? "-")} tone="text-violet-300" />
                  <MetricTile label="CycleGAN decay" value={String(summary.params.cycleGanDecayEpochs ?? "-")} tone="text-violet-300" />
                  <MetricTile label="CycleGAN batch" value={String(summary.params.cycleGanBatchSize ?? "-")} tone="text-violet-300" />
                  <MetricTile label="save_epoch_freq" value={String(summary.params.cycleGanSaveEpochFreq ?? "-")} />
                  <MetricTile label="save_latest_freq" value={String(summary.params.cycleGanSaveLatestFreq ?? "-")} />
                  <MetricTile label="lambda_identity" value={String(summary.params.cycleGanLambdaIdentity ?? "-")} />
                  <MetricTile label="load_size" value={String(summary.params.cycleGanLoadSize ?? "-")} tone="text-sky-300" />
                  <MetricTile label="crop_size" value={String(summary.params.cycleGanCropSize ?? "-")} tone="text-sky-300" />
                </>
              ) : null}
            </CardContent>
          </Card>
        ) : null}

        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-md border border-primary/20 bg-primary/10 text-primary">
                <Gauge className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>基准阶段</CardTitle>
                <CardDescription>仅使用 A 域数据训练</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-3 text-sm sm:grid-cols-2">
            <MetricTile label="最终训练 Loss" value={result.baseline.finalTrainLoss.toFixed(6)} />
            <MetricTile label="最终验证 Loss" value={result.baseline.finalValLoss.toFixed(6)} tone="text-violet-300" />
            <MetricTile label="转向角误差" value={result.baseline.steeringError.toFixed(6)} tone="text-fuchsia-300" />
            <MetricTile label="最佳 Epoch" value={String(result.baseline.bestEpoch ?? "-")} tone="text-sky-300" />
            <MetricTile label="最佳 Val Stress MAE" value={result.baseline.valStressMAE?.toFixed(6) ?? "-"} />
            <div className="flex items-end">
            <Button size="sm" className="mt-4 gap-2" onClick={() => void onDownload("baseline")}>
              <Download className="h-4 w-4" />
              下载 checkpoint
            </Button>
            </div>
          </CardContent>
        </Card>

        {result.augmented ? (
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-violet-300/20 bg-violet-400/10 text-violet-300">
                  <BrainCircuit className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>增强阶段</CardTitle>
                  <CardDescription>A + C 混合训练</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm sm:grid-cols-2">
              <MetricTile label="最终训练 Loss" value={result.augmented.finalTrainLoss.toFixed(6)} />
              <MetricTile label="最终验证 Loss" value={result.augmented.finalValLoss.toFixed(6)} tone="text-violet-300" />
              <MetricTile label="转向角误差" value={result.augmented.steeringError.toFixed(6)} tone="text-fuchsia-300" />
              <MetricTile label="最佳 Epoch" value={String(result.augmented.bestEpoch ?? "-")} tone="text-sky-300" />
              <MetricTile label="最佳 Val Stress MAE" value={result.augmented.valStressMAE?.toFixed(6) ?? "-"} />
              <div className="flex items-end">
              <Button size="sm" className="mt-4 gap-2" onClick={() => void onDownload("augmented")}>
                <Download className="h-4 w-4" />
                下载 checkpoint
              </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-white/10 bg-muted/55 text-muted-foreground">
                  <BrainCircuit className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>增强阶段</CardTitle>
                  <CardDescription>本次任务未开启域增强</CardDescription>
                </div>
              </div>
            </CardHeader>
          </Card>
        )}
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-sky-300/20 bg-sky-400/10 text-sky-300">
              <ImagePlus className="h-4 w-4" />
            </div>
            <div>
              <CardTitle>推理效果对比</CardTitle>
              <CardDescription>上传单张驾驶视角图像，对比基准阶段与增强阶段的预测转向角</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <form onSubmit={onInfer} className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
            <div className="space-y-2">
              <Label htmlFor="infer-file">测试图像</Label>
              <input
                id="infer-file"
                ref={fileRef}
                type="file"
                accept="image/*"
                className="ag-file-input"
              />
            </div>
            <Button type="submit" disabled={inferLoading}>
              {inferLoading ? <Loader2 className="animate-spin" /> : <UploadCloud className="h-4 w-4" />}
              运行对比推理
            </Button>
          </form>

          {previewUrl ? (
            <div className="grid gap-6 md:grid-cols-3">
              <div className="ag-panel-soft p-3">
                <p className="mb-2 text-sm font-medium">输入</p>
                <img src={previewUrl} alt="输入" className="max-h-48 w-full rounded-md border border-white/10 bg-black/25 object-contain" />
              </div>
              <div className="ag-panel-soft p-3">
                <p className="mb-2 text-sm font-medium">基准阶段预测</p>
                <div className="flex h-48 items-center justify-center rounded-md border border-white/10 bg-background/45 font-mono text-3xl font-semibold text-primary">
                  {baselineAngle != null ? `${baselineAngle.toFixed(4)}°` : "—"}
                </div>
              </div>
              <div className="ag-panel-soft p-3">
                <p className="mb-2 text-sm font-medium">增强阶段预测</p>
                <div className="flex h-48 items-center justify-center rounded-md border border-white/10 bg-background/45 font-mono text-3xl font-semibold text-violet-300">
                  {augmentedAngle != null ? `${augmentedAngle.toFixed(4)}°` : result.augmented ? "—" : "未训练"}
                </div>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
