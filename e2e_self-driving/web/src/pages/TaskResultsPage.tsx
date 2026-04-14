import { useEffect, useState, useRef } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchTaskResults, getTask, inferCompare, downloadModelFile } from "@/api/services/tasks";
import type { TaskResultSummary, TrainingTaskSummary } from "@/api/types";
import { showApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Loader2, Download } from "lucide-react";

export function TaskResultsPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const [result, setResult] = useState<TaskResultSummary | null>(null);
  const [summary, setSummary] = useState<TrainingTaskSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [inferLoading, setInferLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [baselineAngle, setBaselineAngle] = useState<number | null>(null);
  const [augAngle, setAugAngle] = useState<number | null>(null);
  const [classAngle, setClassAngle] = useState<number | null>(null);
  const [liteAngle, setLiteAngle] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchTaskResults(taskId);
        if (!cancelled) setResult(data);
        try {
          const meta = await getTask(taskId);
          if (!cancelled) setSummary(meta);
        } catch {
          /* 元信息失败不影响结果展示 */
        }
      } catch (e) {
        showApiError(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  async function onInfer(e: React.FormEvent) {
    e.preventDefault();
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
      const res = await inferCompare(taskId, file);
      setBaselineAngle(res.baselineSteering);
      setAugAngle(res.augmentedSteering ?? null);
      setClassAngle(res.competitionClassSteering ?? null);
      setLiteAngle(res.competitionLiteSteering ?? null);
      toast.success("推理完成");
    } catch (err) {
      showApiError(err);
      setBaselineAngle(null);
      setAugAngle(null);
      setClassAngle(null);
      setLiteAngle(null);
    } finally {
      setInferLoading(false);
    }
  }

  async function onDownload(variant: "baseline" | "augmented") {
    if (!taskId) return;
    try {
      await downloadModelFile(taskId, variant, `${variant}_model.bin`);
      toast.success("开始下载");
    } catch (e) {
      showApiError(e);
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

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">训练结果与对比</h1>
          <p className="text-muted-foreground">
            {summary ? (
              <>
                <span className="font-medium text-foreground">{summary.name}</span>
                {" · "}
              </>
            ) : null}
            ID <code className="text-primary">{taskId}</code>
          </p>
        </div>
        <div className="flex gap-2">
          {summary?.domainAugmentation ? (
            <Button variant="secondary" size="sm" asChild>
              <Link to={`/tasks/${taskId}/domain-compare`}>A/C 图像对比</Link>
            </Button>
          ) : null}
          <Button variant="outline" size="sm" asChild>
            <Link to={`/tasks/${taskId}/monitor`}>返回监控</Link>
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {summary ? (
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>本次训练参数快照</CardTitle>
              <CardDescription>用于任务复盘与多次训练对比</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2 text-sm md:grid-cols-2">
              <p>数据集 A：{summary.params.datasetName || summary.params.datasetId}</p>
              <p>学习率：{summary.params.learningRate}</p>
              <p>Batch Size：{summary.params.batchSize}</p>
              <p>Epochs：{summary.params.epochs}</p>
              <p>域增强：{summary.domainAugmentation ? "开启" : "关闭"}</p>
              <p>竞赛分类模型：{summary.params.useCompetitionClassModel ? "开启" : "关闭"}</p>
              <p>竞赛轻量模型：{summary.params.useCompetitionLiteModel ? "开启" : "关闭"}</p>
              {summary.domainAugmentation ? (
                <>
                  <p>数据集 B：{summary.params.domainBDatasetName || summary.params.domainBDatasetId || "-"}</p>
                  <p>CycleGAN n_epochs：{summary.params.cycleGanEpochs ?? "-"}</p>
                  <p>CycleGAN n_epochs_decay：{summary.params.cycleGanDecayEpochs ?? "-"}</p>
                  <p>CycleGAN batch_size：{summary.params.cycleGanBatchSize ?? "-"}</p>
                  <p>CycleGAN save_epoch_freq：{summary.params.cycleGanSaveEpochFreq ?? "-"}</p>
                  <p>CycleGAN save_latest_freq：{summary.params.cycleGanSaveLatestFreq ?? "-"}</p>
                  <p>CycleGAN load_size：{summary.params.cycleGanLoadSize ?? "-"}</p>
                  <p>CycleGAN crop_size：{summary.params.cycleGanCropSize ?? "-"}</p>
                  <p>CycleGAN lambda_identity：{summary.params.cycleGanLambdaIdentity ?? "-"}</p>
                </>
              ) : null}
            </CardContent>
          </Card>
        ) : null}
        <Card>
          <CardHeader>
            <CardTitle>基准模型</CardTitle>
            <CardDescription>未做域增强</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p>最终训练 Loss：{result.baseline.finalTrainLoss.toFixed(6)}</p>
            <p>最终验证 Loss：{result.baseline.finalValLoss.toFixed(6)}</p>
            <p>转向角误差：{result.baseline.steeringError.toFixed(6)}</p>
            <Button size="sm" className="mt-4 gap-2" onClick={() => void onDownload("baseline")}>
              <Download className="h-4 w-4" />
              下载模型
            </Button>
          </CardContent>
        </Card>
        {result.augmented ? (
          <Card>
            <CardHeader>
              <CardTitle>域增强模型</CardTitle>
              <CardDescription>域增强训练</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p>最终训练 Loss：{result.augmented.finalTrainLoss.toFixed(6)}</p>
              <p>最终验证 Loss：{result.augmented.finalValLoss.toFixed(6)}</p>
              <p>转向角误差：{result.augmented.steeringError.toFixed(6)}</p>
              <Button size="sm" className="mt-4 gap-2" onClick={() => void onDownload("augmented")}>
                <Download className="h-4 w-4" />
                下载模型
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>域增强模型</CardTitle>
              <CardDescription>本次任务未开启域增强</CardDescription>
            </CardHeader>
          </Card>
        )}
        {result.competitionClass ? (
          <Card>
            <CardHeader>
              <CardTitle>竞赛分类模型</CardTitle>
              <CardDescription>e2e_competition/Net_class</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p>最终训练 Loss：{result.competitionClass.finalTrainLoss.toFixed(6)}</p>
              <p>最终验证 Loss：{result.competitionClass.finalValLoss.toFixed(6)}</p>
              <p>最终训练 Acc：{result.competitionClass.finalTrainAcc != null ? result.competitionClass.finalTrainAcc.toFixed(4) : "-"}</p>
              <p>最终验证 Acc：{result.competitionClass.finalValAcc != null ? result.competitionClass.finalValAcc.toFixed(4) : "-"}</p>
              {result.competitionClass.note ? <p className="text-amber-600">{result.competitionClass.note}</p> : null}
            </CardContent>
          </Card>
        ) : null}
        {result.competitionLite ? (
          <Card>
            <CardHeader>
              <CardTitle>竞赛轻量模型</CardTitle>
              <CardDescription>e2e_competition/Net_improve</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p>最终训练 Loss：{result.competitionLite.finalTrainLoss.toFixed(6)}</p>
              <p>最终验证 Loss：{result.competitionLite.finalValLoss.toFixed(6)}</p>
              <p>最终训练 Acc：{result.competitionLite.finalTrainAcc != null ? result.competitionLite.finalTrainAcc.toFixed(4) : "-"}</p>
              <p>最终验证 Acc：{result.competitionLite.finalValAcc != null ? result.competitionLite.finalValAcc.toFixed(4) : "-"}</p>
              {result.competitionLite.note ? <p className="text-amber-600">{result.competitionLite.note}</p> : null}
            </CardContent>
          </Card>
        ) : null}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>推理效果对比</CardTitle>
          <CardDescription>上传单张驾驶视角图像，并排查看两模型预测转向角</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <form onSubmit={onInfer} className="flex flex-wrap items-end gap-4">
            <div className="space-y-2">
              <Label htmlFor="infer-file">测试图像</Label>
              <input
                id="infer-file"
                ref={fileRef}
                type="file"
                accept="image/*"
                className="block text-sm text-muted-foreground file:mr-4 file:rounded-md file:border-0 file:bg-primary file:px-4 file:py-2 file:text-sm file:font-medium file:text-primary-foreground"
              />
            </div>
            <Button type="submit" disabled={inferLoading}>
              {inferLoading ? <Loader2 className="animate-spin" /> : "运行对比推理"}
            </Button>
          </form>

          {previewUrl && (
            <div className="grid gap-6 md:grid-cols-5">
              <div>
                <p className="mb-2 text-sm font-medium">输入</p>
                <img src={previewUrl} alt="输入" className="max-h-48 w-full rounded-md border object-contain" />
              </div>
              <div>
                <p className="mb-2 text-sm font-medium">基准模型预测</p>
                <div className="flex h-48 items-center justify-center rounded-md border bg-muted/30 text-2xl font-semibold text-primary">
                  {baselineAngle != null ? `${baselineAngle.toFixed(4)}°` : "—"}
                </div>
              </div>
              <div>
                <p className="mb-2 text-sm font-medium">域增强模型预测</p>
                <div className="flex h-48 items-center justify-center rounded-md border bg-muted/30 text-2xl font-semibold text-violet-400">
                  {augAngle != null ? `${augAngle.toFixed(4)}°` : result.augmented ? "—" : "未训练"}
                </div>
              </div>
              <div>
                <p className="mb-2 text-sm font-medium">竞赛分类模型预测</p>
                <div className="flex h-48 items-center justify-center rounded-md border bg-muted/30 text-2xl font-semibold text-emerald-500">
                  {classAngle != null ? `${classAngle.toFixed(4)}°` : summary?.params.useCompetitionClassModel ? "—" : "未训练"}
                </div>
              </div>
              <div>
                <p className="mb-2 text-sm font-medium">竞赛轻量模型预测</p>
                <div className="flex h-48 items-center justify-center rounded-md border bg-muted/30 text-2xl font-semibold text-green-600">
                  {liteAngle != null ? `${liteAngle.toFixed(4)}°` : summary?.params.useCompetitionLiteModel ? "—" : "未训练"}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
