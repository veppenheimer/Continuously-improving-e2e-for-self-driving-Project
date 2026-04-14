import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listDatasets } from "@/api/services/datasets";
import { createTask } from "@/api/services/tasks";
import type { DatasetItem } from "@/api/types";
import { showApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

export function NewTrainPage() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [domainBDatasetId, setDomainBDatasetId] = useState("");
  const [lr, setLr] = useState("1e-4");
  const [batch, setBatch] = useState("32");
  const [epochs, setEpochs] = useState("100");
  const [domainAug, setDomainAug] = useState(false);
  const [cycleGanEpochs, setCycleGanEpochs] = useState("20");
  const [cycleGanDecayEpochs, setCycleGanDecayEpochs] = useState("20");
  const [cycleGanBatchSize, setCycleGanBatchSize] = useState("1");
  const [cycleGanSaveEpochFreq, setCycleGanSaveEpochFreq] = useState("5");
  const [cycleGanSaveLatestFreq, setCycleGanSaveLatestFreq] = useState("5000");
  const [cycleGanLoadSize, setCycleGanLoadSize] = useState("286");
  const [cycleGanCropSize, setCycleGanCropSize] = useState("256");
  const [cycleGanLambdaIdentity, setCycleGanLambdaIdentity] = useState("0.5");
  const [useCompetitionClassModel, setUseCompetitionClassModel] = useState(false);
  const [useCompetitionLiteModel, setUseCompetitionLiteModel] = useState(false);
  const [taskName, setTaskName] = useState("");
  const [loading, setLoading] = useState(false);
  const [listLoading, setListLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await listDatasets();
        if (!cancelled) {
          setDatasets(data);
          if (data[0]) {
            setDatasetId(data[0].id);
            const fallbackB = data.find((x) => x.id !== data[0].id) ?? data[0];
            setDomainBDatasetId(fallbackB.id);
          }
        }
      } catch (e) {
        showApiError(e);
      } finally {
        if (!cancelled) setListLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const learningRate = Number(lr);
    const batchSize = Number(batch);
    const epochsN = Number(epochs);
    const cycEpochN = Number(cycleGanEpochs);
    const cycDecayEpochN = Number(cycleGanDecayEpochs);
    const cycBatchN = Number(cycleGanBatchSize);
    const cycSaveEpochN = Number(cycleGanSaveEpochFreq);
    const cycSaveLatestN = Number(cycleGanSaveLatestFreq);
    const cycLoadSizeN = Number(cycleGanLoadSize);
    const cycCropSizeN = Number(cycleGanCropSize);
    const cycLambdaIdentityN = Number(cycleGanLambdaIdentity);
    if (!datasetId || Number.isNaN(learningRate) || Number.isNaN(batchSize) || Number.isNaN(epochsN)) {
      toast.error("请检查表单填写");
      return;
    }
    if (domainAug) {
      if (!domainBDatasetId) {
        toast.error("开启域增强时请选择 B 域数据集");
        return;
      }
      if (domainBDatasetId === datasetId) {
        toast.error("A 域和 B 域数据集不能相同");
        return;
      }
      if (
        Number.isNaN(cycEpochN) ||
        Number.isNaN(cycDecayEpochN) ||
        Number.isNaN(cycBatchN) ||
        Number.isNaN(cycSaveEpochN) ||
        Number.isNaN(cycSaveLatestN) ||
        Number.isNaN(cycLoadSizeN) ||
        Number.isNaN(cycCropSizeN) ||
        Number.isNaN(cycLambdaIdentityN)
      ) {
        toast.error("请检查 CycleGAN 参数");
        return;
      }
    }
    setLoading(true);
    try {
      const trimmed = taskName.trim();
      const task = await createTask({
        datasetId,
        learningRate,
        batchSize,
        epochs: epochsN,
        domainAugmentation: domainAug,
        useCompetitionClassModel,
        useCompetitionLiteModel,
        ...(domainAug
          ? {
              domainBDatasetId,
              cycleGanEpochs: cycEpochN,
              cycleGanDecayEpochs: cycDecayEpochN,
              cycleGanBatchSize: cycBatchN,
              cycleGanSaveEpochFreq: cycSaveEpochN,
              cycleGanSaveLatestFreq: cycSaveLatestN,
              cycleGanLoadSize: cycLoadSizeN,
              cycleGanCropSize: cycCropSizeN,
              cycleGanLambdaIdentity: cycLambdaIdentityN,
            }
          : {}),
        ...(trimmed ? { name: trimmed } : {}),
      });
      toast.success("训练任务已创建");
      navigate(`/tasks/${task.id}/monitor`);
    } catch (err) {
      showApiError(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">新建训练</h1>
        <p className="text-muted-foreground">
          配置超参数并提交到后端；开启域增强后将按 CycleGAN 先生成 C（A→B 风格），再训练 A+C 增强模型
        </p>
      </div>

      <Card className="max-w-xl">
        <CardHeader>
          <CardTitle>训练参数</CardTitle>
          <CardDescription>与 Net/train.py 中逻辑对应，由后端实际执行</CardDescription>
        </CardHeader>
        <CardContent>
          {listLoading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : datasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">请先在「数据集」页上传 ZIP。</p>
          ) : (
            <form onSubmit={onSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="taskName">任务名称（可选）</Label>
                <Input
                  id="taskName"
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                  placeholder="例如：夜间场景 baseline · 留空则自动生成"
                  maxLength={128}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dataset">数据集</Label>
                <select
                  id="dataset"
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.id.slice(0, 8)}…)
                    </option>
                  ))}
                </select>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="lr">学习率</Label>
                  <Input id="lr" value={lr} onChange={(e) => setLr(e.target.value)} placeholder="1e-4" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="bs">Batch Size</Label>
                  <Input id="bs" value={batch} onChange={(e) => setBatch(e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="ep">Epochs</Label>
                  <Input id="ep" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
                </div>
              </div>
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={domainAug}
                  onChange={(e) => setDomainAug(e.target.checked)}
                  className="h-4 w-4 rounded border-input"
                />
                开启域增强（后端训练基准 + 增强双模型）
              </label>
              <div className="space-y-2 rounded-lg border p-4">
                <p className="text-sm font-medium">附加模型（e2e_competition）</p>
                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={useCompetitionClassModel}
                    onChange={(e) => setUseCompetitionClassModel(e.target.checked)}
                    className="h-4 w-4 rounded border-input"
                  />
                  训练分类模型（Net_class/train.py）
                </label>
                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={useCompetitionLiteModel}
                    onChange={(e) => setUseCompetitionLiteModel(e.target.checked)}
                    className="h-4 w-4 rounded border-input"
                  />
                  训练轻量模型（Net_improve/train.py）
                </label>
              </div>
              {domainAug ? (
                <div className="space-y-4 rounded-lg border p-4">
                  <p className="text-sm font-medium">CycleGAN 域增强参数</p>
                  <div className="space-y-2">
                    <Label htmlFor="datasetB">B 域数据集（风格域）</Label>
                    <select
                      id="datasetB"
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={domainBDatasetId}
                      onChange={(e) => setDomainBDatasetId(e.target.value)}
                    >
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} ({d.id.slice(0, 8)}…)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-3">
                    <div className="space-y-2">
                      <Label htmlFor="cycEp">CycleGAN 固定轮数</Label>
                      <Input
                        id="cycEp"
                        value={cycleGanEpochs}
                        onChange={(e) => setCycleGanEpochs(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cycDecayEp">CycleGAN 衰减轮数</Label>
                      <Input
                        id="cycDecayEp"
                        value={cycleGanDecayEpochs}
                        onChange={(e) => setCycleGanDecayEpochs(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cycBs">CycleGAN Batch Size</Label>
                      <Input
                        id="cycBs"
                        value={cycleGanBatchSize}
                        onChange={(e) => setCycleGanBatchSize(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-3">
                    <div className="space-y-2">
                      <Label htmlFor="cycSaveEp">每 x 轮保存</Label>
                      <Input
                        id="cycSaveEp"
                        value={cycleGanSaveEpochFreq}
                        onChange={(e) => setCycleGanSaveEpochFreq(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cycSaveLatest">每 x step 保存 latest</Label>
                      <Input
                        id="cycSaveLatest"
                        value={cycleGanSaveLatestFreq}
                        onChange={(e) => setCycleGanSaveLatestFreq(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cycLambdaId">identity loss 权重</Label>
                      <Input
                        id="cycLambdaId"
                        value={cycleGanLambdaIdentity}
                        onChange={(e) => setCycleGanLambdaIdentity(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="cycLoadSize">load_size</Label>
                      <Input
                        id="cycLoadSize"
                        value={cycleGanLoadSize}
                        onChange={(e) => setCycleGanLoadSize(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cycCropSize">crop_size</Label>
                      <Input
                        id="cycCropSize"
                        value={cycleGanCropSize}
                        onChange={(e) => setCycleGanCropSize(e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              ) : null}
              <Button type="submit" disabled={loading}>
                {loading ? <Loader2 className="animate-spin" /> : "启动训练"}
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
