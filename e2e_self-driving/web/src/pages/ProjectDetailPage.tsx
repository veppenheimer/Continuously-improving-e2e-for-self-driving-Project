import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { showApiError } from "@/api/client";
import { listProjects } from "@/api/services/projects";
import { deleteDataset, listDatasets, uploadDatasetZip } from "@/api/services/datasets";
import { createTask, deleteTask, listTasks } from "@/api/services/tasks";
import type { DatasetItem, ProjectItem, TrainingTaskSummary, TrainModelVariant } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { MODEL_VARIANT_OPTIONS, modelVariantLabel } from "@/lib/modelVariant";
import { taskStatusLabel, taskStatusVariant } from "@/lib/taskStatus";
import {
  Activity,
  ArrowLeft,
  ChevronRight,
  Database,
  FileArchive,
  FolderTree,
  Loader2,
  Play,
  Rocket,
  Settings2,
  Trash2,
  UploadCloud,
  Wand2,
} from "lucide-react";
import { toast } from "sonner";

export function ProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [project, setProject] = useState<ProjectItem | null>(null);
  const [tasks, setTasks] = useState<TrainingTaskSummary[]>([]);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [bootLoading, setBootLoading] = useState(true);
  const [dataLoading, setDataLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const [datasetName, setDatasetName] = useState("");
  const datasetFileRef = useRef<HTMLInputElement>(null);

  const [taskName, setTaskName] = useState("");
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
  const [modelVariant, setModelVariant] = useState<TrainModelVariant>("mobilenet_v2");

  const activeTaskCount = useMemo(
    () => tasks.filter((t) => t.status === "running" || t.status === "paused").length,
    [tasks],
  );
  const completedTaskCount = useMemo(() => tasks.filter((t) => t.status === "completed").length, [tasks]);
  const totalImageCount = useMemo(
    () => datasets.reduce((sum, item) => sum + (item.imageCount ?? 0), 0),
    [datasets],
  );

  async function loadProject() {
    if (!projectId) return;
    const data = await listProjects();
    setProject(data.find((x) => x.id === projectId) ?? null);
  }

  async function refreshProjectData(projectId: string) {
    if (!projectId) return;
    setDataLoading(true);
    try {
      const [taskRows, datasetRows] = await Promise.all([listTasks(projectId), listDatasets(projectId)]);
      setTasks(taskRows);
      setDatasets(datasetRows);
      if (datasetRows.length > 0) {
        if (!datasetRows.some((x) => x.id === datasetId)) {
          setDatasetId(datasetRows[0].id);
        }
        if (!datasetRows.some((x) => x.id === domainBDatasetId)) {
          const fallback = datasetRows.find((x) => x.id !== datasetRows[0].id) ?? datasetRows[0];
          setDomainBDatasetId(fallback.id);
        }
      } else {
        setDatasetId("");
        setDomainBDatasetId("");
      }
    } finally {
      setDataLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await loadProject();
      } catch (e) {
        if (!cancelled) showApiError(e);
      } finally {
        if (!cancelled) setBootLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    if (!projectId) {
      setTasks([]);
      setDatasets([]);
      return;
    }
    void refreshProjectData(projectId).catch(showApiError);
  }, [projectId]);

  async function onUploadDataset(e: React.FormEvent) {
    e.preventDefault();
    if (!projectId) {
      toast.error("请先选择项目");
      return;
    }
    const file = datasetFileRef.current?.files?.[0];
    if (!file) {
      toast.error("请选择 ZIP 文件");
      return;
    }
    setSaving(true);
    try {
      await uploadDatasetZip(projectId, file, datasetName.trim() || undefined);
      toast.success("数据集上传成功");
      setDatasetName("");
      if (datasetFileRef.current) datasetFileRef.current.value = "";
      await refreshProjectData(projectId);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onDeleteDataset(item: DatasetItem) {
    const ok = window.confirm(`确定删除数据集「${item.name}」？`);
    if (!ok) return;
    setSaving(true);
    try {
      await deleteDataset(item.id);
      toast.success("数据集已删除");
      if (projectId) await refreshProjectData(projectId);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onCreateTask(e: React.FormEvent) {
    e.preventDefault();
    if (!projectId) {
      toast.error("请先选择项目");
      return;
    }
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
      toast.error("请检查训练参数");
      return;
    }
    if (domainAug && (!domainBDatasetId || domainBDatasetId === datasetId)) {
      toast.error("开启域增强时请正确选择 B 域数据集");
      return;
    }
    if (
      domainAug &&
      (Number.isNaN(cycEpochN) ||
        Number.isNaN(cycDecayEpochN) ||
        Number.isNaN(cycBatchN) ||
        Number.isNaN(cycSaveEpochN) ||
        Number.isNaN(cycSaveLatestN) ||
        Number.isNaN(cycLoadSizeN) ||
        Number.isNaN(cycCropSizeN) ||
        Number.isNaN(cycLambdaIdentityN))
    ) {
      toast.error("请检查 CycleGAN 参数");
      return;
    }

    setSaving(true);
    try {
      const task = await createTask({
        projectId,
        datasetId,
        modelVariant,
        learningRate,
        batchSize,
        epochs: epochsN,
        domainAugmentation: domainAug,
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
        ...(taskName.trim() ? { name: taskName.trim() } : {}),
      });
      toast.success("训练任务已创建");
      void refreshProjectData(projectId);
      navigate(`/tasks/${task.id}/monitor`);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onDeleteTask(t: TrainingTaskSummary, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    const ok = window.confirm(`确定删除任务「${t.name}」？`);
    if (!ok) return;
    setSaving(true);
    try {
      await deleteTask(t.id);
      toast.success("任务已删除");
      if (projectId) await refreshProjectData(projectId);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  if (bootLoading) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="h-10 w-10 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!projectId) {
    return <Navigate to="/" replace />;
  }

  if (!project) {
    return (
      <div className="space-y-6">
        <Button variant="ghost" asChild>
          <Link to="/">
            <ArrowLeft className="h-4 w-4" />
            返回项目
          </Link>
        </Button>
        <Card>
          <CardContent className="py-10 text-sm text-muted-foreground">项目不存在或已被删除。</CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <section className="ag-page-hero">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div>
            <Button variant="ghost" className="mb-4 px-0" asChild>
              <Link to="/">
                <ArrowLeft className="h-4 w-4" />
                返回项目
              </Link>
            </Button>
            <div className="ag-eyebrow">
              <FolderTree className="h-3.5 w-3.5" />
              Project Detail
            </div>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">{project.name}</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
              管理该项目的数据集、训练配置和任务记录。
            </p>
          </div>
          <div className="grid w-full gap-3 sm:grid-cols-3 lg:w-auto">
            <div className="ag-kpi">
              <Database className="mb-2 h-4 w-4 text-violet-300" />
              <p className="text-xs text-muted-foreground">数据集</p>
              <p className="mt-1 text-2xl font-semibold">{datasets.length}</p>
            </div>
            <div className="ag-kpi">
              <Activity className="mb-2 h-4 w-4 text-primary" />
              <p className="text-xs text-muted-foreground">进行中</p>
              <p className="mt-1 text-2xl font-semibold">{activeTaskCount}</p>
            </div>
            <div className="ag-kpi">
              <FolderTree className="mb-2 h-4 w-4 text-sky-300" />
              <p className="text-xs text-muted-foreground">已完成</p>
              <p className="mt-1 text-2xl font-semibold">{completedTaskCount}</p>
            </div>
          </div>
        </div>
      </section>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-md border border-violet-300/20 bg-violet-400/10 text-violet-300">
                    <Database className="h-4 w-4" />
                  </div>
                  <div>
                    <CardTitle>数据集（当前项目）</CardTitle>
                    <CardDescription>上传 ZIP 后可直接用于该项目训练</CardDescription>
                  </div>
                </div>
                <div className="hidden text-right text-xs text-muted-foreground sm:block">
                  <p>{datasets.length} 个数据集</p>
                  <p>{totalImageCount} 张图像</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={onUploadDataset} className="grid gap-3 md:grid-cols-3">
                <Input
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="数据集名称（可选）"
                />
                <input ref={datasetFileRef} type="file" accept=".zip,application/zip" className="ag-file-input" />
                <Button type="submit" disabled={saving}>
                  <UploadCloud className="h-4 w-4" />上传 ZIP
                </Button>
              </form>
              {dataLoading ? (
                <div className="flex justify-center py-6">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : datasets.length === 0 ? (
                <p className="text-sm text-muted-foreground">当前项目暂无数据集</p>
              ) : (
                <ul className="grid gap-3 sm:grid-cols-2">
                  {datasets.map((d) => (
                    <li key={d.id}>
                      <Card className="bg-card/70">
                        <CardHeader className="pb-2">
                          <div className="flex items-start gap-3">
                            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-white/10 bg-background/45 text-primary">
                              <FileArchive className="h-4 w-4" />
                            </div>
                            <div className="min-w-0">
                              <CardTitle className="truncate text-base">{d.name}</CardTitle>
                              <CardDescription>
                                {d.imageCount != null ? `${d.imageCount} 张图 · ` : ""}
                                {new Date(d.createdAt).toLocaleString()}
                              </CardDescription>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="flex items-center justify-between gap-3">
                          <p className="truncate text-xs text-muted-foreground">ID: {d.id}</p>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                            onClick={() => void onDeleteDataset(d)}
                            disabled={saving}
                          >
                            <Trash2 className="h-4 w-4" />删除
                          </Button>
                        </CardContent>
                      </Card>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-primary/20 bg-primary/10 text-primary">
                  <Rocket className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>新建训练（当前项目）</CardTitle>
                  <CardDescription>配置模型、数据集与域增强参数后启动训练</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {datasets.length === 0 ? (
                <p className="text-sm text-muted-foreground">请先上传数据集。</p>
              ) : (
                <form onSubmit={onCreateTask} className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>任务名称（可选）</Label>
                      <Input value={taskName} onChange={(e) => setTaskName(e.target.value)} placeholder="例如：夜间场景 baseline" />
                    </div>
                    <div className="space-y-2">
                      <Label>数据集 A</Label>
                      <select
                        className="ag-select"
                        value={datasetId}
                        onChange={(e) => setDatasetId(e.target.value)}
                      >
                        {datasets.map((d) => (
                          <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div className="space-y-2">
                      <Label>模型架构</Label>
                      <select
                        className="ag-select"
                        value={modelVariant}
                        onChange={(e) => setModelVariant(e.target.value as TrainModelVariant)}
                      >
                        {MODEL_VARIANT_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                      <p className="text-xs text-muted-foreground">
                        {MODEL_VARIANT_OPTIONS.find((option) => option.value === modelVariant)?.description}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <Label>学习率</Label>
                      <Input value={lr} onChange={(e) => setLr(e.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <Label>Batch Size</Label>
                      <Input value={batch} onChange={(e) => setBatch(e.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <Label>Epochs</Label>
                      <Input value={epochs} onChange={(e) => setEpochs(e.target.value)} />
                    </div>
                  </div>
                  <label className="ag-panel-soft flex cursor-pointer items-center justify-between gap-3 p-3 text-sm">
                    <span className="flex items-center gap-2">
                      <Wand2 className="h-4 w-4 text-violet-300" />
                      <span>
                        <span className="block font-medium">开启域增强</span>
                        <span className="text-xs text-muted-foreground">使用 CycleGAN 生成 C 域图像并进行增强训练</span>
                      </span>
                    </span>
                    <input
                      type="checkbox"
                      checked={domainAug}
                      onChange={(e) => setDomainAug(e.target.checked)}
                      className="h-4 w-4 rounded border-input accent-[hsl(var(--primary))]"
                    />
                  </label>
                  {domainAug ? (
                    <div className="space-y-4 rounded-lg border border-violet-300/20 bg-violet-400/5 p-4">
                      <div className="space-y-2">
                        <Label>数据集 B</Label>
                        <select
                          className="ag-select"
                          value={domainBDatasetId}
                          onChange={(e) => setDomainBDatasetId(e.target.value)}
                        >
                          {datasets.map((d) => (
                            <option key={d.id} value={d.id}>{d.name}</option>
                          ))}
                        </select>
                      </div>
                      <div className="grid gap-4 md:grid-cols-3">
                        <div className="space-y-2">
                          <Label>CycleGAN 固定轮数</Label>
                          <Input value={cycleGanEpochs} onChange={(e) => setCycleGanEpochs(e.target.value)} />
                        </div>
                        <div className="space-y-2">
                          <Label>CycleGAN 衰减轮数</Label>
                          <Input value={cycleGanDecayEpochs} onChange={(e) => setCycleGanDecayEpochs(e.target.value)} />
                        </div>
                        <div className="space-y-2">
                          <Label>CycleGAN Batch Size</Label>
                          <Input value={cycleGanBatchSize} onChange={(e) => setCycleGanBatchSize(e.target.value)} />
                        </div>
                      </div>
                      <div className="grid gap-4 md:grid-cols-3">
                        <div className="space-y-2">
                          <Label>每 x 轮保存</Label>
                          <Input value={cycleGanSaveEpochFreq} onChange={(e) => setCycleGanSaveEpochFreq(e.target.value)} />
                        </div>
                        <div className="space-y-2">
                          <Label>每 x step 保存 latest</Label>
                          <Input value={cycleGanSaveLatestFreq} onChange={(e) => setCycleGanSaveLatestFreq(e.target.value)} />
                        </div>
                        <div className="space-y-2">
                          <Label>identity loss 权重</Label>
                          <Input value={cycleGanLambdaIdentity} onChange={(e) => setCycleGanLambdaIdentity(e.target.value)} />
                        </div>
                      </div>
                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-2">
                          <Label>load_size</Label>
                          <Input value={cycleGanLoadSize} onChange={(e) => setCycleGanLoadSize(e.target.value)} />
                        </div>
                        <div className="space-y-2">
                          <Label>crop_size</Label>
                          <Input value={cycleGanCropSize} onChange={(e) => setCycleGanCropSize(e.target.value)} />
                        </div>
                      </div>
                    </div>
                  ) : null}
                  <Button type="submit" disabled={saving}>
                    <Play className="h-4 w-4" />启动训练
                  </Button>
                </form>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-sky-300/20 bg-sky-400/10 text-sky-300">
                  <Settings2 className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle>训练任务（当前项目）</CardTitle>
                  <CardDescription>查看任务状态、训练曲线与结果产物</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {dataLoading ? (
                <div className="flex justify-center py-6">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : tasks.length === 0 ? (
                <p className="text-sm text-muted-foreground">当前项目暂无训练任务</p>
              ) : (
                <ul className="space-y-3">
                  {tasks.map((t) => (
                    <li key={t.id}>
                      <Card className="bg-card/70 transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/40">
                        <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0 pb-2">
                          <div className="min-w-0 flex-1">
                            <CardTitle className="flex items-center gap-2 text-lg">
                              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-white/10 bg-background/45 text-primary">
                                <Activity className="h-3.5 w-3.5" />
                              </span>
                              <span className="truncate">{t.name}</span>
                            </CardTitle>
                            <CardDescription className="mt-2">
                              {modelVariantLabel(t.params.modelVariant)} · 数据集 {t.params.datasetName || t.params.datasetId}
                              {" · "}LR {t.params.learningRate} · batch {t.params.batchSize} · {t.params.epochs} epochs
                              {t.domainAugmentation ? " · 域增强" : ""}
                            </CardDescription>
                          </div>
                          <div className="flex shrink-0 flex-col items-end gap-2">
                            <Badge variant={taskStatusVariant(t.status)}>{taskStatusLabel(t.status)}</Badge>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                              disabled={saving}
                              onClick={(e) => void onDeleteTask(t, e)}
                            >
                              <Trash2 className="mr-1 h-4 w-4" />删除
                            </Button>
                          </div>
                        </CardHeader>
                        <CardContent className="flex flex-wrap gap-2 pt-2">
                          {(t.status === "running" || t.status === "paused") && (
                            <Button size="sm" asChild>
                              <Link to={`/tasks/${t.id}/monitor`}>
                                监控 <ChevronRight className="h-4 w-4" />
                              </Link>
                            </Button>
                          )}
                          {t.status === "completed" && (
                            <>
                              <Button size="sm" variant="secondary" asChild>
                                <Link to={`/tasks/${t.id}/results`}>
                                  结果与对比 <ChevronRight className="h-4 w-4" />
                                </Link>
                              </Button>
                              <Button size="sm" asChild>
                                <Link to={`/tasks/${t.id}/monitor`}>查看曲线</Link>
                              </Button>
                            </>
                          )}
                        </CardContent>
                      </Card>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
    </div>
  );
}
