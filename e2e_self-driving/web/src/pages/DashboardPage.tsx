import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { showApiError } from "@/api/client";
import { createProject, deleteProject, listProjects, renameProject } from "@/api/services/projects";
import { deleteDataset, listDatasets, uploadDatasetZip } from "@/api/services/datasets";
import { createTask, deleteTask, listTasks } from "@/api/services/tasks";
import type { DatasetItem, ProjectItem, TrainingTaskSummary } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { taskStatusLabel, taskStatusVariant } from "@/lib/taskStatus";
import { ChevronRight, FolderTree, Loader2, Plus, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";

export function DashboardPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [tasks, setTasks] = useState<TrainingTaskSummary[]>([]);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [bootLoading, setBootLoading] = useState(true);
  const [dataLoading, setDataLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const [newProjectName, setNewProjectName] = useState("");
  const [renamingName, setRenamingName] = useState("");

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
  const [useCompetitionClassModel, setUseCompetitionClassModel] = useState(false);
  const [useCompetitionLiteModel, setUseCompetitionLiteModel] = useState(false);

  const selectedProject = useMemo(
    () => projects.find((x) => x.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  async function refreshProjects(keepCurrent = true) {
    const data = await listProjects();
    setProjects(data);
    if (data.length === 0) {
      setSelectedProjectId("");
      setRenamingName("");
      return;
    }
    if (keepCurrent && data.some((x) => x.id === selectedProjectId)) {
      const current = data.find((x) => x.id === selectedProjectId);
      setRenamingName(current?.name ?? "");
      return;
    }
    setSelectedProjectId(data[0].id);
    setRenamingName(data[0].name);
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
        await refreshProjects(false);
      } catch (e) {
        if (!cancelled) showApiError(e);
      } finally {
        if (!cancelled) setBootLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedProjectId) {
      setTasks([]);
      setDatasets([]);
      return;
    }
    void refreshProjectData(selectedProjectId).catch(showApiError);
  }, [selectedProjectId]);

  async function onCreateProject(e: React.FormEvent) {
    e.preventDefault();
    const name = newProjectName.trim();
    if (!name) {
      toast.error("请输入项目名称");
      return;
    }
    setSaving(true);
    try {
      const created = await createProject(name);
      toast.success("项目已创建");
      setNewProjectName("");
      await refreshProjects(false);
      setSelectedProjectId(created.id);
      setRenamingName(created.name);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onRenameProject() {
    if (!selectedProjectId) return;
    const name = renamingName.trim();
    if (!name) {
      toast.error("项目名称不能为空");
      return;
    }
    setSaving(true);
    try {
      await renameProject(selectedProjectId, name);
      toast.success("项目已重命名");
      await refreshProjects(true);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onDeleteProject() {
    if (!selectedProjectId || !selectedProject) return;
    const ok = window.confirm(
      `确定删除项目「${selectedProject.name}」？该项目下的数据集和训练任务会被一并删除，无法恢复。`,
    );
    if (!ok) return;
    setSaving(true);
    try {
      await deleteProject(selectedProjectId);
      toast.success("项目已删除");
      await refreshProjects(false);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onUploadDataset(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedProjectId) {
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
      await uploadDatasetZip(selectedProjectId, file, datasetName.trim() || undefined);
      toast.success("数据集上传成功");
      setDatasetName("");
      if (datasetFileRef.current) datasetFileRef.current.value = "";
      await refreshProjectData(selectedProjectId);
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
      if (selectedProjectId) await refreshProjectData(selectedProjectId);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onCreateTask(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedProjectId) {
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
        projectId: selectedProjectId,
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
        ...(taskName.trim() ? { name: taskName.trim() } : {}),
      });
      toast.success("训练任务已创建");
      void refreshProjectData(selectedProjectId);
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
      if (selectedProjectId) await refreshProjectData(selectedProjectId);
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

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-border/70 bg-card/75 p-5 backdrop-blur">
        <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs text-primary">
          <FolderTree className="h-3.5 w-3.5" />
          项目工作台
        </div>
        <h1 className="mt-3 text-3xl font-bold tracking-tight">项目</h1>
        <p className="mt-1 text-muted-foreground">每个项目拥有独立的数据集与训练任务</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>项目管理</CardTitle>
          <CardDescription>支持新增、重命名、删除项目</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form className="flex flex-wrap gap-2" onSubmit={onCreateProject}>
            <Input
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="新项目名称"
              className="w-72"
            />
            <Button type="submit" disabled={saving}>
              <Plus className="h-4 w-4" />新增项目
            </Button>
            <Button type="button" variant="outline" disabled={saving} onClick={() => void refreshProjects(true)}>
              <RefreshCw className="h-4 w-4" />刷新
            </Button>
          </form>
          {projects.length === 0 ? (
            <p className="text-sm text-muted-foreground">暂无项目，请先创建一个项目。</p>
          ) : (
            <div className="space-y-3">
              <div className="space-y-2">
                <Label htmlFor="project-select">当前项目</Label>
                <select
                  id="project-select"
                  className="flex h-10 w-full max-w-md rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={selectedProjectId}
                  onChange={(e) => {
                    const next = e.target.value;
                    setSelectedProjectId(next);
                    const p = projects.find((x) => x.id === next);
                    setRenamingName(p?.name ?? "");
                  }}
                >
                  {projects.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-wrap gap-2">
                <Input
                  value={renamingName}
                  onChange={(e) => setRenamingName(e.target.value)}
                  placeholder="重命名项目"
                  className="w-72"
                />
                <Button type="button" variant="secondary" onClick={() => void onRenameProject()} disabled={saving || !selectedProjectId}>
                  重命名
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                  onClick={() => void onDeleteProject()}
                  disabled={saving || !selectedProjectId}
                >
                  <Trash2 className="h-4 w-4" />删除项目
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {!!selectedProjectId && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>数据集（当前项目）</CardTitle>
              <CardDescription>上传 ZIP 后可直接用于该项目训练</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={onUploadDataset} className="grid gap-3 md:grid-cols-3">
                <Input
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="数据集名称（可选）"
                />
                <Input ref={datasetFileRef} type="file" accept=".zip,application/zip" />
                <Button type="submit" disabled={saving}>上传 ZIP</Button>
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
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-base">{d.name}</CardTitle>
                          <CardDescription>
                            {d.imageCount != null ? `${d.imageCount} 张图 · ` : ""}
                            {new Date(d.createdAt).toLocaleString()}
                          </CardDescription>
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
              <CardTitle>新建训练（当前项目）</CardTitle>
              <CardDescription>“新建训练”已并入训练任务模块，不再单独页面展示</CardDescription>
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
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                        value={datasetId}
                        onChange={(e) => setDatasetId(e.target.value)}
                      >
                        {datasets.map((d) => (
                          <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="grid gap-4 md:grid-cols-3">
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
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={domainAug}
                      onChange={(e) => setDomainAug(e.target.checked)}
                      className="h-4 w-4 rounded border-input"
                    />
                    开启域增强
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
                      训练分类模型（Net_class）
                    </label>
                    <label className="flex cursor-pointer items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={useCompetitionLiteModel}
                        onChange={(e) => setUseCompetitionLiteModel(e.target.checked)}
                        className="h-4 w-4 rounded border-input"
                      />
                      训练轻量模型（Net_improve）
                    </label>
                  </div>
                  {domainAug ? (
                    <div className="space-y-4 rounded-lg border p-4">
                      <div className="space-y-2">
                        <Label>数据集 B</Label>
                        <select
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
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
                  <Button type="submit" disabled={saving}>启动训练</Button>
                </form>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>训练任务（当前项目）</CardTitle>
              <CardDescription>查看任务状态与结果</CardDescription>
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
                      <Card className="transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/40">
                        <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0 pb-2">
                          <div className="min-w-0 flex-1">
                            <CardTitle className="text-lg">
                              <span className="truncate">{t.name}</span>
                            </CardTitle>
                            <CardDescription className="mt-2">
                              数据集 {t.params.datasetName || t.params.datasetId} · LR {t.params.learningRate} · batch {" "}
                              {t.params.batchSize} · {t.params.epochs} epochs
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
        </>
      )}
    </div>
  );
}
