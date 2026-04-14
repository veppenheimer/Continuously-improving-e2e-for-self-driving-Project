import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { deleteTask, listTasks } from "@/api/services/tasks";
import type { TrainingTaskSummary } from "@/api/types";
import { showApiError } from "@/api/client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { taskStatusLabel, taskStatusVariant } from "@/lib/taskStatus";
import { Loader2, ChevronRight, Trash2, Sparkles, ListTodo } from "lucide-react";
import { toast } from "sonner";

export function DashboardPage() {
  const [tasks, setTasks] = useState<TrainingTaskSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function onDelete(t: TrainingTaskSummary, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    const ok = window.confirm(
      `确定删除任务「${t.name}」？将删除数据库记录与模型文件，且无法恢复。进行中的训练会被终止。`,
    );
    if (!ok) return;
    setDeletingId(t.id);
    try {
      await deleteTask(t.id);
      toast.success("任务已删除");
      setTasks((prev) => prev.filter((x) => x.id !== t.id));
    } catch (err) {
      showApiError(err);
    } finally {
      setDeletingId(null);
    }
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await listTasks();
        if (!cancelled) setTasks(data);
      } catch (e) {
        showApiError(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-border/70 bg-card/75 p-5 backdrop-blur">
        <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs text-primary">
          <Sparkles className="h-3.5 w-3.5" />
          实时训练看板
        </div>
        <h1 className="mt-3 text-3xl font-bold tracking-tight">训练任务</h1>
        <p className="mt-1 text-muted-foreground">查看历史任务的参数、状态与结果摘要</p>
      </div>

      {tasks.length === 0 ? (
        <Card className="border-dashed">
          <CardHeader>
            <CardTitle>暂无任务</CardTitle>
            <CardDescription>上传数据集并创建第一个训练任务</CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild>
              <Link to="/train/new">新建训练</Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <ul className="space-y-3">
          {tasks.map((t) => (
            <li key={t.id}>
              <Card className="transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/40">
                <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0 pb-2">
                  <div className="min-w-0 flex-1">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <ListTodo className="h-4 w-4 text-primary" />
                      <span className="truncate">{t.name}</span>
                    </CardTitle>
                    <CardDescription className="mt-2">
                      ID <code className="text-xs">{t.id.slice(0, 8)}…</code> · 数据集 {t.params.datasetName || t.params.datasetId} · LR{" "}
                      {t.params.learningRate} · batch {t.params.batchSize} · {t.params.epochs} epochs
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
                      disabled={deletingId === t.id}
                      onClick={(e) => void onDelete(t, e)}
                    >
                      {deletingId === t.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>
                          <Trash2 className="mr-1 h-4 w-4" />
                          删除
                        </>
                      )}
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
    </div>
  );
}
