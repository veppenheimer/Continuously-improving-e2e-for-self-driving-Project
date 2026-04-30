import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { showApiError } from "@/api/client";
import { createProject, deleteProject, listProjects } from "@/api/services/projects";
import type { ProjectItem } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { FolderTree, Loader2, Plus, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";

export function DashboardPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [newProjectName, setNewProjectName] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  async function refreshProjects() {
    const data = await listProjects();
    setProjects(data);
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await listProjects();
        if (!cancelled) setProjects(data);
      } catch (err) {
        if (!cancelled) showApiError(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

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
      navigate(`/projects/${created.id}`);
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  async function onDeleteProject(project: ProjectItem, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    const ok = window.confirm(`确定删除项目「${project.name}」？该项目下的数据集和训练任务会被一并删除，无法恢复。`);
    if (!ok) return;
    setSaving(true);
    try {
      await deleteProject(project.id);
      toast.success("项目已删除");
      await refreshProjects();
    } catch (err) {
      showApiError(err);
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="h-10 w-10 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <section className="ag-page-hero">
        <div className="max-w-3xl">
          <div className="ag-eyebrow">
            <FolderTree className="h-3.5 w-3.5" />
            Project Console
          </div>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">项目管理</h1>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            首页只负责创建和进入项目。数据集、训练配置和任务记录统一放在项目详情页。
          </p>
        </div>
      </section>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-md border border-primary/20 bg-primary/10 text-primary">
                <FolderTree className="h-4 w-4" />
              </div>
              <div>
                <CardTitle>新增项目</CardTitle>
                <CardDescription>创建后进入项目详情页管理数据集和训练任务</CardDescription>
              </div>
            </div>
            <Button type="button" variant="outline" disabled={saving} onClick={() => void refreshProjects()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <form className="grid gap-3 md:grid-cols-[minmax(0,1fr)_12rem]" onSubmit={onCreateProject}>
            <Input
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="新项目名称"
            />
            <Button type="submit" disabled={saving}>
              <Plus className="h-4 w-4" />
              新增项目
            </Button>
          </form>
        </CardContent>
      </Card>

      <section className="space-y-3">
        <div>
          <h2 className="text-lg font-semibold">已有项目</h2>
          <p className="mt-1 text-sm text-muted-foreground">点击项目进入详情页。</p>
        </div>
        {projects.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-sm text-muted-foreground">暂无项目，请先创建一个项目。</CardContent>
          </Card>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {projects.map((project) => (
              <div
                key={project.id}
                role="button"
                tabIndex={0}
                className="group block text-left"
                onClick={() => navigate(`/projects/${project.id}`)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    navigate(`/projects/${project.id}`);
                  }
                }}
              >
                <Card className="h-full bg-card/75 transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/45">
                  <CardHeader>
                    <div className="flex items-start gap-3">
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-primary/20 bg-primary/10 text-primary">
                        <FolderTree className="h-4 w-4" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <CardTitle className="truncate text-lg">{project.name}</CardTitle>
                        <CardDescription>{new Date(project.createdAt).toLocaleString()}</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <Button
                      type="button"
                      variant="outline"
                      className="w-full text-destructive hover:bg-destructive/10 hover:text-destructive"
                      disabled={saving}
                      onClick={(e) => void onDeleteProject(project, e)}
                    >
                      <Trash2 className="h-4 w-4" />
                      删除项目
                    </Button>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
