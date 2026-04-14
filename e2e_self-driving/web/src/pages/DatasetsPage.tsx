import { useEffect, useState, useRef } from "react";
import { deleteDataset, listDatasets, uploadDatasetZip } from "@/api/services/datasets";
import type { DatasetItem } from "@/api/types";
import { showApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Loader2, RefreshCw, Trash2 } from "lucide-react";

export function DatasetsPage() {
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    setLoading(true);
    try {
      const data = await listDatasets();
      setItems(data);
    } catch (e) {
      showApiError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function onUpload(e: React.FormEvent) {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) {
      toast.error("请选择 ZIP 文件");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".zip")) {
      toast.error("请上传 ZIP 压缩包");
      return;
    }
    setUploading(true);
    try {
      await uploadDatasetZip(file, name || undefined);
      toast.success("上传成功，后端将自动划分数据集");
      setName("");
      if (fileRef.current) fileRef.current.value = "";
      await refresh();
    } catch (err) {
      showApiError(err);
    } finally {
      setUploading(false);
    }
  }

  async function onDelete(item: DatasetItem) {
    const ok = window.confirm(
      `确定删除数据集「${item.name}」？该操作会移除数据库记录并删除数据目录，且无法恢复。`,
    );
    if (!ok) return;
    setDeletingId(item.id);
    try {
      await deleteDataset(item.id);
      setItems((prev) => prev.filter((x) => x.id !== item.id));
      toast.success("数据集已删除");
    } catch (err) {
      showApiError(err);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">数据集</h1>
        <p className="text-muted-foreground">
          上传 ZIP，图像命名需为 <code className="text-primary">序号_转向角.jpg</code>
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>上传 ZIP</CardTitle>
          <CardDescription>由后端完成解压、校验与训练/验证划分</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onUpload} className="max-w-xl space-y-4">
            <div className="space-y-2">
              <Label htmlFor="ds-name">数据集名称（可选）</Label>
              <Input
                id="ds-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="例如 my_run_01"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="ds-zip">ZIP 文件</Label>
              <Input id="ds-zip" ref={fileRef} type="file" accept=".zip,application/zip" />
            </div>
            <Button type="submit" disabled={uploading}>
              {uploading ? <Loader2 className="animate-spin" /> : "开始上传"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">已上传列表</h2>
        <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw className={loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
          刷新
        </Button>
      </div>

      {loading && items.length === 0 ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : items.length === 0 ? (
        <p className="text-sm text-muted-foreground">暂无数据集</p>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2">
          {items.map((d) => (
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
                    disabled={deletingId === d.id}
                    onClick={() => void onDelete(d)}
                  >
                    {deletingId === d.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Trash2 className="h-4 w-4" />
                        删除
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
