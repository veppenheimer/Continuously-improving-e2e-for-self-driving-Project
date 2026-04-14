import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchDomainAugImageBlob, listDomainAugPairs } from "@/api/services/tasks";
import type { DomainAugPair } from "@/api/types";
import { showApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

export function DomainAugComparePage() {
  const { taskId } = useParams<{ taskId: string }>();
  const [pairs, setPairs] = useState<DomainAugPair[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [aUrl, setAUrl] = useState<string | null>(null);
  const [cUrl, setCUrl] = useState<string | null>(null);
  const [imgLoading, setImgLoading] = useState(false);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await listDomainAugPairs(taskId);
        if (!cancelled) {
          setPairs(data);
          setActiveIndex(data[0]?.index ?? null);
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

  useEffect(() => {
    if (!taskId || activeIndex == null) return;
    let cancelled = false;
    (async () => {
      setImgLoading(true);
      try {
        const [aBlob, cBlob] = await Promise.all([
          fetchDomainAugImageBlob(taskId, activeIndex, "a"),
          fetchDomainAugImageBlob(taskId, activeIndex, "c"),
        ]);
        if (cancelled) return;
        if (aUrl) URL.revokeObjectURL(aUrl);
        if (cUrl) URL.revokeObjectURL(cUrl);
        setAUrl(URL.createObjectURL(aBlob));
        setCUrl(URL.createObjectURL(cBlob));
      } catch (e) {
        showApiError(e);
      } finally {
        if (!cancelled) setImgLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [taskId, activeIndex]);

  const current = useMemo(() => pairs.find((x) => x.index === activeIndex) ?? null, [pairs, activeIndex]);

  if (!taskId) return null;
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">域增强图像对比</h1>
          <p className="text-muted-foreground">任务 {taskId.slice(0, 8)}… 的 A 域图像与生成的 C 域图像对照</p>
        </div>
        <Button variant="outline" size="sm" asChild>
          <Link to={`/tasks/${taskId}/monitor`}>返回监控</Link>
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>样本选择</CardTitle>
          <CardDescription>按样本索引浏览 A/C 对照图像</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex justify-center py-6">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : pairs.length === 0 ? (
            <p className="text-sm text-muted-foreground">暂无可用的域增强样本（请先完成开启域增强的训练任务）。</p>
          ) : (
            <select
              className="flex h-10 w-full max-w-sm rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={activeIndex ?? undefined}
              onChange={(e) => setActiveIndex(Number(e.target.value))}
            >
              {pairs.map((p) => (
                <option key={p.index} value={p.index}>
                  样本 {p.index} · A: {p.aName}
                </option>
              ))}
            </select>
          )}
        </CardContent>
      </Card>

      {current ? (
        <Card>
          <CardHeader>
            <CardTitle>A/C 图像并排对比</CardTitle>
            <CardDescription>
              A：{current.aName} · C：{current.cName}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {imgLoading ? (
              <div className="flex justify-center py-10">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <p className="mb-2 text-sm font-medium">A（原始域）</p>
                  {aUrl ? (
                    <img src={aUrl} alt="A 域图像" className="w-full rounded-md border object-contain" />
                  ) : (
                    <div className="rounded-md border p-8 text-sm text-muted-foreground">无法加载 A 图像</div>
                  )}
                </div>
                <div>
                  <p className="mb-2 text-sm font-medium">C（A→B 风格）</p>
                  {cUrl ? (
                    <img src={cUrl} alt="C 域图像" className="w-full rounded-md border object-contain" />
                  ) : (
                    <div className="rounded-md border p-8 text-sm text-muted-foreground">无法加载 C 图像</div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

