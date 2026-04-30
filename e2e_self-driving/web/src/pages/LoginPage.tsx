import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { toast } from "sonner";
import { loginApi } from "@/api/services/auth";
import { showApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuthStore } from "@/store/authStore";
import { ArrowRight, BrainCircuit, Gauge, Loader2, LockKeyhole, Sparkles, User } from "lucide-react";

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const setAuth = useAuthStore((s) => s.setAuth);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const from = (location.state as { from?: string } | null)?.from || "/";

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await loginApi({ username, password });
      setAuth(res.token, res.user);
      toast.success("登录成功");
      navigate(from, { replace: true });
    } catch (err) {
      showApiError(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4 py-8">
      <div className="grid w-full max-w-5xl gap-5 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="ag-page-hero flex min-h-[520px] flex-col justify-between rounded-lg border border-white/10 bg-background/45 p-6 backdrop-blur-xl">
          <div>
            <div className="ag-eyebrow">
              <Sparkles className="h-3.5 w-3.5" />
              Antigravity Lab
            </div>
            <h1 className="mt-6 max-w-xl text-4xl font-semibold leading-tight tracking-tight text-foreground md:text-5xl">
              构建你个人的端到端自动驾驶模型
            </h1>
            <p className="mt-4 max-w-lg text-sm leading-6 text-muted-foreground">
              上传驾驶视角数据，配置训练参数，并在同一个控制台里观察曲线、比较结果与管理模型产物。
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="ag-kpi">
              <BrainCircuit className="mb-3 h-5 w-5 text-primary" />
              <p className="text-xs text-muted-foreground">Model</p>
              <p className="mt-1 font-medium">E2E Steering</p>
            </div>
            <div className="ag-kpi">
              <Gauge className="mb-3 h-5 w-5 text-violet-300" />
              <p className="text-xs text-muted-foreground">Monitor</p>
              <p className="mt-1 font-medium">Live Loss</p>
            </div>
            <div className="ag-kpi">
              <LockKeyhole className="mb-3 h-5 w-5 text-fuchsia-300" />
              <p className="text-xs text-muted-foreground">Access</p>
              <p className="mt-1 font-medium">Private</p>
            </div>
          </div>
        </section>

        <Card className="flex flex-col justify-center">
          <CardHeader className="space-y-3 pb-5">
            <div className="ag-eyebrow w-fit">
              <User className="h-3.5 w-3.5" />
              智能训练管理台
            </div>
            <CardTitle className="text-2xl">欢迎回来</CardTitle>
            <CardDescription>使用你的账号进入训练控制台</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="user">用户名</Label>
              <Input
                id="user"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="pass">密码</Label>
              <Input
                id="pass"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? <Loader2 className="animate-spin" /> : "登录"}
              {!loading ? <ArrowRight className="h-4 w-4" /> : null}
            </Button>
            <p className="text-center text-sm text-muted-foreground">
              没有账号？{" "}
              <Link to="/register" className="text-primary underline-offset-4 hover:underline">
                注册
              </Link>
            </p>
          </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
