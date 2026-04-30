import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { registerApi } from "@/api/services/auth";
import { showApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuthStore } from "@/store/authStore";
import { ArrowRight, Database, Loader2, Mail, ShieldCheck, Sparkles, UserPlus } from "lucide-react";

export function RegisterPage() {
  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await registerApi({
        username,
        password,
        email: email || undefined,
      });
      setAuth(res.token, res.user);
      toast.success("注册成功");
      navigate("/", { replace: true });
    } catch (err) {
      showApiError(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4 py-8">
      <div className="grid w-full max-w-5xl gap-5 lg:grid-cols-[0.95fr_1.05fr]">
        <Card className="flex flex-col justify-center">
          <CardHeader className="space-y-3 pb-5">
            <div className="ag-eyebrow w-fit">
              <UserPlus className="h-3.5 w-3.5" />
              快速开启训练
            </div>
            <CardTitle className="text-2xl">创建账号</CardTitle>
            <CardDescription>注册后即可上传数据集并启动你的训练任务</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="reg-user">用户名</Label>
              <Input
                id="reg-user"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="reg-email">邮箱（可选）</Label>
              <Input
                id="reg-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="reg-pass">密码</Label>
              <Input
                id="reg-pass"
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
              />
            </div>
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? <Loader2 className="animate-spin" /> : "注册并登录"}
              {!loading ? <ArrowRight className="h-4 w-4" /> : null}
            </Button>
            <p className="text-center text-sm text-muted-foreground">
              已有账号？{" "}
              <Link to="/login" className="text-primary underline-offset-4 hover:underline">
                登录
              </Link>
            </p>
          </form>
          </CardContent>
        </Card>

        <section className="ag-page-hero flex min-h-[520px] flex-col justify-between rounded-lg border border-white/10 bg-background/45 p-6 backdrop-blur-xl">
          <div>
            <div className="ag-eyebrow">
              <Sparkles className="h-3.5 w-3.5" />
              Antigravity Lab
            </div>
            <h1 className="mt-6 max-w-xl text-4xl font-semibold leading-tight tracking-tight text-foreground md:text-5xl">
              从数据集到模型产物，流程更清晰
            </h1>
            <p className="mt-4 max-w-lg text-sm leading-6 text-muted-foreground">
              项目、数据集、域增强、训练任务和推理对比集中在一个控制台里，适合反复实验和复盘。
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="ag-kpi">
              <Database className="mb-3 h-5 w-5 text-primary" />
              <p className="text-xs text-muted-foreground">Dataset</p>
              <p className="mt-1 font-medium">ZIP Upload</p>
            </div>
            <div className="ag-kpi">
              <ShieldCheck className="mb-3 h-5 w-5 text-violet-300" />
              <p className="text-xs text-muted-foreground">Account</p>
              <p className="mt-1 font-medium">Secure</p>
            </div>
            <div className="ag-kpi">
              <Mail className="mb-3 h-5 w-5 text-fuchsia-300" />
              <p className="text-xs text-muted-foreground">Email</p>
              <p className="mt-1 font-medium">Optional</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
