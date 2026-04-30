import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";
import { Activity, Braces, Cpu, Gauge, LayoutDashboard, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/store/authStore";
import { cn } from "@/lib/utils";

const nav = [
  { to: "/", label: "训练工作台", desc: "项目 / 数据 / 任务", icon: LayoutDashboard },
];

export function AppShell() {
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="z-10 border-b border-white/10 bg-background/75 backdrop-blur-2xl md:sticky md:top-0 md:flex md:h-screen md:w-[280px] md:shrink-0 md:flex-col md:border-b-0 md:border-r">
        <Link to="/" className="flex items-center gap-3 p-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-md border border-primary/25 bg-primary/[0.12] text-primary shadow-[0_0_30px_-18px_hsl(var(--primary))]">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <p className="font-semibold leading-tight">端到端训练台</p>
            <p className="text-xs text-muted-foreground">Autonomous Lab Console</p>
          </div>
        </Link>
        <nav className="flex gap-1 px-3 pb-3 md:flex-col md:px-3">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-all",
                  isActive
                    ? "border border-primary/20 bg-primary/[0.12] text-primary shadow-[inset_0_1px_0_hsl(var(--primary)/0.22)]"
                    : "border border-transparent text-muted-foreground hover:bg-white/[0.06] hover:text-foreground",
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="min-w-0">
                <span className="block leading-tight">{label}</span>
                <span className="hidden text-xs text-muted-foreground group-hover:text-muted-foreground md:block">
                  项目 / 数据 / 任务
                </span>
              </span>
            </NavLink>
          ))}
        </nav>
        <div className="hidden px-4 pb-4 md:mt-auto md:block">
          <div className="ag-panel-soft p-4">
            <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
              <Activity className="h-3.5 w-3.5 text-primary" />
              <span>控制台状态</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-md border border-white/10 bg-background/45 p-2">
                <Gauge className="mb-1 h-3.5 w-3.5 text-violet-300" />
                <p className="text-muted-foreground">Pipeline</p>
                <p className="font-medium text-foreground">Ready</p>
              </div>
              <div className="rounded-md border border-white/10 bg-background/45 p-2">
                <Braces className="mb-1 h-3.5 w-3.5 text-primary" />
                <p className="text-muted-foreground">Session</p>
                <p className="truncate font-medium text-foreground">{user?.username ?? "-"}</p>
              </div>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="mt-3 w-full justify-start gap-2"
            onClick={() => {
              logout();
              navigate("/login");
            }}
          >
            <LogOut className="h-4 w-4" />
            退出
          </Button>
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <div className="mx-auto w-full max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
          <Outlet />
        </div>
      </main>
      <footer className="border-t border-white/10 bg-background/70 p-2 text-center text-xs text-muted-foreground backdrop-blur md:hidden">
        <Link to="/login" onClick={() => logout()} className="underline">
          退出登录
        </Link>
      </footer>
    </div>
  );
}
