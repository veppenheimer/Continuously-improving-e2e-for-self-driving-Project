import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";
import { LogOut, LayoutDashboard, Upload, PlayCircle, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/store/authStore";
import { cn } from "@/lib/utils";

const nav = [
  { to: "/", label: "训练任务", icon: LayoutDashboard },
  { to: "/datasets", label: "数据集", icon: Upload },
  { to: "/train/new", label: "新建训练", icon: PlayCircle },
];

export function AppShell() {
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="border-b border-border/60 bg-card/60 backdrop-blur-xl md:flex md:w-64 md:flex-col md:border-b-0 md:border-r md:shrink-0">
        <div className="flex items-center gap-2 p-5">
          <Cpu className="h-8 w-8 text-primary" />
          <div>
            <p className="font-semibold leading-tight">端到端 训练台</p>
            <p className="text-xs text-muted-foreground">Autonomous Lab Console</p>
          </div>
        </div>
        <nav className="flex gap-1 px-3 pb-2 md:flex-col md:px-3">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2 rounded-lg px-3 py-2.5 text-sm transition-all",
                  isActive
                    ? "bg-primary/20 text-primary shadow-[inset_0_1px_0_hsl(var(--primary)/0.35)]"
                    : "text-muted-foreground hover:bg-muted/70 hover:text-foreground",
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden p-4 md:mt-auto md:block">
          <p className="truncate text-xs text-muted-foreground">已登录：{user?.username}</p>
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 w-full justify-start gap-2 text-muted-foreground hover:text-foreground"
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
      <main className="flex-1 overflow-auto p-4 md:p-8">
        <div className="mx-auto max-w-6xl">
          <Outlet />
        </div>
      </main>
      <footer className="border-t border-border p-2 text-center text-xs text-muted-foreground md:hidden">
        <Link to="/login" onClick={() => logout()} className="underline">
          退出登录
        </Link>
      </footer>
    </div>
  );
}
