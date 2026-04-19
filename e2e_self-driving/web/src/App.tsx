import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { LoginPage } from "@/pages/LoginPage";
import { RegisterPage } from "@/pages/RegisterPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { TaskMonitorPage } from "@/pages/TaskMonitorPage";
import { TaskResultsPage } from "@/pages/TaskResultsPage";
import { DomainAugComparePage } from "@/pages/DomainAugComparePage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route
          element={
            <ProtectedRoute>
              <AppShell />
            </ProtectedRoute>
          }
        >
          <Route index element={<DashboardPage />} />
          <Route path="tasks/:taskId/monitor" element={<TaskMonitorPage />} />
          <Route path="tasks/:taskId/results" element={<TaskResultsPage />} />
          <Route path="tasks/:taskId/domain-compare" element={<DomainAugComparePage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
