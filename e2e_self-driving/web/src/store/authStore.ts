import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/api/types";
import { authTokenRef } from "@/store/authTokenRef";

interface AuthState {
  token: string | null;
  user: User | null;
  setAuth: (token: string, user: User) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      setAuth: (token, user) => set({ token, user }),
      logout: () => set({ token: null, user: null }),
    }),
    {
      name: "e2e-training-auth",
      partialize: (s) => ({ token: s.token, user: s.user }),
    },
  ),
);

authTokenRef.getToken = () => useAuthStore.getState().token;
authTokenRef.onUnauthorized = () => {
  useAuthStore.getState().logout();
};
