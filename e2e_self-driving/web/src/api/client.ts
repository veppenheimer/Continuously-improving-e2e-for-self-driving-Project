import axios, {
  type AxiosError,
  type AxiosInstance,
  type InternalAxiosRequestConfig,
} from "axios";
import { toast } from "sonner";
import { API_BASE_URL } from "@/config/env";
import { authTokenRef } from "@/store/authTokenRef";
import type { ApiErrorBody } from "./types";

function getMessage(err: AxiosError<ApiErrorBody>): string {
  const d = err.response?.data;
  if (typeof d?.detail === "string") return d.detail;
  if (typeof d?.message === "string") return d.message;
  if (err.message) return err.message;
  return "请求失败";
}

let client: AxiosInstance | null = null;

export function getApiClient(): AxiosInstance {
  if (client) return client;
  client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 120_000,
    headers: { "Content-Type": "application/json" },
  });

  client.interceptors.request.use((config: InternalAxiosRequestConfig) => {
    const token = authTokenRef.getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  });

  client.interceptors.response.use(
    (res) => res,
    (error: AxiosError<ApiErrorBody>) => {
      const status = error.response?.status;
      if (status === 401) {
        authTokenRef.onUnauthorized?.();
        toast.error("登录已过期，请重新登录");
      } else {
        toast.error(getMessage(error));
      }
      return Promise.reject(error);
    },
  );

  return client;
}

/** 业务层可调用：仅提示、不重复 logout */
export function showApiError(err: unknown, fallback = "操作失败") {
  if (axios.isAxiosError<ApiErrorBody>(err)) {
    toast.error(getMessage(err));
  } else {
    toast.error(fallback);
  }
}
