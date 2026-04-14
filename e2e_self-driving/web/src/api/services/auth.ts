import { getApiClient } from "@/api/client";
import { paths } from "@/api/endpoints";
import type { User } from "@/api/types";

export interface RegisterPayload {
  username: string;
  password: string;
  email?: string;
}

export interface LoginPayload {
  username: string;
  password: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export async function registerApi(body: RegisterPayload): Promise<AuthResponse> {
  const { data } = await getApiClient().post<AuthResponse>(paths.register, body);
  return data;
}

export async function loginApi(body: LoginPayload): Promise<AuthResponse> {
  const { data } = await getApiClient().post<AuthResponse>(paths.login, body);
  return data;
}

export async function fetchMe(): Promise<User> {
  const { data } = await getApiClient().get<User>(paths.me);
  return data;
}
