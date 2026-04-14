/** 供 axios 读取 token / 401 回调，避免 api/client 与 authStore 循环引用 */
export const authTokenRef: {
  getToken: () => string | null;
  onUnauthorized?: () => void;
} = {
  getToken: () => null,
};
