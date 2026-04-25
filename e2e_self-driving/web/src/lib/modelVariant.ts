import type { TrainModelVariant } from "@/api/types";

export interface ModelVariantOption {
  value: TrainModelVariant;
  label: string;
  description: string;
}

export const MODEL_VARIANT_OPTIONS: ModelVariantOption[] = [
  {
    value: "legacy",
    label: "Legacy CNN",
    description: "保留旧版单帧卷积结构，用于历史基线复盘。",
  },
  {
    value: "mobilenet_v2",
    label: "单帧 MobileNetV2",
    description: "当前默认单帧主线模型。",
  },
  {
    value: "temporal3",
    label: "3-frame",
    description: "固定使用 3 帧输入与 stride=1 的时序模型。",
  },
];

export function modelVariantLabel(variant?: string | null): string {
  switch (variant) {
    case "legacy":
      return "Legacy CNN";
    case "mobilenet_v2":
      return "单帧 MobileNetV2";
    case "temporal3":
      return "3-frame";
    default:
      return "旧任务/未知架构";
  }
}
