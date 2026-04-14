import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { LossPoint } from "@/api/types";
import { cn } from "@/lib/utils";

interface SeriesDef {
  name: string;
  train: LossPoint[];
  val: LossPoint[];
  color: string;
}

interface LossChartProps {
  series: SeriesDef[];
  className?: string;
}

export function LossChart({ series, className }: LossChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    if (!chartRef.current) {
      chartRef.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    }
    const chart = chartRef.current;
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || chart.isDisposed()) return;

    const legend: string[] = [];
    const trainSeries: echarts.SeriesOption[] = [];
    const valSeries: echarts.SeriesOption[] = [];

    series.forEach((s) => {
      legend.push(`${s.name} 训练 Loss`, `${s.name} 验证 Loss`);
      trainSeries.push({
        name: `${s.name} 训练 Loss`,
        type: "line",
        smooth: true,
        data: s.train.map((p) => [p.epoch, p.trainLoss]),
        itemStyle: { color: s.color },
      });
      valSeries.push({
        name: `${s.name} 验证 Loss`,
        type: "line",
        smooth: true,
        data: s.val.map((p) => [p.epoch, p.valLoss]),
        itemStyle: { color: s.color },
        lineStyle: { type: "dashed" },
      });
    });

    chart.setOption(
      {
        backgroundColor: "transparent",
        textStyle: { color: "#94a3b8" },
        tooltip: { trigger: "axis" },
        legend: { data: legend, textStyle: { color: "#94a3b8" } },
        grid: { left: 48, right: 24, top: 40, bottom: 32 },
        xAxis: {
          type: "value",
          name: "Epoch",
          nameLocation: "middle",
          nameGap: 28,
          axisLine: { lineStyle: { color: "#334155" } },
          splitLine: { lineStyle: { color: "#1e293b" } },
        },
        yAxis: {
          type: "value",
          name: "Loss",
          axisLine: { lineStyle: { color: "#334155" } },
          splitLine: { lineStyle: { color: "#1e293b" } },
        },
        series: [...trainSeries, ...valSeries],
      },
      true,
    );
  }, [series]);

  return <div ref={ref} className={cn("h-[360px] w-full min-h-[280px]", className)} />;
}
