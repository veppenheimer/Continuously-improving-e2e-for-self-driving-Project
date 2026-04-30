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
        symbol: "none",
        data: s.train.map((p) => [p.epoch, p.trainLoss]),
        itemStyle: { color: s.color },
        lineStyle: { width: 3 },
        areaStyle: { color: s.color, opacity: 0.08 },
      });
      valSeries.push({
        name: `${s.name} 验证 Loss`,
        type: "line",
        smooth: true,
        symbol: "none",
        data: s.val.map((p) => [p.epoch, p.valLoss]),
        itemStyle: { color: s.color },
        lineStyle: { type: "dashed", width: 2 },
      });
    });

    chart.setOption(
      {
        backgroundColor: "transparent",
        textStyle: { color: "#b9c2ff" },
        tooltip: {
          trigger: "axis",
          backgroundColor: "rgba(16, 18, 20, 0.92)",
          borderColor: "rgba(255, 255, 255, 0.12)",
          textStyle: { color: "#eef2ff" },
        },
        legend: { data: legend, textStyle: { color: "#b9c2ff" }, top: 0 },
        grid: { left: 48, right: 24, top: 48, bottom: 36 },
        xAxis: {
          type: "value",
          name: "Epoch",
          nameLocation: "middle",
          nameGap: 28,
          axisLabel: { color: "#9ca7df" },
          axisLine: { lineStyle: { color: "rgba(255,255,255,0.18)" } },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.07)" } },
        },
        yAxis: {
          type: "value",
          name: "Loss",
          axisLabel: { color: "#9ca7df" },
          axisLine: { lineStyle: { color: "rgba(255,255,255,0.18)" } },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.07)" } },
        },
        series: [...trainSeries, ...valSeries],
      },
      true,
    );
  }, [series]);

  return <div ref={ref} className={cn("h-[360px] w-full min-h-[280px]", className)} />;
}
