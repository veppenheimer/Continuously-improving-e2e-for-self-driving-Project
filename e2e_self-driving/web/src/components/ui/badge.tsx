import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring/70 focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary/[0.18] text-primary hover:bg-primary/[0.24]",
        secondary: "border-white/10 bg-secondary/75 text-secondary-foreground hover:bg-secondary",
        outline: "border-border/80 text-foreground",
        success: "border-sky-400/20 bg-sky-500/[0.14] text-sky-300",
        warning: "border-violet-400/[0.22] bg-violet-500/[0.14] text-violet-300",
        muted: "border-white/10 bg-muted/80 text-muted-foreground",
        destructive: "border-red-400/20 bg-red-500/[0.14] text-red-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
