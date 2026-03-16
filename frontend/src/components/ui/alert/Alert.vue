<script setup lang="ts">
import { cva } from "class-variance-authority";
import { computed } from "vue";
import { cn } from "@/lib/utils";

const alertVariants = cva("relative w-full rounded-md border p-4 text-sm", {
  variants: {
    variant: {
      default: "border-border bg-muted text-foreground",
      destructive: "border-destructive/50 bg-destructive/15 text-red-300",
      warning: "border-amber-700/60 bg-amber-950/35 text-amber-200",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

type AlertVariant = "default" | "destructive" | "warning";

type Props = {
  variant?: AlertVariant;
  class?: string;
}

const props = defineProps<Props>();
const alertClass = computed(() => cn(alertVariants({ variant: props.variant }), props.class));
</script>

<template>
  <div :class="alertClass">
    <slot />
  </div>
</template>
