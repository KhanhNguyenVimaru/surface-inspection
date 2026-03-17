<script setup lang="ts">
import { cva } from "class-variance-authority";
import { computed } from "vue";
import { cn } from "@/lib/utils";

const alertVariants = cva("relative w-full rounded-md border p-4 text-sm", {
  variants: {
    variant: {
      default: "border-border bg-muted text-foreground",
      destructive: "border-destructive/30 bg-destructive/10 text-red-700",
      warning: "border-amber-300 bg-amber-50 text-amber-800",
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
