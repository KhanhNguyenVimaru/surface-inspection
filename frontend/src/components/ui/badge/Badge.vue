<script setup lang="ts">
import { cn } from "@/lib/utils";

defineProps<{ class?: string }>();
</script>

<template>
  <span :class="cn('inline-flex items-center rounded-md border border-border bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground', $props.class)">
    <slot />
  </span>
</template>
