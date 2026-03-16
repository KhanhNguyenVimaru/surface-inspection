<script setup lang="ts">
import { cn } from "@/lib/utils";

defineProps<{ class?: string }>();
</script>

<template>
  <div :class="cn('rounded-sm border bg-white text-foreground shadow-sm', $props.class)">
    <slot />
  </div>
</template>
