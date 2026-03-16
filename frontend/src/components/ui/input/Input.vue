<script setup lang="ts">
import { cn } from "@/lib/utils";

defineProps<{ class?: string }>();
</script>

<template>
  <input
    :class="
      cn(
        'flex h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-sm text-foreground shadow-sm outline-none transition-[color,box-shadow] file:mr-2 file:inline-flex file:h-7 file:border-0 file:bg-transparent file:px-0 file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50',
        $props.class
      )
    "
  />
</template>
