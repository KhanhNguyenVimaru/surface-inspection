<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import Button from "@/components/ui/button/Button.vue";
import Card from "@/components/ui/card/Card.vue";
import CardContent from "@/components/ui/card/CardContent.vue";
import CardDescription from "@/components/ui/card/CardDescription.vue";
import CardHeader from "@/components/ui/card/CardHeader.vue";
import CardTitle from "@/components/ui/card/CardTitle.vue";
import Input from "@/components/ui/input/Input.vue";
import Label from "@/components/ui/label/Label.vue";
import Alert from "@/components/ui/alert/Alert.vue";

type PredictResponse = {
  prediction: string;
  confidence: number;
  model_name: string;
  uploaded_file: string;
  preprocessing?: {
    apply_filter: boolean;
    pipeline: string;
  };
};

const checkpointExists = ref(false);
const healthLoading = ref(true);
const selectedFile = ref<File | null>(null);
const previewUrl = ref<string>("");
const loading = ref(false);
const errorMessage = ref("");
const result = ref<PredictResponse | null>(null);
const fileInputKey = ref(0);
const applyFilter = ref(true);

const normalizedConfidence = computed(() => {
  const value = Number(result.value?.confidence ?? 0);
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, value));
});

const confidenceRemainder = computed(() => Number((100 - normalizedConfidence.value).toFixed(2)));

const confidencePieStyle = computed(() => ({
  background: `conic-gradient(rgb(244 244 245) 0% ${normalizedConfidence.value}%, rgb(39 39 42) ${normalizedConfidence.value}% 100%)`,
}));

async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    checkpointExists.value = Boolean(data.checkpoint_exists);
  } catch {
    checkpointExists.value = false;
  } finally {
    healthLoading.value = false;
  }
}

function onFileChange(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
  }
  if (!file) {
    selectedFile.value = null;
    previewUrl.value = "";
    return;
  }
  selectedFile.value = file;
  previewUrl.value = URL.createObjectURL(file);
  errorMessage.value = "";
  result.value = null;
}

async function submitPrediction() {
  if (!selectedFile.value) {
    errorMessage.value = "Please choose an image file first.";
    return;
  }
  loading.value = true;
  errorMessage.value = "";
  result.value = null;

  try {
    const formData = new FormData();
    formData.append("image", selectedFile.value);
    formData.append("apply_filter", applyFilter.value ? "1" : "0");

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      errorMessage.value = data.message || "Prediction failed.";
      return;
    }
    result.value = data as PredictResponse;
  } catch {
    errorMessage.value = "Cannot connect to API server.";
  } finally {
    loading.value = false;
  }
}

function resetForm() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
  }
  selectedFile.value = null;
  previewUrl.value = "";
  errorMessage.value = "";
  result.value = null;
  fileInputKey.value += 1;
}

onMounted(fetchHealth);
onBeforeUnmount(() => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
  }
});
</script>

<template>
  <div class="relative min-h-screen overflow-hidden bg-background text-foreground">
    <div class="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(63,63,70,0.28),_rgba(9,9,11,0.95)_60%)]" />
    <div class="container relative py-8 md:py-12">
      <Card class="border-border/80 bg-card/95 shadow-xl shadow-black/35 backdrop-blur-sm">
        <CardHeader class="space-y-3 border-b border-border pb-5">
          <CardTitle class="text-2xl text-foreground md:text-3xl">Surface Defect Detection</CardTitle>
          <CardDescription class="max-w-2xl text-muted-foreground">
            Upload a metal surface image and get defect class prediction from the ResNet18 model via Flask API.
          </CardDescription>
        </CardHeader>

        <CardContent class="space-y-6 pt-6">
          <Alert v-if="healthLoading">
            Checking backend status...
          </Alert>
          <Alert v-else-if="!checkpointExists" variant="warning">
            Checkpoint not found at <code>outputs/checkpoints/best_resnet18.pt</code>. Train the model first.
          </Alert>
          <Alert v-if="errorMessage" variant="destructive">{{ errorMessage }}</Alert>

          <div class="space-y-3 rounded-lg border border-border bg-zinc-950/50 p-4">
            <div class="flex items-center justify-between gap-3">
              <Label for="image" class="text-foreground">Picture</Label>
              <div class="flex items-center gap-2">
              </div>
            </div>
            <div class="space-y-2">
              <Input
                :key="fileInputKey"
                id="image"
                type="file"
                accept=".jpg,.jpeg,.png,.bmp"
                class="h-10 rounded-md border-input bg-zinc-900/70 file:mr-3 file:rounded-sm file:bg-zinc-800 file:px-2 file:text-zinc-100 hover:file:bg-zinc-700"
                @change="onFileChange"
              />
              <p class="text-xs text-muted-foreground">Select an image to upload for prediction.</p>
              <label class="inline-flex items-center gap-2 text-xs text-muted-foreground">
                <input
                  v-model="applyFilter"
                  type="checkbox"
                  class="h-4 w-4 rounded border border-input bg-transparent accent-zinc-100"
                />
                Apply grayscale enhancement filter (recommended for steel texture images)
              </label>
            </div>
            <div class="flex gap-2 justify-end">
              <Button
                  variant="outline"
                  :disabled="loading"
                  @click="resetForm"
                >
                  Reset
                </Button>
                <Button
                  :disabled="!checkpointExists || loading"
                  @click="submitPrediction"
                >
                  {{ loading ? "Predicting..." : "Run Prediction" }}
                </Button>
            </div>
          </div>

          <div class="grid gap-4 md:grid-cols-2">
            <Card class="border-border bg-zinc-950/55 shadow-sm">
              <CardHeader class="pb-3">
                <CardTitle class="text-lg text-foreground">Input Preview</CardTitle>
                <CardDescription class="text-muted-foreground">Review the image before inference.</CardDescription>
              </CardHeader>
              <CardContent>
                <div class="flex min-h-72 items-center justify-center rounded-md border border-dashed border-border bg-zinc-900/60 p-2">
                  <img
                    v-if="previewUrl"
                    :src="previewUrl"
                    alt="preview"
                    class="max-h-80 w-full rounded-sm border border-border object-contain bg-zinc-950"
                  />
                  <p v-else class="text-sm text-muted-foreground">No image selected.</p>
                </div>
              </CardContent>
            </Card>

            <Card class="border-border bg-zinc-950/55 shadow-sm">
              <CardHeader class="pb-3">
                <CardTitle class="text-lg text-foreground">Prediction Output</CardTitle>
              </CardHeader>
              <CardContent>
                <div v-if="result" class="space-y-3 text-sm">
                  <p class="flex items-center justify-between border-b border-border pb-2">
                    <span class="text-muted-foreground">Prediction</span>
                    <strong class="text-foreground">{{ result.prediction }}</strong>
                  </p>
                  <p class="flex items-center justify-between border-b border-border pb-2">
                    <span class="text-muted-foreground">Confidence</span>
                    <strong class="text-foreground">{{ normalizedConfidence.toFixed(2) }}%</strong>
                  </p>
                  <p class="flex items-center justify-between border-b border-border pb-2">
                    <span class="text-muted-foreground">Preprocess Filter</span>
                    <strong class="text-foreground">{{ result.preprocessing?.apply_filter ? "On" : "Off" }}</strong>
                  </p>
                  <div class="pt-1">
                    <div class="rounded-md border border-border bg-zinc-900/60 p-3">
                      <p class="mb-2 text-xs uppercase tracking-wide text-muted-foreground">Confidence</p>
                      <div class="flex items-center gap-3">
                        <div :style="confidencePieStyle" class="relative h-20 w-20 rounded-full">
                          <div class="absolute inset-[9px] rounded-full bg-zinc-950" />
                        </div>
                        <div class="space-y-1 text-xs">
                          <p class="flex items-center gap-2">
                            <span class="inline-block h-2 w-2 rounded-full bg-zinc-100" />
                            <span class="text-muted-foreground">Confident</span>
                            <strong class="text-foreground">{{ normalizedConfidence.toFixed(2) }}%</strong>
                          </p>
                          <p class="flex items-center gap-2">
                            <span class="inline-block h-2 w-2 rounded-full bg-zinc-700" />
                            <span class="text-muted-foreground">Uncertain</span>
                            <strong class="text-foreground">{{ confidenceRemainder.toFixed(2) }}%</strong>
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <p v-else class="text-sm text-muted-foreground">No prediction yet.</p>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  </div>
</template>
