import type {
  LanguageModelV2Prompt,
  LanguageModelV2StreamPart,
} from "@ai-sdk/provider";

const DEFAULT_RESPONSE = "Hello, world!";
const WEATHER_RESPONSE = "The weather in San Francisco is sunny and 72°F.";
const GREETING_RESPONSE = "Hello! How can I help you today?";

function getPromptText(prompt: LanguageModelV2Prompt): string {
  const parts: string[] = [];

  for (const message of prompt) {
    if (typeof message.content === "string") {
      parts.push(message.content);
      continue;
    }

    for (const part of message.content) {
      if (part.type === "text") {
        parts.push(part.text);
      }
    }
  }

  return parts.join("\n").toLowerCase();
}

function getResponseText(prompt: LanguageModelV2Prompt): string {
  const text = getPromptText(prompt);

  if (text.includes("weather") || text.includes("temperature")) {
    return WEATHER_RESPONSE;
  }

  if (/\b(hello|hi|hey)\b/.test(text)) {
    return GREETING_RESPONSE;
  }

  return DEFAULT_RESPONSE;
}

export function getResponseChunksByPrompt(
  prompt: LanguageModelV2Prompt,
  reasoning = false
): LanguageModelV2StreamPart[] {
  const response = getResponseText(prompt);
  const chunks: LanguageModelV2StreamPart[] = [];

  if (reasoning) {
    chunks.push({ type: "reasoning-start", id: "reasoning-1" });
    chunks.push({
      type: "reasoning-delta",
      id: "reasoning-1",
      delta: "Let me think through this step by step... ",
    });
    chunks.push({ type: "reasoning-end", id: "reasoning-1" });
  }

  chunks.push({ type: "text-start", id: "text-1" });
  chunks.push({ type: "text-delta", id: "text-1", delta: response });
  chunks.push({ type: "text-end", id: "text-1" });
  chunks.push({
    type: "finish",
    finishReason: "stop",
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  });

  return chunks;
}
