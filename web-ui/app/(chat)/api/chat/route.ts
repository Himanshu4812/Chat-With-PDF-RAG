import { streamQuery, type Source, type StreamTimings } from "@/lib/ai/provider";
import { getDefaultChatModel } from "@/lib/ai/models";
import { createUIMessageStream, createUIMessageStreamResponse } from "ai";
import { auth } from "@/app/(auth)/auth";
import {
  getChatById,
  getMessageById,
  saveChat,
  saveMessages,
} from "@/lib/db/queries";
import { generateUUID } from "@/lib/utils";

export const maxDuration = 300;

type StreamContext = {
  resumableStream: (
    streamId: string,
    makeStream: () => ReadableStream
  ) => Promise<ReadableStream | null>;
};

export function getStreamContext(): StreamContext | null {
  return null;
}

interface MessagePart {
  type: string;
  text?: string;
  url?: string;
  name?: string;
  mediaType?: string;
}

interface PostRequestBody {
  id: string;
  message: {
    id?: string;
    role: string;
    parts: MessagePart[];
  };
  selectedChatModel?: string;
  selectedVisibilityType?: string;
  selectedPdfIds?: string[];
}

type ActiveGeneration = {
  requestId: string;
  controller: AbortController;
};

// One generation per chat and per (chat, user message) at a time.
const activeByChat = new Map<string, ActiveGeneration>();
const activeByRequest = new Map<string, AbortController>();

function jsonError(message: string, status: number) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || /abort/i.test(error.message))
  );
}

/**
 * Next.js aborts request.signal when the route handler returns, even though
 * the response is still streaming. Detect real client cancellation from the
 * response body's cancel() instead (Stop button, navigation, disconnect).
 */
function responseWithAbort(
  response: Response,
  controller: AbortController
): Response {
  if (!response.body) {
    return response;
  }

  const reader = response.body.getReader();
  const body = new ReadableStream<Uint8Array>({
    async pull(streamController) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          streamController.close();
          return;
        }
        streamController.enqueue(value);
      } catch (error) {
        streamController.error(error);
      }
    },
    cancel(reason) {
      controller.abort();
      return reader.cancel(reason).catch(() => {});
    },
  });

  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

function formatSourcesMarkdown(sources: Source[]): string {
  if (sources.length === 0) {
    return "";
  }
  const lines = sources.map((source) => {
    const page =
      source.page_number !== null &&
      source.page_number !== undefined &&
      source.page_number !== ""
        ? `, page ${source.page_number}`
        : "";
    return `- ${source.pdf_name} (chunk ${source.chunk_index}${page})`;
  });
  return `\n\n**Sources:**\n${lines.join("\n")}`;
}

// Simple question classifier was removed: without selected PDFs the backend
// answers directly (pdf_ids=[]), so general chat is never blocked by a
// static warning and Stop can cancel real generation.

export async function POST(request: Request) {
  let cleanup = () => {};

  try {
    const body: PostRequestBody = await request.json();
    const chatId = body.id;

    const session = await auth();
    if (!session?.user?.id) {
      return jsonError("Unauthorized", 401);
    }
    const userId = session.user.id;

    const selectedChatModel =
      body.selectedChatModel || (await getDefaultChatModel());

    const textPart = body.message.parts.find((p) => p.type === "text");
    const textContent = textPart?.text || "";

    if (!textContent) {
      return jsonError("No text content found in message", 400);
    }

    const userMessageId = body.message.id || generateUUID();
    const chatKey = `${userId}:${chatId}`;
    const requestKey = `${userId}:${chatId}:${userMessageId}`;

    if (activeByRequest.has(requestKey)) {
      return jsonError(
        "A response for this message is already being generated.",
        409
      );
    }

    // A new message supersedes any in-flight generation for the same chat.
    activeByChat.get(chatKey)?.controller.abort();

    const controller = new AbortController();
    activeByChat.set(chatKey, { requestId: requestKey, controller });
    activeByRequest.set(requestKey, controller);

    cleanup = () => {
      if (activeByRequest.get(requestKey) === controller) {
        activeByRequest.delete(requestKey);
      }
      if (activeByChat.get(chatKey)?.controller === controller) {
        activeByChat.delete(chatKey);
      }
    };

    let chat = await getChatById({ id: chatId });
    if (!chat) {
      const title =
        textContent.slice(0, 100) + (textContent.length > 100 ? "..." : "");
      await saveChat({
        id: chatId,
        userId,
        title,
        visibility:
          (body.selectedVisibilityType as "private" | "public") || "private",
      });
      chat = await getChatById({ id: chatId });
    }

    // Avoid duplicate user-message inserts on retries/regenerations.
    const existingUserMessage = await getMessageById({ id: userMessageId });
    if (existingUserMessage.length === 0) {
      await saveMessages({
        messages: [
          {
            id: userMessageId,
            chatId,
            role: "user",
            content: textContent,
            parts: JSON.stringify(body.message.parts),
            createdAt: new Date(),
          },
        ],
      });
    }

    const selectedPdfIds = Array.isArray(body.selectedPdfIds)
      ? body.selectedPdfIds
      : [];
    const useRAG = selectedPdfIds.length > 0;

    console.log(
      `[chat] model=${selectedChatModel} mode=${
        useRAG ? "RAG" : "direct"
      } pdfs=${selectedPdfIds.length}`
    );

    const assistantMessageId = generateUUID();
    const reasoningId = "reasoning-1";
    const textId = "text-1";

    const messageStream = createUIMessageStream({
      execute: async ({ writer }) => {
        let reasoningStarted = false;
        let reasoningClosed = false;
        let textStarted = false;
        let textClosed = false;
        let reasoningText = "";

        const startReasoning = () => {
          if (!reasoningStarted) {
            writer.write({ type: "reasoning-start", id: reasoningId });
            reasoningStarted = true;
          }
        };

        const startText = () => {
          if (!textStarted) {
            writer.write({ type: "text-start", id: textId });
            textStarted = true;
          }
        };

        const closeParts = () => {
          if (reasoningStarted && !reasoningClosed) {
            writer.write({ type: "reasoning-end", id: reasoningId });
            reasoningClosed = true;
          }
          if (textStarted && !textClosed) {
            writer.write({ type: "text-end", id: textId });
            textClosed = true;
          }
        };

        try {
          let completed = false;
          let answer = "";
          let sources: Source[] = [];
          let timings: StreamTimings = {};

          for await (const event of streamQuery(textContent, {
            chatId,
            model: selectedChatModel,
            pdfIds: useRAG ? selectedPdfIds : [],
            requestId: requestKey,
            signal: controller.signal,
          })) {
            if (controller.signal.aborted) {
              break;
            }

            switch (event.event) {
              case "meta": {
                timings = event.data.timings;
                break;
              }
              case "reasoning": {
                startReasoning();
                const delta = event.data.text.endsWith("\n")
                  ? event.data.text
                  : `${event.data.text}\n`;
                reasoningText += delta;
                writer.write({
                  type: "reasoning-delta",
                  id: reasoningId,
                  delta,
                });
                break;
              }
              case "reasoning_delta": {
                startReasoning();
                reasoningText += event.data.text;
                writer.write({
                  type: "reasoning-delta",
                  id: reasoningId,
                  delta: event.data.text,
                });
                break;
              }
              case "sources": {
                sources = event.data.sources;
                if (sources.length > 0) {
                  writer.write({
                    type: "data-sources" as any,
                    data: sources,
                  });
                }
                break;
              }
              case "token": {
                startText();
                answer += event.data.text;
                writer.write({
                  type: "text-delta",
                  id: textId,
                  delta: event.data.text,
                });
                break;
              }
              case "done": {
                completed = true;
                answer = event.data.answer;
                sources = event.data.sources;
                timings = event.data.timings;
                if (event.data.reasoning && !reasoningText) {
                  reasoningText = event.data.reasoning;
                }

                const sourceMarkdown = formatSourcesMarkdown(sources);
                if (sourceMarkdown) {
                  startText();
                  answer += sourceMarkdown;
                  writer.write({
                    type: "text-delta",
                    id: textId,
                    delta: sourceMarkdown,
                  });
                }

                if (sources.length > 0) {
                  writer.write({
                    type: "data-sources" as any,
                    data: sources,
                  });
                }
                break;
              }
              case "error": {
                throw new Error(event.data.error);
              }
            }
          }

          if (!completed) {
            throw new Error(
              "The model stream ended before the answer was completed."
            );
          }

          // Persist only after a completed stream, and before closing the
          // text part so the client cannot reload before the save commits.
          const assistantParts: any[] = [];
          if (reasoningText.trim()) {
            assistantParts.push({
              type: "reasoning",
              text: reasoningText.trim(),
            });
          }
          assistantParts.push({ type: "text", text: answer });
          if (sources.length > 0) {
            assistantParts.push({ type: "data-sources", data: sources });
          }
          if (Object.keys(timings).length > 0) {
            assistantParts.push({ type: "data-timings", data: timings });
          }

          try {
            await saveMessages({
              messages: [
                {
                  id: assistantMessageId,
                  chatId,
                  role: "assistant",
                  content: answer,
                  parts: JSON.stringify(assistantParts),
                  createdAt: new Date(),
                },
              ],
            });
          } catch (saveError) {
            console.error("[chat] Failed to save assistant message", saveError);
            writer.write({
              type: "error",
              errorText:
                "The answer completed but could not be saved. Please retry.",
            });
            return;
          }

          closeParts();

          console.log(
            `[chat] completed ttft=${timings.ttft_seconds ?? "n/a"}s total=${
              timings.total_seconds ?? "n/a"
            }s`
          );
        } catch (error) {
          if (isAbortError(error) || controller.signal.aborted) {
            console.log("[chat] generation aborted; nothing saved");
            return;
          }

          const message =
            error instanceof Error ? error.message : "Unknown error occurred";
          console.error("[chat] streaming error:", message);

          startText();
          const errorText = `❌ **Error**: ${message}\n\nPlease check:\n- Model is installed and running\n- PDF documents are uploaded\n- Backend service is accessible`;
          writer.write({
            type: "text-delta",
            id: textId,
            delta: errorText,
          });
          closeParts();

          // Emit the error last so the client renders the message above
          // before useChat transitions into its error state.
          writer.write({ type: "error", errorText: message });

          // Not saved: this response is incomplete/errored.
        } finally {
          try {
            closeParts();
          } catch {
            // The stream may already be closed after a client abort.
          }
          cleanup();
        }
      },
    });

    return responseWithAbort(
      createUIMessageStreamResponse({ stream: messageStream }),
      controller
    );
  } catch (error) {
    cleanup();
    console.error("[chat] request error:", error);
    const message =
      error instanceof Error ? error.message : "Unknown error occurred";
    return jsonError(message, 500);
  }
}
