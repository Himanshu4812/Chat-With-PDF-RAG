/**
 * Custom Ollama provider integration for FastAPI backend
 */

interface Message {
  role: string;
  content: string;
}

export interface Source {
  pdf_name: string;
  pdf_id: string;
  chunk_index: number;
  page_number?: number | string | null;
}

export interface StreamTimings {
  [key: string]: number | string | boolean | null | undefined;
}

export type StreamQueryEvent =
  | {
      event: "meta";
      data: {
        model: string;
        pdfs_queried: number;
        use_thinking: boolean;
        timings: StreamTimings;
      };
    }
  | { event: "reasoning" | "reasoning_delta" | "token"; data: { text: string } }
  | { event: "sources"; data: { sources: Source[] } }
  | {
      event: "done";
      data: {
        answer: string;
        sources: Source[];
        reasoning: string;
        timings: StreamTimings;
        message_id?: number;
        session_id?: string;
      };
    }
  | { event: "error"; data: { error: string } };

export interface StreamQueryOptions {
  chatId: string;
  model: string;
  pdfIds: string[];
  requestId: string;
  signal?: AbortSignal;
  provider?: string;
  mode?: string;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
  metadata: {
    model_used: string;
    chunks_retrieved: number;
    pdfs_queried: number;
    reasoning_steps?: string[];
  };
  session_id: string;
  message_id: number;
}

function parseStreamEvent(raw: string): StreamQueryEvent | null {
  let eventName = "";
  const dataLines: string[] = [];

  for (const line of raw.replace(/\r\n/g, "\n").split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!eventName || dataLines.length === 0) {
    return null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }
  const data = (parsed ?? {}) as Record<string, unknown>;
  const text = typeof data.text === "string" ? data.text : "";

  switch (eventName) {
    case "meta":
      return {
        event: "meta",
        data: {
          model: typeof data.model === "string" ? data.model : "",
          pdfs_queried:
            typeof data.pdfs_queried === "number" ? data.pdfs_queried : 0,
          use_thinking: Boolean(data.use_thinking),
          timings: (data.timings as StreamTimings) ?? {},
        },
      };
    case "reasoning":
    case "reasoning_delta":
    case "token":
      return { event: eventName, data: { text } };
    case "sources":
      return {
        event: "sources",
        data: { sources: Array.isArray(data.sources) ? (data.sources as Source[]) : [] },
      };
    case "done":
      if (typeof data.answer !== "string") {
        return null;
      }
      return {
        event: "done",
        data: {
          answer: data.answer,
          sources: Array.isArray(data.sources) ? (data.sources as Source[]) : [],
          reasoning: typeof data.reasoning === "string" ? data.reasoning : "",
          timings: (data.timings as StreamTimings) ?? {},
          message_id:
            typeof data.message_id === "number" ? data.message_id : undefined,
          session_id:
            typeof data.session_id === "string" ? data.session_id : undefined,
        },
      };
    case "error":
      return {
        event: "error",
        data: {
          error:
            typeof data.error === "string" ? data.error : "Unknown stream error",
        },
      };
    default:
      return null;
  }
}

/**
 * Stream a genuine token-by-token answer from the FastAPI backend.
 * Closing/aborting this generator cancels the upstream HTTP request.
 */
export async function* streamQuery(
  question: string,
  options: StreamQueryOptions
): AsyncGenerator<StreamQueryEvent> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const response = await fetch(`${API_URL}/api/v1/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      question,
      model: options.model,
      pdf_ids: options.pdfIds,
      session_id: options.chatId,
      request_id: options.requestId,
      provider: options.provider,
      mode: options.mode,
    }),
    signal: options.signal,
    cache: "no-store",
  });

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorDetail;
    } catch {
      // Keep statusText when the body is not JSON.
    }
    throw new Error(errorDetail);
  }

  if (!response.body) {
    throw new Error("Backend returned no stream body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.match(/\r?\n\r?\n/);
      while (boundary && boundary.index !== undefined) {
        const rawEvent = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const event = parseStreamEvent(rawEvent);
        if (event) {
          yield event;
        }
        boundary = buffer.match(/\r?\n\r?\n/);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim().length > 0) {
      const event = parseStreamEvent(buffer);
      if (event) {
        yield event;
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // Already closed.
    }
    reader.releaseLock();
  }
}

export async function ollamaChat(
  messages: Message[],
  model: string = "mistral:latest",
  pdfIds?: string[]
): Promise<QueryResponse> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  // Extract last user message
  const lastMessage = messages[messages.length - 1];
  const question = lastMessage.content;

  // Query backend with optional PDF filter
  const response = await fetch(`${API_URL}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      model,
      session_id: null,
      pdf_ids: pdfIds, // Filter to specific PDFs if provided
    }),
  });

  if (!response.ok) {
    // Try to get error details from response
    let errorDetail = response.statusText;
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorDetail;
    } catch {
      // If parsing fails, use statusText
    }

    throw new Error(errorDetail);
  }

  const data: QueryResponse = await response.json();

  return data;
}

export async function uploadPDF(file: File): Promise<any> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/v1/pdfs/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
}

export async function listPDFs(): Promise<any[]> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const response = await fetch(`${API_URL}/api/v1/pdfs`);

  if (!response.ok) {
    throw new Error(`Failed to list PDFs: ${response.statusText}`);
  }

  return response.json();
}

export async function deletePDF(pdfId: string): Promise<void> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const response = await fetch(`${API_URL}/api/v1/pdfs/${pdfId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(`Delete failed: ${response.statusText}`);
  }
}
