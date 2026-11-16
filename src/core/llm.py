"""LLM configuration and setup."""
import logging
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

logger = logging.getLogger(__name__)


def build_chat_llm(
    model_name: str,
    num_ctx: int = 2048,
    num_predict: int = 512,
    keep_alive: str = "30m",
    temperature: float = 0.1,
) -> ChatOllama:
    """Build a ChatOllama client tuned for CPU-only, low-RAM machines.

    Explicit num_ctx/num_predict keep prompt-eval and generation bounded
    instead of relying on Ollama's potentially large defaults.
    """
    return ChatOllama(
        model=model_name,
        num_ctx=num_ctx,
        num_predict=num_predict,
        keep_alive=keep_alive,
        temperature=temperature,
    )


class LLMManager:
    """Manages LLM configuration and prompts."""
    
    def __init__(
        self,
        model_name: str = "phi4-mini:3.8b",
        num_ctx: int = 2048,
        num_predict: int = 512,
        keep_alive: str = "30m",
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.llm = build_chat_llm(
            model_name=model_name,
            num_ctx=num_ctx,
            num_predict=num_predict,
            keep_alive=keep_alive,
            temperature=temperature,
        )
        
    def get_query_prompt(self) -> PromptTemplate:
        """Get query generation prompt (single source of truth)."""
        return PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )

    # Models whose Ollama chat API supports think=True streaming
    THINKING_MODEL_SUBSTRINGS = ("qwen3", "deepseek-r1", "qwen", "deepseek")

    @classmethod
    def is_thinking_model(cls, model_name: str) -> bool:
        """Check if a model supports thinking/chain-of-thought mode."""
        lowered = (model_name or "").lower()
        return any(tm in lowered for tm in cls.THINKING_MODEL_SUBSTRINGS)

    def get_rag_prompt(self) -> ChatPromptTemplate:
        """Get concise RAG prompt with structured citations (no CoT preamble)."""
        template = """Answer the question based ONLY on the context below.
Each block starts with [Source: filename.pdf p.N #chunk].

Rules:
- Cite every claim inline as [Source: filename.pdf p.N].
- Always include at least one citation, even in short answers.
- Be concise: at most 5 sentences unless a list or detail is requested.
- Do not repeat the question or add a preamble.
- If the answer is not in the context, reply exactly: 'The provided documents do not contain this information.'

Context:
{context}

Question: {question}

Answer:"""
        return ChatPromptTemplate.from_template(template)

    def get_cot_system_message(self, formatted_context: str) -> str:
        """System message for thinking-capable models (qwen3/deepseek-r1)."""
        return f"""You are an expert AI assistant that uses chain-of-thought reasoning.

Answer the question based ONLY on the provided context from PDF documents.
Cite sources inline as [Source: filename.pdf p.N] for every claim.
If the answer is NOT in the context, say: 'The provided documents do not contain this information.'

CHAIN-OF-THOUGHT PROCESS:
1. **Read and understand** the question carefully
2. **Scan the context** to identify all relevant information
3. **Break down** the information by source document
4. **Analyze** how each piece relates to the question
5. **Synthesize** a comprehensive answer
6. **Cite sources** explicitly for every claim

Context from PDF documents:
{formatted_context}

Think through each step carefully, showing your reasoning process.""" 