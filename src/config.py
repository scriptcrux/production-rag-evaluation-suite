"""Configuration module for FinSight Enterprise RAG & Evaluation Framework.
Uses Pydantic Settings for type-safe environment variable management.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for application, RAG pipeline, and DeepEval."""

    # LLM Settings
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    max_output_tokens: int = Field(default=1500, alias="MAX_OUTPUT_TOKENS")

    # Retrieval & RAG Parameters
    dense_top_k: int = Field(default=6, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(default=6, alias="SPARSE_TOP_K")
    rerank_top_k: int = Field(default=3, alias="RERANK_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    hybrid_alpha: float = Field(default=0.5, alias="HYBRID_ALPHA")

    # Conversation Memory
    max_memory_turns: int = Field(default=5, alias="MAX_MEMORY_TURNS")
    session_timeout_minutes: int = Field(default=60, alias="SESSION_TIMEOUT_MINUTES")

    # DeepEval Evaluation Metric Thresholds
    confident_ai_api_key: Optional[str] = Field(default=None, alias="CONFIDENT_AI_API_KEY")
    deepeval_eval_model: str = Field(default="gpt-4o-mini", alias="DEEPEVAL_EVAL_MODEL")
    threshold_faithfulness: float = Field(default=0.85, alias="THRESHOLD_FAITHFULNESS")
    threshold_answer_relevancy: float = Field(default=0.70, alias="THRESHOLD_ANSWER_RELEVANCY")
    threshold_contextual_precision: float = Field(default=0.70, alias="THRESHOLD_CONTEXTUAL_PRECISION")
    threshold_contextual_recall: float = Field(default=0.80, alias="THRESHOLD_CONTEXTUAL_RECALL")
    threshold_hallucination: float = Field(default=0.15, alias="THRESHOLD_HALLUCINATION")

    # Performance & Cost Constraints (SLAs)
    max_latency_seconds: float = Field(default=4.0, alias="MAX_LATENCY_SECONDS")
    max_query_cost_usd: float = Field(default=0.03, alias="MAX_QUERY_COST_USD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global singleton settings instance
settings = Settings()

