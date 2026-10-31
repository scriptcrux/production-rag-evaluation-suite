"""Pydantic domain models and schemas for FinSight Enterprise System."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.core.telemetry import ExecutionTelemetry


class DocumentChunk(BaseModel):
    """Represents an atomic financial document chunk with rich metadata."""
    id: str
    ticker: str
    company_name: str
    fiscal_year: str
    period_ended: str
    section: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class FinancialMetric(BaseModel):
    """Structured financial data point extracted from SEC filings."""
    metric_name: str = Field(description="Name of financial metric (e.g. Total Revenue, Free Cash Flow)")
    value: str = Field(description="Exact value as reported (e.g. '$245.1 billion')")
    unit: str = Field(default="USD", description="Currency or metric unit")
    period: str = Field(description="Reporting period (e.g. FY2024, Q4 FY24)")
    comparison_period: Optional[str] = Field(default=None, description="Prior comparison period (e.g. FY2023)")
    yoy_change: Optional[str] = Field(default=None, description="YoY percentage change (e.g. '+16%')")
    audit_status: str = Field(default="Audited", description="Audit classification: Audited, Unaudited, Reported")


class Citation(BaseModel):
    """Verifiable source citation mapped directly to a filing chunk."""
    chunk_id: str
    ticker: str
    section: str
    excerpt: str = Field(description="Direct text quote supporting the statement")


class GuardrailStatus(BaseModel):
    """Results of security, safety, and compliance checks."""
    is_safe: bool = True
    input_sanitized: bool = False
    prompt_injection_detected: bool = False
    pii_redacted: bool = False
    regulatory_disclaimer_appended: bool = True
    issues: List[str] = Field(default_factory=list)


class FinancialAnalystResponse(BaseModel):
    """Strictly validated structured output schema for Financial Analyst Agent."""
    executive_summary: str = Field(
        description="Concise, high-impact executive summary answering the user query directly"
    )
    financial_metrics: List[FinancialMetric] = Field(
        default_factory=list,
        description="Structured key financial numbers extracted strictly from retrieved context"
    )
    risk_assessment: List[str] = Field(
        default_factory=list,
        description="Identified regulatory, operational, or financial risks from SEC disclosures"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Direct citations pointing to the exact 10-K sections and chunk IDs"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence based on context completeness and retrieval relevance"
    )
    regulatory_disclaimer: str = Field(
        description="Mandatory SEC non-investment-advice compliance disclaimer"
    )
    guardrail_status: GuardrailStatus = Field(default_factory=GuardrailStatus)
    telemetry: Optional[ExecutionTelemetry] = None


class UserQuery(BaseModel):
    """User query payload with optional metadata filters and session context."""
    text: str
    session_id: str = "default_session"
    ticker_filter: Optional[str] = None
    fiscal_year_filter: Optional[str] = None


class ConversationTurn(BaseModel):
    """Single turn within multi-turn conversation memory."""
    turn_id: str
    user_query: str
    rewritten_query: Optional[str] = None
    agent_response: str
    citations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

