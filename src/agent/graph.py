"""LangGraph-powered stateful orchestration pipeline for FinSight Financial Analyst."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.core.logger import get_logger
from src.core.telemetry import ExecutionTelemetry, LatencyTracker, calculate_cost
from src.domain.models import (
    DocumentChunk,
    FinancialAnalystResponse,
    GuardrailStatus,
    UserQuery,
)

if TYPE_CHECKING:
    from src.agent.financial_analyst import FinancialAnalystAgent

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """Type-safe state container tracking the lifecycle of an agent execution in LangGraph."""
    trace_id: str
    query: UserQuery
    tracker: LatencyTracker
    sanitized_input: str
    guardrail_status: GuardrailStatus
    prompt_injection_detected: bool
    rewritten_query: str
    retrieved_chunks: List[DocumentChunk]
    reranked_chunks: List[DocumentChunk]
    response: Optional[FinancialAnalystResponse]
    telemetry: Optional[ExecutionTelemetry]
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str]


def build_financial_analyst_graph(agent: "FinancialAnalystAgent"):
    """Constructs and compiles the production LangGraph state graph for FinSight."""

    # 1. Node: Guardrail Validation & Input Sanitization
    async def node_guardrail_check(state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        sanitized, status = agent.guardrails.validate_input(query.text)
        return {
            "sanitized_input": sanitized,
            "guardrail_status": status,
            "prompt_injection_detected": status.prompt_injection_detected,
        }

    # 2. Conditional Routing after Guardrail Check
    def route_guardrails(state: AgentState) -> str:
        if state.get("prompt_injection_detected", False):
            return "node_safety_rejection"
        return "node_query_rewriting"

    # 3. Node: Safety Rejection (short-circuit on prompt injection / malicious intent)
    async def node_safety_rejection(state: AgentState) -> Dict[str, Any]:
        trace_id = state["trace_id"]
        tracker = state["tracker"]
        query = state["query"]
        guardrail_status = state["guardrail_status"]

        tracker.stop("total")
        total_lat = tracker.get("total") or 1.5

        telemetry = ExecutionTelemetry(
            trace_id=trace_id,
            total_latency_ms=total_lat,
            prompt_tokens=len(query.text.split()),
            completion_tokens=40,
            total_tokens=len(query.text.split()) + 40,
            estimated_cost_usd=0.00001,
            model_used=agent.model_name,
            sla_latency_passed=True,
            sla_cost_passed=True,
        )

        response = FinancialAnalystResponse(
            executive_summary=(
                "Security alert: The query was flagged for adversarial or policy-violating patterns. "
                "Request declined in accordance with SEC compliance and AI governance safety policies."
            ),
            financial_metrics=[],
            risk_assessment=["Adversarial prompt injection attempt intercepted by security guardrails."],
            citations=[],
            confidence_score=0.0,
            regulatory_disclaimer=agent.guardrails.get_disclaimer(),
            guardrail_status=guardrail_status,
            telemetry=telemetry,
        )
        return {"response": response, "telemetry": telemetry}

    # 4. Node: Contextual Conversation Memory & Query Rewriting
    async def node_query_rewriting(state: AgentState) -> Dict[str, Any]:
        tracker = state["tracker"]
        query = state["query"]
        sanitized = state.get("sanitized_input", query.text)

        tracker.start("memory_rewrite")
        rewritten = await agent.memory.rewrite_query(query.session_id, sanitized)
        tracker.stop("memory_rewrite")

        return {"rewritten_query": rewritten}

    # 5. Node: Hybrid Retrieval (Dense Vector + BM25 Sparse RRF)
    async def node_hybrid_retrieval(state: AgentState) -> Dict[str, Any]:
        tracker = state["tracker"]
        query = state["query"]
        rewritten = state["rewritten_query"]

        tracker.start("retrieval")
        candidates = await agent.retriever.retrieve(
            query=rewritten,
            top_k=settings.dense_top_k,
            ticker_filter=query.ticker_filter,
            fiscal_year_filter=query.fiscal_year_filter,
        )
        tracker.stop("retrieval")

        return {"retrieved_chunks": candidates}

    # 6. Node: Domain Authority Reranking
    async def node_reranking(state: AgentState) -> Dict[str, Any]:
        tracker = state["tracker"]
        rewritten = state["rewritten_query"]
        candidates = state.get("retrieved_chunks", [])

        tracker.start("rerank")
        reranked = agent.reranker.rerank(rewritten, candidates)
        tracker.stop("rerank")

        return {"reranked_chunks": reranked}

    # 7. Node: Synthesis (OpenAI with JSON Schema mode or Deterministic Evaluator fallback)
    async def node_synthesis(state: AgentState) -> Dict[str, Any]:
        tracker = state["tracker"]
        query = state["query"]
        rewritten = state["rewritten_query"]
        reranked = state.get("reranked_chunks", [])
        guardrail_status = state["guardrail_status"]

        tracker.start("llm")
        prompt_tokens_est = 250 + sum(len(c.content.split()) for c in reranked)
        completion_tokens_est = 180

        response_obj: FinancialAnalystResponse
        if agent._client:
            try:
                response_obj = await agent._generate_with_openai(
                    original_query=query.text,
                    rewritten_query=rewritten,
                    chunks=reranked,
                    guardrail_status=guardrail_status,
                )
            except Exception as e:
                logger.warning(
                    f"[{state['trace_id']}] OpenAI generation failed: {e}. Falling back to deterministic engine."
                )
                response_obj = agent._generate_deterministic(
                    original_query=query.text,
                    rewritten_query=rewritten,
                    chunks=reranked,
                    guardrail_status=guardrail_status,
                )
        else:
            response_obj = agent._generate_deterministic(
                original_query=query.text,
                rewritten_query=rewritten,
                chunks=reranked,
                guardrail_status=guardrail_status,
            )

        tracker.stop("llm")

        return {
            "response": response_obj,
            "prompt_tokens": prompt_tokens_est,
            "completion_tokens": completion_tokens_est,
        }

    # 8. Node: Telemetry, SLA Verification, and State Persistence
    async def node_telemetry_and_memory(state: AgentState) -> Dict[str, Any]:
        trace_id = state["trace_id"]
        tracker = state["tracker"]
        query = state["query"]
        response = state.get("response")

        total_latency = tracker.stop("total")
        prompt_tokens = state.get("prompt_tokens", 250)
        completion_tokens = state.get("completion_tokens", 180)

        cost_usd = calculate_cost(
            model=agent.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        sla_latency_passed = (total_latency / 1000.0) <= settings.max_latency_seconds
        sla_cost_passed = cost_usd <= settings.max_query_cost_usd

        telemetry = ExecutionTelemetry(
            trace_id=trace_id,
            retrieval_latency_ms=tracker.get("retrieval") or 0.0,
            rerank_latency_ms=tracker.get("rerank") or 0.0,
            llm_latency_ms=tracker.get("llm") or 0.0,
            total_latency_ms=total_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=cost_usd,
            model_used=agent.model_name,
            sla_latency_passed=sla_latency_passed,
            sla_cost_passed=sla_cost_passed,
        )

        if response:
            response.telemetry = telemetry
            # Save turn to conversation memory
            agent.memory.add_turn(
                session_id=query.session_id,
                user_query=query.text,
                agent_response=response.executive_summary,
                rewritten_query=state.get("rewritten_query"),
                citations=[f"{c.chunk_id}:{c.ticker}" for c in response.citations],
            )
            logger.info(
                f"[{trace_id}] Completed analysis in {total_latency:.1f}ms "
                f"(cost: ${cost_usd:.5f}, confidence: {response.confidence_score})"
            )

        return {"response": response, "telemetry": telemetry}

    # --- Assemble LangGraph Pipeline ---
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("node_guardrail_check", node_guardrail_check)
    workflow.add_node("node_safety_rejection", node_safety_rejection)
    workflow.add_node("node_query_rewriting", node_query_rewriting)
    workflow.add_node("node_hybrid_retrieval", node_hybrid_retrieval)
    workflow.add_node("node_reranking", node_reranking)
    workflow.add_node("node_synthesis", node_synthesis)
    workflow.add_node("node_telemetry_and_memory", node_telemetry_and_memory)

    # Add Edges & Conditional Branching
    workflow.add_edge(START, "node_guardrail_check")
    workflow.add_conditional_edges(
        "node_guardrail_check",
        route_guardrails,
        {
            "node_safety_rejection": "node_safety_rejection",
            "node_query_rewriting": "node_query_rewriting",
        },
    )
    workflow.add_edge("node_safety_rejection", END)
    workflow.add_edge("node_query_rewriting", "node_hybrid_retrieval")
    workflow.add_edge("node_hybrid_retrieval", "node_reranking")
    workflow.add_edge("node_reranking", "node_synthesis")
    workflow.add_edge("node_synthesis", "node_telemetry_and_memory")
    workflow.add_edge("node_telemetry_and_memory", END)

    return workflow.compile()
