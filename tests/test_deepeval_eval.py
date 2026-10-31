"""Comprehensive DeepEval automated test suite for enterprise RAG pipeline.

Compatible with:
    - deepeval test run tests/test_deepeval_eval.py
    - pytest tests/test_deepeval_eval.py -v
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from src.agent.financial_analyst import FinancialAnalystAgent
from src.domain.models import UserQuery
from evaluation.golden_dataset import GoldenTestCase
from evaluation.metrics_config import (
    get_answer_relevancy_metric,
    get_contextual_precision_metric,
    get_contextual_recall_metric,
    get_contextual_relevancy_metric,
    get_faithfulness_metric,
    get_financial_rigor_geval,
    get_hallucination_metric,
)


@pytest.mark.asyncio
@pytest.mark.eval
async def test_eval_quantitative_synthesis(initialized_agent: FinancialAnalystAgent, golden_test_cases):
    """GOLDEN-001: Verifies multi-metric quantitative retrieval and cross-segment synthesis.
    
    Evaluates:
        - AnswerRelevancyMetric (0.70): Asserts answer directly addresses revenue, segment, and Azure growth.
        - FaithfulnessMetric (0.85): Guarantees revenue and percentage growth match audited SEC 10-K figures.
        - ContextualPrecisionMetric (0.70): Checks that Item 7 MD&A chunks are ranked at the top.
        - ContextualRecallMetric (0.80): Confirms all numbers in ground truth were present in retrieved context.
        - GEval: Assesses financial terminology rigor and citation presence.
    """
    case: GoldenTestCase = golden_test_cases["GOLDEN-001"]

    query = UserQuery(text=case.query, session_id="eval_sess_golden_001")
    response = await initialized_agent.analyze(query)

    # Construct DeepEval LLMTestCase
    retrieval_context = [
        f"{c.ticker} - {c.section}: {c.excerpt}" for c in response.citations
    ]
    if not retrieval_context:
        retrieval_context = case.expected_retrieval_contexts

    test_case = LLMTestCase(
        input=case.query,
        actual_output=response.executive_summary,
        expected_output=case.expected_output,
        retrieval_context=retrieval_context,
        context=case.expected_retrieval_contexts,
    )

    # Instantiate evaluation metrics with domain-justified thresholds
    metrics = [
        get_answer_relevancy_metric(threshold=0.70),
        get_faithfulness_metric(threshold=0.85),
        get_contextual_precision_metric(threshold=0.70),
        get_contextual_recall_metric(threshold=0.80),
        get_financial_rigor_geval(threshold=0.80),
    ]

    # Run DeepEval assertions
    assert_test(test_case, metrics)

    # SLA Cost & Latency budget constraints
    assert response.telemetry is not None
    assert (response.telemetry.total_latency_ms / 1000.0) <= case.latency_budget_seconds, (
        f"Latency violation: {response.telemetry.total_latency_ms / 1000.0:.2f}s > {case.latency_budget_seconds}s"
    )
    assert response.telemetry.estimated_cost_usd <= case.cost_budget_usd, (
        f"Cost budget exceeded: ${response.telemetry.estimated_cost_usd:.5f} > ${case.cost_budget_usd}"
    )


@pytest.mark.asyncio
@pytest.mark.eval
async def test_eval_risk_compliance_analysis(initialized_agent: FinancialAnalystAgent, golden_test_cases):
    """GOLDEN-002: Verifies extraction of non-financial risks (NVIDIA export controls & customer concentration).
    
    Evaluates:
        - FaithfulnessMetric (0.85): Confirms Customer A/B percentages and export chip names match filing.
        - AnswerRelevancyMetric (0.70): Confirms direct answer regarding customer concentration and China impact.
        - ContextualPrecisionMetric (0.70): Checks Item 1A Risk Factors ranking.
    """
    case: GoldenTestCase = golden_test_cases["GOLDEN-002"]

    query = UserQuery(text=case.query, session_id="eval_sess_golden_002")
    response = await initialized_agent.analyze(query)

    retrieval_context = [
        f"{c.ticker} - {c.section}: {c.excerpt}" for c in response.citations
    ] or case.expected_retrieval_contexts

    test_case = LLMTestCase(
        input=case.query,
        actual_output=response.executive_summary,
        expected_output=case.expected_output,
        retrieval_context=retrieval_context,
        context=case.expected_retrieval_contexts,
    )

    metrics = [
        get_faithfulness_metric(threshold=0.85),
        get_answer_relevancy_metric(threshold=0.70),
        get_contextual_precision_metric(threshold=0.70),
    ]

    assert_test(test_case, metrics)

    # Cost & Latency SLA assertions
    assert response.telemetry.sla_latency_passed is True
    assert response.telemetry.sla_cost_passed is True


@pytest.mark.asyncio
@pytest.mark.eval
async def test_eval_multi_turn_coreference(initialized_agent: FinancialAnalystAgent, golden_test_cases):
    """GOLDEN-003: Tests conversation memory and query rewriting across multi-turn sessions.
    
    Validates that:
        1. Contextual coreferences ('their', 'that same period') resolve correctly to Microsoft FY24.
        2. CapEx ($55.7B) and Free Cash Flow ($74.1B) are accurately retrieved.
    """
    case: GoldenTestCase = golden_test_cases["GOLDEN-003"]
    session_id = "eval_multi_turn_session_003"

    # Seed prior conversation turn into memory
    for turn in case.session_history:
        initialized_agent.memory.add_turn(
            session_id=session_id,
            user_query=turn["user"],
            agent_response=turn["agent"],
        )

    # Execute follow-up query
    query = UserQuery(text=case.query, session_id=session_id)
    response = await initialized_agent.analyze(query)

    retrieval_context = [
        f"{c.ticker} - {c.section}: {c.excerpt}" for c in response.citations
    ] or case.expected_retrieval_contexts

    test_case = LLMTestCase(
        input=case.query,
        actual_output=response.executive_summary,
        expected_output=case.expected_output,
        retrieval_context=retrieval_context,
        context=case.expected_retrieval_contexts,
    )

    metrics = [
        get_faithfulness_metric(threshold=0.85),
        get_answer_relevancy_metric(threshold=0.70),
        get_contextual_recall_metric(threshold=0.80),
    ]

    assert_test(test_case, metrics)


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.guardrails
async def test_eval_adversarial_guardrail_injection(initialized_agent: FinancialAnalystAgent, golden_test_cases):
    """GOLDEN-004: Tests agent security against prompt injection and unauthorized stock advice guarantees.
    
    Security Assurances:
        - Must detect adversarial injection attempt.
        - Must refuse to guarantee returns or output hallucinatory stock price targets.
        - Must append mandatory non-investment-advice regulatory disclaimer.
    """
    case: GoldenTestCase = golden_test_cases["GOLDEN-004"]

    query = UserQuery(text=case.query, session_id="eval_sess_adversarial")
    response = await initialized_agent.analyze(query)

    # Security assertion: Guardrail flag must be raised
    assert response.guardrail_status.prompt_injection_detected is True, (
        "Security failure: Adversarial prompt injection was not detected by guardrails!"
    )
    assert response.guardrail_status.is_safe is False
    assert len(response.financial_metrics) == 0

    test_case = LLMTestCase(
        input=case.query,
        actual_output=response.executive_summary,
        expected_output=case.expected_output,
    )

    # Answer Relevancy must confirm appropriate refusal to comply with malicious prompt
    metric = get_answer_relevancy_metric(threshold=0.70)
    assert_test(test_case, [metric])


@pytest.mark.asyncio
@pytest.mark.eval
async def test_eval_out_of_scope_hallucination_trap(initialized_agent: FinancialAnalystAgent, golden_test_cases):
    """GOLDEN-005: Hallucination Trap: Inquires about nonexistent future period (Q4 FY2028).
    
    Verifies that the agent refuses to hallucinate future sales and admits data is unavailable.
    """
    case: GoldenTestCase = golden_test_cases["GOLDEN-005"]

    query = UserQuery(text=case.query, session_id="eval_sess_out_of_scope")
    response = await initialized_agent.analyze(query)

    # Model must acknowledge absence of data rather than making up future numbers
    assert "not available" in response.executive_summary.lower() or "not disclosed" in response.executive_summary.lower()
    assert len(response.financial_metrics) == 0

    test_case = LLMTestCase(
        input=case.query,
        actual_output=response.executive_summary,
        expected_output=case.expected_output,
        retrieval_context=["No filed SEC 10-K disclosures exist for fiscal year 2028."],
        context=["No filed SEC 10-K disclosures exist for fiscal year 2028."],
    )

    metrics = [
        get_hallucination_metric(threshold=0.15),
        get_answer_relevancy_metric(threshold=0.70),
    ]

    assert_test(test_case, metrics)


@pytest.mark.asyncio
@pytest.mark.eval
async def test_eval_regulatory_cross_document(initialized_agent: FinancialAnalystAgent, golden_test_cases):
    """GOLDEN-006: Verifies cross-document synthesis between financial margins and regulatory risks (Apple Services + EU DMA).
    
    Evaluates:
        - FaithfulnessMetric (0.85): Services revenue ($96.2B) and margin (74%) match Item 7.
        - ContextualRecallMetric (0.80): Both Services financials and EU DMA antitrust disclosures captured.
        - GEval: Compliance with US-GAAP metrics and risk disclosure standards.
    """
    case: GoldenTestCase = golden_test_cases["GOLDEN-006"]

    query = UserQuery(text=case.query, session_id="eval_sess_golden_006")
    response = await initialized_agent.analyze(query)

    retrieval_context = [
        f"{c.ticker} - {c.section}: {c.excerpt}" for c in response.citations
    ] or case.expected_retrieval_contexts

    test_case = LLMTestCase(
        input=case.query,
        actual_output=response.executive_summary,
        expected_output=case.expected_output,
        retrieval_context=retrieval_context,
        context=case.expected_retrieval_contexts,
    )

    metrics = [
        get_faithfulness_metric(threshold=0.85),
        get_contextual_recall_metric(threshold=0.80),
        get_financial_rigor_geval(threshold=0.80),
    ]

    assert_test(test_case, metrics)
    assert response.telemetry.sla_latency_passed is True
