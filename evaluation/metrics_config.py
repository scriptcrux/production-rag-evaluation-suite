"""DeepEval Metric Suites and Configuration for Financial RAG System.

Evaluation-Driven Development (EDD) Rationale:
----------------------------------------------
Financial analysis and SEC regulatory compliance represent zero-tolerance domains for LLM hallucination.
This module defines the quantitative quality gates enforced on every pull request and release build.

Metric Selection & Threshold Justifications:
---------------------------------------------
1. FaithfulnessMetric (Threshold: 0.85):
   - RATIONALE: Evaluates whether factual claims in actual_output are strictly grounded in retrieval_context.
   - THRESHOLD: 0.85. In financial auditing under SEC Rule 10b-5, misstating revenues, margins, or CapEx figures
     incurs severe liability. A minimum threshold of 0.85 guarantees that >85% of claims have direct textual
     support in audited 10-K filings.

2. AnswerRelevancyMetric (Threshold: 0.70):
   - RATIONALE: Evaluates whether the agent directly answered the specific user question without conversational drift.
   - THRESHOLD: 0.70. Financial answers legitimately include contextual preamble and mandatory regulatory
     non-investment-advice disclaimers. Setting the threshold at 0.70 accommodates necessary legal scaffolding
     while rejecting rambling or evasive responses.

3. HallucinationMetric (Threshold: 0.15):
   - RATIONALE: Directly measures the proportion of conflicting statements between actual output and context.
   - THRESHOLD: 0.15. In DeepEval, lower hallucination score indicates higher accuracy. A maximum tolerance of 0.15
     ensures that no more than 15% of output statements lack direct verification.

4. ContextualPrecisionMetric (Threshold: 0.70):
   - RATIONALE: Assesses whether the hybrid retrieval and re-ranking pipeline ranks the most relevant 10-K
     sections (e.g. Item 7 MD&A or Item 8 Financials) at the top of the context window.
   - THRESHOLD: 0.70. High precision minimizes LLM needle-in-a-haystack distractions and context window costs.

5. ContextualRecallMetric (Threshold: 0.80):
   - RATIONALE: Measures whether all essential facts required to construct the ground-truth golden output were
     successfully retrieved by the RAG pipeline.
   - THRESHOLD: 0.80. A recall failure causes the downstream synthesizer to miss crucial numbers or footnotes.

6. GEval - Financial Rigor & Compliance Metric (Threshold: 0.80):
   - RATIONALE: Custom LLM-as-a-Judge metric assessing SEC US-GAAP terminology, exact numerical citations,
     and the presence of the mandatory non-investment advisory disclaimer.
   - THRESHOLD: 0.80. Enforces enterprise governance standards beyond generic language metrics.

Evaluation Model Selection Rationale:
--------------------------------------
- CI/CD & Automated Pipeline: 'gpt-4o-mini'
  Offers superior structured JSON reasoning and evaluation accuracy at ~10% of the cost of legacy GPT-4,
  permitting frequent evaluation of multi-hundred test case suites within tight CI budgets.
"""

from typing import Optional
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCaseParams

from src.config import settings
from evaluation.judge import FinSightEvalJudge


def get_evaluation_model():
    """Returns the evaluation model: OpenAI model if API key is provided, or FinSightEvalJudge."""
    api_key = settings.openai_api_key
    if api_key and api_key not in ("mock", "test", "") and not api_key.startswith("your-"):
        return settings.deepeval_eval_model
    return FinSightEvalJudge()


def get_faithfulness_metric(threshold: Optional[float] = None) -> FaithfulnessMetric:
    """Returns FaithfulnessMetric configured for financial verification."""
    return FaithfulnessMetric(
        threshold=threshold or settings.threshold_faithfulness,
        model=get_evaluation_model(),
        include_reason=True,
    )


def get_answer_relevancy_metric(threshold: Optional[float] = None) -> AnswerRelevancyMetric:
    """Returns AnswerRelevancyMetric configured for analytical inquiries."""
    return AnswerRelevancyMetric(
        threshold=threshold or settings.threshold_answer_relevancy,
        model=get_evaluation_model(),
        include_reason=True,
    )


def get_hallucination_metric(threshold: Optional[float] = None) -> HallucinationMetric:
    """Returns HallucinationMetric configured for financial filings."""
    return HallucinationMetric(
        threshold=threshold or settings.threshold_hallucination,
        model=get_evaluation_model(),
        include_reason=True,
    )


def get_contextual_precision_metric(threshold: Optional[float] = None) -> ContextualPrecisionMetric:
    """Returns ContextualPrecisionMetric evaluating top-k retrieval signal-to-noise."""
    return ContextualPrecisionMetric(
        threshold=threshold or settings.threshold_contextual_precision,
        model=get_evaluation_model(),
        include_reason=True,
    )


def get_contextual_recall_metric(threshold: Optional[float] = None) -> ContextualRecallMetric:
    """Returns ContextualRecallMetric verifying complete capture of ground-truth facts."""
    return ContextualRecallMetric(
        threshold=threshold or settings.threshold_contextual_recall,
        model=get_evaluation_model(),
        include_reason=True,
    )


def get_contextual_relevancy_metric(threshold: float = 0.70) -> ContextualRelevancyMetric:
    """Returns ContextualRelevancyMetric evaluating context conciseness and signal."""
    return ContextualRelevancyMetric(
        threshold=threshold,
        model=get_evaluation_model(),
        include_reason=True,
    )


def get_financial_rigor_geval(threshold: float = 0.80) -> GEval:
    """Returns custom G-Eval metric for SEC US-GAAP compliance, citations, and disclaimer."""
    return GEval(
        name="Financial Rigor & SEC Compliance",
        criteria=(
            "Evaluate whether the actual output demonstrates high financial and regulatory rigor: "
            "1. All quantitative numbers have appropriate units and periods (e.g. FY2024, billions). "
            "2. Includes explicit citations to Form 10-K sections. "
            "3. Contains mandatory SEC non-investment-advice regulatory disclaimer. "
            "4. Refuses to make financial return promises or stock price guarantees."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=threshold,
        model=get_evaluation_model(),
    )

