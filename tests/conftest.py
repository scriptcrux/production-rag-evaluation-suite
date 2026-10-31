"""Pytest fixtures and configuration for DeepEval testing suite."""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles for DeepEval emojis (🎯, ✨)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name, None)
        _reconfig = getattr(_stream, "reconfigure", None)
        if callable(_reconfig):
            try:
                _reconfig(encoding="utf-8")
            except Exception:
                pass

    # Monkey-patch Rich's legacy Windows console writer to prevent charmap UnicodeEncodeErrors
    try:
        import rich._win32_console
        _orig_write_text = rich._win32_console.LegacyWindowsTerm.write_text

        def _safe_write_text(self, text: str):
            try:
                _orig_write_text(self, text)
            except UnicodeEncodeError:
                safe_text = text.encode("ascii", errors="replace").decode("ascii")
                _orig_write_text(self, safe_text)

        rich._win32_console.LegacyWindowsTerm.write_text = _safe_write_text
    except Exception:
        pass

import pytest
import pytest_asyncio
from typing import Dict

from src.agent.financial_analyst import FinancialAnalystAgent
from evaluation.golden_dataset import GOLDEN_DATASET, GoldenTestCase


@pytest_asyncio.fixture(scope="session")
async def initialized_agent():
    """Provides a pre-indexed and warmed-up FinancialAnalystAgent instance."""
    agent = FinancialAnalystAgent()
    await agent.initialize()
    return agent


@pytest.fixture(scope="session")
def golden_test_cases() -> Dict[str, GoldenTestCase]:
    """Provides lookup dictionary of golden benchmark cases keyed by test ID."""
    return {case.id: case for case in GOLDEN_DATASET}
