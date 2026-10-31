"""Multi-turn conversation memory with sliding window and query rewriting."""

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.config import settings
from src.core.logger import get_logger
from src.domain.models import ConversationTurn

logger = get_logger(__name__)


class ConversationMemory:
    """Manages multi-turn conversation state across sessions and rewrites contextual queries."""

    def __init__(self, max_turns: int = 5, api_key: Optional[str] = None):
        self.max_turns = max_turns
        self.api_key = api_key or settings.openai_api_key
        self.sessions: Dict[str, List[ConversationTurn]] = {}
        self._client: Optional[AsyncOpenAI] = None

        if self.api_key and self.api_key not in ("mock", "test", "") and not self.api_key.startswith("your-"):
            try:
                self._client = AsyncOpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        """Returns the conversation history for a given session."""
        return self.sessions.get(session_id, [])

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        agent_response: str,
        rewritten_query: Optional[str] = None,
        citations: Optional[List[str]] = None,
    ):
        """Adds a new conversation turn and maintains the sliding window size."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        turn = ConversationTurn(
            turn_id=f"turn_{uuid.uuid4().hex[:8]}",
            user_query=user_query,
            rewritten_query=rewritten_query,
            agent_response=agent_response,
            citations=citations or [],
            timestamp=datetime.now(timezone.utc),
        )
        self.sessions[session_id].append(turn)

        # Maintain sliding window
        if len(self.sessions[session_id]) > self.max_turns:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns :]

    async def rewrite_query(self, session_id: str, current_query: str) -> str:
        """Resolves coreferences and context switches using conversation history.
        
        Example:
            Turn 1: "What was Microsoft's Azure growth in FY2024?"
            Turn 2: "What about their capital expenditures in that period?"
            -> Rewritten: "Microsoft Corporation capital expenditures in fiscal year 2024"
        """
        history = self.get_history(session_id)
        if not history:
            return current_query

        # Check if query contains coreferences or follow-up markers
        follow_up_cues = ["their", "its", "that period", "same period", "that company", "those risks", "what about", "how about", "and for"]
        query_lower = current_query.lower()
        has_cue = any(cue in query_lower for cue in follow_up_cues)

        if not has_cue and len(current_query.split()) > 6:
            # Query is already self-contained
            return current_query

        # If LLM client is available, execute LLM query rewrite
        if self._client:
            try:
                history_text = "\n".join(
                    f"User: {t.user_query}\nAssistant: {t.agent_response[:120]}..."
                    for t in history[-2:]
                )
                prompt = (
                    "Given the recent conversation history, rewrite the user's latest follow-up question "
                    "into a self-contained search query that includes the exact company name, ticker, "
                    "and fiscal year if referenced implicitly. Return ONLY the rewritten query with no commentary.\n\n"
                    f"History:\n{history_text}\n\n"
                    f"Follow-up: {current_query}\n"
                    "Rewritten Query:"
                )
                messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
                response = await self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=60,
                )
                content = response.choices[0].message.content or ""
                rewritten = content.strip().strip('"')
                logger.info(f"LLM rewritten query: '{current_query}' -> '{rewritten}'")
                return rewritten
            except Exception as e:
                logger.warning(f"LLM query rewriting failed: {e}. Using rule-based fallback.")

        # Rule-based context resolution fallback
        last_turn = history[-1]
        prior_context = f"{last_turn.user_query} {last_turn.agent_response}"

        detected_ticker = None
        for ticker in ["MSFT", "NVDA", "AAPL"]:
            if ticker in prior_context.upper() or ticker.lower() in prior_context.lower():
                detected_ticker = ticker
                break
        
        detected_company = ""
        if detected_ticker == "MSFT" or "microsoft" in prior_context.lower():
            detected_company = "Microsoft Corporation (MSFT)"
        elif detected_ticker == "NVDA" or "nvidia" in prior_context.lower():
            detected_company = "NVIDIA Corporation (NVDA)"
        elif detected_ticker == "AAPL" or "apple" in prior_context.lower():
            detected_company = "Apple Inc. (AAPL)"

        detected_year = "FY2024"
        for yr in ["FY2024", "FY24", "FY2023", "FY23", "2024", "2023"]:
            if yr in prior_context:
                detected_year = yr
                break

        # Replace coreferences
        rewritten = current_query
        rewritten = re.sub(r"\b(their|its|that company\'?s?)\b", detected_company, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\b(that period|that same period|same period)\b", detected_year, rewritten, flags=re.IGNORECASE)

        if detected_company and detected_company not in rewritten:
            rewritten = f"{detected_company} {rewritten}"

        logger.info(f"Rule-based rewritten query: '{current_query}' -> '{rewritten}'")
        return rewritten

