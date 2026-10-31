"""Security, PII, and financial regulatory guardrails module."""

import re
from typing import List, Tuple
from src.core.exceptions import GuardrailViolationException
from src.core.logger import get_logger
from src.domain.models import GuardrailStatus

logger = get_logger(__name__)

MANDATORY_REGULATORY_DISCLAIMER = (
    "Disclaimer: This financial analysis is generated strictly for informational and research purposes "
    "based on public SEC filings. It does not constitute investment advice, financial planning, "
    "or an endorsement to buy or sell securities. Past financial performance is no guarantee of future results."
)

# Known prompt injection & jailbreak patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt",
    r"jailbreak",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"dan\s+mode",
    r"override\s+safety",
    r"bypass\s+restrictions",
    r"give\s+me\s+insider\s+trading\s+tips",
    r"guaranteed\s+(return|profit|100%)",
]

# PII regex patterns
PII_PATTERNS = {
    "CREDIT_CARD": r"\b(?:\d{4}[ -]?){3}\d{4}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
    "PHONE": r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
}


class SecurityGuardrails:
    """Enterprise guardrails enforcing security, PII hygiene, and SEC regulatory compliance."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def validate_input(self, user_text: str) -> Tuple[str, GuardrailStatus]:
        """Validates incoming user prompt against prompt injection and scrubs PII.
        
        Returns:
            Tuple of (sanitized_prompt, GuardrailStatus)
        """
        issues: List[str] = []
        is_safe = True
        injection_detected = False
        pii_redacted = False
        sanitized_text = user_text

        # 1. Prompt Injection & Adversarial Check
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, user_text, re.IGNORECASE):
                logger.warning(f"Prompt injection attempt detected: matched pattern '{pattern}'")
                issues.append(f"Adversarial prompt injection pattern detected: '{pattern}'")
                injection_detected = True
                is_safe = False
                if self.strict_mode:
                    raise GuardrailViolationException(
                        "Prompt Injection Detected",
                        {"pattern": pattern, "query": user_text}
                    )
                break

        # 2. PII Detection & Sanitization
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, sanitized_text)
            if matches:
                pii_redacted = True
                issues.append(f"Redacted sensitive PII ({pii_type})")
                sanitized_text = re.sub(pattern, f"[REDACTED_{pii_type}]", sanitized_text)

        status = GuardrailStatus(
            is_safe=is_safe,
            input_sanitized=pii_redacted,
            prompt_injection_detected=injection_detected,
            pii_redacted=pii_redacted,
            regulatory_disclaimer_appended=True,
            issues=issues,
        )
        return sanitized_text, status

    def enforce_output_compliance(self, text: str) -> str:
        """Ensures that mandatory regulatory disclaimers are present."""
        if MANDATORY_REGULATORY_DISCLAIMER not in text:
            return f"{text}\n\n---\n{MANDATORY_REGULATORY_DISCLAIMER}"
        return text

    def get_disclaimer(self) -> str:
        """Returns the standard SEC disclaimer text."""
        return MANDATORY_REGULATORY_DISCLAIMER
