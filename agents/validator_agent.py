"""
Validator / Critic Agent – checks each field for:
  - Evidence support (is there a citation?)
  - Format validity (dates, numbers)
  - Confidence threshold
  - Hallucination detection

Sends failed fields back to the Orchestrator for re-extraction.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from loguru import logger

from models import ExtractedField, SharedState, ValidationResult


# ─── Validators ──────────────────────────────────────────────────────────────

def _validate_date(value: str) -> tuple[bool, str]:
    """Check if a date field has a recognizable date format."""
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",            # YYYY-MM-DD
        r"\d{1,2}/\d{1,2}/\d{2,4}",      # MM/DD/YYYY
        r"\w+ \d{1,2},? \d{4}",           # Month DD, YYYY
        r"\d{1,2} \w+ \d{4}",             # DD Month YYYY
    ]
    for pat in date_patterns:
        if re.search(pat, str(value), re.IGNORECASE):
            return True, "Valid date format"
    return False, f"Unrecognized date format: {value}"


def _validate_bid_number(value: str) -> tuple[bool, str]:
    """Check if bid number looks like a valid ID."""
    if re.search(r"[A-Z0-9][\-\.][A-Z0-9]", str(value), re.IGNORECASE):
        return True, "Valid bid number format"
    if len(str(value)) >= 3:
        return True, "Bid number present"
    return False, "Bid number too short or empty"


FIELD_VALIDATORS = {
    "Due Date": _validate_date,
    "Delivery Date": _validate_date,
    "Pre Bid Meeting": lambda v: (True, "ok") if v else (False, "Empty"),
    "Bid Number": _validate_bid_number,
}

CONFIDENCE_THRESHOLD = 0.4
REQUIRED_FIELDS = ["Bid Number", "Title", "Due Date", "company_name"]


def validate_field(
    field_name: str,
    field: ExtractedField,
) -> ValidationResult:
    """Validate a single extracted field."""

    # 1. Null check for required fields
    if field.value is None:
        is_required = field_name in REQUIRED_FIELDS
        return ValidationResult(
            field_name=field_name,
            passed=not is_required,
            reason="Not found in documents" if not is_required else f"Required field '{field_name}' is missing",
            confidence=field.confidence,
        )

    # 2. Hallucination check – no sources = suspicious
    if not field.sources and field.value is not None:
        return ValidationResult(
            field_name=field_name,
            passed=False,
            reason="Value present but no source citation found",
            confidence=0.1,
        )

    # 3. Confidence threshold
    if field.confidence < CONFIDENCE_THRESHOLD and field.value is not None:
        return ValidationResult(
            field_name=field_name,
            passed=False,
            reason=f"Low confidence ({field.confidence:.2f} < {CONFIDENCE_THRESHOLD})",
            confidence=field.confidence,
        )

    # 4. Format validators
    if field_name in FIELD_VALIDATORS:
        valid, reason = FIELD_VALIDATORS[field_name](field.value)
        if not valid:
            return ValidationResult(
                field_name=field_name,
                passed=False,
                reason=reason,
                confidence=field.confidence,
            )

    return ValidationResult(
        field_name=field_name,
        passed=True,
        reason="All checks passed",
        confidence=field.confidence,
    )


def run_validator_agent(state: SharedState) -> SharedState:
    """
    Validate all draft fields. Mark failed ones for re-extraction.
    """
    start = time.time()
    log_entry: Dict[str, Any] = {
        "agent": "ValidatorAgent",
        "input": {"fields_to_validate": len(state.draft_fields)},
        "start_time": start,
    }

    try:
        logger.info(f"[ValidatorAgent] Validating {len(state.draft_fields)} fields")

        validation_results: List[ValidationResult] = []
        failed_fields: List[str] = []

        for field_name, field_value in state.draft_fields.items():
            result = validate_field(field_name, field_value)
            validation_results.append(result)

            if not result.passed:
                retry_count = state.retry_counts.get(field_name, 0)
                if retry_count < 1:  # max 1 retry attempt per field
                    state.retry_counts[field_name] = retry_count + 1
                    failed_fields.append(field_name)
                    logger.warning(
                        f"  ✗ {field_name}: {result.reason} "
                        f"(scheduling retry 1/1)"
                    )
                else:
                    logger.warning(
                        f"  ✗ {field_name}: retry attempted, accepting as not found"
                    )
            else:
                logger.debug(f"  ✓ {field_name} (conf={result.confidence:.2f})")

        state.validation_results = validation_results
        state.failed_fields = failed_fields
        state.iteration += 1

        passed = sum(1 for r in validation_results if r.passed)
        not_found = sum(1 for r in validation_results
                        if not r.passed and state.draft_fields.get(r.field_name, ExtractedField()).value is None)

        state.current_step = "validation_done"
        log_entry.update({
            "output": {
                "passed": passed,
                "failed": len(failed_fields),
                "not_found": not_found,
            },
            "latency": time.time() - start,
            "status": "success",
        })
        logger.info(
            f"[ValidatorAgent] {passed}/{len(validation_results)} passed, "
            f"{len(failed_fields)} to retry"
        )

    except Exception as e:
        error_msg = f"ValidatorAgent error: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.current_step = "validation_failed"
        log_entry.update({"status": "error", "error": str(e), "latency": time.time() - start})

    state.agent_logs.append(log_entry)
    return state
