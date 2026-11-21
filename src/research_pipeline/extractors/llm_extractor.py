"""LLM Extractor Client with Strict Budget Tracking and Mock Capabilities.

Supports:
- Gemini 2.5 Flash / Gemini 2.5 Pro via google-genai
- Pre-call Reservation and Post-call Settlement via BudgetTracker
- Full parameter sanitization via Allowlist (R12 / Section 12)
- Mock transport for deterministic tests
"""

import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel

from research_pipeline.extractors.budget_tracker import BudgetTracker
from research_pipeline.extractors.raw_artifact import RawExtractionArtifact
from research_pipeline.extractors.models import (
    canonicalize_text,
    compute_canonical_hash,
)

ALLOWED_GENERATION_PARAM_KEYS = {
    "model_checkpoint",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "max_output_tokens",
    "response_mime_type",
}


def sanitize_generation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Strict Allowlist filter for generation parameters (R12). Discards any unlisted keys."""
    return {k: v for k, v in params.items() if k in ALLOWED_GENERATION_PARAM_KEYS}


# Rule-based C0 Keyword extraction baseline
C0_KEYWORD_PATTERNS = {
    "models": [
        ("Minority Game", "agent_based", "agent-based financial market model"),
        ("Hawkes Process", "stochastic_process", "self-exciting point process"),
        ("Transformer", "deep_learning", "attention-based sequential model"),
        ("Agent-Based Model", "agent_based", "computational simulation with autonomous agents"),
        ("GARCH", "stochastic_process", "autoregressive conditional heteroskedasticity"),
    ],
    "methods": [
        ("Neural Posterior Estimation", "parameter_inference", "simulation-based inference technique"),
        ("NPE", "parameter_inference", "neural posterior estimation"),
        ("Maximum Likelihood Estimation", "parameter_inference", "parametric optimization"),
        ("Reinforcement Learning", "optimization", "policy optimization framework"),
        ("Dickey-Fuller", "hypothesis_testing", "unit root test"),
    ],
    "phenomena": [
        ("Volatility Clustering", "clustering of high volatility periods"),
        ("Fat Tails", "heavy-tailed return distributions"),
        ("Order Flow Imbalance", "asymmetry in limit order book arrivals"),
        ("Price Impact", "shift in mid-price caused by market orders"),
    ],
}


class MockLLMTransport:
    """Mock LLM transport providing deterministic responses for tests."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        condition_id: str,
    ) -> Tuple[str, int, Dict[str, int]]:
        """Simulate LLM generation returning (raw_text, http_status, token_usage)."""
        self.call_history.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "condition_id": condition_id,
        })
        
        # Check custom mock response
        for k, v in self.responses.items():
            if k in prompt:
                return v, 200, {"prompt_tokens": 500, "completion_tokens": 250, "total_tokens": 750}

        # Default mocked responses per condition
        if condition_id == "C1":
            raw_text = (
                "### Phenomenon\n"
                "- Volatility Clustering: periods of high volatility cluster together\n\n"
                "### Model\n"
                "- Minority Game: an agent-based simulation of market dynamics\n\n"
                "### Method\n"
                "- Neural Posterior Estimation: simulation-based parameter inference\n\n"
                "### Capability\n"
                "- Price Simulation: generate price series from behavioral parameters\n"
            )
            return raw_text, 200, {"prompt_tokens": 400, "completion_tokens": 120, "total_tokens": 520}

        # Default structured JSON for C2, C3, C4
        default_payload = {
            "phenomenas": [{
                "local_id": "phe_01",
                "name": "Volatility Clustering",
                "description": "Periods of high volatility cluster together in financial series.",
                "evidence": {
                    "target_field_name": "abstract_text",
                    "target_field_hash": "dummy_hash",
                    "verbatim_quote": "volatility clustering",
                    "start_char": 0,
                    "end_char": 21,
                }
            }],
            "models": [{
                "local_id": "mod_01",
                "name": "Minority Game",
                "category": "agent_based",
                "description": "Agent-based computational model.",
                "evidence": {
                    "target_field_name": "abstract_text",
                    "target_field_hash": "dummy_hash",
                    "verbatim_quote": "Minority Game",
                    "start_char": 25,
                    "end_char": 38,
                }
            }],
            "methods": [{
                "local_id": "met_01",
                "name": "Neural Posterior Estimation",
                "category": "parameter_inference",
                "description": "Inference algorithm for intractable likelihoods.",
                "evidence": {
                    "target_field_name": "abstract_text",
                    "target_field_hash": "dummy_hash",
                    "verbatim_quote": "Neural Posterior Estimation",
                    "start_char": 40,
                    "end_char": 67,
                }
            }],
            "capabilities": [{
                "local_id": "cap_01",
                "name": "Price Series Simulation",
                "operation": "generate",
                "inputs": [{
                    "data_type_l1": "parameter_vector",
                    "data_type_l2": "agent_behavior_parameters",
                    "description": "Agent memory and choice parameters",
                    "time_resolution": "static",
                    "dimensionality": "10D",
                    "is_unspecified": False,
                    "evidence": {
                        "target_field_name": "abstract_text",
                        "target_field_hash": "dummy_hash",
                        "verbatim_quote": "agent parameters",
                        "start_char": 70,
                        "end_char": 86,
                    }
                }],
                "outputs": [{
                    "data_type_l1": "timeseries_1d",
                    "data_type_l2": "price_timeseries",
                    "description": "Synthetic price series",
                    "time_resolution": "tick",
                    "dimensionality": "1D",
                    "is_unspecified": False,
                    "evidence": {
                        "target_field_name": "abstract_text",
                        "target_field_hash": "dummy_hash",
                        "verbatim_quote": "synthetic prices",
                        "start_char": 90,
                        "end_char": 106,
                    }
                }],
                "preconditions": ["Requires parameter configuration"],
                "limitations": ["Sensitive to initialization"],
                "evidence": {
                    "target_field_name": "abstract_text",
                    "target_field_hash": "dummy_hash",
                    "verbatim_quote": "simulate synthetic prices",
                    "start_char": 80,
                    "end_char": 105,
                }
            }],
            "assumptions": [],
            "limitations": [],
            "local_relations": [{
                "relation_id": "rel_01",
                "src_local_id": "mod_01",
                "dst_local_id": "phe_01",
                "relation_type": "EXPLAINS",
                "evidence": {
                    "target_field_name": "abstract_text",
                    "target_field_hash": "dummy_hash",
                    "verbatim_quote": "Minority Game reproduces volatility clustering",
                    "start_char": 25,
                    "end_char": 71,
                }
            }],
        }
        return json.dumps(default_payload), 200, {"prompt_tokens": 600, "completion_tokens": 300, "total_tokens": 900}


class ExtractionService:
    """Extraction orchestrator managing C0〜C4 runs with budget protection."""

    def __init__(
        self,
        budget_tracker: BudgetTracker,
        model_checkpoint: str = "gemini-2.5-flash",
        transport: Optional[Any] = None,
    ):
        self.budget_tracker = budget_tracker
        self.model_checkpoint = model_checkpoint
        self.transport = transport or MockLLMTransport()

    def run_c0_rule_based(
        self,
        source_id: str,
        version_id: str,
        canonical_fields: Dict[str, str],
        extraction_run_id: str,
    ) -> RawExtractionArtifact:
        """Execute C0 rule-based baseline extraction without LLM API call."""
        text = " ".join(canonical_fields.values())
        parsed_candidates: Dict[str, List[Dict[str, Any]]] = {
            "phenomenons": [],
            "models": [],
            "methods": [],
            "capabilities": [],
            "assumptions": [],
            "limitations": [],
        }

        # Match keywords
        for name, cat, desc in C0_KEYWORD_PATTERNS["models"]:
            if name.lower() in text.lower():
                parsed_candidates["models"].append({
                    "local_id": f"mod_{len(parsed_candidates['models'])+1:02d}",
                    "name": name,
                    "category": cat,
                    "description": desc,
                })

        for name, cat, desc in C0_KEYWORD_PATTERNS["methods"]:
            if name.lower() in text.lower():
                parsed_candidates["methods"].append({
                    "local_id": f"met_{len(parsed_candidates['methods'])+1:02d}",
                    "name": name,
                    "category": cat,
                    "description": desc,
                })

        for name, desc in C0_KEYWORD_PATTERNS["phenomena"]:
            if name.lower() in text.lower():
                parsed_candidates["phenomenons"].append({
                    "local_id": f"phe_{len(parsed_candidates['phenomenons'])+1:02d}",
                    "name": name,
                    "description": desc,
                })

        target_field_hashes = {k: compute_canonical_hash(v) for k, v in canonical_fields.items()}
        prompt_hash = hashlib.sha256(b"C0_RULE_BASED").hexdigest()

        return RawExtractionArtifact(
            extraction_run_id=extraction_run_id,
            source_id=source_id,
            version_id=version_id,
            condition_id="C0",
            raw_response_text=json.dumps(parsed_candidates),
            http_status=200,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            executed_at="2026-11-15T00:00:00Z",
            model_checkpoint="rule-based-v1",
            prompt_hash=prompt_hash,
            target_field_hashes=target_field_hashes,
            json_parse_status="success",
            parse_error=None,
            parsed_candidates=parsed_candidates,
        )

    def run_llm_condition(
        self,
        condition_id: str,
        source_id: str,
        version_id: str,
        canonical_fields: Dict[str, str],
        extraction_run_id: str,
        prompt_template: str,
        system_instruction: str,
        max_output_tokens: int = 4096,
    ) -> RawExtractionArtifact:
        """Execute LLM extraction (C1, C2, C3, C4) with Pre-call Reservation and Post-call Settlement."""
        target_field_hashes = {k: compute_canonical_hash(v) for k, v in canonical_fields.items()}
        content_text = "\n\n".join(f"[{k}]\n{v}" for k, v in canonical_fields.items())
        full_prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", content_text)
        prompt_hash = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()

        # 1. Conservative token counting (User Instruction 4)
        # Includes system instruction, prompt text, and schema
        estimated_input_tokens = len(full_prompt.split()) * 2 + len(system_instruction.split()) * 2

        reservation_id = f"{extraction_run_id}:{source_id}:{condition_id}"

        # 2. Phase 1: Pre-call Reservation
        self.budget_tracker.reserve(
            reservation_id=reservation_id,
            model_id=self.model_checkpoint,
            estimated_input_tokens=estimated_input_tokens,
            max_output_tokens=max_output_tokens,
            allowance_tokens=250,
        )

        try:
            # 3. Network dispatch
            raw_text, http_status, token_usage = self.transport.generate(
                prompt=full_prompt,
                system_instruction=system_instruction,
                condition_id=condition_id,
            )
        except Exception as e:
            # Release reservation on failure
            self.budget_tracker.release_reservation(reservation_id)
            raise e

        # 4. Phase 2: Post-call Settlement
        actual_in = token_usage.get("prompt_tokens", estimated_input_tokens)
        actual_out = token_usage.get("completion_tokens", 500)
        self.budget_tracker.settle(reservation_id, actual_in, actual_out)

        # 5. Syntax Parse (without schema model validation yet)
        parsed_candidates = None
        parse_error = None
        json_parse_status = "not_applicable"

        if condition_id == "C1":
            json_parse_status = "not_applicable"
        else:
            try:
                parsed_candidates = json.loads(raw_text)
                json_parse_status = "success"
            except Exception as e:
                json_parse_status = "invalid_json"
                parse_error = str(e)

        return RawExtractionArtifact(
            extraction_run_id=extraction_run_id,
            source_id=source_id,
            version_id=version_id,
            condition_id=condition_id,  # type: ignore
            raw_response_text=raw_text,
            http_status=http_status,
            token_usage=token_usage,
            executed_at="2026-11-15T00:00:00Z",
            model_checkpoint=self.model_checkpoint,
            prompt_hash=prompt_hash,
            target_field_hashes=target_field_hashes,
            json_parse_status=json_parse_status,
            parse_error=parse_error,
            parsed_candidates=parsed_candidates,
        )
