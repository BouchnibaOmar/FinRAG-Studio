from finrag.generation.answers import MISTRAL_LARGE_4
from finrag.generation.model_catalogue import parse_catalogue


def synthetic_catalogue():
    return parse_catalogue(
        {
            "data": [
                {
                    "id": identifier,
                    "name": identifier,
                    "context_length": 65536,
                    "top_provider": {"max_completion_tokens": 32768},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "supported_parameters": [
                        "response_format",
                        "structured_outputs",
                        "max_tokens",
                        "reasoning",
                    ],
                    "pricing": {"prompt": "0.000001", "completion": "0.000002", "request": "0"},
                    "reasoning": {"supported_efforts": ["none", "high"], "mandatory": False},
                }
                for identifier in ("fictional/generator", "fictional/other", MISTRAL_LARGE_4)
            ]
        },
        "2026-10-15T00:00:00+00:00",
    )


def accept_uncertain_cost(ui):
    for checkbox in ui.checkbox:
        if checkbox.key.startswith("cost_accept_"):
            checkbox.check().run()
            return
    raise AssertionError("Expected explicit uncertain-cost acceptance")
