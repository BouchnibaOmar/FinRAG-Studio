from pathlib import Path
from unittest.mock import patch

import pytest

from finrag.config import FinragError
from finrag.evaluation.split_guard import authorize_split, load_dev_questions


@pytest.mark.parametrize(
    "allow,reason", [(False, None), (True, None), (False, "future reason"), (True, " ")]
)
def test_test_split_requires_both_explicit_controls(allow: bool, reason: str | None) -> None:
    with pytest.raises(FinragError, match="--allow-test"):
        authorize_split("test", allow, reason)


def test_phase2a_never_opens_test_even_with_future_controls() -> None:
    authorize_split("test", True, "Future independent evaluation protocol")
    with patch.object(Path, "read_bytes", side_effect=AssertionError("must not read")):
        with pytest.raises(FinragError, match="Phase 2A excludes"):
            load_dev_questions(Path("nonexistent"), "test", True, "Future protocol")


def test_dev_guard_does_not_require_test_authorization() -> None:
    authorize_split("dev")
