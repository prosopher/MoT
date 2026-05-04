from __future__ import annotations

"""Backward-compatible entry point for the Doc2Dial multi-agent QA experiment.

The original test runner used SQuAD. The main multi-agent QA entry point now
uses Doc2Dial, so this file delegates to exp.multi_agents_qa to avoid keeping a
second SQuAD-based path around.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.multi_agents_qa import main


if __name__ == "__main__":
    main()
