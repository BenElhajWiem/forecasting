from __future__ import annotations

import argparse
import json

from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent
from agents.verification_agent import VerificationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="deepseek-chat",
                         help="Registry preset key, e.g. deepseek-chat / openai-mini / claude-api / gemini-flash-native")
    parser.add_argument("--n", type=int, default=3, help="Number of queries to test")
    parser.add_argument("--queries-file", default="experiments/queries/queries_verif_25.json")
    parser.add_argument("--max-revisions", type=int, default=2)
    args = parser.parse_args()

    with open(args.queries_file) as f:
        queries = json.load(f)

    registry = Registry()
    adapter = LLMClientAdapter(registry.presets[args.preset])
    vcfg = VerificationConfig(max_revisions=args.max_revisions)

    for q in queries[: args.n]:
        print("\n" + "=" * 90)
        print(f"QUERY {q['id']}: {q['text']}")
        print("=" * 90)
        result = orchestration_agent(
            user_query=q["text"],
            adapter=adapter,
            enable_verification=True,
            verification_cfg=vcfg,
        )
        print("\n--- Verification history ---")
        for step in result["verification_history"]:
            status = "CONSISTENT" if step["consistent"] else "FLAGGED"
            print(f"  attempt {step['revision']}: {status}")
            for issue in step["issues"]:
                print(f"    - {issue}")
        print(f"\nRevisions used: {result['revisions_used']}")
        print(f"\nFinal forecast:\n{result['forecast']}")


if __name__ == "__main__":
    main()
