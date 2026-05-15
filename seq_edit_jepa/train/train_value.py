from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Value-head training placeholder for verifier/value supervised stages.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise SystemExit(
        "Standalone value-head training is intentionally deferred. "
        "Use train_one_step configs with model.lambda_val once value targets are added."
    )


if __name__ == "__main__":
    main()
