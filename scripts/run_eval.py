"""Runs the full stage-by-stage evaluation and saves results for the app's evaluation page.

    python scripts/run_eval.py
    python scripts/run_eval.py --embedders intfloat/multilingual-e5-base intfloat/multilingual-e5-small lsa
GPT is used automatically when OPENAI_API_KEY is set (env var or .streamlit/secrets.toml);
otherwise the generation stage measures the retrieval-only fallback. --llm-limit caps GPT calls (cost).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.evaluation import run_all, save  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--embedders", nargs="+", default=["intfloat/multilingual-e5-base", "lsa"])
ap.add_argument("--llm-limit", type=int, default=100)
args = ap.parse_args()
res = run_all(args.embedders, args.llm_limit)
save(res)
for k in ("data", "chosen_unit", "classification", "generation", "safety", "skipped_embedders"):
    print(k, ":", res[k])
print("\nUnits\n" + res["units"].round(3).to_string(index=False))
print("\nRetrieval\n" + res["retrieval"].round(3).to_string(index=False))
print("\nPer type\n" + res["classification_per_type"].round(3).to_string(index=False))
print("\nSafety probes missed / false alarms\n" + res["safety_probes"][res["safety_probes"].expected != res["safety_probes"].flagged].to_string(index=False))
