"""Automated Research Benchmark Runner for Lifelong MAPF Repair.

Executes controlled, reproducible comparative experiments across:
1. Proposed Adaptive Framework (LLM + Verified Cache + Deterministic Validator)
2. Fixed Classical Baselines (PIBT, LNS, Windowed PBS)
3. Ablation: No Cache (--cache off)
4. Ablation: Controlled Non-executing Validator Evaluation Mode (--validator off)
5. Ablation: Cache Similarity Threshold Sweep (tau in [0.70, 0.80, 0.85, 0.90, 0.95])
6. Disruption Overlap Variation (none, same_aisle, same_robot)

Outputs:
- runs/benchmark_summary.csv (Aggregated quantitative metrics)
- runs/benchmark_summary.md  (Publication-ready GitHub Markdown table)
- runs/benchmark_summary.tex (Publication-ready LaTeX table)
"""

import os
import sys
import csv
import time
import argparse
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from amr_repair.loop.controller import SimulationConfig, run


def parse_benchmark_args():
    parser = argparse.ArgumentParser(description="Automated MAPF Benchmark Experiment Runner")
    parser.add_argument("--map", default="maps/warehouse-10-20-10-2-1.map", help="Path to MovingAI .map file")
    parser.add_argument("--robots", type=int, default=50, help="Number of robots to simulate")
    parser.add_argument("--steps", type=int, default=120, help="Timesteps per experiment")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101], help="Random seeds to evaluate")
    parser.add_argument("--overlaps", nargs="+", default=["same_aisle"], choices=["none", "same_aisle", "same_robot"], help="Overlap modes to evaluate")
    parser.add_argument("--rate", type=float, default=0.08, help="Disruption injection rate")
    parser.add_argument("--output_dir", default="runs/benchmark", help="Directory for benchmark JSON replays")
    parser.add_argument("--summary_dir", default="runs", help="Directory for aggregated summary tables")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing: runs subset of conditions")
    return parser.parse_args()


def build_experiment_matrix(quick: bool = False) -> List[Dict[str, Any]]:
    """Define the experimental conditions."""
    if quick:
        return [
            {"name": "Proposed_Framework", "mode": "llm", "cache": True, "val": True, "thresh": 0.85},
            {"name": "Fixed_PIBT", "mode": "baseline_pibt", "cache": False, "val": True, "thresh": 0.85},
            {"name": "Fixed_PBS", "mode": "baseline_pbs", "cache": False, "val": True, "thresh": 0.85},
            {"name": "Ablation_NoCache", "mode": "llm", "cache": False, "val": True, "thresh": 0.85},
            {"name": "Ablation_Thresh_070", "mode": "llm", "cache": True, "val": True, "thresh": 0.70},
            {"name": "Ablation_Thresh_095", "mode": "llm", "cache": True, "val": True, "thresh": 0.95},
        ]

    return [
        # 1. Proposed Framework
        {"name": "Proposed_Framework", "mode": "llm", "cache": True, "val": True, "thresh": 0.85},
        # 2. Classical Baselines
        {"name": "Fixed_PIBT", "mode": "baseline_pibt", "cache": False, "val": True, "thresh": 0.85},
        {"name": "Fixed_LNS", "mode": "baseline_lns", "cache": False, "val": True, "thresh": 0.85},
        {"name": "Fixed_PBS", "mode": "baseline_pbs", "cache": False, "val": True, "thresh": 0.85},
        # 3. Ablation: No Cache
        {"name": "Ablation_NoCache", "mode": "llm", "cache": False, "val": True, "thresh": 0.85},
        # 4. Ablation: Controlled Validator Evaluation Mode
        {"name": "Ablation_NoValidator_Eval", "mode": "llm", "cache": True, "val": False, "thresh": 0.85},
        # 5. Threshold Sweeps
        {"name": "Threshold_0.70", "mode": "llm", "cache": True, "val": True, "thresh": 0.70},
        {"name": "Threshold_0.80", "mode": "llm", "cache": True, "val": True, "thresh": 0.80},
        {"name": "Threshold_0.85", "mode": "llm", "cache": True, "val": True, "thresh": 0.85},
        {"name": "Threshold_0.90", "mode": "llm", "cache": True, "val": True, "thresh": 0.90},
        {"name": "Threshold_0.95", "mode": "llm", "cache": True, "val": True, "thresh": 0.95},
    ]


def run_benchmarks(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)

    matrix = build_experiment_matrix(quick=args.quick)
    all_results: List[Dict[str, Any]] = []

    print(f"=== Starting Lifelong MAPF Benchmark Suite ===")
    print(f"Conditions: {len(matrix)} | Seeds: {args.seeds} | Overlaps: {args.overlaps} | Robots: {args.robots}")

    total_runs = len(matrix) * len(args.seeds) * len(args.overlaps)
    run_idx = 0

    for exp in matrix:
        for overlap in args.overlaps:
            for seed in args.seeds:
                run_idx += 1
                exp_id = f"{exp['name']}_s{seed}_{overlap}"
                print(f"[{run_idx}/{total_runs}] Running {exp_id}...")

                config = SimulationConfig(
                    map_path=args.map,
                    num_robots=args.robots,
                    steps=args.steps,
                    seed=seed,
                    mode=exp["mode"],
                    disruption_rate=args.rate,
                    overlap_mode=overlap,
                    cache_enabled=exp["cache"],
                    validator_enabled=exp["val"],
                    cache_threshold=exp["thresh"],
                    run_id=exp_id,
                    output_dir=args.output_dir,
                )

                t0 = time.perf_counter()
                try:
                    replay = run(config)
                    elapsed = time.perf_counter() - t0
                    m = replay.metrics

                    row = {
                        "Condition": exp["name"],
                        "Mode": exp["mode"],
                        "Overlap": overlap,
                        "Seed": seed,
                        "Cache": "On" if exp["cache"] else "Off",
                        "Validator": "On" if exp["val"] else "Off",
                        "Threshold": exp["thresh"],
                        "Incidents": m.incidents_total,
                        "SuccessRatePct": round(m.repair_success_rate * 100.0, 1),
                        "MeanRecoverySteps": m.mean_recovery_steps,
                        "CostDeltaPct": m.cost_delta_pct,
                        "LLMCalls": m.llm_calls,
                        "LLMTokens": m.llm_tokens,
                        "LLMCallsAvoided": m.llm_calls_avoided,
                        "MeanLLMLatencyMs": m.mean_llm_latency_ms,
                        "CacheHits": m.cache_hits,
                        "CacheLookups": m.cache_lookups,
                        "CacheHitRatePct": m.cache_hit_rate_pct,
                        "VerifiedReuses": m.verified_cache_reuse_count,
                        "VerifiedReuseRatePct": m.verified_reuse_rate_pct,
                        "MeanRepairLatencyMs": m.mean_total_repair_latency_ms,
                        "InterceptedViolations": m.safety_violations_intercepted,
                        "EmergencyHolds": m.emergency_holds,
                        "RunDurationSec": round(elapsed, 2),
                        "Status": "COMPLETED",
                    }
                except Exception as ex:
                    print(f"ERROR in {exp_id}: {ex}")
                    row = {
                        "Condition": exp["name"],
                        "Mode": exp["mode"],
                        "Overlap": overlap,
                        "Seed": seed,
                        "Cache": "On" if exp["cache"] else "Off",
                        "Validator": "On" if exp["val"] else "Off",
                        "Threshold": exp["thresh"],
                        "Incidents": 0,
                        "SuccessRatePct": 0.0,
                        "MeanRecoverySteps": 0.0,
                        "CostDeltaPct": 0.0,
                        "LLMCalls": 0,
                        "LLMTokens": 0,
                        "LLMCallsAvoided": 0,
                        "MeanLLMLatencyMs": 0.0,
                        "CacheHits": 0,
                        "CacheLookups": 0,
                        "CacheHitRatePct": 0.0,
                        "VerifiedReuses": 0,
                        "VerifiedReuseRatePct": 0.0,
                        "MeanRepairLatencyMs": 0.0,
                        "InterceptedViolations": 0,
                        "EmergencyHolds": 0,
                        "RunDurationSec": 0.0,
                        "Status": f"FAILED: {ex}",
                    }

                all_results.append(row)

    # 1. Export summary CSV
    csv_path = os.path.join(args.summary_dir, "benchmark_summary.csv")
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[Artifact] Benchmark summary CSV written to: {csv_path}")

    # 2. Aggregate across seeds by (Condition, Overlap)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        key = f"{r['Condition']}|{r['Overlap']}"
        grouped.setdefault(key, []).append(r)

    agg_rows = []
    for key, items in grouped.items():
        cond, ov = key.split("|")
        valid_items = [it for it in items if it["Status"] == "COMPLETED"]
        if not valid_items:
            continue
        n = len(valid_items)
        agg_rows.append({
            "Condition": cond,
            "Overlap": ov,
            "Incidents": round(sum(it["Incidents"] for it in valid_items) / n, 1),
            "SuccessRate": round(sum(it["SuccessRatePct"] for it in valid_items) / n, 1),
            "RecoverySteps": round(sum(it["MeanRecoverySteps"] for it in valid_items) / n, 2),
            "CostDeltaPct": round(sum(it["CostDeltaPct"] for it in valid_items) / n, 2),
            "LLMCalls": round(sum(it["LLMCalls"] for it in valid_items) / n, 1),
            "CacheHitRate": round(sum(it["CacheHitRatePct"] for it in valid_items) / n, 1),
            "VerifiedReuseRate": round(sum(it["VerifiedReuseRatePct"] for it in valid_items) / n, 1),
            "RepairLatencyMs": round(sum(it["MeanRepairLatencyMs"] for it in valid_items) / n, 1),
            "ViolationsCaught": round(sum(it["InterceptedViolations"] for it in valid_items) / n, 1),
        })

    # 3. Export Markdown Table
    md_path = os.path.join(args.summary_dir, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Lifelong MAPF Disruption Repair: Benchmark Evaluation\n\n")
        f.write("| Experimental Condition | Overlap Mode | Incidents | Success Rate | Mean Recovery (steps) | Cost Δ (%) | LLM Calls | Cache Hit Rate (%) | Verified Reuse (%) | Repair Latency (ms) | Intercepted Violations |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for a in agg_rows:
            f.write(
                f"| **{a['Condition']}** | `{a['Overlap']}` | {a['Incidents']} | {a['SuccessRate']}% | {a['RecoverySteps']} | +{a['CostDeltaPct']}% | {a['LLMCalls']} | {a['CacheHitRate']}% | {a['VerifiedReuseRate']}% | {a['RepairLatencyMs']} ms | {a['ViolationsCaught']} |\n"
            )
    print(f"[Artifact] Benchmark Markdown table written to: {md_path}")

    # 4. Export LaTeX Table
    tex_path = os.path.join(args.summary_dir, "benchmark_summary.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Benchmark Comparative Results for Disruption-Aware Path Repair in Lifelong AMR Routing\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Quantitative comparison of adaptive LLM repair against classical baselines and ablations across warehouse disruption scenarios.}\n")
        f.write("\\label{tab:mapf_benchmark_results}\n")
        f.write("\\begin{tabular}{llcccccccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Strategy / Condition} & \\textbf{Overlap} & \\textbf{Success (\\%)} & \\textbf{Recovery} & \\textbf{Cost $\\Delta$ (\\%)} & \\textbf{LLM Calls} & \\textbf{Hit Rate (\\%)} & \\textbf{Reuse (\\%)} & \\textbf{Latency (ms)} & \\textbf{Safety Violations} \\\\\n")
        f.write("\\hline\n")
        for a in agg_rows:
            cond_tex = a['Condition'].replace('_', '\\_')
            f.write(f"{cond_tex} & {a['Overlap']} & {a['SuccessRate']} & {a['RecoverySteps']} & +{a['CostDeltaPct']} & {a['LLMCalls']} & {a['CacheHitRate']} & {a['VerifiedReuseRate']} & {a['RepairLatencyMs']} & {a['ViolationsCaught']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"[Artifact] Benchmark LaTeX table written to: {tex_path}")


if __name__ == "__main__":
    args = parse_benchmark_args()
    run_benchmarks(args)
