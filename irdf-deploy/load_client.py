import time
import json
import requests
import argparse
import random
import sys

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def pct(vals, p):
    if not vals:
        return 0.0
    k = max(0, int(round((p / 100.0) * (len(vals) - 1))))
    return sorted(vals)[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=300)

    # Accept BOTH --target_url and --url for compatibility
    ap.add_argument(
        "--target_url", "--url",
        dest="url",
        default="http://localhost:8000/predict"
    )

    ap.add_argument("--health_url", default=None)
    ap.add_argument("--dim", type=int, default=90)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--output_plot", default=None)
    args = ap.parse_args()

    def payload(dim):
        return {"x": [random.random() for _ in range(dim)]}

    # Optional initial health snapshot
    initial_health = None
    if args.health_url:
        try:
            initial_health = requests.get(args.health_url, timeout=3).json()
        except Exception as e:
            print(f"[WARN] could not fetch initial health: {e}", file=sys.stderr)

    lat_ms = []
    cpu_s = []
    rss_b = []

    n_ok = 0
    t0_wall = time.perf_counter()

    for i in range(args.N):
        try:
            t0 = time.perf_counter()
            r = requests.post(args.url, json=payload(args.dim), timeout=10)
            t1 = time.perf_counter()
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"[WARN] request {i} failed: {e}", file=sys.stderr)
            continue

        # Prefer server-reported latency, else use our own measurement
        lat = d.get("latency_ms", (t1 - t0) * 1000.0)
        lat_ms.append(lat)

        cpu_s.append(float(d.get("proc_cpu_s", 0.0)))
        rss_b.append(int(d.get("rss_bytes", 0)))

        n_ok += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    t1_wall = time.perf_counter()
    total_wall_s = t1_wall - t0_wall

    if n_ok == 0:
        print("[ERROR] No successful requests", file=sys.stderr)
        sys.exit(1)

    avg_lat = sum(lat_ms) / len(lat_ms)
    p50_lat = pct(lat_ms, 50)
    p95_lat = pct(lat_ms, 95)
    avg_cpu = sum(cpu_s) / len(cpu_s)
    avg_rss = sum(rss_b) / len(rss_b)
    peak_rss = max(rss_b) if rss_b else 0

    throughput = n_ok / total_wall_s if total_wall_s > 0 else 0.0

    result = {
        "timestamp": time.time(),
        "count": n_ok,
        "latency_ms": {
            "p50": p50_lat,
            "p95": p95_lat,
            "avg": avg_lat,
        },
        "avg_throughput_rps": throughput,
        "cpu_s_per_req_avg": avg_cpu,
        "rss_bytes_avg": avg_rss,
        "peak_memory_bytes": peak_rss,
        "health_initial": initial_health,
    }

    # Print JSON to stdout (useful if you run it manually)
    print(json.dumps(result, indent=2))

    # Save to file for run_experiment.py
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)

    # Optional simple latency plot
    if args.output_plot and plt is not None:
        plt.figure()
        plt.plot(lat_ms)
        plt.xlabel("Request index")
        plt.ylabel("Latency (ms)")
        plt.tight_layout()
        plt.savefig(args.output_plot)


if __name__ == "__main__":
    main()
