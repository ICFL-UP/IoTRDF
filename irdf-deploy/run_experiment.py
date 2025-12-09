import os
import subprocess
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import requests

# --- Import our custom module ---
try:
    from energy_calculator import calculate_energy_from_file
except ImportError:
    print("Error: `energy_calculator.py` not found.")
    print("Please make sure it's in the same directory.")
    sys.exit(1)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# --- Test Parameters ---
N_REQUESTS = 300            # Number of requests per test run
FEATURE_DIM = 90            # Total features (for 'predict' payload). Change if needed.
ASSUMED_POWER_W = 1.5       # Power coefficient for energy calculation
SERVER_URL = "http://localhost:8000"
SERVER_HEALTH = f"{SERVER_URL}/health"
SERVER_PREDICT = f"{SERVER_URL}/predict"

# --- Docker & File Setup ---
RESULTS_DIR = "results"              # All reports and graphs will go here
SERVER_IMAGE = "irdf-infer"          # Name to build (matches run.sh/manual tests)
SERVER_NAME = "irdf-node"            # Name for the running container
BG_LOAD_NAME = "bg-load"             # Name for the interference container
MODELS_DIR = os.path.abspath("models") # Absolute path to your models folder

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def run_command(cmd, shell=False, check=True):
    """Runs a shell command, prints it, and returns its output."""
    print(f"\n[CMD] {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR RUNNING COMMAND ---")
        print(f"COMMAND: {' '.join(cmd)}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise e


def build_docker_image():
    """Cleans old containers/images and builds a fresh irdf-infer image."""
    print("--- (0/6) Cleaning old containers/images ---")
    # Stop & remove any leftover containers (ignore errors)
    run_command(["docker", "rm", "-f", SERVER_NAME], check=False)
    run_command(["docker", "rm", "-f", BG_LOAD_NAME], check=False)

    # Remove old image if it exists (ignore errors)
    run_command(["docker", "rmi", "-f", SERVER_IMAGE], check=False)

    print("--- (1/6) Building Docker Image ---")
    # Force a no-cache rebuild so server.py changes are always picked up
    run_command(["docker", "build", "--no-cache", "-t", SERVER_IMAGE, "."])


def start_server(with_interference=False):
    """Starts the irdf-infer container with resource limits."""
    print(f"--- Starting Server (Interference: {with_interference}) ---")
    stop_server()  # Clean up any old ones

    # For now, use the same resources that worked in your manual test.
    # You can later reduce to --cpus=0.25 --memory=256m if it still starts fine.
    run_command([
        "docker", "run", "-d", "--rm", "--name", SERVER_NAME,
        "--cpus=1", "--memory=2g",
        "-p", "8000:8000",
        "-v", f"{MODELS_DIR}:/models",
        SERVER_IMAGE
    ])

    if with_interference:
        print("--- Starting Background Interference (Aggressive Load: 0.25 CPU) ---")
        run_command([
            "docker", "run", "-d", "--rm", "--name", BG_LOAD_NAME,
            "--cpus=0.25", "alpine", "sh", "-c", # <-- UPDATED TO 0.25
            'i=0; while true; do i=$((i+1)); test $((i%1000000)) -eq 0 && printf "."; done'
        ])

    # Wait for /health to be ready instead of a blind sleep
    print("Waiting for server /health to become ready...")
    for i in range(40):  # up to ~40 seconds
        try:
            r = requests.get(SERVER_HEALTH, timeout=2)
            if r.status_code == 200:
                print("Server is healthy.")
                return
        except Exception:
            pass
        time.sleep(1)

    print("WARNING: server /health never became ready within timeout; continuing anyway.")


def stop_server():
    """Stops and removes all test containers."""
    print("--- Stopping All Test Containers ---")
    run_command(["docker", "stop", SERVER_NAME], check=False, shell=False)
    run_command(["docker", "stop", BG_LOAD_NAME], check=False, shell=False)
    time.sleep(2)  # Give Docker time to release ports


def set_model(model_name):
    """Tells the server to reload a specific model set ('policy' or 'baseline')."""
    if model_name in ["policy", "baseline"]:
        print(f"--- Setting server model to: {model_name} ---")
        run_command(["curl", "-X", "POST", f"{SERVER_URL}/model/reload?set={model_name}"])


def run_test(test_id, target_url):
    """Runs load_client.py for a single test scenario."""
    print(f"\n--- (RUNNING TEST: {test_id}) ---")
    json_path = os.path.join(RESULTS_DIR, f"{test_id}_report.json")
    plot_path = os.path.join(RESULTS_DIR, f"{test_id}_plot.png")

    cmd = [
        "python3", "load_client.py",
        "--N", str(N_REQUESTS),
        "--target_url", target_url,
        "--health_url", SERVER_HEALTH,
        "--dim", str(FEATURE_DIM),
        "--output_json", json_path,
        "--output_plot", plot_path
    ]
    run_command(cmd)
    return json_path


def get_model_size(model_name):
    """Gets the file size of a model in bytes."""
    try:
        # NOTE: Returning size in bytes now (not MB)
        return os.path.getsize(os.path.join(MODELS_DIR, model_name, "model.joblib"))
    except FileNotFoundError:
        return 0


def analyze_results(all_data):
    """Generates a final report (table and graphs) from all test data."""
    print("\n--- (5/6) Generating Final Report and Graphs ---")

    # --- 1. Create DataFrame ---
    data_list = []
    for test_name, metrics in all_data.items():
        metrics['Test'] = test_name
        data_list.append(metrics)

    df = pd.DataFrame(data_list)
    df = df.set_index('Test')  # Use test name as the row index

    # --- 2. Format DataFrame for Printing and Plotting ---
    report_df = df[[
        'p95_latency_ms',
        'avg_throughput_rps',
        'peak_memory_mb',
        'energy_j_per_req',
        'model_size_bytes', # Changed from model_size_mb
        'cpu_s_per_req'
    ]].copy()

    # Rename column headers
    report_df.columns = [
        "p95 Latency (ms)",
        "Throughput (rps)",
        "Peak Memory (MB)",
        "Energy/Req (J)",
        "Model Size (Bytes)", # Changed header to Bytes
        "CPU-sec/Req"
    ]

    report_df = report_df.round({
        "p95 Latency (ms)": 2,
        "Throughput (rps)": 2,
        "Peak Memory (MB)": 2,
        "Energy/Req (J)": 6,
        "Model Size (Bytes)": 0, # Round to nearest byte (0 decimals)
        "CPU-sec/Req": 6
    })

    print("\n" + "=" * 80)
    print(" " * 25 + "FINAL EXPERIMENT RESULTS")
    print("=" * 80)
    print(report_df.to_string())
    print("=" * 80)

    # Save table to file
    report_df.to_csv(os.path.join(RESULTS_DIR, "final_report.csv"))
    print(f"\nReport table saved to {RESULTS_DIR}/final_report.csv")

    # --- 3. Generate Comparison Graphs for all metrics (CLEAN RESEARCH ARTICLE STYLE) ---
    print("\nGenerating comparison graphs for all metrics (Clean Style)...")
    
    # Define plot properties
    plot_df = report_df.copy()
    
    # Define colors based on test type (Base vs. Interf, Policy vs. Baseline)
    test_colors = {
        'NoModel_Base': '#AEC6CF', 'NoModel_Interf': '#AEC6CF',
        'Policy_Base': '#C3E6CB', 'Policy_Interf': '#C3E6CB',
        'Baseline_Base': '#FFC3A0', 'Baseline_Interf': '#FFC3A0'
    }
    colors = [test_colors.get(test, '#CCCCCC') for test in plot_df.index]
    
    metrics_plotted = []
    
    for column in plot_df.columns:
        plt.figure(figsize=(10, 6))
        plot_df[column].plot(kind='bar', color=colors)
        
        file_name = f"final_graph_{column.replace('/', 'per').replace(' ', '_').replace('-', '_').lower()}.png"

        plt.title("") # Title is empty
        plt.ylabel(column) # Y-axis label is the metric name
        plt.xlabel("Scenario") # X-axis label is "Scenario"
        
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, file_name))
        plt.close() # Close figure to free up memory
        metrics_plotted.append(file_name)

    print(f"\nComparison graphs saved to {RESULTS_DIR}/ (Titles removed, Y-label is metric name, X-label is 'Scenario').")
    print(f"Graphs created: {', '.join([m.split('_')[-1].split('.')[0] for m in metrics_plotted])}.")


# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def main():
    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory not found: {MODELS_DIR}")
        sys.exit(1)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    all_results = {}

    # Get model sizes once
    # *** NOW RETURNING SIZE IN BYTES ***
    policy_size_bytes = get_model_size("policy")
    baseline_size_bytes = get_model_size("baseline")

    # Define all test scenarios
    # (test_id, target_url, model_to_set, use_interference, model_size)
    scenarios = [
        ("NoModel_Base",    SERVER_PREDICT, None,      False, 0),
        ("NoModel_Interf",  SERVER_PREDICT, None,      True,  0),
        ("Policy_Base",     SERVER_PREDICT, "policy",  False, policy_size_bytes),
        ("Policy_Interf",   SERVER_PREDICT, "policy",  True,  policy_size_bytes),
        ("Baseline_Base",   SERVER_PREDICT, "baseline", False, baseline_size_bytes),
        ("Baseline_Interf", SERVER_PREDICT, "baseline", True,  baseline_size_bytes),
    ]

    try:
        # --- (2/6) Build Image ---
        build_docker_image()

        # --- (3/6) Run All Scenarios ---
        for test_id, target_url, model_name, use_interf, model_size_bytes in scenarios:
            print("\n" + "=" * 80)
            print(f"STARTING SCENARIO: {test_id}")
            print("=" * 80)

            start_server(with_interference=use_interf)

            if model_name:
                set_model(model_name)

            # Run the test
            report_file = run_test(test_id, target_url)

            # --- (4/6) Collect & Process Results ---
            with open(report_file, 'r') as f:
                report_data = json.load(f)

            energy_data = calculate_energy_from_file(report_file, ASSUMED_POWER_W)
            if energy_data is None:
                energy_data = {"energy_j_per_req": 0, "cpu_s_per_req": 0}

            # Aggregate all metrics
            all_results[test_id] = {
                'p95_latency_ms':      report_data['latency_ms']['p95'],
                'avg_throughput_rps':  report_data['avg_throughput_rps'],
                'peak_memory_mb':      report_data['peak_memory_bytes'] / (1024 * 1024),
                'energy_j_per_req':    energy_data['energy_j_per_req'],
                'cpu_s_per_req':       energy_data['cpu_s_per_req'],
                'model_size_bytes':    model_size_bytes # Key name changed
            }

            stop_server()

        # --- (5/6) Generate Final Report ---
        analyze_results(all_results)

        print("\n" + "=" * 80)
        print(" " * 28 + "EXPERIMENT HARNESS COMPLETE")
        print("=" * 80)

    except Exception as e:
        print("\n\n--- !!! AN ERROR OCCURRED !!! ---")
        print(f"Error: {e}")
        print("Attempting to clean up containers...")
    finally:
        # --- (6/6) Final Cleanup ---
        stop_server()


if __name__ == "__main__":
    main()
