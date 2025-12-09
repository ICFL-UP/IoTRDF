# energy_calculator.py

import json

def calculate_energy_from_file(report_file, power_w=1.5):
    """
    Reads the JSON report produced by load_client.py and computes:
      - cpu_s_per_req
      - energy_j_per_req = cpu_s_per_req * power_w
    """
    try:
        with open(report_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[energy_calculator] ERROR reading {report_file}: {e}")
        return None

    # This key must be in the JSON produced by load_client.py
    cpu_s = data.get("cpu_s_per_req_avg")
    if cpu_s is None:
        print(f"[energy_calculator] 'cpu_s_per_req_avg' not found in {report_file}")
        return None

    energy_j = cpu_s * power_w

    return {
        "cpu_s_per_req": cpu_s,
        "energy_j_per_req": energy_j,
    }
