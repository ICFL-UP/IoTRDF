import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import font_manager

# --- Configuration ---
RESULTS_DIR = 'results'

# 2. Scenarios and Colors for Professional Presentation
# Define the order and labels for the X-axis of the plots
SCENARIO_ORDER = [
    'NoModel_Base', 'NoModel_Interf',
    'Policy_Base', 'Policy_Interf',
    'Baseline_Base', 'Baseline_Interf'
]

# Define a palette with distinct colors for each scenario
COLORS = {
    'NoModel_Base': '#7D98A1',   # Light Slate (Gray/Blue)
    'NoModel_Interf': '#4F6D7A', # Dark Slate (Gray/Blue)
    'Policy_Base': '#39B34A',    # Bright Green (Base)
    'Policy_Interf': '#2C8E38',  # Dark Green (Interference)
    'Baseline_Base': '#F18F01',  # Bright Orange (Base)
    'Baseline_Interf': '#D27B00' # Dark Orange (Interference)
}

# --- Hardcoded Experiment Data ---
raw_data = {
    'Test': SCENARIO_ORDER,
    'p95 Latency (ms)': [1.88, 2.19, 1.94, 2.03, 1.91, 2.04],
    'Throughput (rps)': [121.70, 118.01, 122.80, 114.78, 121.12, 114.34],
    'Peak Memory (MB)': [163.09, 163.23, 163.26, 163.30, 162.93, 163.55],
    'Energy/Req (J)': [0.00175, 0.00195, 0.00205, 0.00185, 0.00200, 0.00175],
    'Model Size (MB)': [0.000, 0.000, 0.002, 0.002, 0.002, 0.002],
    'CPU-sec/Req': [0.001167, 0.001300, 0.001367, 0.001233, 0.001333, 0.001167]
}

# --- Data Preparation ---
def load_data_from_dict(data_dict):
    """Loads the experiment data from the hardcoded dictionary."""
    df = pd.DataFrame(data_dict).set_index('Test')
    df = df.reindex(SCENARIO_ORDER)
    return df

# --- Plotting Functions ---
def create_bar_chart(df, metric, ylabel, filename):
    """Generates a professional bar chart for a single metric with academic formatting."""
    
    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(8, 5)) 

    # 2. Get Data and Colors
    data = df[metric]
    bar_colors = [COLORS[scenario] for scenario in SCENARIO_ORDER]
    
    # 3. Plot Bars (Removed edgecolor and linewidth for no outlines)
    bars = ax.bar(data.index, data.values, color=bar_colors)

    # 4. Add Value Labels (MOVED TO BOTTOM OF BAR, TEXT IS WHITE)
    for bar in bars:
        height = bar.get_height()
        # Custom formatting for small numbers
        if metric in ['Energy/Req (J)', 'CPU-sec/Req']:
             label_format = f'{height:.5f}'
        else:
             label_format = f'{height:.2f}'

        ax.annotate(label_format,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -5),  # Negative offset moves text INSIDE the bar
                    textcoords="offset points",
                    ha='center', va='top', # 'va=top' positions the text at the top of the bar, pulled down by offset
                    fontsize=10, 
                    color='white') # Set text color to white for contrast

    # 5. Customize X-axis Labels (Using global font settings)
    short_labels = [
        'NoModel\nBase', 'NoModel\nInterf', 
        'Policy\nBase', 'Policy\nInterf', 
        'Baseline\nBase', 'Baseline\nInterf'
    ]
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels(short_labels, rotation=0, fontsize=10)

    # 6. Customize Axes and Title (Using global font settings, thin lines)
    ax.set_title(ylabel, fontsize=12, loc='left')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='y', labelsize=10) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5) 
    
    # 7. Legend is intentionally removed

    # 8. Save Figure
    output_path = os.path.join(RESULTS_DIR, filename)
    plt.tight_layout() 
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {output_path}")

# --- Main Execution ---
if __name__ == '__main__':
    # Setting default font configuration for the entire matplotlib session
    plt.rcParams['font.family'] = 'sans-serif'
    # Try to set Arial or fall back to standard sans-serif fonts
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Helvetica', 'Arial']
    plt.rcParams['font.size'] = 10 
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    df_results = load_data_from_dict(raw_data)

    if df_results is not None:
        plot_definitions = [
            ('p95 Latency (ms)', 'p95 Latency (ms)', 'final_graph_p95_latency.png'),
            ('Throughput (rps)', 'Throughput (requests/sec)', 'final_graph_throughput.png'),
            ('Peak Memory (MB)', 'Peak Memory Usage (MB)', 'final_graph_peak_memory.png'),
            ('Energy/Req (J)', 'Energy Per Request (J)', 'final_graph_energy_per_req.png'),
            ('CPU-sec/Req', 'CPU Seconds Per Request (s)', 'final_graph_cpu_per_req.png')
        ]

        for metric, ylabel, filename in plot_definitions:
            create_bar_chart(df_results, metric, ylabel, filename)

    print("Plotting complete. Check the 'results' folder.")
