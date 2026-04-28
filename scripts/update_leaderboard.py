import os
import csv
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
README_PATH = os.path.join(REPO_ROOT, "README.md")

def get_average_tps(log_file):
    if not os.path.exists(log_file):
        return "N/A"
    
    tps_values = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'tps' in row:
                try:
                    tps_values.append(float(row['tps']))
                except ValueError:
                    continue
                
    if not tps_values:
        return "N/A"
    return f"{sum(tps_values)/len(tps_values):.0f}"

def get_best_ppl(eval_log_file):
    if not os.path.exists(eval_log_file):
        return "N/A"
    
    best_ppl = float('inf')
    with open(eval_log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'val_ppl' in row and row['val_ppl'] != 'N/A' and row['val_ppl'].strip() != '':
                try:
                    best_ppl = min(best_ppl, float(row['val_ppl']))
                except ValueError:
                    continue
                
    if best_ppl == float('inf'):
        return "N/A"
    return f"{best_ppl:.2f}"

def update_readme():
    mamba1_tps = get_average_tps(os.path.join(LOGS_DIR, "mamba1_metrics.csv"))
    mamba2_tps = get_average_tps(os.path.join(LOGS_DIR, "mamba2_metrics.csv"))
    mamba3_tps = get_average_tps(os.path.join(LOGS_DIR, "mamba3_siso_metrics.csv"))
    
    mamba1_ppl = get_best_ppl(os.path.join(LOGS_DIR, "mamba1_metrics.csv"))
    mamba2_ppl = get_best_ppl(os.path.join(LOGS_DIR, "mamba2_metrics.csv"))
    mamba3_ppl = get_best_ppl(os.path.join(LOGS_DIR, "mamba3_siso_metrics.csv"))
    
    table = (
        "## Model Leaderboard\n\n"
        "| Model | Throughput (TPS) | Validation PPL |\n"
        "| :--- | :--- | :--- |\n"
        f"| Vanilla Mamba | {mamba1_tps} | {mamba1_ppl} |\n"
        f"| Mamba-2 | {mamba2_tps} | {mamba2_ppl} |\n"
        f"| Mamba-3 SISO | {mamba3_tps} | {mamba3_ppl} |\n\n"
    )
    
    if not os.path.exists(README_PATH):
        with open(README_PATH, "w") as f:
            f.write("# Mamba Review\n\n" + table)
        return
        
    with open(README_PATH, "r") as f:
        content = f.read()
        
    pattern = r"## Model Leaderboard\n\n.*?\|.*?\n(?:\|.*?\n)*\n?"
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, table, content, flags=re.DOTALL)
    else:
        content = content.rstrip() + "\n\n" + table
        
    with open(README_PATH, "w") as f:
        f.write(content)
        
    print("Leaderboard updated in README.md")

if __name__ == "__main__":
    update_readme()
