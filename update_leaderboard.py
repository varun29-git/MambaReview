import os
import csv
import re

def get_average_tps(log_file):
    if not os.path.exists(log_file):
        return "N/A"
    
    tps_values = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'TPS' in row:
                try:
                    tps_values.append(float(row['TPS']))
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
            if 'Val_PPL' in row:
                try:
                    best_ppl = min(best_ppl, float(row['Val_PPL']))
                except ValueError:
                    continue
                
    if best_ppl == float('inf'):
        return "N/A"
    return f"{best_ppl:.2f}"

def update_readme():
    mamba1_tps = get_average_tps("Vanilla-Mamba/metrics_log.csv")
    mamba2_tps = get_average_tps("mamba-2/metrics_log.csv")
    
    mamba1_ppl = get_best_ppl("Vanilla-Mamba/eval_log.csv")
    mamba2_ppl = get_best_ppl("mamba-2/eval_log.csv")
    
    # We will also check quality_metrics.csv if available as a fallback
    if mamba1_ppl == "N/A" and os.path.exists("quality_metrics.csv"):
        with open("quality_metrics.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Model') == 'Mamba-1':
                    mamba1_ppl = row.get('Perplexity', "N/A")
                if row.get('Model') == 'Mamba-2':
                    mamba2_ppl = row.get('Perplexity', "N/A")
    
    table = (
        "## Model Leaderboard\n\n"
        "| Model | Throughput (TPS) | Validation PPL |\n"
        "| :--- | :--- | :--- |\n"
        f"| Vanilla Mamba | {mamba1_tps} | {mamba1_ppl} |\n"
        f"| Mamba-2 | {mamba2_tps} | {mamba2_ppl} |\n\n"
    )
    
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("# Mamba Review\n\n" + table)
        return
        
    with open(readme_path, "r") as f:
        content = f.read()
        
    pattern = r"## Model Leaderboard\n\n.*?\|.*?\n(?:\|.*?\n)*\n?"
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, table, content, flags=re.DOTALL)
    else:
        content = content.rstrip() + "\n\n" + table
        
    with open(readme_path, "w") as f:
        f.write(content)
        
    print("Leaderboard updated in README.md")

if __name__ == "__main__":
    update_readme()
