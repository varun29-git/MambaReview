import os
import csv
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

def read_metrics(csv_path):
    if not os.path.exists(csv_path):
        return [], []
        
    tokens = []
    ppls = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'val_ppl' in row and row['val_ppl'] != 'N/A' and row['val_ppl'].strip() != '':
                try:
                    tokens.append(float(row['tokens_seen']))
                    ppls.append(float(row['val_ppl']))
                except ValueError:
                    continue
                    
    return tokens, ppls

def main():
    m1_tokens, m1_ppls = read_metrics(os.path.join(LOGS_DIR, "mamba1_metrics.csv"))
    m2_tokens, m2_ppls = read_metrics(os.path.join(LOGS_DIR, "mamba2_metrics.csv"))
    
    if not m1_tokens and not m2_tokens:
        print("No validation perplexity data found to plot.")
        return
        
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.axes()
    ax.set_facecolor('white')
    
    # High contrast styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(colors='black')
    
    if m1_tokens:
        plt.plot(m1_tokens, m1_ppls, color='black', linestyle='-', linewidth=2, label='Vanilla Mamba')
    if m2_tokens:
        # Using a dashed black line to differentiate while keeping it monochrome
        plt.plot(m2_tokens, m2_ppls, color='black', linestyle='--', linewidth=2, label='Mamba-2')
        
    plt.xlabel('Tokens Seen', fontsize=12, color='black', fontweight='bold')
    plt.ylabel('Validation Perplexity', fontsize=12, color='black', fontweight='bold')
    plt.title('TinyStories Validation Perplexity (Mamba-1 vs Mamba-2)', fontsize=14, color='black', fontweight='bold')
    
    plt.grid(True, linestyle=':', color='gray', alpha=0.5)
    
    # Format x-axis as millions
    def format_millions(x, pos):
        return f'{x/1e6:.1f}M'
        
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_millions))
    
    plt.legend(frameon=True, facecolor='white', edgecolor='black', fontsize=11)
    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ppl_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_path}")

if __name__ == "__main__":
    main()
