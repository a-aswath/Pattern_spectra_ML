import glob
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[2:]  # skip header + column indices

    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        numbers = [float(x) for x in tokens[1:]]  # skip row index
        data.append(numbers)

    return np.array(data)
def collect_bins(file_pattern):
    files = sorted(glob.glob(file_pattern))
    bins = [[[] for _ in range(11)] for _ in range(11)]
    for f in files:
        mat = load_granulometry_m_file(f)
        for i in range(11):
            for j in range(11):
                if mat[i, j] != 0:  # optional: ignore zeros
                    bins[i][j].append(mat[i, j])
    return bins, len(files)

# --- Collect H and NH bins ---
bins_Bh, n_Bh_files = collect_bins("granulometryhB*.m")
bins_Bnh, n_Bnh_files = collect_bins("granulometryBuh*.m")

bins_Gh, n_Gh_files = collect_bins("granulometryhG*.m")
bins_Gnh, n_Gnh_files = collect_bins("granulometryGuh*.m")

bins_Rh, n_Rh_files = collect_bins("granulometryhR*.m")
bins_Rnh, n_Rnh_files = collect_bins("granulometryRuh*.m")


# Example: your bin values (replace with actual lists)

Bh_values = np.array(bins_Bh[7][8])
Gh_values = np.array(bins_Gh[7][8])
Rh_values = np.array(bins_Rh[7][8])

BNH_values = np.array(bins_Bnh[7][8])
GNH_values = np.array(bins_Gnh[7][8])
RNH_values = np.array(bins_Rnh[7][8])

# Define bins for histogram (shared bins for fair comparison)
all_values = np.concatenate([Bh_values, Gh_values, Rh_values,
                             BNH_values, GNH_values, RNH_values])
bin_edges = np.linspace(all_values.min(), all_values.max(), 21)  # 20 bins

# Compute histograms (density=True for probability hist)
Bh_hist, _   = np.histogram(Bh_values,   bins=bin_edges, density=True)
Gh_hist, _   = np.histogram(Gh_values,   bins=bin_edges, density=True)
Rh_hist, _   = np.histogram(Rh_values,   bins=bin_edges, density=True)

BNH_hist, _  = np.histogram(BNH_values,  bins=bin_edges, density=True)
GNH_hist, _  = np.histogram(GNH_values,  bins=bin_edges, density=True)
RNH_hist, _  = np.histogram(RNH_values,  bins=bin_edges, density=True)

# Concatenate histograms
H_hist  = np.concatenate([Bh_hist, Gh_hist, Rh_hist])
NH_hist = np.concatenate([BNH_hist, GNH_hist, RNH_hist])

# Log-transform histograms
epsilon = 1e-6
log_H  = np.log(H_hist  + epsilon)
log_NH = np.log(NH_hist + epsilon)

# Plot
plt.figure(figsize=(10,5))
plt.plot(log_H,  label="H",  color="skyblue", marker="o")
plt.plot(log_NH, label="NH", color="orange",  marker="o")
plt.xlabel("Concatenated Histogram Bin Index")
plt.ylabel("Log(Frequency Density)")
plt.title("Concatenated Histograms (H vs NH)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
from scipy.stats import ttest_ind
effect_sizes = np.zeros((11, 11))
p_values = np.ones((11, 11))

for i in range(11):
    for j in range(11):
        # Extract values for each channel (raw values)
        Bh_vals  = np.array(bins_Bh[i][j])
        Gh_vals  = np.array(bins_Gh[i][j])
        Rh_vals  = np.array(bins_Rh[i][j])

        Bnh_vals = np.array(bins_Bnh[i][j])
        Gnh_vals = np.array(bins_Gnh[i][j])
        Rnh_vals = np.array(bins_Rnh[i][j])

        # Only proceed if all channels have enough values
        if (len(Bh_vals) > 1 and len(Gh_vals) > 1 and len(Rh_vals) > 1 and
            len(Bnh_vals) > 1 and len(Gnh_vals) > 1 and len(Rnh_vals) > 1):

            # Define common bin edges for fair histogram comparison
            all_vals = np.concatenate([Bh_vals, Gh_vals, Rh_vals,
                                       Bnh_vals, Gnh_vals, Rnh_vals])
            bin_edges = np.linspace(all_vals.min(), all_vals.max(), 21)  # 20 bins

            # Compute histograms per channel
            Bh_hist, _  = np.histogram(Bh_vals,  bins=bin_edges, density=True)
            Gh_hist, _  = np.histogram(Gh_vals,  bins=bin_edges, density=True)
            Rh_hist, _  = np.histogram(Rh_vals,  bins=bin_edges, density=True)

            Bnh_hist, _ = np.histogram(Bnh_vals, bins=bin_edges, density=True)
            Gnh_hist, _ = np.histogram(Gnh_vals, bins=bin_edges, density=True)
            Rnh_hist, _ = np.histogram(Rnh_vals, bins=bin_edges, density=True)

            # Concatenate histograms across channels
            H_hist  = np.concatenate([Bh_hist, Gh_hist, Rh_hist])
            NH_hist = np.concatenate([Bnh_hist, Gnh_hist, Rnh_hist])

            # Log-transform histograms
            H_log  = np.log(H_hist  + 1e-6)
            NH_log = np.log(NH_hist + 1e-6)

            # --- Effect size (Cohen’s d with pooled std) ---
            n_H, n_NH = len(H_log), len(NH_log)
            s_H, s_NH = np.std(H_log, ddof=1), np.std(NH_log, ddof=1)
            s_pooled = np.sqrt(((n_H-1)*s_H**2 + (n_NH-1)*s_NH**2) / (n_H+n_NH-2))

            effect_sizes[i, j] = (np.mean(H_log) - np.mean(NH_log)) / s_pooled

            # --- t-test ---
            _, p_val = ttest_ind(H_log, NH_log, equal_var=False)
            p_values[i, j] = p_val

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(np.abs(effect_sizes), cmap='viridis')
plt.colorbar(label='|Cohen\'s d|')
plt.title('Effect size per bin')
plt.xlabel('j (column)')
plt.ylabel('i (row)')

plt.subplot(1,2,2)
plt.imshow(-np.log10(p_values), cmap='magma')
plt.colorbar(label='-log10(p-value)')
plt.title('Significance per bin')
plt.xlabel('j (column)')
plt.ylabel('i (row)')

plt.tight_layout()
plt.show()
#Bins with highest |effect size| and lowest p-value are the most suitable features for classification, We can make a ranking table of all 21×21 bins by effect size and p-value,

# then select top N bins as features.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Prepare list to store bin stats
bin_stats = []

for i in range(11):
    for j in range(11):
        # Raw values from each channel
        Bh_vals  = np.array(bins_Bh[i][j])
        Gh_vals  = np.array(bins_Gh[i][j])
        Rh_vals  = np.array(bins_Rh[i][j])

        Bnh_vals = np.array(bins_Bnh[i][j])
        Gnh_vals = np.array(bins_Gnh[i][j])
        Rnh_vals = np.array(bins_Rnh[i][j])

        if (len(Bh_vals) > 1 and len(Gh_vals) > 1 and len(Rh_vals) > 1 and
            len(Bnh_vals) > 1 and len(Gnh_vals) > 1 and len(Rnh_vals) > 1):

            # Define common bin edges for histogram (fair comparison)
            all_vals = np.concatenate([Bh_vals, Gh_vals, Rh_vals,
                                       Bnh_vals, Gnh_vals, Rnh_vals])
            bin_edges = np.linspace(all_vals.min(), all_vals.max(), 21)  # 20 bins

            # Compute histograms for each channel
            Bh_hist, _  = np.histogram(Bh_vals,  bins=bin_edges, density=True)
            Gh_hist, _  = np.histogram(Gh_vals,  bins=bin_edges, density=True)
            Rh_hist, _  = np.histogram(Rh_vals,  bins=bin_edges, density=True)

            Bnh_hist, _ = np.histogram(Bnh_vals, bins=bin_edges, density=True)
            Gnh_hist, _ = np.histogram(Gnh_vals, bins=bin_edges, density=True)
            Rnh_hist, _ = np.histogram(Rnh_vals, bins=bin_edges, density=True)

            # Concatenate histograms across channels
            H_hist  = np.concatenate([Bh_hist, Gh_hist, Rh_hist])
            NH_hist = np.concatenate([Bnh_hist, Gnh_hist, Rnh_hist])

            # Log-transform histograms
            H_log  = np.log(H_hist  + 1e-6)
            NH_log = np.log(NH_hist + 1e-6)

            # --- Effect size (Cohen’s d) ---
            n_H, n_NH = len(H_log), len(NH_log)
            s_H, s_NH = np.std(H_log, ddof=1), np.std(NH_log, ddof=1)
            s_pooled = np.sqrt(((n_H-1)*s_H**2 + (n_NH-1)*s_NH**2) / (n_H + n_NH - 2))

            effect_size = (np.mean(H_log) - np.mean(NH_log)) / s_pooled

            # --- t-test ---
            _, p_val = ttest_ind(H_log, NH_log, equal_var=False)

            # Save stats
            bin_stats.append({
                'bin_i': i,
                'bin_j': j,
                'effect_size': effect_size,
                'p_value': p_val
            })

# Convert to DataFrame
df_bins = pd.DataFrame(bin_stats)
df_bins['abs_effect_size'] = np.abs(df_bins['effect_size'])
df_sorted = df_bins.sort_values(by='abs_effect_size', ascending=False)

# Show top 10 most discriminative bins
print(df_sorted[['bin_i','bin_j','effect_size','p_value']].head(10))

# Build effect size matrix
effect_matrix = np.zeros((11, 11))
for _, row in df_bins.iterrows():
    i = int(row['bin_i'])
    j = int(row['bin_j'])
    effect_matrix[i, j] = np.abs(row['effect_size'])

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(effect_matrix, cmap='viridis', origin='lower')
plt.colorbar(label="|Cohen's d| (Effect Size)")
plt.title('Discriminative Power per Bin (H vs NH)')
plt.xlabel('Column (j)')
plt.ylabel('Row (i)')

# Highlight top 10 bins
top_bins = df_sorted.head(10)
for _, row in top_bins.iterrows():
    i = int(row['bin_i'])
    j = int(row['bin_j'])
    plt.text(j, i, '★', color='red', ha='center', va='center', fontsize=12)

plt.show()

# Concatenate corresponding bins from B, G, R channels
bins_h = [[bins_Bh[i][j] + bins_Gh[i][j] + bins_Rh[i][j] for j in range(11)] for i in range(11)]
n_h_files = n_Bh_files + n_Gh_files + n_Rh_files    

bins_nh = [[bins_Bnh[i][j] + bins_Gnh[i][j] + bins_Rnh[i][j] for j in range(11)] for i in range(11)]
n_nh_files = n_Bnh_files + n_Gnh_files + n_Rnh_files

print(f"Total H files: {n_h_files}, Total NH files: {n_nh_files}")



