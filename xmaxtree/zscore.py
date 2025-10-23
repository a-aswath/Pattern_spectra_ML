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
    bins = [[[] for _ in range(21)] for _ in range(21)]
    for f in files:
        mat = load_granulometry_m_file(f)
        for i in range(21):
            for j in range(21):
                if mat[i, j] != 0:  # optional: ignore zeros
                    bins[i][j].append(mat[i, j])
    return bins, len(files)

# --- Collect H and NH bins ---
bins_h, n_h_files = collect_bins("granulometryh*.m")
bins_nh, n_nh_files = collect_bins("granulometrynh*.m")

# --- Example: plot a single bin distribution ---
bin_i, bin_j = 7, 8  # choose any bin

plt.figure(figsize=(12,5))

# H group
plt.hist(bins_h[bin_i][bin_j], bins=20, alpha=0.6, color='skyblue', density=True, label=f'H bin ({bin_i},{bin_j})')

# NH group
plt.hist(bins_nh[bin_i][bin_j], bins=20, alpha=0.6, color='orange', density=True, label=f'NH bin ({bin_i},{bin_j})')

plt.xlabel('Granulometry Value')
plt.ylabel('Density')
plt.title(f'Distribution of bin ({bin_i},{bin_j}) across H ({n_h_files}) and NH ({n_nh_files}) files')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

import numpy as np

def summary_stats(values):
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'count': len(values)
    }

stats_h = summary_stats(bins_h[bin_i][bin_i])
stats_nh = summary_stats(bins_nh[bin_i][bin_i])

print(f'H bin (7, 8):', stats_h)
print(f'NH bin (7, 8):', stats_nh)


# Example: your bin values (replace with actual lists)
H_values  = bins_h[7][8]
NH_values = bins_nh[7][8]

# Log-transform to compress skewed data
epsilon = 1e-6  # to avoid log(0)
log_H  = np.log(np.array(H_values) + epsilon)
log_NH = np.log(np.array(NH_values) + epsilon)

# Plot
plt.figure(figsize=(10,5))
plt.hist(log_H, bins=20, alpha=0.6, color='skyblue', density=True, label='H')
plt.hist(log_NH, bins=20, alpha=0.6, color='orange', density=True, label='NH')
plt.xlabel('Log(Granulometry Value)')
plt.ylabel('Density')
plt.title('Distribution of bin (7,8) for H vs NH (log scale)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()



from scipy.stats import ttest_ind

effect_sizes = np.zeros((21,21))
p_values = np.ones((21,21))

for i in range(21):
    for j in range(21):
        H_vals = np.array(bins_h[i][j])
        NH_vals = np.array(bins_nh[i][j])
        
        if len(H_vals) > 1 and len(NH_vals) > 1:
            # Log-transform (optional)
            H_log = np.log(H_vals + 1e-6)
            NH_log = np.log(NH_vals + 1e-6)
            
            # Pooled std
            n_H, n_NH = len(H_log), len(NH_log)
            s_H, s_NH = np.std(H_log, ddof=1), np.std(NH_log, ddof=1)
            s_pooled = np.sqrt(((n_H-1)*s_H**2 + (n_NH-1)*s_NH**2)/(n_H+n_NH-2))
            
            effect_sizes[i,j] = (np.mean(H_log) - np.mean(NH_log)) / s_pooled
            
            # t-test
            t_stat, p_val = ttest_ind(H_log, NH_log, equal_var=False)
            p_values[i,j] = p_val
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
from scipy.stats import ttest_ind

# Prepare list to store bin stats
bin_stats = []

for i in range(21):
    for j in range(21):
        H_vals = np.array(bins_h[i][j])
        NH_vals = np.array(bins_nh[i][j])
        
        if len(H_vals) > 1 and len(NH_vals) > 1:
            # Log-transform to reduce skew
            H_log = np.log(H_vals + 1e-6)
            NH_log = np.log(NH_vals + 1e-6)
            
            # Pooled standard deviation
            n_H, n_NH = len(H_log), len(NH_log)
            s_H, s_NH = np.std(H_log, ddof=1), np.std(NH_log, ddof=1)
            s_pooled = np.sqrt(((n_H-1)*s_H**2 + (n_NH-1)*s_NH**2) / (n_H + n_NH - 2))
            
            # Effect size (Cohen's d)
            effect_size = (np.mean(H_log) - np.mean(NH_log)) / s_pooled
            
            # t-test
            t_stat, p_val = ttest_ind(H_log, NH_log, equal_var=False)
            
            bin_stats.append({
                'bin_i': i,
                'bin_j': j,
                'effect_size': effect_size,
                'p_value': p_val
            })

# Convert to DataFrame
df_bins = pd.DataFrame(bin_stats)

# Sort by absolute effect size (descending)
df_bins['abs_effect_size'] = np.abs(df_bins['effect_size'])
df_sorted = df_bins.sort_values(by='abs_effect_size', ascending=False)

# Show top 10 most discriminative bins
print(df_sorted[['bin_i','bin_j','effect_size','p_value']].head(10))
# Initialize a 21x21 array for effect sizes
# Initialize a 21x21 array for effect sizes
effect_matrix = np.zeros((21, 21))

# Fill effect_matrix with the absolute effect sizes
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

def compute_zscore(flat):
    """Compute z-scores of a 1D numpy array, ignoring 0 values"""
    nonzero = flat[flat != 0]   # filter out zeros
    if len(nonzero) == 0:
        return np.array([])
    mean = np.mean(nonzero)
    std = np.std(nonzero)
    if std == 0:
        return np.zeros_like(nonzero)
    return (nonzero - mean) / std

# Collect files
files_h  = sorted(glob.glob("granulometryh*.m"))
files_nh = sorted(glob.glob("granulometrynh*.m"))

# --- Group 1: H files ---
all_h = []
for f in files_h:
    mat = load_granulometry_m_file(f)
    flat = mat.flatten()
    all_h.extend(flat[flat != 0])   # add only nonzero values

all_h = np.array(all_h)
z_h= compute_zscore(all_h)

print(f"Group H: {len(all_h)} values, mean(z)={np.mean(z_h):.3f}, std(z)={np.std(z_h):.3f}")

# --- Group 2: NH files ---
all_nh = []
for f in files_nh:
    mat = load_granulometry_m_file(f)
    flat = mat.flatten()
    all_nh.extend(flat[flat != 0])

all_nh = np.array(all_nh)
z_nh = compute_zscore(all_nh)

print(f"Group NH: {len(all_nh)} values, mean(z)={np.mean(z_nh):.3f}, std(z)={np.std(z_nh):.3f}")
# --- Plotting means ± std ---
groups = ['H', 'NH']
means = [np.mean(all_h), np.mean(all_nh)]
stds  = [np.std(all_h), np.std(all_nh)]

plt.figure(figsize=(6,5))
plt.bar(groups, means, yerr=stds, capsize=10, color=['skyblue','orange'])
plt.ylabel('Mean granulometry value')
plt.title('Mean ± STD for H and NH groups')
plt.grid(alpha=0.3, axis='y')
plt.show()

from scipy.stats import zscore
all_h_z = zscore(all_h)
all_nh_z = zscore(all_nh)
# --- Plotting histograms of z-scores ---
# --- Plot histograms ---
plt.figure(figsize=(10,5))
plt.hist(all_h_z, bins=500, alpha=0.6, label='H', density=True)
plt.hist(all_nh_z, bins=500, alpha=0.6, label='NH', density=True)
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.title('Distribution of H vs NH (Z-scores)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# Add small epsilon to avoid log(0)
epsilon = 1e-6
all_h_log = np.log(all_h + epsilon)
all_nh_log = np.log(all_nh + epsilon)

# --- Plot histograms of log-transformed values ---
plt.figure(figsize=(10,5))
plt.hist(all_h_log, bins=500, alpha=0.6, label='H', density=True)
plt.hist(all_nh_log, bins=500, alpha=0.6, label='NH', density=True)
plt.xlabel('Log(Value)')
plt.ylabel('Density')
plt.title('Log-Transformed Distribution of H vs NH')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(all_h_log, all_nh_log, equal_var=False)

print(f"T-test on log-transformed data: t-statistic={t_stat:.3f}, p-value={p_val:.3e}")
# # Histograms
# plt.hist(all_h, bins=100, alpha=0.6, label="H", density=True)
# plt.hist(all_nh, bins=100, alpha=0.6, label="NH", density=True)

# # Labels
# plt.xlabel("Z-score")
# plt.ylabel("Density")
# plt.title("Distribution of H vs NH")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()