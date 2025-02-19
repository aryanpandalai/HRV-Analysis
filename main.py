import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.signal import welch
from scipy.spatial.distance import pdist
import nolds

file_path = "/Users/aryanpandalai/Desktop/PycharmProjects/fNIRS/Heartrate_Variability_Analysis/HRV_Analysis_Data.csv"
df = pd.read_csv(file_path)
df = df.dropna(axis=1, how="all") # getting rid of columns & rows w/ no data
df = df.iloc[2:].reset_index(drop=True) # instead of deleting rows from raw file, just implementing here
df["Time"] = pd.to_timedelta(df["Time"]) - pd.Timedelta(seconds=1) # simplicity to start times at first second
# Convert back to time format (if necessary)
df["Time"] = df["Time"].astype(str)
df.to_csv("cleaned_HRV_Analysis_Data.csv", index=False)
rr_intervals = 60000 / df["HR (bpm)"].dropna().values
# Function for Time-Domain Metrics
def compute_time_domain(rr_intervals):
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_intervals, ddof=1)
    mean_nn = np.mean(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100

    return {
        "RMSSD (ms)": rmssd,
        "SDNN (ms)": sdnn,
        "Mean NN Interval (ms)": mean_nn,
        "NN50 count": nn50,
        "pNN50 (%)": pnn50
    }


# Function for Frequency-Domain Metrics
def compute_frequency_domain(rr_intervals, fs=4):
    freqs, psd = welch(rr_intervals, fs=fs, nperseg=len(rr_intervals) // 2)

    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    lf_power = np.trapezoid(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])],
                        freqs[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
    hf_power = np.trapezoid(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])],
                        freqs[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    return {
        "LF Power (ms²)": lf_power,
        "HF Power (ms²)": hf_power,
        "LF/HF Ratio": lf_hf_ratio
    }, freqs, psd


# Function for Sample Entropy
def sample_entropy(U, m=2, r=0.2):
    U = np.array(U)
    N = len(U)

    def _phi(m):
        x = np.array([U[i: i + m] for i in range(N - m + 1)])
        C = np.sum(np.less_equal(pdist(x, metric="chebyshev"), r), axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)

    return _phi(m) - _phi(m + 1)


# Compute Metrics
time_domain_metrics = compute_time_domain(rr_intervals)
freq_domain_metrics, freqs, psd = compute_frequency_domain(rr_intervals)
sampen = sample_entropy(rr_intervals)
dfa_alpha1 = nolds.dfa(rr_intervals)
variance = np.var(rr_intervals)
cv = np.std(rr_intervals) / np.mean(rr_intervals)

# Converting results to dataframe for better visualization
metrics_df = pd.DataFrame({
    "Metric": ["RMSSD", "SDNN", "Mean NN Interval", "NN50", "pNN50", "LF Power", "HF Power", "LF/HF Ratio",
               "Sample Entropy", "DFA α1", "Variance", "Coefficient of Variation"],
    "Value": [time_domain_metrics["RMSSD (ms)"], time_domain_metrics["SDNN (ms)"],
              time_domain_metrics["Mean NN Interval (ms)"],
              time_domain_metrics["NN50 count"], time_domain_metrics["pNN50 (%)"],
              freq_domain_metrics["LF Power (ms²)"],
              freq_domain_metrics["HF Power (ms²)"], freq_domain_metrics["LF/HF Ratio"], sampen, dfa_alpha1, variance,
              cv]
})

# Print HRV Metrics Table
col1_width = 30  # dependent on the longest metric name
col2_width = 20  # dependent longest value

# table header
print("\n  Heart Rate Variability Metrics  \n")
print(f"{'Metric'.ljust(col1_width)} {'Value'.ljust(col2_width)}")
print("-" * (col1_width + col2_width))

for index, row in metrics_df.iterrows():
    metric = str(row["Metric"]).ljust(col1_width)
    value = str(row["Value"]).ljust(col2_width)
    print(f"{metric}{value}")


#Plots

# Poincaré Plot
def plot_poincare(rr_intervals):
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=rr_n, y=rr_n1, alpha=0.5)
    plt.xlabel("RR(n) (ms)")
    plt.ylabel("RR(n+1) (ms)")
    plt.title("Poincaré Plot")
    plt.grid(True)
    plt.show()


# Power Spectral Density Plot
def plot_psd(freqs, psd):
    plt.figure(figsize=(8, 5))
    plt.semilogy(freqs, psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (ms²/Hz)")
    plt.title("Power Spectral Density (Welch’s Method)")
    plt.grid(True)
    plt.show()


# RR Interval Trend Plot
def plot_rr_intervals(rr_intervals):
    plt.figure(figsize=(8, 5))
    plt.plot(rr_intervals, marker="o", linestyle="-", markersize=2, alpha=0.7)
    plt.xlabel("Beat Number")
    plt.ylabel("RR Interval (ms)")
    plt.title("RR Interval Trends")
    plt.grid(True)
    plt.show()


plot_rr_intervals(rr_intervals)
plot_poincare(rr_intervals)
plot_psd(freqs, psd)