import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# CONFIGURATION
# ================================

DETAILS_FILE = "evaluation_output/detailed_results_20260220_152342.json"
SUMMARY_FILE = "evaluation_output/summary_20260220_152342.json"

OUTPUT_DIR = "evaluation_output/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")


# ================================
# LOAD DATA
# ================================

with open(DETAILS_FILE, "r", encoding="utf-8") as f:
    details = json.load(f)

with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    summary = json.load(f)

df = pd.DataFrame(details)

latencies = df["latency"]
accuracies = df["accuracy"]
sources_used = df["sources_used"]
grounded = df["grounded"]


# ================================
# GRAPH 1: LATENCY HISTOGRAM
# ================================

plt.figure(figsize=(10,6))
sns.histplot(latencies, bins=10, kde=True)
plt.title("Latency Distribution")
plt.xlabel("Latency (seconds)")
plt.ylabel("Frequency")

plt.savefig(f"{OUTPUT_DIR}/latency_histogram.png")
plt.close()


# ================================
# GRAPH 2: ACCURACY DISTRIBUTION
# ================================

plt.figure(figsize=(8,5))
sns.countplot(x=accuracies)
plt.title("Answer Accuracy Distribution")
plt.xlabel("Accuracy Score")
plt.ylabel("Count")

plt.savefig(f"{OUTPUT_DIR}/accuracy_distribution.png")
plt.close()


# ================================
# GRAPH 3: LATENCY PERCENTILES
# ================================

percentiles = [50, 75, 90, 95, 99]
values = np.percentile(latencies, percentiles)

plt.figure(figsize=(8,5))
plt.plot(percentiles, values, marker="o")

plt.title("Latency Percentiles")
plt.xlabel("Percentile")
plt.ylabel("Latency (seconds)")

plt.savefig(f"{OUTPUT_DIR}/latency_percentiles.png")
plt.close()


# ================================
# GRAPH 4: SOURCES USED DISTRIBUTION
# ================================

plt.figure(figsize=(8,5))
sns.countplot(x=sources_used)
plt.title("Sources Used per Query")
plt.xlabel("Sources Used")
plt.ylabel("Count")

plt.savefig(f"{OUTPUT_DIR}/sources_used_distribution.png")
plt.close()


# ================================
# GRAPH 5: SYSTEM PERFORMANCE DASHBOARD
# ================================

fig, axs = plt.subplots(2, 2, figsize=(12,10))

sns.histplot(latencies, bins=10, ax=axs[0,0])
axs[0,0].set_title("Latency Distribution")

sns.countplot(x=accuracies, ax=axs[0,1])
axs[0,1].set_title("Accuracy Distribution")

sns.countplot(x=sources_used, ax=axs[1,0])
axs[1,0].set_title("Sources Used")

grounded_counts = df["grounded"].value_counts()
axs[1,1].bar(["Grounded", "Not Grounded"],
             [grounded_counts.get(True,0), grounded_counts.get(False,0)])

axs[1,1].set_title("Grounded Answers")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/system_dashboard.png")
plt.close()


# ================================
# GRAPH 6: LATENCY VS QUERY INDEX
# ================================

plt.figure(figsize=(10,5))
plt.plot(latencies.values, marker="o")
plt.title("Latency per Query")
plt.xlabel("Query Index")
plt.ylabel("Latency (seconds)")

plt.savefig(f"{OUTPUT_DIR}/latency_per_query.png")
plt.close()


print("\nGraphs generated successfully.")
print(f"Saved in: {OUTPUT_DIR}")