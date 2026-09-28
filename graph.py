import matplotlib.pyplot as plt
import numpy as np

# ROUGE scores before and after improvement
metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
before_values = [0.9077, 0.8407, 0.8913]
after_values = [0.9139, 0.8497, 0.8977]

x = np.arange(len(metrics))  # the label locations
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bar_width/2, before_values, bar_width, label="Before", color="skyblue")
ax.bar(x + bar_width/2, after_values, bar_width, label="After", color="orange")

ax.set_xlabel("ROUGE Metrics")
ax.set_ylabel("Scores")
ax.set_title("Improvement in ROUGE Scores")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.ylim(0.8, 0.92)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
