import matplotlib.pyplot as plt
import numpy as np

# Define the labels and the data
labels = ["TP", "FN", "FP", "TN", "2019-2022"]
dist_2 = [97.13, 88.78, 82.60, 40.23, 89.01]
dist_3 = [2.87, 11.22, 17.40, 59.60, 8.58]
dist_4 = [0.00, 0.00, 0.00, 0.17, 0.06]


def add_hatch(ax):
    bars = ax.patches
    bars[-1].set_hatch("//")


# Define the bar width and positions
barWidth = 0.25
r1 = np.arange(len(dist_2))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

hatches = ["", "", "", "", "//"]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plotting data on the first subplot
axs[1].bar(
    r1, dist_2, color="b", width=barWidth, edgecolor="black", label="2", hatch=hatches
)
axs[1].bar(
    r2, dist_3, color="r", width=barWidth, edgecolor="black", label="3", hatch=hatches
)
# axs[0].bar(
#     r3, dist_4, color="g", width=barWidth, edgecolor="black", label="4", hatch=hatches
# )
axs[1].axhline(y=89.01, color="k", linestyle="dashed", alpha=0.5)
axs[1].axhline(y=8.58, color="k", linestyle="dashed", alpha=0.5)
axs[1].set_title("Concept Embeddings")

plt.ylabel("Percentage %", fontweight="bold")

# Modify the data as per your second plot
dist_2_second_plot = [99.53, 82.98, 95.93, 38.80, 89.01]
dist_3_second_plot = [0.47, 17.02, 4.07, 60.97, 8.58]
dist_4_second_plot = [0.00, 0.00, 0.00, 0.23, 0.06]

# 99,53%	0,47%	0,00%
# 95,93%	4,07%	0,00%
# 82,98%	17,02%	0,00%
# 38,80%	60,97%	0,23%

# Plotting data on the second subplot
axs[0].bar(
    r1,
    dist_2_second_plot,
    color="b",
    width=barWidth,
    edgecolor="black",
    label="2",
    hatch=hatches,
)
axs[0].bar(
    r2,
    dist_3_second_plot,
    color="r",
    width=barWidth,
    edgecolor="black",
    label="3",
    hatch=hatches,
)
# axs[1].bar(
#     r3, dist_4_second_plot, color="g", width=barWidth, edgecolor="black", label="4"
# )
axs[0].axhline(y=89.01, color="k", linestyle="dashed", alpha=0.5)
axs[0].axhline(y=8.58, color="k", linestyle="dashed", alpha=0.5)
axs[0].set_title("Baseline")

# Setting the labels and title
plt.setp(
    axs, xticks=[r + barWidth * 0.5 for r in range(len(dist_2))], xticklabels=labels
)
plt.suptitle("Distance Distribution for Confusion Matrix and Real Data", fontsize=20)


# Adding the legend
plt.legend(title="Distance")

# Showing the plots
# plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
