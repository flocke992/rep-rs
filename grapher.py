# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# mpl.use('pgf')
mpl.rcParams.update({
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    "font.family": "serif",
    'lines.linewidth': 1.0,
     'pgf.preamble': r'\usepackage{amssymb}',
})

# %%
ex1 = pd.read_csv("30_50_exact.csv")
ex1_omp = ex1.loc[ex1['algorithm'] == 'omp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()
ex1_ls_omp = ex1.loc[ex1['algorithm'] == 'ls-omp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()
ex1_mp = ex1.loc[ex1['algorithm'] == 'mp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()
ex1_weak_mp = ex1.loc[ex1['algorithm'] == 'weak-mp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()
ex1_thresh = ex1.loc[ex1['algorithm'] == 'thresholding', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()

# Performance1
# %%
fig, ax = plt.subplots(3, figsize=(5,12), sharex=True)

ax[0].plot(ex1_omp.index, ex1_omp.l2_recovery_error, marker=".", label="omp")
ax[0].plot(ex1_ls_omp.index, ex1_ls_omp.l2_recovery_error, marker="o", label="ls-omp")
ax[0].plot(ex1_mp.index, ex1_mp.l2_recovery_error, marker="x", label="mp")
ax[0].plot(ex1_weak_mp.index, ex1_weak_mp.l2_recovery_error, marker="*", label="weak-mp")
ax[0].plot(ex1_thresh.index, ex1_thresh.l2_recovery_error, marker="*", label="thresholding")

ax[0].set_ylim([0.0, 0.2])
ax[0].set_ylabel("Average relative $l_2$ error")
ax[0].set_title("recovery error")
ax[0].legend()
ax[0].grid()

ax[1].plot(ex1_omp.index, ex1_omp.solution_support_distance, marker=".", label="omp")
ax[1].plot(ex1_ls_omp.index, ex1_ls_omp.solution_support_distance, marker="o", label="ls-omp")
ax[1].plot(ex1_mp.index, ex1_mp.solution_support_distance, marker="x", label="mp")
ax[1].plot(ex1_weak_mp.index, ex1_weak_mp.solution_support_distance, marker="*", label="weak-mp")
ax[1].plot(ex1_thresh.index, ex1_thresh.solution_support_distance, marker="*", label="thresholding")
  
ax[1].set_ylabel("Average solution support distance")
ax[1].set_title("True support recovery")
ax[1].legend()
ax[1].grid()

ax[2].plot(ex1_omp.index, ex1_omp.time, marker=".", label="omp")
ax[2].plot(ex1_ls_omp.index, ex1_ls_omp.time, marker="o", label="ls-omp")
ax[2].plot(ex1_mp.index, ex1_mp.time, marker="x", label="mp")
ax[2].plot(ex1_weak_mp.index, ex1_weak_mp.time, marker="*", label="weak-mp")
ax[2].plot(ex1_thresh.index, ex1_thresh.time, marker="*", label="thresholding")
  
ax[2].set_xlabel("Cardinality of true solution")
ax[2].set_ylabel("Average compute time in $s$")
ax[2].set_title("Performance speed")
ax[2].legend()
ax[2].grid()

fig.suptitle("Greedy algorithm performance")
fig.tight_layout()
fig.savefig("performance1")

# Performance2
# %%
ex2 = pd.read_csv("60_100_exact.csv")
ex2_omp = ex2.loc[ex2['algorithm'] == 'omp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()
ex2_ls_omp = ex2.loc[ex2['algorithm'] == 'ls-omp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()
ex2_mp = ex2.loc[ex2['algorithm'] == 'mp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()
ex2_weak_mp = ex2.loc[ex2['algorithm'] == 'weak-mp', ["cardinality", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["cardinality"]).mean()


# %%
fig, ax = plt.subplots(3, figsize=(5,12), sharex=True)

ax[0].plot(ex2_omp.index, ex2_omp.l2_recovery_error, marker=".", label="omp")
ax[0].plot(ex2_ls_omp.index, ex2_ls_omp.l2_recovery_error, marker="o", label="ls-omp")
ax[0].plot(ex2_mp.index, ex2_mp.l2_recovery_error, marker="x", label="mp")
ax[0].plot(ex2_weak_mp.index, ex2_weak_mp.l2_recovery_error, marker="*", label="weak-mp")

ax[0].set_ylabel("Average relative $L_2$ error")
ax[0].set_title("recovery error")
ax[0].legend()
ax[0].grid()

ax[1].plot(ex2_omp.index, ex2_omp.solution_support_distance, marker=".", label="omp")
ax[1].plot(ex2_ls_omp.index, ex2_ls_omp.solution_support_distance, marker="o", label="ls-omp")
ax[1].plot(ex2_mp.index, ex2_mp.solution_support_distance, marker="x", label="mp")
ax[1].plot(ex2_weak_mp.index, ex2_weak_mp.solution_support_distance, marker="*", label="weak-mp")
  
ax[1].set_ylabel("Average solution support distance")
ax[1].set_title("True support recovery")
ax[1].legend()
ax[1].grid()

ax[2].plot(ex2_omp.index, ex2_omp.time, marker=".", label="omp")
ax[2].plot(ex2_ls_omp.index, ex2_ls_omp.time, marker="o", label="ls-omp")
ax[2].plot(ex2_mp.index, ex2_mp.time, marker="x", label="mp")
ax[2].plot(ex2_weak_mp.index, ex2_weak_mp.time, marker="*", label="weak-mp")
  
ax[2].set_xlabel("Cardinality of true solution")
ax[2].set_ylabel("Average compute time in $s$")
ax[2].set_title("Performance speed")
ax[2].legend()
ax[2].grid()

fig.suptitle("Greedy algorithm performance")
fig.tight_layout()
fig.savefig("performance2.png")


ex3 = pd.read_csv("60_100_thresholds.csv")
ex3_omp = ex3.loc[ex3['algorithm'] == 'omp', ["threshold", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["threshold"]).mean()
ex3_ls_omp = ex3.loc[ex3['algorithm'] == 'ls-omp', ["threshold", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["threshold"]).mean()
ex3_mp = ex3.loc[ex3['algorithm'] == 'mp', ["threshold", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["threshold"]).mean()
ex3_weak_mp = ex3.loc[ex3['algorithm'] == 'weak-mp', ["threshold", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["threshold"]).mean()
ex3_thresh = ex3.loc[ex3['algorithm'] == 'thresholding', ["threshold", "time", "l2_recovery_error", "solution_support_distance"]].groupby(["threshold"]).mean()


# %%
fig, ax = plt.subplots(3, figsize=(5,12), sharex=True)

ax[0].plot(ex3_omp.index, ex3_omp.l2_recovery_error, marker=".", label="omp")
ax[0].plot(ex3_ls_omp.index, ex3_ls_omp.l2_recovery_error, marker="o", label="ls-omp")
ax[0].plot(ex3_mp.index, ex3_mp.l2_recovery_error, marker="x", label="mp")
ax[0].plot(ex3_weak_mp.index, ex3_weak_mp.l2_recovery_error, marker="*", label="weak-mp")

ax[0].set_ylabel("Average relative $L_2$ error")
ax[0].set_title("recovery error")
ax[0].set_xscale("log")
ax[0].legend()
ax[0].grid()

ax[1].plot(ex3_omp.index, ex3_omp.solution_support_distance, marker=".", label="omp")
ax[1].plot(ex3_ls_omp.index, ex3_ls_omp.solution_support_distance, marker="o", label="ls-omp")
ax[1].plot(ex3_mp.index, ex3_mp.solution_support_distance, marker="x", label="mp")
ax[1].plot(ex3_weak_mp.index, ex3_weak_mp.solution_support_distance, marker="*", label="weak-mp")
  
ax[1].set_ylabel("Average solution support distance")
ax[1].set_title("True support recovery")
ax[1].set_xscale("log")
ax[1].legend()
ax[1].grid()

ax[2].plot(ex3_omp.index, ex3_omp.time, marker=".", label="omp")
ax[2].plot(ex3_ls_omp.index, ex3_ls_omp.time, marker="o", label="ls-omp")
ax[2].plot(ex3_mp.index, ex3_mp.time, marker="x", label="mp")
ax[2].plot(ex3_weak_mp.index, ex3_weak_mp.time, marker="*", label="weak-mp")
  
ax[2].set_xlabel("residual error threshold")
ax[2].set_ylabel("Average compute time in $s$")
ax[2].set_title("Performance speed")
ax[2].set_xscale("log")
ax[2].legend()
ax[2].grid()

fig.suptitle("Greedy algorithm performance")
fig.tight_layout()
fig.savefig("performance3.png")
