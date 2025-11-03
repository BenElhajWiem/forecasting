# Performs statistical comparison across ablations.

# What it does:

# Reads the CSVs produced by ablate.py.
# Groups by experiment ID, computes deltas vs baseline (A0).
# Applies Wilcoxon signed-rank tests and bootstrap confidence intervals.
# Optionally produces summary tables (ΔMAE, p-values).

# Why important:
# Gives you the scientific validity of your results — shows which changes matter.