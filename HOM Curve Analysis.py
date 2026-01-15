import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from scipy.special import erf, erfcx
from scipy.signal import savgol_filter

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# === Paths ===
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "Data")
file_names = [f for f in os.listdir(data_path) if f.endswith(".csv")]
data_dict = {}


def exp_erf_stable(z1, z2):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ez1 = np.exp(z1)
        return ez1 - np.exp(z1 - z2*z2) * erfcx(z2)

# === Fit function (stable) ===
def fit_function(x, r, b, eta):
    t, T, L = x

    rp = r / (1 + (b * L * r) ** 2)
    rho = np.maximum(r - rp, 0.0)
    eta2 = (2 * eta - 1) ** 2  # keep your mapping

    # Real-only erf terms (these are stable)
    a = np.sqrt(rp / 2.0)
    term_real = (1 + eta2) * (erf(a * (T + t)) + erf(a * (T - t)))

    # Build the complex argument for erf(...)
    # z = sqrt(rp/2)*T + i*sqrt(rho/2)*t
    z = a * T + 1j * np.sqrt(rho / 2.0) * t

    # We need: exp(-r t^2/2) * ( erf(z) + erf(conj(z)) )
    # Since exp(-r t^2/2) is real, this equals:
    # 2 * Re( exp(-r t^2/2) * erf(z) )
    z1 = -0.5 * r * t**2  # log prefactor

    # Stable evaluation of exp(z1) * erf(z), then take 2*Re(...)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        exp_erf_val = exp_erf_stable(z1, z)          # complex array
        complex_block = 2.0 * np.real(exp_erf_val)   # real array

    term_complex = (1 - eta2) * complex_block

    ans = 0.25 * (term_real - term_complex)

    # Final cleanup (just in case)
    ans = np.where(np.isfinite(ans), ans, 0.0)
    return ans


# === FWHM function ===
def fwhm_interp(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of the same length")

    y_work = -y

    i0 = int(np.argmax(y_work))           # location of peak in y_work
    y0 = y_work[i0]
    yb = np.min(y_work)                   # baseline estimate (simple)
    half = yb + 0.5 * (y0 - yb)           # half-maximum level

    # Find left crossing: last index < i0 where y crosses half
    left_idxs = np.where(y_work[:i0] < half)[0]
    if left_idxs.size == 0:
        raise ValueError("No left half-maximum crossing found")
    i1 = left_idxs[-1]     # above half
    i2 = i1 + 1            # below half (closer to trough)

    # Linear interpolation for left crossing
    x_left = x[i1] + (half - y_work[i1]) * (x[i2] - x[i1]) / (y_work[i2] - y_work[i1])

    # Find right crossing: first index > i0 where y crosses half
    right_idxs = np.where(y_work[i0+1:] < half)[0]
    if right_idxs.size == 0:
        raise ValueError("No right half-maximum crossing found")
    j2 = i0 + 1 + right_idxs[0]  # above half
    j1 = j2 - 1                  # below half

    # Linear interpolation for right crossing
    x_right = x[j1] + (half - y_work[j1]) * (x[j2] - x[j1]) / (y_work[j2] - y_work[j1])

    return x_right - x_left

# === Collect all data ===
t_all, T_all, L_all, y_all, dataset_idx = [], [], [], [], []
dataset_counter = 0

for file_name in file_names:
    s = file_name.split()
    fiber_length = float(s[5].lstrip("length=").rstrip(".csv"))
    exp_time = float(s[3].lstrip("exptime="))
    if fiber_length <= 0:
        continue

    abs_path = os.path.join(data_path, file_name)
    stage_position, coincidences, singles1, singles2 = np.genfromtxt(
        abs_path, delimiter=",", skip_header=1, unpack=True
    )
    coincidence_window = float(s[4].lstrip("ccwin="))

    coincidences /= np.max(coincidences)
    # 1. Find the index of the trough (minimum)
    N = int(len(coincidences) / 10)
    if N // 2 == 0:
        N += 1
    trough_idx = np.argmin(savgol_filter(coincidences, N, 1))

    center = stage_position[trough_idx]

    # 4. Center the data around the parabola minimum
    stage_position -= center
    stage_position *= 1e1 / 2.998  # ps

    max_stage_position = 20
    bm = np.logical_and(stage_position <= max_stage_position, stage_position >= -max_stage_position)
    stage_position = stage_position[bm]
    coincidences = coincidences[bm]

    n_points = len(stage_position)
    t_all.append(stage_position)
    T_all.append(np.full(n_points, coincidence_window))
    L_all.append(np.full(n_points, fiber_length))
    y_all.append(coincidences)
    dataset_idx.append(np.full(n_points, dataset_counter))

    data_dict[file_name] = {
        "dataset_index": dataset_counter,
        "fiber_length": fiber_length,
        "coincidence_window": coincidence_window,
        "stage_position": stage_position,
        "coincidences": coincidences,
        "exposure_time": exp_time,
    }

    dataset_counter += 1

# stack all into 1D arrays
t_all = np.concatenate(t_all)
T_all = np.concatenate(T_all)
L_all = np.concatenate(L_all)
y_all = np.concatenate(y_all)
dataset_idx = np.concatenate(dataset_idx)

# === Residuals with automatic scaling ===
def scaled_residuals(params, t, T, L, dataset_idx, y):
    r, b = params[0], params[1]
    eta_vec = np.asarray(params[2:])  # length = n_datasets

    residuals = np.zeros_like(y)

    # loop by dataset to use dataset-specific η and analytic scale
    for idx in np.unique(dataset_idx):
        mask = dataset_idx == idx
        eta_k = eta_vec[int(idx)]

        y_pred_sub = fit_function((t[mask], T[mask], L[mask]), r, b, eta_k)
        y_sub = y[mask]

        # analytic per-dataset scale (same as your current approach)
        numerator = np.sum(y_pred_sub * y_sub)
        denominator = np.sum(y_pred_sub**2)
        scale = numerator / denominator if denominator != 0 else 1.0

        residuals[mask] = y_pred_sub * scale - y_sub

    return residuals

# === Initial guess & bounds ===
p0 = [10, 10] + [0.5] * dataset_counter

# === Run optimization ===
res = least_squares(
    scaled_residuals,
    p0,
    method="lm",
    args=(t_all, T_all, L_all, dataset_idx, y_all),
)
fit_params = res.x
jacobian = res.jac

# === Estimate parameter uncertainties ===
residuals = scaled_residuals(fit_params, t_all, T_all, L_all, dataset_idx, y_all)
dof = len(residuals) - len(fit_params)
res2 = np.sum(residuals**2)
sigma2 = res2 / max(dof, 1)

if jacobian is not None:
    cov = np.linalg.inv(jacobian.T @ jacobian) * sigma2
    param_errors = np.sqrt(np.diag(cov))
else:
    cov = None
    param_errors = np.full_like(fit_params, np.nan, dtype=float)

r, b = fit_params[0], fit_params[1]
r_err, b_err = param_errors[0], param_errors[1]
eta_vec = fit_params[2:]
eta_errs = param_errors[2:]

print("\n=== Global Fit Results ===")
print(f"r = {r:.4g} ± {r_err:.4g} 1/ps²")
print(f"β₂ = {b:.4g} ± {b_err:.4g} ps²/km")

# === Collect per-dataset parameters ===
L_list = []
T_list = []
eta_list = []
eta_err_list = []
s_list = []
exp_list = [] 

for file_name, data in data_dict.items():
    idx = data["dataset_index"]
    L_i = data["fiber_length"]
    T_i = data["coincidence_window"]
    exp_i = data["exposure_time"]
    eta_i = eta_vec[idx]
    eta_i_err = eta_errs[idx]

    # Compute per-dataset analytic scale
    mask = dataset_idx == idx
    y_pred_sub = fit_function((t_all[mask], T_all[mask], L_all[mask]), r, b, eta_i)
    y_sub = y_all[mask]
    numerator = np.sum(y_pred_sub * y_sub)
    denominator = np.sum(y_pred_sub**2)
    s_i = numerator / denominator if denominator != 0 else 1.0

    L_list.append(L_i)
    T_list.append(T_i)
    eta_list.append(eta_i)
    eta_err_list.append(eta_i_err)
    s_list.append(s_i)
    exp_list.append(exp_i)

# === Print per-dataset summary ===
print("\n=== Per-Dataset Parameters ===")
print(f"{'Idx':>3} | {'L (km)':>8} | {'T (ps)':>8} | {'η':>8} ± {'ση':<8} | {'s_i':>8} | {'exp_time (ms)':>10}")
print("-" * 65)
for k, (L_i, T_i, eta_i, eta_err_i, s_i, exp_i) in enumerate(
    zip(L_list, T_list, eta_list, eta_err_list, s_list, exp_list)
):
    print(f"{k:>3} | {L_i:8.3f} | {T_i:8.3f} | {eta_i:8.4f} ± {eta_err_i:<8.4f} | {s_i:8.4f} | {str(exp_i):>10}")

# === Plotting per dataset ===
for file_name, data in data_dict.items():
    idx = data["dataset_index"]
    r, b = fit_params[0], fit_params[1]
    eta_k = fit_params[2 + idx]
    eta_k_err = eta_errs[idx]

    x_data = (
        data["stage_position"],
        np.full_like(data["stage_position"], data["coincidence_window"]),
        np.full_like(data["stage_position"], data["fiber_length"]),
    )
    y = data["coincidences"]

    y_pred = fit_function(x_data, r, b, eta_k)

    data["coincidences_predicted"] = y_pred

    numerator = np.sum(y_pred * y)
    denominator = np.sum(y**2)
    scale = numerator / denominator if denominator != 0 else 1.0
    y_scaled = y * scale

    data["coincidences_scaled"] = y_scaled

    data_dict[file_name] = data

    plt.figure(figsize=(4.5, 3.5), dpi=300)

    # Plot data points
    plt.plot(x_data[0], y_scaled, "+", color="black", label="data", markersize=4, markeredgewidth=1)

    # Plot fitted line
    plt.plot(x_data[0], y_pred, "-", color="red", label="fit", linewidth=1.5)

    plt.xlabel("Delay [ps]", fontsize=18)
    plt.ylabel("Coincidence count [-]", fontsize=18)

    x_margin = 0.05 * (np.max(x_data[0]) - np.min(x_data[0]))
    y_margin = 0.05 * (np.max(y_scaled) - np.min(y_scaled))
    plt.xlim(np.min(x_data[0]) - x_margin, np.max(x_data[0]) + x_margin)
    plt.ylim(np.min(y_scaled) - y_margin, np.max(y_scaled) + y_margin)

    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.minorticks_on()
    plt.tick_params(which="both", direction="in", top=True, right=True)
    plt.tick_params(axis="both", which="major", length=6, width=1)
    plt.tick_params(axis="both", which="minor", length=3, width=0.8)

    plt.legend(frameon=False, fontsize=14)
    plt.tight_layout()

    os.makedirs("Figures", exist_ok=True)
    plt.savefig(f"Figures/Fit_{file_name}.png", dpi=500, bbox_inches="tight")
    plt.close()


# === Single plot: FWHM vs L, multiple T (colored) ===
f_vals, T_vals, L_vals = [], [], []

# dataset names (keys in data_dict) to exclude
exclude_datasets = {
    "Tieme 2025-07-16 11.48 exptime=200 ccwin=1000 length=29.csv",
    "Tieme 2025-07-16 13.53 exptime=500 ccwin=100 length=10.csv"
}

r, b = fit_params[0], fit_params[1]
taup = np.linspace(-50, 50, 20000)
for file_name, data in data_dict.items():
    if file_name in exclude_datasets:
        continue

    tau_i = data["stage_position"]
    T_i   = data["coincidence_window"]   # ps
    L_i   = data["fiber_length"]          # km
    c_i   = data["coincidences_scaled"]

    f_vals.append(fwhm_interp(tau_i, c_i))
    T_vals.append(T_i)
    L_vals.append(L_i)

T_vals = np.array(T_vals)
L_vals = np.array(L_vals)
f_vals = np.array(f_vals)

Ts = np.unique(T_vals)

L_min = 0
L_max = np.max(L_vals) + 1

L_grid_global = np.linspace(L_min, L_max, 400)  # prediction range for ALL T

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

# optional: color map for many curves
cmap = plt.get_cmap("tab10")  # or "viridis"
colors = [cmap(i % cmap.N) for i in range(len(Ts))]

for k, T in enumerate(Ts):
    mask = (T_vals == T)
    L_data = L_vals[mask]
    f_data = f_vals[mask]

    # sort data points by L (important for clean plotting)
    order = np.argsort(L_data)
    L_data = L_data[order]
    f_data = f_data[order]

    if len(L_data) > 0:
        # predicted FWHM(L)
        fp_grid = np.empty_like(L_grid_global, dtype=float)
        for j, Lg in enumerate(L_grid_global):
            x_pred = (
                taup,
                np.full_like(taup, T, dtype=float),
                np.full_like(taup, Lg, dtype=float),
            )
            cp = fit_function(x_pred, r, b, 1 / 2)
            fp_grid[j] = fwhm_interp(taup, cp)

        col = colors[k]

        # measured points
        ax.plot(
            L_data,
            f_data,
            "+",
            color=col,
            markersize=6,
            markeredgewidth=2,
            alpha=0.85,
        )

        # predicted curve
        ax.plot(
            L_grid_global,
            fp_grid,
            "-",
            color=col,
            label=f"T = {T/1000:.3g} ns",
            linewidth=1.5,
            alpha=0.85,
        )

# axes formatting
ax.set_xlim(L_min, L_max)
ax.set_ylim(0.5, 3.5)

ax.set_xlabel("Fiber Length [km]", fontsize=18)
ax.set_ylabel("FWHM [ps]", fontsize=18)

ax.grid(True, which="both", linestyle=":", linewidth=0.6)
ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.tick_params(axis="both", which="major", length=6, width=1)
ax.tick_params(axis="both", which="minor", length=3, width=0.8)

ax.set_yscale("linear")

# -------------------------
# Legend on top (3 x 3 grid)
# -------------------------
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncols=3,
    frameon=False,
    fontsize=13,
    columnspacing=1.2,
    handlelength=2.2,
)

# leave room for top legend
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("Figures/FWHM_all_T.png", dpi=500, bbox_inches="tight")
plt.close()

# === 2D Scatter plot: T RMS residuals per L ===
rms_residuals = []
T_vals = []
L_vals = []

for file_name, data in data_dict.items():
    idx = data["dataset_index"]
    T_i = data["coincidence_window"] / 1000  # ns
    L_i = data["fiber_length"]               # km

    mask = dataset_idx == idx
    res_i = residuals[mask]

    rms_i = np.sqrt(np.mean(res_i**2))

    rms_residuals.append(rms_i)
    T_vals.append(T_i)
    L_vals.append(L_i)

T_vals = np.array(T_vals)
L_vals = np.array(L_vals)
rms_residuals = np.array(rms_residuals)

# To be plotted fiber lengths
L_plot = [5.0, 10.0]

# Not to be plotted coincidence rates
T_plot = [1.0]

# Plot
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

for i, L in enumerate(L_plot):
    mask = np.logical_and(L_vals == L, T_vals != T_plot)
    ax.scatter(
        T_vals[mask],
        rms_residuals[mask],
        s=70,
        edgecolor="k",
        alpha=0.85,
        label=fr"$L = {L}$ km"
    )

ax.set_xlabel(r"$T$ [ns]", fontsize=18)
ax.set_ylabel("RMS Residual", fontsize=18)

ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

csv_path = os.path.join(base_path, "dataset_fit_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # --- Header ---
    writer.writerow([
        "Entry Type",
        "Dataset Index",
        "Fiber Length (km)",
        "Coincidence Window (ps)",
        "η",
        "ση",
        "Scale Factor s_i",
        "Exposure Time (ms)",
        "N Points",
        "RMS Residual",
        "r (1/ps^2)",
        "σ_r",
        "beta2 (ps^2/km)",
        "σ_beta2"
    ])

    # --- Global parameters row ---
    writer.writerow([
        "GLOBAL", "", "", "", "", "", "", "", "", "",
        r, r_err, b, b_err
    ])

    # --- Compute RMS residuals per dataset ---
    rms_residuals_dict = {}
    for file_name, data in data_dict.items():
        idx = data["dataset_index"]
        mask = dataset_idx == idx
        res_i = residuals[mask]
        rms_residuals_dict[idx] = np.sqrt(np.mean(res_i**2))

    # --- Per-dataset rows ---
    for file_name, data in data_dict.items():
        idx = data["dataset_index"]

        L_i = data["fiber_length"]
        T_i = data["coincidence_window"]
        exp_i = data["exposure_time"]
        eta_i = eta_vec[idx]
        eta_err_i = eta_errs[idx]

        # Analytic scale factor
        mask = dataset_idx == idx
        y_pred_sub = fit_function(
            (t_all[mask], T_all[mask], L_all[mask]),
            r, b, eta_i
        )
        y_sub = y_all[mask]
        numerator = np.sum(y_pred_sub * y_sub)
        denominator = np.sum(y_pred_sub**2)
        s_i = numerator / denominator if denominator != 0 else 1.0

        writer.writerow([
            "DATASET",
            idx,
            L_i,
            T_i,
            eta_i,
            eta_err_i,
            s_i,
            exp_i,
            np.sum(mask),         # Number of points
            rms_residuals_dict[idx],
            "", "", "", ""        # Global columns empty
        ])

print(f"\nCSV summary written to: {csv_path}")