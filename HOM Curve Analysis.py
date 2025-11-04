import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.signal import savgol_filter

# === Paths ===
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "Data")
file_names = [f for f in os.listdir(data_path) if f.endswith(".csv")]
data_dict = {}

# === Fit function ===
def fit_function(x, r, b, eta):
    t, T, L = x
    rp = r / (1 + (b * L * r) ** 2)
    rho = np.maximum(r - rp, 0)
    eta = (2*eta - 1)**2

    ans = (
        1 / 4
        * (
            (1 + eta)*erf(np.sqrt(rp / 2) * (T + t))
            + (1 + eta)*erf(np.sqrt(rp / 2) * (T - t))
            - (1 - eta)*np.exp(-r * t**2 / 2)
            * (
                erf(np.sqrt(rp / 2) * T + 1j * np.sqrt(rho / 2) * t)
                + erf(np.sqrt(rp / 2) * T - 1j * np.sqrt(rho / 2) * t)
            )
        )
    )
    ans[np.isnan(ans.real)] = 0 + 0j
    return ans.real

# === Collect all data ===
t_all, T_all, L_all, y_all, dataset_idx = [], [], [], [], []
dataset_counter = 0

for file_name in file_names:
    s = file_name.split()
    fiber_length = float(s[5].lstrip("length=").rstrip(".csv"))
    if fiber_length <= 0:
        continue

    abs_path = os.path.join(data_path, file_name)
    stage_position, coincidences, singles1, singles2 = np.genfromtxt(
        abs_path, delimiter=",", skip_header=1, unpack=True
    )
    coincidence_window = float(s[4].lstrip("ccwin="))

    coincidences /= np.max(coincidences)
    # 1. Find the index of the trough (minimum)
    N = int(len(coincidences)/10)
    if N // 2 == 0:
        N += 1
    trough_idx = np.argmin(savgol_filter(coincidences, N,1))

    center = stage_position[trough_idx]

    # 4. Center the data around the parabola minimum
    stage_position -= center
    stage_position *= 1e1 / 3.0  # ps

    max_stage_position = 9
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
        mask = (dataset_idx == idx)
        eta_k = eta_vec[int(idx)]

        y_pred_sub = fit_function((t[mask], T[mask], L[mask]), r, b, eta_k)
        y_sub = y[mask]

        # analytic per-dataset scale (same as your current approach)
        numerator = np.sum(y_pred_sub * y_sub)
        denominator = np.sum(y_pred_sub**2)
        scale = numerator / denominator if denominator != 0 else 1.0

        residuals[mask] = y_pred_sub * scale - y_sub

    return residuals


def objective(params, t, T, L, dataset_idx, y):
    resids = scaled_residuals(params, t, T, L, dataset_idx, y)
    return np.sum(resids**2)

# === Initial guess & bounds ===
p0 = [10, 10] + [0.5] * dataset_counter

# === Run optimization ===
res = least_squares(
    scaled_residuals, p0,  method="lm",
    args=(t_all, T_all, L_all, dataset_idx, y_all),
    # Optional robustness:
    # loss='soft_l1', f_scale=0.5
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

print("\n=== Global Fit Results (auto-scaled per dataset) ===")
print(f"r = {r:.4g} ± {r_err:.4g}")
print(f"b = {b:.4g} ± {b_err:.4g}")
for k, (e, ee) in enumerate(zip(eta_vec, eta_errs)):
    print(f"η[{k}] = {e:.4g} ± {ee:.4g}")


# === Plotting per dataset (with infimum scaling and shaded uncertainties) ===
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

    numerator = np.sum(y_pred * y)
    denominator = np.sum(y**2)
    scale = numerator / denominator if denominator != 0 else 1.0
    y_scaled = y * scale

    plt.figure(figsize=(4.5, 3.5), dpi=300)

    # Plot data points
    plt.plot(x_data[0], y_scaled, "+", color="black", label="data", markersize=4, markeredgewidth=1)

    # Plot fitted line
    plt.plot(x_data[0], y_pred, "-", color="red", label="fit", linewidth=1.5)

    # Axis labels and title
    plt.xlabel("Delay [ps]", fontsize=11)
    plt.ylabel("Coincidence count [-]", fontsize=11)

    # Axis span chosen based on data range with margins
    x_margin = 0.05 * (np.max(x_data[0]) - np.min(x_data[0]))
    y_margin = 0.05 * (np.max(y_scaled) - np.min(y_scaled))
    plt.xlim(np.min(x_data[0]) - x_margin, np.max(x_data[0]) + x_margin)
    plt.ylim(np.min(y_scaled) - y_margin, np.max(y_scaled) + y_margin)

    # Grid and ticks
    plt.grid(True, which='both', linestyle=':', linewidth=0.6)
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', top=True, right=True)
    plt.tick_params(axis='both', which='major', length=6, width=1)
    plt.tick_params(axis='both', which='minor', length=3, width=0.8)

    # Legend
    plt.legend(frameon=False, fontsize=9)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save figure
    os.makedirs("Figures", exist_ok=True)
    plt.savefig(f"Figures/Fit_{file_name}.png", dpi=500, bbox_inches="tight")
    plt.close()