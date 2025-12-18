#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# c = 299792.458 nm/ps
C_NM_PER_PS = 299792.458
TWO_PI = 2.0 * np.pi


def lambda_nm_to_omega_radps(lambda_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength (nm) to angular frequency ω (rad/ps)."""
    return TWO_PI * C_NM_PER_PS / lambda_nm


def gaussian_y_omega(omega, A, mu, s):
    """Gaussian transmittance in angular frequency with s = sigma^2."""
    return A * np.exp(-((omega - mu) ** 2) / (2*s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input CSV path")
    ap.add_argument("-o", "--output", default="gaussian_fit_parameters.csv")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    lambda_nm = df["Wavelength (nm)"].to_numpy()
    if np.any(lambda_nm <= 0):
        raise ValueError("Wavelength values must be > 0.")

    T = np.clip(df["Transmission (%)"].to_numpy() / 100.0, 0.0, 1.0)
    y = np.sqrt(T)

    # Convert to angular frequency
    omega = lambda_nm_to_omega_radps(lambda_nm)

    p0 = [1, lambda_nm_to_omega_radps(1550), 10]

    bounds = (
        [0.0, omega.min(), 1e-30], 
        [1.0, omega.max(), np.inf],
    )

    popt, _ = curve_fit(
        gaussian_y_omega,
        omega,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
    )

    A, mu, s = popt
    sigma_omega = np.sqrt(s)
    fwhm_omega = 2.354820045 * sigma_omega

    # Convert center omega back to wavelength for reference
    center_lambda_nm = TWO_PI * C_NM_PER_PS / mu


    # residuals
    yfit = gaussian_y_omega(omega, *popt)
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = pd.DataFrame([{
        "peak_transmittance": A,
        "mu_rad_per_ps": mu,
        "center_wavelength_nm": center_lambda_nm,
        "s_rad2_per_ps2": s,
        "fwhm_omega_rad_per_ps": fwhm_omega,
        "r2_T": r2,
        "n_points": len(y),
    }])

    out.to_csv(args.output, index=False)
    print(f"Wrote fit parameters to: {args.output}")
    print(out.to_string(index=False))

    if not args.no_plot:
        omega_dense = np.linspace(omega.min(), omega.max(), 10000)
        plt.figure()
        plt.scatter(omega, y, s=20, label="Measured transmittance (0–1)")
        plt.plot(omega_dense, gaussian_y_omega(omega_dense, *popt),
                 label="Gaussian fit (ω-space)")
        plt.xlabel("Angular frequency ω (rad/ps)")
        plt.ylabel("Transmittance")
        plt.title("Gaussian Transmittance Fit in Angular Frequency Space")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
