"""Focused parsing and analysis helpers for the DLS walkthrough notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import uniform_filter1d


BOLTZMANN = 1.380649e-23
WATER_VISCOSITY_TEMPERATURES_C = np.array([25.0, 37.0])
WATER_VISCOSITIES_PA_S = np.array([0.890e-3, 0.691e-3])


@dataclass(frozen=True)
class Acquisition:
    measurement_id: int
    acquisition_id: int
    measurement_name: str
    elapsed_s: float
    temperature_c: float
    delay_s: np.ndarray
    correlation: np.ndarray


@dataclass(frozen=True)
class DLSExperiment:
    path: Path
    wavelength_nm: float
    measurement_names: dict[int, str]
    acquisitions: tuple[Acquisition, ...]


def load_exp_file(path: str | Path) -> DLSExperiment:
    """Read measurement metadata, lag times, and c1 curves from a Dynamics .exp file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="latin-1").splitlines()

    delay_line = next((line for line in lines if line.startswith("Delay Times=")), None)
    wavelength_line = next(
        (line for line in lines if line.startswith("Laser Wavelength (nm)=")),
        None,
    )
    if delay_line is None or wavelength_line is None:
        raise ValueError(f"{path}: missing delay-time or wavelength metadata")

    delay_fields = delay_line.split("=", 1)[1].split("|")
    delay_s = np.asarray([float(value) for value in delay_fields[1:]], dtype=float)
    wavelength_nm = float(wavelength_line.split("=", 1)[1])

    measurement_names: dict[int, str] = {}
    acquisition_metadata: dict[tuple[int, int], tuple[float, float]] = {}
    current_measurement: int | None = None
    for line in lines:
        if line.startswith("[Measurement "):
            current_measurement = int(line.strip("[]").split()[1])
            measurement_names[current_measurement] = ""
        elif current_measurement is not None and line.startswith("Name="):
            measurement_names[current_measurement] = line.split("=", 1)[1]
        elif (
            current_measurement is not None
            and line.startswith("Data Row ")
            and "=" in line
        ):
            fields = line.split("=", 1)[1].split("|")
            label = fields[1].split("$", 1)[0].strip()
            if label.startswith("Acq "):
                acquisition_id = int(label.split()[1])
                elapsed_s = float(fields[6].split("$", 1)[0])
                temperature_c = float(fields[7].split("$", 1)[0])
                acquisition_metadata[(current_measurement, acquisition_id)] = (
                    elapsed_s,
                    temperature_c,
                )

    if not measurement_names:
        raise ValueError(f"{path}: no measurement sections found")
    for measurement_id, name in measurement_names.items():
        if not name:
            measurement_names[measurement_id] = f"Measurement {measurement_id}"

    acquisitions: list[Acquisition] = []
    for index, line in enumerate(lines):
        if not line.startswith("[Acquisition "):
            continue
        parts = line.strip("[]").split()
        measurement_id, acquisition_id = int(parts[1]), int(parts[2])
        correlation_line = None
        for candidate in lines[index + 1 :]:
            if candidate.startswith(("[Acquisition ", "[Measurement ")):
                break
            if candidate.startswith("c1="):
                correlation_line = candidate
                break
        if correlation_line is None:
            raise ValueError(
                f"{path}: acquisition {measurement_id}/{acquisition_id} has no c1 curve"
            )

        fields = correlation_line.split("=", 1)[1].split("|")
        declared_count = int(fields[0])
        point_count = min(declared_count, len(fields) - 1, len(delay_s))
        if point_count <= 0:
            raise ValueError(
                f"{path}: acquisition {measurement_id}/{acquisition_id} has no points"
            )
        correlation = np.asarray(fields[1 : point_count + 1], dtype=float)

        metadata = acquisition_metadata.get((measurement_id, acquisition_id))
        if metadata is None:
            raise ValueError(
                f"{path}: missing metadata for acquisition "
                f"{measurement_id}/{acquisition_id}"
            )
        if measurement_id not in measurement_names:
            raise ValueError(
                f"{path}: acquisition references unknown measurement {measurement_id}"
            )
        elapsed_s, temperature_c = metadata
        acquisitions.append(
            Acquisition(
                measurement_id=measurement_id,
                acquisition_id=acquisition_id,
                measurement_name=measurement_names[measurement_id],
                elapsed_s=elapsed_s,
                temperature_c=temperature_c,
                delay_s=delay_s[:point_count],
                correlation=correlation,
            )
        )

    if not acquisitions:
        raise ValueError(f"{path}: no acquisition curves found")
    return DLSExperiment(
        path=path,
        wavelength_nm=wavelength_nm,
        measurement_names=measurement_names,
        acquisitions=tuple(acquisitions),
    )


def measurement_overview(experiment: DLSExperiment) -> list[dict[str, object]]:
    """Summarize acquisition counts, time span, and measured temperature per block."""
    overview = []
    for measurement_id, name in experiment.measurement_names.items():
        acquisitions = [
            item
            for item in experiment.acquisitions
            if item.measurement_id == measurement_id
        ]
        temperatures = np.asarray([item.temperature_c for item in acquisitions])
        elapsed = np.asarray([item.elapsed_s for item in acquisitions])
        overview.append(
            {
                "measurement_id": measurement_id,
                "sample": f"Sample {measurement_id}",
                "measurement": name,
                "acquisitions": len(acquisitions),
                "acquisition_duration_s": 30,
                "temperature_min_c": float(temperatures.min()),
                "temperature_max_c": float(temperatures.max()),
                "elapsed_start_s": float(elapsed.min()),
                "elapsed_end_s": float(elapsed.max()),
                "curve_points": len(acquisitions[0].correlation),
            }
        )
    return overview


def plot_autocorrelations(experiment: DLSExperiment):
    """Plot all exported c1 curves and the median curve for each measurement block."""
    measurement_ids = sorted(experiment.measurement_names)
    figure, axes = plt.subplots(
        1,
        len(measurement_ids),
        figsize=(7 * len(measurement_ids), 5),
        squeeze=False,
        sharey=True,
    )
    colors = plt.get_cmap("tab10").colors
    for axis, measurement_id, color in zip(axes[0], measurement_ids, colors):
        acquisitions = [
            item
            for item in experiment.acquisitions
            if item.measurement_id == measurement_id
        ]
        curves = np.stack([item.correlation for item in acquisitions])
        delay_s = acquisitions[0].delay_s
        for curve in curves:
            axis.plot(delay_s, curve, color=color, alpha=0.16, linewidth=0.8)
        axis.plot(
            delay_s,
            np.median(curves, axis=0),
            color=color,
            linewidth=2.2,
            label="Median",
        )
        temperatures = np.asarray([item.temperature_c for item in acquisitions])
        axis.set_title(
            f"Sample {measurement_id}: "
            f"{experiment.measurement_names[measurement_id]}\n"
            f"{temperatures.mean():.2f} °C, n={len(acquisitions)}"
        )
        axis.set_xscale("log")
        axis.set_xlabel("Lag time (s)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("Exported correlation (c1)")
    figure.suptitle("Autocorrelation curves by measurement block")
    figure.tight_layout()
    return figure, axes[0]


def plot_median_correlation_comparison(experiment: DLSExperiment):
    """Compare each measurement block's median exported correlation curve."""
    measurement_ids = sorted(experiment.measurement_names)
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("tab10").colors
    for measurement_id, color in zip(measurement_ids, colors):
        acquisitions = [
            item
            for item in experiment.acquisitions
            if item.measurement_id == measurement_id
        ]
        curves = np.stack([item.correlation for item in acquisitions])
        axis.plot(
            acquisitions[0].delay_s,
            np.median(curves, axis=0),
            color=color,
            linewidth=2.2,
            label=f"Sample {measurement_id}: "
            f"{experiment.measurement_names[measurement_id]}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Lag time (s)")
    axis.set_ylabel("Exported correlation (c1)")
    axis.set_title("Median correlation curves by measurement")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure, axis


def plot_autocorrelations_by_time(experiment: DLSExperiment):
    """Plot each exported curve colored by acquisition elapsed time."""
    measurement_ids = sorted(experiment.measurement_names)
    elapsed_times = np.asarray(
        [item.elapsed_s for item in experiment.acquisitions], dtype=float
    )
    color_map = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(elapsed_times.min()), vmax=float(elapsed_times.max()))
    figure, axes = plt.subplots(
        1,
        len(measurement_ids),
        figsize=(7 * len(measurement_ids), 5),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for axis, measurement_id in zip(axes[0], measurement_ids):
        acquisitions = [
            item
            for item in experiment.acquisitions
            if item.measurement_id == measurement_id
        ]
        curves = np.stack([item.correlation for item in acquisitions])
        for acquisition in acquisitions:
            axis.plot(
                acquisition.delay_s,
                acquisition.correlation,
                color=color_map(norm(acquisition.elapsed_s)),
                alpha=0.75,
                linewidth=0.8,
            )
        axis.plot(
            acquisitions[0].delay_s,
            np.median(curves, axis=0),
            color="black",
            linewidth=2,
            label="Median",
        )
        axis.set_xscale("log")
        axis.set_title(
            f"Sample {measurement_id}: "
            f"{experiment.measurement_names[measurement_id]}"
        )
        axis.set_xlabel("Lag time (s)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("Exported correlation (c1)")
    figure.colorbar(
        ScalarMappable(norm=norm, cmap=color_map),
        ax=axes[0].tolist(),
        label="Acquisition elapsed time (s)",
    )
    figure.suptitle("Correlation curves colored by acquisition time")
    return figure, axes[0]


def _fit_cumulants(
    correlation: np.ndarray,
    delay_s: np.ndarray,
    min_r_squared: float,
) -> tuple[dict[str, float] | None, str | None]:
    if len(correlation) < 30 or len(correlation) != len(delay_s):
        return None, "too few correlation points"

    baseline = float(np.median(correlation[-30:]))
    amplitude = float(np.median(correlation[:10]) - baseline)
    if not np.isfinite(amplitude) or amplitude <= 0:
        return None, "non-positive correlation amplitude"

    g1 = np.sqrt(np.clip((correlation - baseline) / amplitude, 1e-8, None))
    smoothed_g1 = uniform_filter1d(g1, size=5)
    above_half = np.flatnonzero(smoothed_g1 < 0.8)
    below_half = np.flatnonzero(smoothed_g1 < 0.2)
    if not len(above_half) or not len(below_half):
        return None, "decay does not span fit window"

    start = above_half[0]
    end_candidates = below_half[below_half > start]
    if not len(end_candidates):
        return None, "decay does not span fit window"
    fit_indices = np.arange(start, end_candidates[0])
    fit_indices = fit_indices[np.isfinite(smoothed_g1[fit_indices])]
    if len(fit_indices) < 8:
        return None, "too few points in fit window"

    x = delay_s[fit_indices]
    if np.any(x <= 0):
        return None, "non-positive delay time"
    y = np.log(smoothed_g1[fit_indices])
    scale = float(np.median(x))
    quadratic, linear, intercept = np.polyfit(x / scale, y, 2)
    predicted = np.polyval((quadratic, linear, intercept), x / scale)
    variance = float(np.sum((y - y.mean()) ** 2))
    if variance <= 0:
        return None, "no variation in fit window"

    r_squared = float(1 - np.sum((y - predicted) ** 2) / variance)
    gamma_s_inv = float(-linear / scale)
    pdi = float(2 * quadratic / linear**2) if linear != 0 else np.nan
    if not np.isfinite(gamma_s_inv) or gamma_s_inv <= 0:
        return None, "non-positive decay rate"
    if not np.isfinite(pdi):
        return None, "non-finite PDI"

    fit = {
        "pdi": pdi,
        "gamma_s_inv": gamma_s_inv,
        "r_squared": r_squared,
    }
    if not 0 <= pdi <= 1:
        return fit, "PDI outside 0-1 fit range"
    if not np.isfinite(r_squared) or r_squared < min_r_squared:
        return fit, "fit R-squared below threshold"
    return fit, None


def estimate_hydrodynamic_radius_nm(
    gamma_s_inv: float,
    temperature_c: float,
    wavelength_nm: float,
    scattering_angle_deg: float = 90.0,
    refractive_index: float = 1.333,
    viscosity_pa_s: float | None = None,
) -> float:
    """Estimate Rh via Stokes-Einstein from a cumulants decay rate."""
    if not 0 < scattering_angle_deg <= 180:
        raise ValueError("scattering_angle_deg must be in (0, 180]")
    if refractive_index <= 0 or wavelength_nm <= 0 or gamma_s_inv <= 0:
        raise ValueError("refractive index, wavelength, and decay rate must be positive")
    if viscosity_pa_s is None:
        viscosity_pa_s = float(
            np.interp(
                temperature_c,
                WATER_VISCOSITY_TEMPERATURES_C,
                WATER_VISCOSITIES_PA_S,
            )
        )
    if viscosity_pa_s <= 0:
        raise ValueError("viscosity_pa_s must be positive")

    wavevector = (
        4
        * np.pi
        * refractive_index
        / (wavelength_nm * 1e-9)
        * np.sin(np.deg2rad(scattering_angle_deg) / 2)
    )
    diffusion_m2_s = gamma_s_inv / wavevector**2
    radius_m = (
        BOLTZMANN
        * (temperature_c + 273.15)
        / (6 * np.pi * viscosity_pa_s * diffusion_m2_s)
    )
    return float(radius_m * 1e9)


def analyze_hydrodynamic_radii(
    experiment: DLSExperiment,
    min_r_squared: float = 0.80,
    scattering_angle_deg: float = 90.0,
    refractive_index: float = 1.333,
    viscosity_pa_s: float | None = None,
) -> list[dict[str, object]]:
    """Fit each acquisition and estimate Rh, retaining fit failures and reasons."""
    if not 0 <= min_r_squared <= 1:
        raise ValueError("min_r_squared must be in [0, 1]")
    results = []
    for acquisition in experiment.acquisitions:
        fit, reason = _fit_cumulants(
            acquisition.correlation,
            acquisition.delay_s,
            min_r_squared,
        )
        row: dict[str, object] = {
            "measurement_id": acquisition.measurement_id,
            "sample": f"Sample {acquisition.measurement_id}",
            "measurement": acquisition.measurement_name,
            "acquisition": acquisition.acquisition_id,
            "elapsed_s": acquisition.elapsed_s,
            "temperature_c": acquisition.temperature_c,
            "gamma_s_inv": np.nan,
            "pdi": np.nan,
            "fit_r_squared": np.nan,
            "hydrodynamic_radius_nm": np.nan,
            "fit_accepted": False,
            "rejection_reason": reason,
        }
        if fit is not None:
            row.update(
                {
                    "gamma_s_inv": fit["gamma_s_inv"],
                    "pdi": fit["pdi"],
                    "fit_r_squared": fit["r_squared"],
                    "hydrodynamic_radius_nm": estimate_hydrodynamic_radius_nm(
                        fit["gamma_s_inv"],
                        acquisition.temperature_c,
                        experiment.wavelength_nm,
                        scattering_angle_deg,
                        refractive_index,
                        viscosity_pa_s,
                    ),
                    "fit_accepted": reason is None,
                }
            )
            row["hydrodynamic_diameter_nm"] = (
                2 * float(row["hydrodynamic_radius_nm"])
            )
        else:
            row["hydrodynamic_diameter_nm"] = np.nan
        results.append(row)
    return results


def summarize_exclusions(
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Count rejected acquisitions by sample and report whether Rh was estimable."""
    summaries = []
    measurement_ids = sorted({int(row["measurement_id"]) for row in results})
    for measurement_id in measurement_ids:
        rows = [
            row
            for row in results
            if row["measurement_id"] == measurement_id and not row["fit_accepted"]
        ]
        reasons = sorted({str(row["rejection_reason"]) for row in rows})
        for reason in reasons:
            rejected = [row for row in rows if row["rejection_reason"] == reason]
            summaries.append(
                {
                    "sample": f"Sample {measurement_id}",
                    "measurement": rejected[0]["measurement"],
                    "exclusion_reason": reason,
                    "acquisitions_excluded": len(rejected),
                    "radius_calculated_but_excluded": sum(
                        np.isfinite(float(row["hydrodynamic_radius_nm"]))
                        for row in rejected
                    ),
                }
            )
    return summaries


def summarize_hydrodynamic_radii(
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize fit acceptance and median accepted Rh for each measurement block."""
    summaries = []
    measurement_ids = sorted({int(row["measurement_id"]) for row in results})
    for measurement_id in measurement_ids:
        rows = [row for row in results if row["measurement_id"] == measurement_id]
        accepted = [
            float(row["hydrodynamic_radius_nm"])
            for row in rows
            if row["fit_accepted"]
            and np.isfinite(float(row["hydrodynamic_radius_nm"]))
        ]
        summaries.append(
            {
                "measurement_id": measurement_id,
                "sample": f"Sample {measurement_id}",
                "measurement": rows[0]["measurement"],
                "temperature_mean_c": float(
                    np.mean([float(row["temperature_c"]) for row in rows])
                ),
                "acquisitions": len(rows),
                "fits_accepted": len(accepted),
                "fits_rejected": len(rows) - len(accepted),
                "median_accepted_Rh_nm": (
                    float(np.median(accepted)) if accepted else np.nan
                ),
                "median_accepted_diameter_nm": (
                    float(2 * np.median(accepted)) if accepted else np.nan
                ),
            }
        )
    return summaries


def plot_hydrodynamic_radii(results: list[dict[str, object]]):
    """Plot per-acquisition Rh estimates; accepted and rejected fits use distinct marks."""
    measurement_ids = sorted({int(row["measurement_id"]) for row in results})
    figure, axes = plt.subplots(
        1,
        len(measurement_ids),
        figsize=(7 * len(measurement_ids), 5),
        squeeze=False,
        sharey=True,
    )
    colors = plt.get_cmap("tab10").colors
    for axis, measurement_id, color in zip(axes[0], measurement_ids, colors):
        rows = [row for row in results if row["measurement_id"] == measurement_id]
        accepted_radii = [
            float(row["hydrodynamic_radius_nm"])
            for row in rows
            if row["fit_accepted"]
            and np.isfinite(float(row["hydrodynamic_radius_nm"]))
            and float(row["hydrodynamic_radius_nm"]) > 0
        ]
        for accepted, marker in ((True, "o"), (False, "x")):
            selected = [
                row
                for row in rows
                if row["fit_accepted"] is accepted
                and np.isfinite(float(row["hydrodynamic_radius_nm"]))
                and float(row["hydrodynamic_radius_nm"]) > 0
            ]
            if selected:
                axis.scatter(
                    [row["acquisition"] for row in selected],
                    [row["hydrodynamic_radius_nm"] for row in selected],
                    marker=marker,
                    color=color,
                    alpha=0.8,
                    label="Accepted fit" if accepted else "Rejected fit",
                )
        if accepted_radii:
            median_radius = float(np.median(accepted_radii))
            axis.axhline(
                median_radius,
                color=color,
                linestyle=":",
                linewidth=1.5,
                label="Median accepted Rh",
            )
            axis.text(
                0.02,
                median_radius,
                f"Median Rh = {median_radius:,.1f} nm",
                transform=axis.get_yaxis_transform(),
                color=color,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
        axis.set_yscale("log")
        axis.set_title(f"{rows[0]['sample']}: {rows[0]['measurement']}")
        axis.set_xlabel("Acquisition number")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("Estimated hydrodynamic radius, Rh (nm; log scale)")
    figure.suptitle("Per-acquisition hydrodynamic radius estimates")
    figure.tight_layout()
    return figure, axes[0]


def plot_accepted_polydispersity(results: list[dict[str, object]]):
    """Plot accepted per-acquisition PDI and its median for each measurement."""
    measurement_ids = sorted({int(row["measurement_id"]) for row in results})
    figure, axes = plt.subplots(
        1,
        len(measurement_ids),
        figsize=(7 * len(measurement_ids), 5),
        squeeze=False,
        sharey=True,
    )
    colors = plt.get_cmap("tab10").colors
    for axis, measurement_id, color in zip(axes[0], measurement_ids, colors):
        rows = [row for row in results if row["measurement_id"] == measurement_id]
        accepted = [
            row
            for row in rows
            if row["fit_accepted"] and np.isfinite(float(row["pdi"]))
        ]
        if accepted:
            pdi_values = [float(row["pdi"]) for row in accepted]
            median_pdi = float(np.median(pdi_values))
            axis.scatter(
                [row["acquisition"] for row in accepted],
                pdi_values,
                marker="o",
                color=color,
                alpha=0.8,
                label="Accepted fit PDI",
            )
            axis.axhline(
                median_pdi,
                color=color,
                linestyle=":",
                linewidth=1.5,
                label="Median accepted PDI",
            )
            axis.text(
                0.02,
                median_pdi,
                f"Median PDI = {median_pdi:.3f}",
                transform=axis.get_yaxis_transform(),
                color=color,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
            axis.legend(frameon=False)
        else:
            axis.text(
                0.5,
                0.5,
                "No accepted fits",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_ylim(0, 1)
        axis.set_title(f"{rows[0]['sample']}: {rows[0]['measurement']}")
        axis.set_xlabel("Acquisition number")
        axis.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Polydispersity index (PDI)")
    figure.suptitle("Accepted-fit polydispersity by acquisition")
    figure.tight_layout()
    return figure, axes[0]
