#!/usr/bin/env python3
"""Estimate DLS PDI and hydrodynamic diameter from Dynamics .exp files."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


BOLTZMANN = 1.380649e-23
WATER_VISCOSITY_TEMPERATURES_C = np.array([25.0, 37.0])
WATER_VISCOSITIES_PA_S = np.array([0.890e-3, 0.691e-3])


def parse_exp_file(path: Path):
    lines = path.read_text(encoding="latin-1").splitlines()
    delay_line = next((line for line in lines if line.startswith("Delay Times=")), None)
    wavelength_line = next(
        (line for line in lines if line.startswith("Laser Wavelength (nm)=")),
        None,
    )
    if delay_line is None or wavelength_line is None:
        raise ValueError(f"{path}: missing delay-time or laser-wavelength metadata")

    delay_fields = delay_line.split("=", 1)[1].split("|")
    delay_times = np.asarray([float(value) for value in delay_fields[1:]], dtype=float)
    wavelength_nm = float(wavelength_line.split("=", 1)[1])

    measurements = {}
    current_measurement = None
    for line in lines:
        if line.startswith("[Measurement "):
            current_measurement = int(line.strip("[]").split()[1])
            measurements[current_measurement] = {"name": None, "acquisitions": {}}
        elif current_measurement is not None and line.startswith("Name="):
            measurements[current_measurement]["name"] = line.split("=", 1)[1]
        elif (
            current_measurement is not None
            and line.startswith("Data Row ")
            and "=" in line
        ):
            fields = line.split("=", 1)[1].split("|")
            acquisition_label = fields[1].split("$", 1)[0].strip()
            if not acquisition_label.startswith("Acq "):
                continue
            acquisition_id = int(acquisition_label.split()[1])
            measurements[current_measurement]["acquisitions"][acquisition_id] = {
                "time_s": float(fields[6].split("$", 1)[0]),
                "temperature_c": float(fields[7].split("$", 1)[0]),
            }

    if not measurements:
        raise ValueError(f"{path}: no measurement sections found")
    for measurement_id, measurement in measurements.items():
        if not measurement["name"]:
            measurement["name"] = f"Measurement {measurement_id}"

    acquisitions = []
    for index, line in enumerate(lines):
        if not line.startswith("[Acquisition "):
            continue
        parts = line.strip("[]").split()
        measurement_id, acquisition_id = int(parts[1]), int(parts[2])
        correlation_line = None
        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].startswith("[Acquisition "):
            if lines[next_index].startswith("c1="):
                correlation_line = lines[next_index]
                break
            next_index += 1
        if correlation_line is None:
            acquisitions.append((measurement_id, acquisition_id, None))
            continue

        correlation_fields = correlation_line.split("=", 1)[1].split("|")
        declared_points = int(correlation_fields[0])
        point_count = min(
            declared_points,
            len(correlation_fields) - 1,
            len(delay_times),
        )
        correlation = np.asarray(correlation_fields[1 : 1 + point_count], dtype=float)
        acquisitions.append((measurement_id, acquisition_id, correlation))

    return measurements, acquisitions, delay_times, wavelength_nm


def fit_cumulants(correlation: np.ndarray, delay_s: np.ndarray, min_r_squared: float):
    if len(correlation) < 30 or len(correlation) != len(delay_s):
        return None, "too few correlation points"

    baseline = np.median(correlation[-30:])
    amplitude = np.median(correlation[:10]) - baseline
    if not np.isfinite(amplitude) or amplitude <= 0:
        return None, "non-positive correlation amplitude"

    g1 = np.sqrt(np.clip((correlation - baseline) / amplitude, 1e-8, None))
    smooth_g1 = uniform_filter1d(g1, size=5)
    above_half = np.flatnonzero(smooth_g1 < 0.8)
    below_half = np.flatnonzero(smooth_g1 < 0.2)
    if not len(above_half) or not len(below_half):
        return None, "decay does not span fit window"

    start = above_half[0]
    end_candidates = below_half[below_half > start]
    if not len(end_candidates):
        return None, "decay does not span fit window"
    fit_indices = np.arange(start, end_candidates[0])
    fit_indices = fit_indices[np.isfinite(smooth_g1[fit_indices])]
    if len(fit_indices) < 8:
        return None, "too few points in fit window"

    x = delay_s[fit_indices]
    if np.any(x <= 0):
        return None, "non-positive delay time"
    y = np.log(smooth_g1[fit_indices])
    scale = np.median(x)
    quadratic, linear, intercept = np.polyfit(x / scale, y, 2)
    predicted = np.polyval((quadratic, linear, intercept), x / scale)
    variance = np.sum((y - y.mean()) ** 2)
    if variance <= 0:
        return None, "no variation in fit window"
    r_squared = 1 - np.sum((y - predicted) ** 2) / variance
    gamma_s_inv = -linear / scale
    pdi = 2 * quadratic / linear**2 if linear != 0 else np.nan

    if not np.isfinite(gamma_s_inv) or gamma_s_inv <= 0:
        return None, "non-positive decay rate"
    if not np.isfinite(pdi):
        return None, "non-finite PDI"

    fit = {"pdi": pdi, "gamma_s_inv": gamma_s_inv, "r_squared": r_squared}
    if not 0 <= pdi <= 1:
        return fit, "PDI outside 0-1 fit range"
    if not np.isfinite(r_squared) or r_squared < min_r_squared:
        return fit, "fit R-squared below threshold"
    return fit, None


def estimate_diameter_nm(
    gamma_s_inv: float,
    temperature_c: float,
    wavelength_nm: float,
    scattering_angle_deg: float,
    refractive_index: float,
    viscosity_pa_s: float | None,
):
    if viscosity_pa_s is None:
        viscosity_pa_s = float(
            np.interp(
                temperature_c,
                WATER_VISCOSITY_TEMPERATURES_C,
                WATER_VISCOSITIES_PA_S,
            )
        )
    wavevector = (
        4
        * np.pi
        * refractive_index
        / (wavelength_nm * 1e-9)
        * np.sin(np.deg2rad(scattering_angle_deg) / 2)
    )
    diffusion_m2_s = gamma_s_inv / wavevector**2
    return (
        BOLTZMANN
        * (temperature_c + 273.15)
        / (3 * np.pi * viscosity_pa_s * diffusion_m2_s)
        * 1e9
    )


def format_diameter(diameter_nm: float):
    if not np.isfinite(diameter_nm):
        return "n/a"
    if diameter_nm >= 10_000:
        return f"{diameter_nm / 1_000:,.0f} um"
    return f"{diameter_nm:,.0f} nm"


def format_measurement_name(name: str):
    for original, replacement in (
        ("RNA-origami_", ""),
        ("_annealed", ""),
        ("3HAE", "3H-AE"),
        ("5HTB", "5HT-B"),
        ("25C", "25°C"),
        ("37C", "37°C"),
    ):
        name = name.replace(original, replacement)
    return name.replace("_", " ")


def expand_inputs(inputs: list[Path], default_dir: Path):
    paths = []
    for input_path in inputs or [default_dir]:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("*.exp")))
        elif input_path.suffix.lower() == ".exp":
            paths.append(input_path)
        else:
            raise ValueError(f"Expected an .exp file or directory: {input_path}")

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(resolved)
            seen.add(resolved)
    if not unique_paths:
        raise ValueError(f"No .exp files found in {default_dir}")
    return unique_paths


def process_file(path: Path, args, output_dir: Path):
    measurements, acquisitions, delay_times, wavelength_nm = parse_exp_file(path)
    delay_scale = 1.0 if args.lag_time_unit == "s" else 1e-3
    results = []
    exclusions = defaultdict(Counter)
    acquisition_counts = Counter()

    for measurement_id, acquisition_id, correlation in acquisitions:
        acquisition_counts[measurement_id] += 1
        measurement = measurements.get(measurement_id)
        if measurement is None:
            raise ValueError(f"{path}: acquisition references unknown measurement {measurement_id}")
        label = measurement["name"]
        metadata = measurement["acquisitions"].get(acquisition_id)
        if metadata is None:
            exclusions[measurement_id]["missing acquisition metadata"] += 1
            continue
        if correlation is None:
            exclusions[measurement_id]["missing correlation curve"] += 1
            continue

        point_count = min(len(correlation), len(delay_times))
        delay_s = delay_times[:point_count] * delay_scale
        fit, reason = fit_cumulants(
            correlation[:point_count],
            delay_s,
            args.min_r_squared,
        )
        if reason is not None:
            exclusions[measurement_id][reason] += 1
        if fit is None:
            continue

        diameter_nm = estimate_diameter_nm(
            fit["gamma_s_inv"],
            metadata["temperature_c"],
            wavelength_nm,
            args.scattering_angle_deg,
            args.refractive_index,
            args.viscosity_pa_s,
        )
        results.append(
            {
                "measurement_id": measurement_id,
                "measurement": label,
                "acquisition": acquisition_id,
                "time_s": metadata["time_s"],
                "pdi": fit["pdi"],
                "diameter_nm": diameter_nm,
                "fit_r_squared": fit["r_squared"],
                "fit_accepted": reason is None,
            }
        )

    if not results:
        raise ValueError(f"{path}: no acquisitions passed the cumulants-fit checks")

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    summary = []
    for color_index, measurement_id in enumerate(sorted(measurements)):
        measurement = measurements[measurement_id]
        sample_name = format_measurement_name(measurement["name"])
        sample_results = sorted(
            (row for row in results if row["measurement_id"] == measurement_id),
            key=lambda row: row["time_s"],
        )
        accepted_results = [row for row in sample_results if row["fit_accepted"]]
        diameters = np.asarray([row["diameter_nm"] for row in accepted_results])
        median_diameter = float(np.median(diameters)) if len(diameters) else np.nan
        pdi_values = np.asarray([row["pdi"] for row in accepted_results])
        summary.append(
            {
                "file": path.name,
                "measurement": sample_name,
                "acquisitions_total": acquisition_counts[measurement_id],
                "acquisitions_plotted": len(sample_results),
                "acquisitions_used": len(accepted_results),
                "acquisitions_excluded": acquisition_counts[measurement_id] - len(accepted_results),
                "mean_pdi": float(np.mean(pdi_values)) if len(pdi_values) else "",
                "median_diameter_nm": median_diameter if len(diameters) else "",
            }
        )
        if accepted_results:
            ax.plot(
                [row["time_s"] for row in accepted_results],
                [row["pdi"] for row in accepted_results],
                marker="o",
                markersize=4,
                linewidth=1.2,
                color=colors[color_index % len(colors)],
                label=f"{sample_name} (median dH {format_diameter(median_diameter)})",
            )

    ax.set_title(f"Estimated DLS polydispersity: {path.name}")
    ax.set_xlabel("Elapsed measurement time (s)")
    ax.set_ylabel("Estimated polydispersity index (PDI)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    plot_path = output_dir / f"{path.stem}_pdi.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    scatter_fig, scatter_ax = plt.subplots(figsize=(12, 7), dpi=150)
    for color_index, measurement_id in enumerate(sorted(measurements)):
        measurement = measurements[measurement_id]
        sample_name = format_measurement_name(measurement["name"])
        sample_results = [
            row for row in results if row["measurement_id"] == measurement_id
        ]
        accepted_results = [row for row in sample_results if row["fit_accepted"]]
        diameters = np.asarray([row["diameter_nm"] for row in accepted_results])
        median_diameter = float(np.median(diameters)) if len(diameters) else np.nan
        label = f"{sample_name} (median dH {format_diameter(median_diameter)})"
        color = colors[color_index % len(colors)]

        for fit_accepted, marker in ((True, "o"), (False, "x")):
            fit_results = [
                row for row in sample_results if row["fit_accepted"] == fit_accepted
            ]
            if fit_results:
                scatter_ax.scatter(
                    [row["diameter_nm"] for row in fit_results],
                    [np.clip(row["pdi"], 0, 1) for row in fit_results],
                    marker=marker,
                    color=color,
                    alpha=0.8 if fit_accepted else 0.55,
                    label=label if marker == "o" or not accepted_results else "_nolegend_",
                )
                label = "_nolegend_"

    scatter_ax.set_xscale("log")
    scatter_ax.set_title(f"Estimated DLS diameter and PDI: {path.name}")
    scatter_ax.set_xlabel("Estimated hydrodynamic diameter (nm, log scale)")
    scatter_ax.set_ylabel("Estimated polydispersity index (PDI)")
    scatter_ax.set_ylim(0, 1)
    scatter_ax.grid(True, which="both", alpha=0.25)
    scatter_ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    scatter_fig.text(
        0.5,
        0.01,
        "Circles: accepted fits; x: rejected fits. Rejected PDI values outside [0, 1] are clipped to the axis limits.",
        ha="center",
    )
    scatter_fig.tight_layout(rect=(0, 0.04, 0.78, 1))
    scatter_path = output_dir / f"{path.stem}_diameter_vs_pdi.png"
    scatter_fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(scatter_fig)

    print(f"{path.name}: saved {plot_path}")
    print(f"{path.name}: saved {scatter_path}")
    for measurement_id, row in zip(sorted(measurements), summary):
        print(
            f"  {row['measurement']}: {row['acquisitions_used']}/"
            f"{row['acquisitions_total']} fits accepted; "
            f"{row['acquisitions_plotted']} plotted; "
            f"median dH={format_diameter(row['median_diameter_nm'])}; "
            f"mean PDI={row['mean_pdi']:.3f}"
            if row["acquisitions_used"]
            else f"  {row['measurement']}: no accepted fits; "
            f"{row['acquisitions_plotted']} plotted"
        )
        print(f"    excluded: {dict(exclusions[measurement_id])}")
    return summary, plot_path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Estimate PDI from DLS .exp correlation curves and save one plot per file."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="One or more .exp files or directories. Default: all .exp files beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "plots",
        help="Directory for PNG plots and the summary CSV (default: DLS/plots).",
    )
    parser.add_argument(
        "--lag-time-unit",
        choices=("s", "ms"),
        default="s",
        help="Unit of the file's Delay Times values (default: s; verify your instrument export).",
    )
    parser.add_argument("--scattering-angle-deg", type=float, default=90.0)
    parser.add_argument("--refractive-index", type=float, default=1.333)
    parser.add_argument(
        "--viscosity-pa-s",
        type=float,
        default=None,
        help="Optional constant solvent viscosity; default approximates water at each measurement temperature.",
    )
    parser.add_argument("--min-r-squared", type=float, default=0.80)
    args = parser.parse_args()
    if not 0 < args.scattering_angle_deg <= 180:
        parser.error("--scattering-angle-deg must be in (0, 180].")
    if args.refractive_index <= 0:
        parser.error("--refractive-index must be positive.")
    if not 0 <= args.min_r_squared <= 1:
        parser.error("--min-r-squared must be in [0, 1].")
    if args.viscosity_pa_s is not None and args.viscosity_pa_s <= 0:
        parser.error("--viscosity-pa-s must be positive.")
    return args


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    inputs = expand_inputs(args.inputs, script_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summary = []
    for path in inputs:
        summary, _ = process_file(path, args, args.output_dir)
        all_summary.extend(summary)

    summary_path = args.output_dir / "dls_pdi_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_summary[0].keys())
        writer.writeheader()
        writer.writerows(all_summary)

    print(f"Summary saved to {summary_path}")
    viscosity_description = (
        f"{args.viscosity_pa_s:g} Pa s"
        if args.viscosity_pa_s is not None
        else "water approximation"
    )
    print(
        "Assumptions: "
        f"lag times in {args.lag_time_unit}, {args.scattering_angle_deg:g}-degree scattering, "
        f"n={args.refractive_index:g}, {viscosity_description}; "
        f"fits require R-squared >= {args.min_r_squared:g} and PDI in [0, 1]."
    )


if __name__ == "__main__":
    main()
