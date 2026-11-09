
import csv
import numpy as np
import matplotlib.pyplot as plt

from nupack import Strand, Complex, Model, complex_analysis

# ============================================================
# INPUT SEQUENCES
# Entered as 5' -> 3' sequences.
# T is converted to U for the RNA thermodynamic model.
# ============================================================

seq1 = """
AACTCTCGATCAGCGGACAGCGCTAATACGACTCACTATAGGGAGATACCTGGGAATGTGCTCACGTCTCGAGCATATTCGGCCTGCTTCGGCAGGCGCTCGCGAGACGTGCGAGCGCAGGGTGAAGCCTCCACGCCCTGCCCTCACTTCGGTGAGCCAGGTGTCTCCCGCAGGGTGCGCGTGGCTTCGGCCACGACCTGAAGGAGGCACGGGTCGACGGTACCCAGTGGACGGGTATCGTCGCGGAGCTTCGGCTCCGGACCGCGTCCACTGCGGTCCGCACTCTGC
"""

seq2 = """
GCTCGAATTCTAATACGACTCACTATAGggGGAATTAGAGTGTGTTCCTGAACTGCTTCGGCGGTTCGCTACGTTCTTCGGAATGTATATAGTGTTCGCATTATACCGTAGTCCAA
"""


def clean_sequence(sequence):
    sequence = "".join(sequence.split()).upper().replace("T", "U")
    invalid = set(sequence) - set("ACGU")
    if invalid:
        raise ValueError(f"Invalid nucleotide characters: {invalid}")
    return sequence


seq1 = clean_sequence(seq1)
seq2 = clean_sequence(seq2)

print(f"3H-AE length: {len(seq1)} nt")
print(f"5H-TB length: {len(seq2)} nt")

# ============================================================
# EXPERIMENTAL CONDITIONS -- CHANGE AS NEEDED
# ============================================================

T_MIN = 20
T_MAX = 90
T_STEP = 2

# ============================================================
# DEFINE STRANDS
# ============================================================

A = Strand(seq1, name="3H-AE")
B = Strand(seq2, name="5H-TB")
complex_A = Complex([A], name="3H-AE")
complex_B = Complex([B], name="5H-TB")

temperatures = np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP)
fractions_paired_A = []
fractions_paired_B = []

# ============================================================
# TEMPERATURE SCAN
# ============================================================

for temp in temperatures:
    print(f"Calculating at {temp:.1f} °C ...")

    model = Model(
        material="rna",
        celsius=float(temp)
    )

    result = complex_analysis(
        complexes=[complex_A, complex_B],
        model=model,
        compute=["pairs"]
    )

    # The diagonal of each pair-probability matrix is the probability that
    # each nucleotide is unpaired. Report the fraction paired within that
    # strand as one minus the mean unpaired probability.
    pairs_A = result[complex_A].pairs
    pairs_B = result[complex_B].pairs
    fraction_paired_A = 1.0 - float(np.asarray(pairs_A.diagonal).mean())
    fraction_paired_B = 1.0 - float(np.asarray(pairs_B.diagonal).mean())
    fractions_paired_A.append(fraction_paired_A)
    fractions_paired_B.append(fraction_paired_B)

# ============================================================
# ESTIMATE 50% MELTING TEMPERATURE FOR EACH STRAND
# ============================================================

fractions_paired_A = np.asarray(fractions_paired_A)
fractions_paired_B = np.asarray(fractions_paired_B)


def estimate_tm(fractions, strand_name):
    crossings = np.where(
        (fractions[:-1] >= 0.5) &
        (fractions[1:] < 0.5)
    )[0]

    if len(crossings) > 0:
        i = crossings[0]

        # Linear interpolation between adjacent temperatures.
        t1, t2 = temperatures[i], temperatures[i + 1]
        f1, f2 = fractions[i], fractions[i + 1]
        tm = t1 + (0.5 - f1) * (t2 - t1) / (f2 - f1)
        print(f"Estimated {strand_name} 50% paired Tm: {tm:.2f} °C")
        return tm

    if fractions[0] < 0.5:
        print(f"{strand_name} is already below 50% paired at the minimum temperature.")
    elif fractions[-1] >= 0.5:
        print(f"{strand_name} remains at or above 50% paired at the maximum temperature.")
    else:
        print(f"No downward 50% paired crossing was detected for {strand_name}.")
    return None


tm_A = estimate_tm(fractions_paired_A, "3H-AE")
tm_B = estimate_tm(fractions_paired_B, "5H-TB")

# ============================================================
# ADDITIONAL SALT CONDITIONS
# ============================================================

# NUPACK 4.0.2 has no potassium parameter, so sodium=0.12 M is used as a
# monovalent-salt proxy for 120 mM KCl. Its salt correction also requires at
# least 50 mM sodium, so lower-salt conditions use that minimum and are
# approximations rather than true salt-free or Mg-only simulations.
MIN_NUPACK_SODIUM_M = 0.05
additional_conditions = [
    {
        "label": "3H-AE - 120 mM KCl (NUPACK Na+ proxy)",
        "requested": "120 mM KCl; no Mg; no Na",
        "model_note": "sodium=120 mM used as a monovalent-salt proxy for KCl",
        "complex": complex_A,
        "sodium": 0.12,
        "magnesium": 0.0,
    },
    {
        "label": "3H-AE - water requested (50 mM Na+ floor)",
        "requested": "nuclease-free water; no ionic salts",
        "model_note": "modeled at NUPACK minimum sodium=50 mM; not salt-free",
        "complex": complex_A,
        "sodium": MIN_NUPACK_SODIUM_M,
        "magnesium": 0.0,
    },
    {
        "label": "5H-TB - 12 mM Mg + 50 mM Na+ floor",
        "requested": "12 mM Mg; no Na",
        "model_note": "modeled with 12 mM Mg and NUPACK minimum sodium=50 mM",
        "complex": complex_B,
        "sodium": MIN_NUPACK_SODIUM_M,
        "magnesium": 0.012,
    },
    {
        "label": "5H-TB - water requested (50 mM Na+ floor)",
        "requested": "nuclease-free water; no Mg; no Na",
        "model_note": "modeled at NUPACK minimum sodium=50 mM; not salt-free",
        "complex": complex_B,
        "sodium": MIN_NUPACK_SODIUM_M,
        "magnesium": 0.0,
    },
]

additional_results = []
for condition in additional_conditions:
    print(f"\nRunning additional condition: {condition['label']}")
    print(f"  Requested: {condition['requested']}")
    print(f"  NUPACK model: {condition['model_note']}")

    fractions_paired = []

    for temp in temperatures:
        model = Model(
            material="rna",
            celsius=float(temp),
            sodium=condition["sodium"],
            magnesium=condition["magnesium"],
        )

        pairing_result = complex_analysis(
            complexes=[condition["complex"]],
            model=model,
            compute=["pairs"],
        )
        pairs = pairing_result[condition["complex"]].pairs
        fractions_paired.append(
            1.0 - float(np.asarray(pairs.diagonal).mean())
        )

    fractions_paired = np.asarray(fractions_paired)
    tm_paired = estimate_tm(fractions_paired, condition["label"])
    additional_results.append({
        **condition,
        "fractions_paired": fractions_paired,
        "tm_paired": tm_paired,
    })


# ============================================================
# SAVE RESULTS
# ============================================================

with open("rna_melting_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Temperature_C",
        "3H-AE_fraction_paired",
        "5H-TB_fraction_paired",
    ])

    for values in zip(
        temperatures,
        fractions_paired_A,
        fractions_paired_B,
    ):
        writer.writerow(values)

plt.figure(figsize=(8, 5))
plt.plot(temperatures, fractions_paired_A, "o-", label="3H-AE (single strand)")
plt.plot(temperatures, fractions_paired_B, "s-", label="5H-TB (single strand)")
plt.axhline(0.5, color="gray", linestyle="--", label="50% paired")

if tm_A is not None:
    plt.axvline(tm_A, linestyle=":", label=f"3H-AE Tm = {tm_A:.2f} °C")
if tm_B is not None:
    plt.axvline(tm_B, linestyle="-.", label=f"5H-TB Tm = {tm_B:.2f} °C")

plt.xlabel("Temperature (°C)")
plt.ylabel("Fraction of nucleotides paired within each strand")
plt.title("NUPACK Single-Strand RNA Melting Curves")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("rna_melting_curve.png", dpi=300)

with open("rna_additional_conditions_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Condition",
        "Requested_conditions",
        "NUPACK_model_note",
        "NUPACK_sodium_model_parameter_mM",
        "NUPACK_magnesium_parameter_mM",
        "Temperature_C",
        "Fraction_paired_within_strand",
        "Single_strand_Tm_C",
    ])
    for result in additional_results:
        for i, temp in enumerate(temperatures):
            writer.writerow([
                result["label"],
                result["requested"],
                result["model_note"],
                result["sodium"] * 1000,
                result["magnesium"] * 1000,
                temp,
                result["fractions_paired"][i],
                result["tm_paired"],
            ])

plt.figure(figsize=(9, 6))
for result in additional_results:
    tm_label = (
        f'{result["tm_paired"]:.2f} °C'
        if result["tm_paired"] is not None else "not detected"
    )
    plt.plot(
        temperatures,
        result["fractions_paired"],
        "o-",
        label=f'{result["label"]} (Tm {tm_label})',
    )
plt.axhline(0.5, color="gray", linestyle="--", label="50% paired")
plt.xlabel("Temperature (°C)")
plt.ylabel("Fraction of nucleotides paired within strand")
plt.title("Additional NUPACK Salt-Condition Melting Curves")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("rna_additional_conditions_single_strand_melting.png", dpi=300)

plt.show()

print("\nSaved rna_melting_results.csv")
print("Saved rna_melting_curve.png")
print("Saved rna_additional_conditions_results.csv")
print("Saved rna_additional_conditions_single_strand_melting.png")
