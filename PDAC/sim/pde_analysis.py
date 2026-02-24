#!/usr/bin/env python3
"""
PDE Center Voxel Analysis - Extract (0,0,0) concentrations across all timesteps
Pure Python (no pandas required)
"""

import os
import glob
import csv
from pathlib import Path

# Define output directory
output_dir = Path("/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/outputs/pde")
result_file = Path("/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/pde_analysis.csv")
stats_file = Path("/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/pde_analysis_stats.csv")

# Chemical names in order
chemicals = ["O2", "IFN", "IL2", "IL10", "TGFB", "CCL2", "ARGI", "NO", "IL12", "VEGFA"]

# Collect all step files
step_files = sorted(glob.glob(str(output_dir / "pde_step_*.csv")))
print(f"Found {len(step_files)} PDE output files")

# Extract center voxel (0,0,0) data
data = []
chemical_data = {chem: [] for chem in chemicals}

for step_idx, filepath in enumerate(step_files):
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            found = False
            for row in reader:
                if row['x'] == '0' and row['y'] == '0' and row['z'] == '0':
                    row_data = {'step': step_idx}
                    for chem in chemicals:
                        val = float(row[chem])
                        row_data[chem] = val
                        chemical_data[chem].append(val)
                    data.append(row_data)
                    found = True
                    break

            if not found:
                print(f"Warning: Step {step_idx} missing center voxel (0,0,0)")
    except Exception as e:
        print(f"Error reading step {step_idx}: {e}")

# Save detailed timeline
with open(result_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['step'] + chemicals)
    writer.writeheader()
    writer.writerows(data)

print(f"\nDetailed timeline saved to: {result_file}")

# Calculate statistics
def safe_divide(a, b):
    """Safe division avoiding division by zero"""
    if abs(b) > 1e-15:
        return a / b
    return float('nan')

stats_data = []
for chem in chemicals:
    if not chemical_data[chem]:
        continue

    initial = chemical_data[chem][0]
    final = chemical_data[chem][-1]
    max_val = max(chemical_data[chem])
    max_step = chemical_data[chem].index(max_val)

    # Determine trend
    if abs(final - initial) < 1e-15:
        trend = "Stable"
    elif final > initial:
        trend = "Increasing"
    else:
        trend = "Decreasing"

    # Calculate fold-change
    fold_change = safe_divide(final, initial)

    stats_data.append({
        'Chemical': chem,
        'Initial_Conc': f"{initial:.6e}",
        'Final_Conc': f"{final:.6e}",
        'Fold_Change': f"{fold_change:.2e}" if fold_change == fold_change else "N/A",
        'Max_Conc': f"{max_val:.6e}",
        'Max_Step': max_step,
        'Trend': trend
    })

# Save stats table
with open(stats_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Chemical', 'Initial_Conc', 'Final_Conc', 'Fold_Change', 'Max_Conc', 'Max_Step', 'Trend'])
    writer.writeheader()
    writer.writerows(stats_data)

print(f"Statistics summary saved to: {stats_file}")

# Print summary
print("\n" + "="*110)
print("PDE CENTER VOXEL (0,0,0) ANALYSIS SUMMARY")
print("="*110)
print(f"Steps analyzed: {len(data)} (0-{len(data)-1})")
print(f"Simulation time: ~{len(data) * 10} minutes (assuming 600s/step)")
print("\n")

# Print statistics table
print(f"{'Chemical':<10} {'Initial (M)':<15} {'Final (M)':<15} {'Fold-Change':<15} {'Peak (M)':<15} {'Trend':<12}")
print("-" * 110)
for row in stats_data:
    print(f"{row['Chemical']:<10} {row['Initial_Conc']:<15} {row['Final_Conc']:<15} {row['Fold_Change']:<15} {row['Max_Conc']:<15} {row['Trend']:<12}")

# Detailed analysis
print("\n" + "="*110)
print("DETAILED INTERPRETATION")
print("="*110)

for row in stats_data:
    chem = row['Chemical']
    initial = float(row['Initial_Conc'].split('e')[0] if 'e' in row['Initial_Conc'].lower() else row['Initial_Conc'])

    print(f"\n{chem}:")
    print(f"  Initial:     {row['Initial_Conc']} M")
    print(f"  Final:       {row['Final_Conc']} M")
    print(f"  Fold-change: {row['Fold_Change']}")
    print(f"  Peak:        {row['Max_Conc']} M at step {row['Max_Step']}")
    print(f"  Trend:       {row['Trend']}")

    # Interpretation
    trend = row['Trend']
    if trend == "Stable":
        print(f"  → Reached steady-state early or unchanged")
    elif trend == "Increasing":
        try:
            final_val = float(row['Final_Conc'].split('e')[0] if 'e' in row['Final_Conc'].lower() else row['Final_Conc'])
            init_val = float(row['Initial_Conc'].split('e')[0] if 'e' in row['Initial_Conc'].lower() else row['Initial_Conc'])
            if final_val < init_val * 1e-8:
                print(f"  → Very slow accumulation")
            else:
                print(f"  → Steady accumulation toward equilibrium")
        except:
            print(f"  → Accumulating")
    else:  # Decreasing
        print(f"  → Decay/consumption exceeds production")

print("\n" + "="*110)
print("COMPARISON WITH HCC REFERENCE VALUES")
print("="*110)
print("""
Expected HCC CPU values (typical tumor center voxel):
  O2:      ~20 mmHg ≈ 0.67 M (or ~0.0067 M if lower in tumor)
  IFN-γ:   ~1-100 pM = 1e-12 to 1e-10 M
  IL-2:    ~10-100 pM = 1e-11 to 1e-10 M
  IL-10:   ~10-100 pM = 1e-11 to 1e-10 M
  TGF-β:   ~1-10 pM = 1e-12 to 1e-11 M
  CCL2:    ~1-10 pM = 1e-12 to 1e-11 M
  ArgI:    ~100 nM = 1e-7 M
  NO:      ~1-10 µM = 1e-6 to 1e-5 M
  IL-12:   ~1-10 pM = 1e-12 to 1e-11 M
  VEGF-A:  ~10-100 pM = 1e-11 to 1e-10 M

Current GPU values (final concentrations):
""")

for row in stats_data:
    print(f"  {row['Chemical']:6s}: {row['Final_Conc']} M")

print("""
MAGNITUDES CHECK:
If GPU values are still 100-500x too small compared to reference:
  1. Check uptake collection kernel (lines ~260-275 in pde_solver.cu)
  2. Verify uptake values are non-zero and correct
  3. Check operator application (lines ~438-487 in pde_solver.cu)
  4. Ensure dt is applied only once, not twice
  5. Verify source terms are reaching the RHS correctly
""")

print(f"\nAnalysis complete! Check output files:")
print(f"  Timeline: {result_file}")
print(f"  Summary:  {stats_file}")
