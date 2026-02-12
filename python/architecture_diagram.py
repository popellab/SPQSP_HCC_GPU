"""
GPU/CPU Architecture Diagram for SPQSP PDAC Model
Visualizes parallelization strategies across different components
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np

# Set up the figure with multiple subplots (increased spacing and size)
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1], hspace=0.4, wspace=0.35)

# Color scheme
cpu_color = '#E8F4F8'
gpu_color = '#FFF4E6'
agent_color = '#A8D5E2'
pde_color = '#FFD6A5'
qsp_color = '#B8E6B8'

# ============================================================================
# Panel 1: Overall Architecture (CPU vs GPU)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
ax1.set_title('SPQSP PDAC Model Architecture: CPU/GPU Split',
              fontsize=16, fontweight='bold', pad=20)

# GPU Box
gpu_box = FancyBboxPatch((0.5, 0.5), 6, 5,
                          boxstyle="round,pad=0.1",
                          edgecolor='#FF8C00', linewidth=3,
                          facecolor=gpu_color, alpha=0.3)
ax1.add_patch(gpu_box)
ax1.text(3.5, 5.2, 'GPU (CUDA)', fontsize=14, fontweight='bold', ha='center')

# ABM Agents box within GPU
abm_box = FancyBboxPatch((0.8, 3.2), 2.5, 1.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#0077BE', linewidth=2,
                          facecolor=agent_color, alpha=0.5)
ax1.add_patch(abm_box)
ax1.text(2.05, 4.7, 'Agent-Based Model', fontsize=11, fontweight='bold', ha='center')
ax1.text(2.05, 4.3, 'FLAME GPU 2', fontsize=9, ha='center', style='italic')
ax1.text(2.05, 3.9, '• Cancer cells', fontsize=8, ha='center')
ax1.text(2.05, 3.6, '• T cells, TRegs, MDSCs', fontsize=8, ha='center')

# Parallelization note for ABM (moved down to avoid overlap)
ax1.text(2.05, 2.95, '1 thread = 1 agent', fontsize=8, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# PDE Solver box within GPU
pde_box = FancyBboxPatch((3.7, 3.2), 2.5, 1.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#D2691E', linewidth=2,
                          facecolor=pde_color, alpha=0.5)
ax1.add_patch(pde_box)
ax1.text(4.95, 4.7, 'PDE Solver', fontsize=11, fontweight='bold', ha='center')
ax1.text(4.95, 4.3, 'Implicit CG', fontsize=9, ha='center', style='italic')
ax1.text(4.95, 3.9, '10 chemicals', fontsize=8, ha='center')
ax1.text(4.95, 3.6, '(O₂, IFNγ, CCL2, etc.)', fontsize=8, ha='center')

# Parallelization note for PDE (moved down to avoid overlap)
ax1.text(4.95, 2.95, '1 thread = 1 voxel', fontsize=8, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Coupling arrows
arrow1 = FancyArrowPatch((3.3, 4.1), (3.7, 4.1),
                        arrowstyle='<->', mutation_scale=20,
                        linewidth=2, color='red')
ax1.add_patch(arrow1)
ax1.text(3.5, 4.4, 'Coupling', fontsize=8, ha='center', color='red', fontweight='bold')

# Data flow box
flow_box = FancyBboxPatch((0.8, 1.0), 5.4, 1.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='gray', linewidth=1.5,
                          facecolor='white', alpha=0.8, linestyle='--')
ax1.add_patch(flow_box)
ax1.text(3.5, 2.6, 'Data Flow per ABM Step (10 min)', fontsize=10, fontweight='bold', ha='center')

# Flow steps (adjusted label positions for better spacing)
flow_x = [1.2, 2.2, 3.2, 4.2, 5.2]
flow_labels = ['PDE→Agents\nRead [C]',
               'Agent\nBehavior',
               'Agents→PDE\nSources',
               'PDE Solve\n(CG iter)',
               'Repeat']
for i, (x, label) in enumerate(zip(flow_x, flow_labels)):
    circle = Circle((x, 1.7), 0.3, facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(x, 1.7, str(i+1), fontsize=11, ha='center', va='center', fontweight='bold')
    ax1.text(x, 1.0, label, fontsize=7, ha='center', va='center')
    if i < len(flow_x) - 1:
        ax1.arrow(x + 0.35, 1.7, 0.3, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# CPU Box
cpu_box = FancyBboxPatch((7.0, 0.5), 2.5, 5,
                          boxstyle="round,pad=0.1",
                          edgecolor='#2E8B57', linewidth=3,
                          facecolor=cpu_color, alpha=0.3)
ax1.add_patch(cpu_box)
ax1.text(8.25, 5.2, 'CPU', fontsize=14, fontweight='bold', ha='center')

# QSP box within CPU
qsp_box = FancyBboxPatch((7.3, 3.2), 1.9, 1.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#228B22', linewidth=2,
                          facecolor=qsp_color, alpha=0.5)
ax1.add_patch(qsp_box)
ax1.text(8.25, 4.7, 'QSP Model', fontsize=11, fontweight='bold', ha='center')
ax1.text(8.25, 4.3, 'SUNDIALS CVODE', fontsize=9, ha='center', style='italic')
ax1.text(8.25, 3.9, '153 species', fontsize=8, ha='center')
ax1.text(8.25, 3.6, '277 parameters', fontsize=8, ha='center')

# Parallelization note for QSP (moved down to avoid overlap with box)
ax1.text(8.25, 2.95, 'Sequential\n(1 CPU thread)', fontsize=8, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

# I/O box
io_box = FancyBboxPatch((7.3, 1.0), 1.9, 1.0,
                          boxstyle="round,pad=0.05",
                          edgecolor='gray', linewidth=1.5,
                          facecolor='lightgray', alpha=0.3)
ax1.add_patch(io_box)
ax1.text(8.25, 1.75, 'Parameter I/O', fontsize=9, fontweight='bold', ha='center')
ax1.text(8.25, 1.45, 'XML config', fontsize=8, ha='center')
ax1.text(8.25, 1.15, 'CSV output', fontsize=8, ha='center')

# GPU-CPU coupling arrow
arrow2 = FancyArrowPatch((6.5, 2.5), (7.0, 2.5),
                        arrowstyle='<->', mutation_scale=20,
                        linewidth=2, color='purple', linestyle='--')
ax1.add_patch(arrow2)
ax1.text(6.75, 2.85, 'Partial\nCoupling', fontsize=8, ha='center',
         color='purple', fontweight='bold')

# ============================================================================
# Panel 2: Agent Parallelization Detail
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_title('Agent Functions: 1 Thread per Agent',
              fontsize=12, fontweight='bold', pad=10)

# Draw a grid representing threads
n_rows, n_cols = 6, 8
cell_size = 1.0
start_x, start_y = 1.0, 1.5

# Draw agent cells
agent_types = ['Cancer', 'Cancer', 'T cell', 'Cancer', 'TReg', 'Cancer', 'MDSC', 'Cancer']
colors = {'Cancer': '#FF6B6B', 'T cell': '#4ECDC4', 'TReg': '#95E1D3', 'MDSC': '#F38181'}

for row in range(n_rows):
    for col in range(n_cols):
        agent_idx = row * n_cols + col
        if agent_idx < 48:  # Show 48 agents
            agent_type = agent_types[col % len(agent_types)]
            color = colors[agent_type]
            rect = Rectangle((start_x + col * cell_size, start_y + row * cell_size),
                           cell_size * 0.9, cell_size * 0.9,
                           facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
            ax2.add_patch(rect)
            ax2.text(start_x + col * cell_size + 0.45,
                    start_y + row * cell_size + 0.45,
                    f'T{agent_idx}', fontsize=5, ha='center', va='center')

# Labels (adjusted y-positions to avoid overlap)
ax2.text(5.0, 8.2, 'Each box = 1 CUDA thread', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
ax2.text(5.0, 0.5, f'Example: {n_rows * n_cols} agents = {n_rows * n_cols} threads',
         fontsize=9, ha='center', style='italic')

# Legend (moved down slightly)
legend_x = 0.5
legend_y = 7.0
for i, (agent_type, color) in enumerate(colors.items()):
    rect = Rectangle((legend_x, legend_y - i * 0.5), 0.4, 0.4,
                     facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
    ax2.add_patch(rect)
    ax2.text(legend_x + 0.6, legend_y - i * 0.5 + 0.2, agent_type,
            fontsize=8, va='center')

# ============================================================================
# Panel 3: PDE 3D Parallelization Detail
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 8)
ax3.axis('off')
ax3.set_title('PDE Spatial Operations: 1 Thread per Voxel',
              fontsize=12, fontweight='bold', pad=10)

# Draw 3D grid representation (isometric-ish view)
grid_size = 5  # Show 5x5x5 grid
voxel_size = 0.8
offset_x, offset_y = 2.0, 2.0

# Draw voxels in 3D-ish projection
for z in range(grid_size):
    for y in range(grid_size):
        for x in range(grid_size):
            # Only draw outer shell for clarity
            if x == 0 or x == grid_size - 1 or \
               y == 0 or y == grid_size - 1 or \
               z == 0 or z == grid_size - 1:
                px = offset_x + x * voxel_size + z * 0.3
                py = offset_y + y * voxel_size + z * 0.3

                # Color by depth
                alpha = 0.3 + 0.7 * (z / grid_size)
                rect = Rectangle((px, py), voxel_size * 0.85, voxel_size * 0.85,
                               facecolor=pde_color, edgecolor='black',
                               linewidth=0.5, alpha=alpha)
                ax3.add_patch(rect)

# Labels (adjusted spacing to avoid overlap)
ax3.text(5.0, 7.6, f'{grid_size}×{grid_size}×{grid_size} = {grid_size**3} voxels',
         fontsize=10, ha='center', fontweight='bold')
ax3.text(5.0, 7.0, f'= {grid_size**3} threads (one per voxel)',
         fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
ax3.text(5.0, 0.5, 'Each voxel computed in parallel', fontsize=9, ha='center', style='italic')

# Thread block annotation (moved up to avoid overlap with yellow box)
ax3.text(5.0, 6.4, 'Organized as 8×8×8 thread blocks', fontsize=8, ha='center', color='blue')

# ============================================================================
# Panel 4: Parallelization Comparison Table
# ============================================================================
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 5)

# Title
ax4.text(5.0, 4.5, 'Parallelization Summary (50³ grid, 10,000 agents)',
         fontsize=12, fontweight='bold', ha='center')

# Table data
table_data = [
    ['Operation', 'Parallelization Strategy', 'Thread Count', 'Config'],
    ['Agent functions', '1 thread = 1 agent', '10,000', 'Auto (FLAME GPU)'],
    ['PDE diffusion (3D)', '1 thread = 1 voxel', '125,000', '7×7×7 blocks × 8³ threads'],
    ['PDE vectors (1D)', '1 thread = 1 element', '125,000', '489 blocks × 256 threads'],
    ['Agent→PDE coupling', '1 thread = 1 agent', '10,000', '40 blocks × 256 threads'],
    ['PDE→Agent coupling', '1 thread = 1 agent', '10,000', '40 blocks × 256 threads'],
    ['QSP (CPU)', 'Sequential (single thread)', '1', 'CVODE integrator']
]

# Draw table
row_height = 0.5
col_widths = [2.0, 3.0, 2.0, 3.0]
col_positions = [0.5, 2.5, 5.5, 7.5]
start_y = 3.5

# Header
for col_idx, (col_pos, width) in enumerate(zip(col_positions, col_widths)):
    rect = Rectangle((col_pos, start_y), width, row_height,
                     facecolor='lightblue', edgecolor='black', linewidth=1)
    ax4.add_patch(rect)
    ax4.text(col_pos + width / 2, start_y + row_height / 2,
            table_data[0][col_idx], fontsize=9, ha='center', va='center',
            fontweight='bold')

# Data rows
for row_idx, row in enumerate(table_data[1:], 1):
    y_pos = start_y - row_idx * row_height

    # Alternate row colors
    if row_idx % 2 == 0:
        row_color = 'white'
    else:
        row_color = 'lightgray'

    # Special color for QSP row
    if 'QSP' in row[0]:
        row_color = cpu_color
    elif 'Agent' in row[0]:
        row_color = agent_color
    elif 'PDE' in row[0]:
        row_color = pde_color

    for col_idx, (col_pos, width) in enumerate(zip(col_positions, col_widths)):
        rect = Rectangle((col_pos, y_pos), width, row_height,
                        facecolor=row_color, edgecolor='gray', linewidth=0.5, alpha=0.5)
        ax4.add_patch(rect)
        ax4.text(col_pos + width / 2, y_pos + row_height / 2,
                row[col_idx], fontsize=8, ha='center', va='center')

# Save figure
plt.tight_layout()
plt.savefig('/home/chase/SPQSP/SPQSP_PDAC-main/python/architecture_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to: python/architecture_diagram.png")
plt.show()
