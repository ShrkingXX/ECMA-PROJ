import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'table_m1_summary.csv'), index_col='outcome')

outcomes   = ['log_wage', 'log_emp', 'log_wbill']
labels     = ['Log Wage', 'Log Employment', 'Log Wage Bill']
betas      = df.loc[outcomes, 'beta'].values
ses        = df.loc[outcomes, 'se'].values
pvals      = df.loc[outcomes, 'p_value'].values
stars      = df.loc[outcomes, 'stars'].str.strip().values

ci95 = 1.96 * ses
ci90 = 1.645 * ses

y = np.array([2, 1, 0])   # vertical positions

fig, ax = plt.subplots(figsize=(7, 3.5))

colors = ['#2166ac' if p < 0.05 else '#b0b0b0' for p in pvals]

# 90% CI (thicker)
for i in range(len(outcomes)):
    ax.plot([betas[i] - ci90[i], betas[i] + ci90[i]], [y[i], y[i]],
            color=colors[i], lw=3, solid_capstyle='round', zorder=2)

# 95% CI (thinner whiskers)
for i in range(len(outcomes)):
    ax.plot([betas[i] - ci95[i], betas[i] + ci95[i]], [y[i], y[i]],
            color=colors[i], lw=1.2, solid_capstyle='round', zorder=2)

# Point estimates
ax.scatter(betas, y, color=colors, s=60, zorder=3)

# Stars
for i in range(len(outcomes)):
    if stars[i]:
        ax.text(betas[i] + ci95[i] + 0.002, y[i], stars[i],
                va='center', ha='left', fontsize=10, color=colors[i])

# Zero line
ax.axvline(0, color='black', lw=0.8, linestyle='--')

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('β  (effect of 1-unit increase in LLM exposure after ChatGPT)', fontsize=10)
ax.set_title('OLS DiD: Effect of LLM Exposure on Labor Market Outcomes\n'
             '(TWFE, clustered SE at occupation level, 90% and 95% CI)',
             fontsize=10, pad=10)

ax.set_xlim(betas.min() - ci95.max() - 0.015,
            betas.max() + ci95.max() + 0.025)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2166ac', lw=2, label='p < 0.05'),
    Line2D([0], [0], color='#b0b0b0', lw=2, label='p ≥ 0.05'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'figure_m1_coef_plot.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
