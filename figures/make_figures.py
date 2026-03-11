"""Generate paper figures for RLVR reward design paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

# ============================================================
# Figure 1: Overall success by variant (30-task eval, all models)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.5))

models = ['v1', 'v3a', 'v5', 'v2c', 'v2b', 'v2a', 'v4']
short_desc = ['format only', 'SFT', 'milestone hyb.', 'naive partial', 'outc-dom+cur', 'outcome-dom', 'partial cont.']
inits = ['scratch', 'scratch', 'scratch', 'scratch', 'cont.', 'scratch', 'cont.']
means = [63, 46, 0, 14.4, 75.6, 78.9, 82.2]
stds = [0, 0, 0, 3.1, 4.2, 3.1, 1.6]
# Nature/Cell academic palette
# v1=gold, v3a=dark gray, v5=medium gray, v2c=terracotta, v2b=sage green, v2a=steel blue, v4=muted purple
colors = ['#F2C45A', '#A0A0A0', '#C4C4C4', '#E8734A', '#5BA85B', '#3C78A8', '#8E6DB5']
edgecolors = ['#C9A038', '#707070', '#999999', '#C45A32', '#468346', '#2D5F87', '#6E5490']

x = np.arange(len(models))
bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
              edgecolor=edgecolors, linewidth=1.2, zorder=3)

# Single combined x-tick label: model name + description + init
xlabels = [f'{m}\n{d}\n({init})' for m, d, init in zip(models, short_desc, inits)]
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=8.5, linespacing=1.3)

ax.set_ylabel('Overall Success Rate (%)')
ax.set_title('Training Variant Comparison (30-task eval, 3 seeds)')
ax.set_ylim(0, 100)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, zorder=1)
ax.grid(axis='y', alpha=0.3, zorder=0)

# Annotate key values
for bar, mean, std in zip(bars, means, stds):
    y_pos = max(mean + (std if std > 0 else 0) + 2, 5)
    label = f'{mean:.1f}%' if std > 0 else f'{mean}%'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            label, ha='center', va='bottom', fontsize=9,
            fontweight='bold' if std > 0 else 'normal')

plt.tight_layout()
plt.savefig('/tmp/fig1_overall.pdf', bbox_inches='tight')
plt.savefig('/tmp/fig1_overall.png', bbox_inches='tight')
print("[OK] Figure 1: Overall success by variant")

# ============================================================
# Figure 2: Expanded eval — v2a vs v2c by difficulty
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

# Fold step info into the x-tick label
difficulties = ['D1 (Easy)\n1-2 steps', 'D2 (Medium)\n2-3 steps', 'D3 (Hard)\n4-5 steps', 'Overall']
v2a_means = [90.0, 99.3, 31.3, 73.6]
v2a_stds = [0.8, 0.5, 4.1, 1.0]
v2c_means = [33.0, 0.0, 0.0, 11.0]
v2c_stds = [2.9, 0.0, 0.0, 1.0]

x = np.arange(len(difficulties))
width = 0.35

bars1 = ax.bar(x - width/2, v2a_means, width, yerr=v2a_stds, capsize=4,
               label='v2a (outcome-dominant)', color='#3C78A8', edgecolor='#2D5F87', linewidth=1.2, zorder=3)
bars2 = ax.bar(x + width/2, v2c_means, width, yerr=v2c_stds, capsize=4,
               label='v2c (naive partial)', color='#E8734A', edgecolor='#C45A32', linewidth=1.2, zorder=3)

ax.set_ylabel('Success Rate (%)')
ax.set_title('Expanded Evaluation: 100 tasks/difficulty × 3 seeds')
ax.set_xticks(x)
ax.set_xticklabels(difficulties, fontsize=9)
ax.set_ylim(0, 118)
ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
ax.grid(axis='y', alpha=0.3, zorder=0)

# Annotate v2a values — offset slightly left
for bar, mean in zip(bars1, v2a_means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 3,
            f'{mean:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2D5F87')
# Annotate v2c values — offset slightly right
for bar, mean in zip(bars2, v2c_means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 3,
            f'{mean:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#C45A32')

# Gap annotation — between the two "Overall" bars, pointing down to v2c bar
ax.annotate('6.7×', xy=(3 + width/2, 14), xytext=(3 + width/2, 50),
            fontsize=13, fontweight='bold', color='#8E6DB5',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#8E6DB5', lw=1.5))

plt.tight_layout()
plt.savefig('/tmp/fig2_expanded.pdf', bbox_inches='tight')
plt.savefig('/tmp/fig2_expanded.png', bbox_inches='tight')
print("[OK] Figure 2: Expanded eval v2a vs v2c")

# ============================================================
# Figure 3: D3 Failure Taxonomy — stacked bar (Type A / Type B / Other)
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5))

models_tax = ['v2a\n(outc-dom,\nscratch)', 'v2b\n(outc-dom,\ncont.)', 'v2c\n(naive partial,\nscratch)', 'v5\n(milestone hyb.,\nscratch)', 'v4\n(partial,\ncont.)']

# D3 failures out of 30 total D3 tasks (10 × 3 seeds)
#                v2a  v2b  v2c  v5   v4
success =       [14,  9,   0,   0,   14]
type_a =        [8,   18,  0,   0,   8]
type_b =        [8,   0,   30,  10,  8]
other =         [0,   3,   0,   20,  0]   # v5: 20 param_error (semantic failure)

x = np.arange(len(models_tax))
width = 0.55

p1 = ax.bar(x, success, width, label='Success', color='#3C78A8', edgecolor='white', linewidth=0.5, zorder=3)
p2 = ax.bar(x, type_a, width, bottom=success, label='Type A (tries, fails verif.)',
            color='#F2C45A', edgecolor='white', linewidth=0.5, zorder=3)
p3 = ax.bar(x, type_b, width, bottom=[s+a for s,a in zip(success, type_a)],
            label='Type B (early stop)', color='#E8734A', edgecolor='white', linewidth=0.5, zorder=3)
p4 = ax.bar(x, other, width, bottom=[s+a+b for s,a,b in zip(success, type_a, type_b)],
            label='Other (semantic failure)', color='#C4C4C4', edgecolor='white', linewidth=0.5, zorder=3)

ax.set_ylabel('D3 Tasks (30 total = 10 × 3 seeds)')
ax.set_title('D3 Failure Taxonomy by Model', pad=50)
ax.set_xticks(x)
ax.set_xticklabels(models_tax, fontsize=8.5, linespacing=1.2)
ax.set_ylim(0, 44)
ax.legend(loc='lower center', fontsize=7.5, framealpha=0.9, ncol=4,
          bbox_to_anchor=(0.5, 1.01))
ax.grid(axis='y', alpha=0.3, zorder=0)

# Highlight v2c — annotation
ax.annotate('100% Type B\n(early stop)', xy=(2, 31), xytext=(2.7, 41),
            fontsize=8.5, fontweight='bold', color='#C45A32',
            arrowprops=dict(arrowstyle='->', color='#C45A32', lw=1.5),
            ha='center')

# Highlight v2b — annotation
ax.annotate('0% Type B\n(all attempts)', xy=(1, 28), xytext=(0.15, 41),
            fontsize=8.5, fontweight='bold', color='#2D5F87',
            arrowprops=dict(arrowstyle='->', color='#2D5F87', lw=1.5),
            ha='center')

# Highlight v5 — annotation for its unique failure mode
ax.annotate('67% semantic\nfailure', xy=(3, 31), xytext=(3.85, 41),
            fontsize=8.5, fontweight='bold', color='#888888',
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5),
            ha='center')

plt.subplots_adjust(top=0.80)
plt.savefig('/tmp/fig3_taxonomy.pdf', bbox_inches='tight')
plt.savefig('/tmp/fig3_taxonomy.png', bbox_inches='tight')
print("[OK] Figure 3: D3 Failure Taxonomy")

print("\n[DONE] All 3 figures saved to /tmp/fig{1,2,3}_{overall,expanded,taxonomy}.{pdf,png}")
