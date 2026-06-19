"""
مقایسه سه مدل پروژه Speech Psychology
1. Baseline: TF-IDF + Ridge
2. BERT Embedding + Ridge
3. BERT Embedding + LightGBM

این اسکریپت:
- نتایج RMSE هر ستون برای هر مدل را می‌خواند (مستقیم از خروجی Colab کپی شده)
- جدول مقایسه‌ای می‌سازد
- چند نمودار برای گزارش تولید می‌کند
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. داده‌های خام RMSE — مستقیماً از خروجی هر سه نوت‌بوک (txt ها) گرفته شده
# -------------------------------------------------------------------

TARGET_COLS = [
    'sense', 'honor', 'curse', 'despise', 'situation',
    'antihuman', 'roughness', 'slaughter', 'strike_support', 'depression_rate'
]

rmse_baseline = {
    'curse': 0.937496, 'despise': 0.913755, 'antihuman': 0.910976,
    'honor': 0.867704, 'sense': 0.823247, 'roughness': 0.771388,
    'strike_support': 0.750165, 'slaughter': 0.666900,
    'situation': 0.591697, 'depression_rate': 0.511738
}

rmse_bert_ridge = {
    'curse': 0.902801, 'despise': 0.878910, 'antihuman': 0.871408,
    'honor': 0.833413, 'sense': 0.781158, 'roughness': 0.762728,
    'strike_support': 0.731834, 'slaughter': 0.669480,
    'situation': 0.573249, 'depression_rate': 0.498222
}

rmse_lgbm = {
    'sense': 0.7878, 'honor': 0.8416, 'curse': 0.9119, 'despise': 0.8859,
    'situation': 0.5776, 'antihuman': 0.8768, 'roughness': 0.7571,
    'slaughter': 0.6567, 'strike_support': 0.7357, 'depression_rate': 0.5019
}

# -------------------------------------------------------------------
# 2. ساخت DataFrame مقایسه‌ای
# -------------------------------------------------------------------

df = pd.DataFrame({
    'Baseline (TF-IDF+Ridge)': [rmse_baseline[c] for c in TARGET_COLS],
    'BERT+Ridge':              [rmse_bert_ridge[c] for c in TARGET_COLS],
    'BERT+LightGBM':           [rmse_lgbm[c] for c in TARGET_COLS],
}, index=TARGET_COLS)

print("=" * 70)
print("Per-column RMSE comparison across the three models")
print("=" * 70)
print(df.round(4).to_string())

# -------------------------------------------------------------------
# 3. محاسبه MCRMSE و امتیاز نهایی هر مدل
# -------------------------------------------------------------------

def score_from_mcrmse(mcrmse):
    return (1.5 - mcrmse) * (100 / 150) * 150

summary = pd.DataFrame({
    'Model': df.columns,
    'MCRMSE': df.mean().values,
})
summary['Score (/150)'] = summary['MCRMSE'].apply(score_from_mcrmse)
summary = summary.sort_values('MCRMSE')

print("\n" + "=" * 70)
print("Final Summary — MCRMSE and Score per model")
print("=" * 70)
print(summary.round(4).to_string(index=False))

best_model = summary.iloc[0]['Model']
print(f"\nBest model: {best_model}")

# -------------------------------------------------------------------
# 4. نمودار ۱ — مقایسه MCRMSE سه مدل (Bar chart)
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#e74c3c', '#f39c12', '#27ae60']
bars = ax.bar(summary['Model'], summary['MCRMSE'], color=colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, summary['MCRMSE']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}',
            ha='center', fontsize=11, fontweight='bold')

ax.axhline(y=1.5, color='gray', linestyle='--', linewidth=1, label='Reject threshold (1.5)')
ax.set_ylabel('MCRMSE (lower is better)')
ax.set_title('MCRMSE Comparison Across Models', fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig('fig1_mcrmse_comparison.png', dpi=150)
plt.close()
print("\n✅ fig1_mcrmse_comparison.png saved")

# -------------------------------------------------------------------
# 5. نمودار ۲ — مقایسه امتیاز نهایی (Score) سه مدل
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(summary['Model'], summary['Score (/150)'], color=colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, summary['Score (/150)']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}',
            ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Score (out of 150)')
ax.set_title('Final Score Comparison Across Models', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('fig2_score_comparison.png', dpi=150)
plt.close()
print("✅ fig2_score_comparison.png saved")

# -------------------------------------------------------------------
# 6. نمودار ۳ — مقایسه RMSE هر ستون بین سه مدل (Grouped bar chart)
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(TARGET_COLS))
width = 0.25

ax.bar(x - width, df['Baseline (TF-IDF+Ridge)'], width, label='Baseline (TF-IDF+Ridge)', color='#e74c3c')
ax.bar(x,          df['BERT+Ridge'],              width, label='BERT+Ridge',              color='#f39c12')
ax.bar(x + width,  df['BERT+LightGBM'],           width, label='BERT+LightGBM',           color='#27ae60')

ax.set_xticks(x)
ax.set_xticklabels(TARGET_COLS, rotation=30, ha='right')
ax.set_ylabel('RMSE')
ax.set_title('Per-Feature RMSE Comparison Across Models', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('fig3_per_column_comparison.png', dpi=150)
plt.close()
print("✅ fig3_per_column_comparison.png saved")

# -------------------------------------------------------------------
# 7. نمودار ۴ — Heatmap از RMSE ها
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(df.values, cmap='RdYlGn_r', aspect='auto', vmin=0.4, vmax=1.0)

ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=20, ha='right')
ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index)

for i in range(len(df.index)):
    for j in range(len(df.columns)):
        ax.text(j, i, f'{df.values[i,j]:.3f}', ha='center', va='center', fontsize=9)

ax.set_title('RMSE Heatmap — Model × Feature', fontsize=13, fontweight='bold')
fig.colorbar(im, label='RMSE')
plt.tight_layout()
plt.savefig('fig4_heatmap.png', dpi=150)
plt.close()
print("✅ fig4_heatmap.png saved")

# -------------------------------------------------------------------
# 8. ذخیره جدول‌ها به صورت CSV (برای استفاده در گزارش)
# -------------------------------------------------------------------

df.round(4).to_csv('/per_column_rmse_comparison.csv')
summary.round(4).to_csv('model_summary.csv', index=False)
print("✅ CSV tables saved")

print("\nDone!")
