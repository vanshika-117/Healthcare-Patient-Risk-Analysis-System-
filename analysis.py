"""
================================================================================
  Healthcare Patient Data Analysis System
  Resume Project | Python + Pandas + NumPy + Matplotlib
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION  (simulates a real hospital dataset)
# ─────────────────────────────────────────────────────────────────────────────

def generate_patient_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic patient records for analysis."""
    np.random.seed(seed)

    ages        = np.random.normal(loc=50, scale=15, size=n).clip(18, 90).astype(int)
    genders     = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.50, 0.02])
    bmi         = np.random.normal(loc=27, scale=5, size=n).clip(15, 50).round(1)
    bp_systolic = (120 + (ages - 40) * 0.4 + np.random.normal(0, 12, n)).clip(90, 200).astype(int)
    bp_diastolic= (80  + (ages - 40) * 0.2 + np.random.normal(0, 8,  n)).clip(60, 120).astype(int)
    cholesterol = (180 + bmi * 1.5 + np.random.normal(0, 20, n)).clip(120, 320).astype(int)
    glucose     = (90  + bmi * 0.8 + ages * 0.3 + np.random.normal(0, 15, n)).clip(70, 300).astype(int)
    smoker      = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    exercise_days = np.random.randint(0, 8, size=n)   # days/week

    # Risk score: weighted combination
    risk_score = (
        (ages > 55).astype(int) * 2 +
        (bmi > 30).astype(int)  * 2 +
        smoker * 3 +
        (bp_systolic > 140).astype(int) * 2 +
        (cholesterol > 240).astype(int) * 1 +
        (glucose > 125).astype(int) * 2 +
        (exercise_days < 2).astype(int) * 1
    )

    # Diagnosis
    conditions = []
    for rs in risk_score:
        if rs >= 8:
            conditions.append("High Risk")
        elif rs >= 5:
            conditions.append("Moderate Risk")
        elif rs >= 2:
            conditions.append("Low Risk")
        else:
            conditions.append("Healthy")

    df = pd.DataFrame({
        "patient_id"    : [f"P{str(i+1).zfill(4)}" for i in range(n)],
        "age"           : ages,
        "gender"        : genders,
        "bmi"           : bmi,
        "bp_systolic"   : bp_systolic,
        "bp_diastolic"  : bp_diastolic,
        "cholesterol"   : cholesterol,
        "glucose"       : glucose,
        "smoker"        : smoker,
        "exercise_days" : exercise_days,
        "risk_score"    : risk_score,
        "risk_category" : conditions,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and add derived features."""

    # ── Validation
    assert df["age"].between(0, 120).all(),      "Invalid age values"
    assert df["bmi"].between(10, 80).all(),       "Invalid BMI values"
    assert df["bp_systolic"].gt(0).all(),         "Invalid BP values"

    # ── Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, 90],
        labels=["18-30", "31-45", "46-60", "61+"]
    )

    # ── BMI classification (WHO)
    df["bmi_class"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

    # ── Hypertension flag
    df["hypertension"] = (
        (df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)
    ).astype(int)

    # ── Diabetes risk flag
    df["diabetes_risk"] = (df["glucose"] >= 125).astype(int)

    # ── Pulse pressure
    df["pulse_pressure"] = df["bp_systolic"] - df["bp_diastolic"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(df: pd.DataFrame) -> dict:
    """Return key descriptive statistics."""
    numeric_cols = ["age", "bmi", "bp_systolic", "cholesterol", "glucose", "risk_score"]

    stats = {
        "overview": {
            "total_patients"   : len(df),
            "avg_age"          : round(df["age"].mean(), 1),
            "avg_bmi"          : round(df["bmi"].mean(), 1),
            "hypertension_pct" : round(df["hypertension"].mean() * 100, 1),
            "diabetes_risk_pct": round(df["diabetes_risk"].mean() * 100, 1),
            "smoker_pct"       : round(df["smoker"].mean() * 100, 1),
        },
        "risk_distribution": df["risk_category"].value_counts().to_dict(),
        "by_age_group"     : df.groupby("age_group", observed=True)["risk_score"].mean().round(2).to_dict(),
        "correlation"      : df[numeric_cols].corr().round(3),
        "describe"         : df[numeric_cols].describe().round(2),
    }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg"      : "#0D1117",
    "panel"   : "#161B22",
    "accent1" : "#58A6FF",
    "accent2" : "#3FB950",
    "accent3" : "#F78166",
    "accent4" : "#D2A8FF",
    "text"    : "#C9D1D9",
    "muted"   : "#8B949E",
}

RISK_COLORS = {
    "Healthy"       : "#3FB950",
    "Low Risk"      : "#58A6FF",
    "Moderate Risk" : "#E3B341",
    "High Risk"     : "#F78166",
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor"  : PALETTE["bg"],
        "axes.facecolor"    : PALETTE["panel"],
        "axes.edgecolor"    : "#30363D",
        "axes.labelcolor"   : PALETTE["text"],
        "xtick.color"       : PALETTE["muted"],
        "ytick.color"       : PALETTE["muted"],
        "text.color"        : PALETTE["text"],
        "grid.color"        : "#21262D",
        "grid.linewidth"    : 0.6,
        "font.family"       : "DejaVu Sans",
        "axes.titlesize"    : 11,
        "axes.labelsize"    : 9,
    })


def plot_dashboard(df: pd.DataFrame, stats: dict, output_path: str = "dashboard.png"):
    setup_style()

    fig = plt.figure(figsize=(18, 13), facecolor=PALETTE["bg"])
    fig.suptitle(
        "🏥  Healthcare Patient Risk Analysis Dashboard",
        fontsize=20, fontweight="bold", color=PALETTE["text"],
        y=0.97
    )

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.45, wspace=0.38,
        left=0.06, right=0.97,
        top=0.92, bottom=0.07
    )

    # ── [A] KPI Cards (top row, span all 4 cols)
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_axis_off()
    kpi_data = [
        ("Total Patients",    stats["overview"]["total_patients"],   PALETTE["accent1"], "👤"),
        ("Avg Age (yrs)",     stats["overview"]["avg_age"],          PALETTE["accent2"], "🎂"),
        ("Avg BMI",           stats["overview"]["avg_bmi"],          PALETTE["accent4"], "⚖️"),
        ("Hypertension %",    f"{stats['overview']['hypertension_pct']}%", PALETTE["accent3"], "❤️"),
        ("Diabetes Risk %",   f"{stats['overview']['diabetes_risk_pct']}%", "#E3B341", "🩺"),
        ("Smokers %",         f"{stats['overview']['smoker_pct']}%",  PALETTE["muted"], "🚬"),
    ]
    for i, (label, value, color, icon) in enumerate(kpi_data):
        x = i / 6
        rect = FancyBboxPatch((x + 0.005, 0.08), 0.155, 0.84,
                              boxstyle="round,pad=0.02",
                              linewidth=1.5, edgecolor=color,
                              facecolor=PALETTE["panel"],
                              transform=ax_kpi.transAxes, clip_on=False)
        ax_kpi.add_patch(rect)
        ax_kpi.text(x + 0.083, 0.72, icon,     ha="center", va="center",
                    fontsize=18, transform=ax_kpi.transAxes)
        ax_kpi.text(x + 0.083, 0.42, str(value), ha="center", va="center",
                    fontsize=17, fontweight="bold", color=color,
                    transform=ax_kpi.transAxes)
        ax_kpi.text(x + 0.083, 0.15, label,    ha="center", va="center",
                    fontsize=8,  color=PALETTE["muted"],
                    transform=ax_kpi.transAxes)

    # ── [B] Risk Distribution donut
    ax_donut = fig.add_subplot(gs[1, 0])
    risk_dist = stats["risk_distribution"]
    labels  = list(risk_dist.keys())
    sizes   = list(risk_dist.values())
    colors  = [RISK_COLORS.get(l, "#888") for l in labels]
    wedges, texts, autotexts = ax_donut.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
        pctdistance=0.78
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color(PALETTE["bg"]); at.set_fontweight("bold")
    ax_donut.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                    ncol=2, fontsize=7.5, frameon=False, labelcolor=PALETTE["text"])
    ax_donut.set_title("Risk Category Distribution", pad=10)

    # ── [C] Age distribution histogram
    ax_age = fig.add_subplot(gs[1, 1])
    ax_age.hist(df["age"], bins=20, color=PALETTE["accent1"],
                edgecolor=PALETTE["bg"], linewidth=0.5, alpha=0.9)
    ax_age.axvline(df["age"].mean(), color=PALETTE["accent3"],
                   linestyle="--", linewidth=1.5, label=f"Mean: {df['age'].mean():.0f}")
    ax_age.set_xlabel("Age (years)"); ax_age.set_ylabel("Count")
    ax_age.set_title("Age Distribution")
    ax_age.legend(fontsize=8, frameon=False)
    ax_age.grid(True, axis="y", alpha=0.4)

    # ── [D] BMI by Risk Category box plot
    ax_box = fig.add_subplot(gs[1, 2])
    categories = ["Healthy", "Low Risk", "Moderate Risk", "High Risk"]
    data_by_risk = [df[df["risk_category"] == c]["bmi"].values for c in categories]
    bp = ax_box.boxplot(data_by_risk, patch_artist=True, notch=True,
                        medianprops=dict(color=PALETTE["bg"], linewidth=2))
    for patch, cat in zip(bp["boxes"], categories):
        patch.set_facecolor(RISK_COLORS[cat])
        patch.set_alpha(0.85)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color(PALETTE["muted"])
    ax_box.set_xticklabels(["Healthy", "Low", "Mod.", "High"], fontsize=7.5)
    ax_box.set_ylabel("BMI"); ax_box.set_title("BMI by Risk Category")
    ax_box.grid(True, axis="y", alpha=0.4)

    # ── [E] Cholesterol vs Glucose scatter
    ax_scatter = fig.add_subplot(gs[1, 3])
    for cat in categories:
        sub = df[df["risk_category"] == cat]
        ax_scatter.scatter(sub["cholesterol"], sub["glucose"],
                           c=RISK_COLORS[cat], s=12, alpha=0.55,
                           label=cat, edgecolors="none")
    ax_scatter.set_xlabel("Cholesterol (mg/dL)")
    ax_scatter.set_ylabel("Glucose (mg/dL)")
    ax_scatter.set_title("Cholesterol vs Glucose")
    ax_scatter.axhline(125, color=PALETTE["accent3"], linestyle="--",
                       linewidth=1, alpha=0.7, label="Diabetes threshold")
    ax_scatter.legend(fontsize=6.5, frameon=False, markerscale=1.5)
    ax_scatter.grid(True, alpha=0.3)

    # ── [F] Risk score by age group bar
    ax_bar = fig.add_subplot(gs[2, 0:2])
    age_risk = df.groupby("age_group", observed=True)["risk_score"].mean()
    bars = ax_bar.bar(age_risk.index.astype(str), age_risk.values,
                      color=[PALETTE["accent1"], PALETTE["accent2"],
                             PALETTE["accent4"], PALETTE["accent3"]],
                      edgecolor=PALETTE["bg"], linewidth=0.8, width=0.6)
    for bar, val in zip(bars, age_risk.values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=PALETTE["text"], fontweight="bold")
    ax_bar.set_xlabel("Age Group"); ax_bar.set_ylabel("Avg Risk Score")
    ax_bar.set_title("Average Risk Score by Age Group")
    ax_bar.grid(True, axis="y", alpha=0.4)

    # ── [G] Correlation heatmap
    ax_heat = fig.add_subplot(gs[2, 2:4])
    numeric_cols = ["age", "bmi", "bp_systolic", "cholesterol", "glucose", "risk_score"]
    corr = df[numeric_cols].corr()
    im = ax_heat.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(numeric_cols)))
    ax_heat.set_yticks(range(len(numeric_cols)))
    short = ["Age", "BMI", "SysBP", "Chol.", "Glucose", "Risk"]
    ax_heat.set_xticklabels(short, fontsize=8, rotation=30, ha="right")
    ax_heat.set_yticklabels(short, fontsize=8)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax_heat.text(j, i, f"{corr.values[i,j]:.2f}",
                         ha="center", va="center", fontsize=7,
                         color="black" if abs(corr.values[i,j]) > 0.4 else PALETTE["text"])
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
    ax_heat.set_title("Feature Correlation Heatmap")

    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  ✅  Dashboard saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. REPORT EXPORT  (CSV + TXT)
# ─────────────────────────────────────────────────────────────────────────────

def export_reports(df: pd.DataFrame, stats: dict, out_dir: str = "."):
    """Save cleaned data and text summary."""
    df.to_csv(f"{out_dir}/patient_data_cleaned.csv", index=False)

    lines = [
        "=" * 60,
        "  HEALTHCARE PATIENT RISK ANALYSIS — SUMMARY REPORT",
        "=" * 60,
        "",
        "── DATASET OVERVIEW ──────────────────────────────────",
        f"  Total Patients     : {stats['overview']['total_patients']}",
        f"  Average Age        : {stats['overview']['avg_age']} yrs",
        f"  Average BMI        : {stats['overview']['avg_bmi']}",
        f"  Hypertension       : {stats['overview']['hypertension_pct']}%",
        f"  Diabetes Risk      : {stats['overview']['diabetes_risk_pct']}%",
        f"  Smokers            : {stats['overview']['smoker_pct']}%",
        "",
        "── RISK DISTRIBUTION ─────────────────────────────────",
    ]
    for cat, cnt in stats["risk_distribution"].items():
        pct = cnt / stats["overview"]["total_patients"] * 100
        lines.append(f"  {cat:<18}: {cnt:>4} patients ({pct:.1f}%)")

    lines += [
        "",
        "── RISK SCORE BY AGE GROUP ───────────────────────────",
    ]
    for grp, score in stats["by_age_group"].items():
        lines.append(f"  Age {grp:<8}: Avg risk score = {score}")

    lines += ["", "=" * 60, "  Generated by Healthcare Analysis System (Python)", "=" * 60]

    with open(f"{out_dir}/summary_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✅  CSV + TXT reports saved → {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n🏥  Healthcare Patient Risk Analysis System")
    print("─" * 48)

    print("  Generating patient data...")
    df = generate_patient_data(n=500)

    print("  Cleaning & engineering features...")
    df = clean_and_engineer(df)

    print("  Computing statistics...")
    stats = compute_statistics(df)

    ov = stats["overview"]
    print(f"\n  📊 Quick Stats:")
    print(f"     Patients        : {ov['total_patients']}")
    print(f"     Avg Age         : {ov['avg_age']} yrs")
    print(f"     Hypertension    : {ov['hypertension_pct']}%")
    print(f"     Diabetes Risk   : {ov['diabetes_risk_pct']}%")

    print("\n  Building dashboard...")
    plot_dashboard(df, stats, output_path="healthcare_dashboard.png")

    print("  Exporting reports...")
    export_reports(df, stats, out_dir=".")

    print("\n  🎉  All done!\n")
    return df, stats


if __name__ == "__main__":
    df, stats = main()