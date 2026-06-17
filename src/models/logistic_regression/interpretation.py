"""Interpretation helpers for logistic regression models."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def save_coefficient_analysis(coef_df, output_dir, model_name):
    top_coef = coef_df.sort_values("abs_coefficient", ascending=False).head(20)
    colors = ["red" if x < 0 else "green" for x in top_coef["coefficient"]]

    plt.figure(figsize=(12, 10))
    plt.barh(top_coef["feature"], top_coef["coefficient"], color=colors)
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("Coefficient Value")
    plt.title(f"{model_name} Top 20 Feature Coefficients")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Increases mortality risk"),
        Patch(facecolor="red", label="Decreases mortality risk"),
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coefficient_analysis.png"), dpi=300)
    plt.close()

    coef_df.to_csv(os.path.join(output_dir, "model_coefficients.csv"), index=False)


def calculate_odds_ratios(coef_df):
    odds_df = coef_df.copy()
    odds_df["odds_ratio"] = np.exp(odds_df["coefficient"])
    odds_df["percent_change"] = (odds_df["odds_ratio"] - 1) * 100
    odds_df["abs_percent_change"] = np.abs(odds_df["percent_change"])
    return odds_df.sort_values("abs_percent_change", ascending=False)


def save_odds_ratio_analysis(odds_df, output_dir, model_name):
    top_odds = odds_df.head(20).copy()

    plt.figure(figsize=(12, 10))
    plt.barh(top_odds["feature"], top_odds["percent_change"])
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("Effect on Odds of Mortality (%)")
    plt.title(f"{model_name} Feature Effects on Mortality Risk")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "odds_ratio_analysis.png"), dpi=300)
    plt.close()

    odds_df.to_csv(os.path.join(output_dir, "odds_ratios.csv"), index=False)
