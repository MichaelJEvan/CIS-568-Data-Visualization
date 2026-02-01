#!/usr/bin/env python3
"""
gdp_enrollment_scatter_html.py

Reads merged_clean.csv (created by the earlier pipeline) and produces:
 - static small-multiples scatter (PNG)
 - interactive faceted scatter (HTML)

Expect merged_clean.csv in current directory. 
We can adjust YEARS list if we want specific years.
"""
import os
import pandas as pd
import numpy as np

# plotting libs
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------- CONFIG ----------
MERGED_CSV = "merged_clean.csv"
OUT_PNG = "gdp_vs_enrollment_scatter.png"
OUT_HTML = "gdp_vs_enrollment_scatter.html"
# If you want to limit to specific years, set here (integers). If None, will auto-select up to 6 years.
YEARS = None   # e.g. [2019,2020,2021,2022,2023] or None to auto-select top 5-6 years present
MAX_PANELS = 6
# point size: scale GDP integer to marker size for plotting (tweak multiplier as needed)
SIZE_SCALE = 0.00002
# ---------------------------

def load_and_prepare(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    # ensure columns exist
    for c in ("country","year","gdp","enrollment","gdp_pct_change","enrollment_pct_change","both_increase"):
        if c not in df.columns:
            print(f"Warning: column '{c}' not present in merged file.")
    # coerce types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    # gdp and enrollment might be Int64 nullable — convert to numeric floats for plotting size calculations
    df["gdp_numeric"] = pd.to_numeric(df.get("gdp", pd.Series()), errors="coerce")
    df["enrollment_numeric"] = pd.to_numeric(df.get("enrollment", pd.Series()), errors="coerce")
    df["gdp_pct_change"] = pd.to_numeric(df.get("gdp_pct_change", pd.Series()), errors="coerce")
    df["enrollment_pct_change"] = pd.to_numeric(df.get("enrollment_pct_change", pd.Series()), errors="coerce")
    
    # boolean normalize
    if df.get("both_increase") is not None:
        df["both_increase"] = df["both_increase"].astype(bool)
    return df

def choose_years(df):
    yrs = sorted(df["year"].dropna().unique())
    if YEARS:
        use = [y for y in YEARS if y in yrs]
        if not use:
            raise ValueError("No requested YEARS are present in the data.")
        return use
    # auto-select up to MAX_PANELS most-recent years
    use = sorted(yrs)[-MAX_PANELS:]
    return use

def make_static_plot(df, years, out_png):
    sns.set(style="whitegrid")
    n = len(years)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    palette = {True: "#1f77b4", False: "#cccccc"}  # True blue, False gray
    for i, year in enumerate(years):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        sub = df[df["year"] == year].copy()
        if sub.empty:
            ax.set_title(f"{year} (no data)")
            ax.axis("off")
            continue
        # marker size: proportional to GDP (use absolute + clamp)
        sizes = (sub["gdp_numeric"].fillna(0).abs() * SIZE_SCALE).clip(lower=10, upper=400)
        sns.scatterplot(
            data=sub,
            x="gdp_pct_change",
            y="enrollment_pct_change",
            hue="both_increase",
            palette=palette,
            legend=(i==0),
            ax=ax,
            s=sizes,
            alpha=0.75,
            edgecolor="k",
            linewidth=0.2
        )
        ax.axvline(0, color="black", lw=0.6, linestyle="--")
        ax.axhline(0, color="black", lw=0.6, linestyle="--")
        ax.set_xlabel("GDP % change (YoY)")
        ax.set_ylabel("Enrollment % change (YoY)")
        ax.set_title(str(year))
        # annotate top outliers by distance from origin
        sub = sub.assign(dist = (sub["gdp_pct_change"].fillna(0)**2 + sub["enrollment_pct_change"].fillna(0)**2)**0.5)
        top = sub.sort_values("dist", ascending=False).head(5)
        for _, row in top.iterrows():
            if pd.isna(row["gdp_pct_change"]) or pd.isna(row["enrollment_pct_change"]):
                continue
            ax.text(row["gdp_pct_change"], row["enrollment_pct_change"], str(row.get("country","")), fontsize=7)
    # remove empty axes
    for j in range(n, rows*cols):
        r = j // cols; c = j % cols
        axes[r][c].axis("off")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved static scatter PNG:", out_png)

def make_interactive_plot(df, years, out_html):
    sub = df[df["year"].isin(years)].copy()
    # size scaling for plotly: normalize gdp_numeric to a reasonable marker size range
    gdp_nonnull = sub["gdp_numeric"].dropna()
    if not gdp_nonnull.empty:
        min_g, max_g = gdp_nonnull.min(), gdp_nonnull.max()
        # avoid division by zero
        if max_g > min_g:
            sub["size"] = 15 + 85 * ((sub["gdp_numeric"].fillna(min_g) - min_g) / (max_g - min_g))
        else:
            sub["size"] = 35
    else:
        sub["size"] = 25
    # build hover fields
    hover = ["country", "year", "gdp_numeric", "enrollment_numeric", "gdp_pct_change", "enrollment_pct_change", "both_increase"]
    fig = px.scatter(
        sub,
        x="gdp_pct_change",
        y="enrollment_pct_change",
        color=sub["both_increase"].astype(str),
        facet_col="year",
        facet_col_wrap=3,
        size="size",
        hover_data=hover,
        labels={
            "gdp_pct_change":"GDP % change (YoY)",
            "enrollment_pct_change":"Enrollment % change (YoY)",
            "both_increase":"Both increased?"
        },
        title="GDP % change vs Enrollment % change (faceted by year)"
    )
    # add zero lines to each facet via shapes for each x-domain / y-domain (Plotly facet domains are tricky; use annotations)
    fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0.3, color='black')))
    fig.update_layout(legend_title_text="both_increase", width=1200, height=600)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print("Saved interactive scatter HTML:", out_html)

def main():
    df = load_and_prepare(MERGED_CSV)
    years = choose_years(df)
    print("Plotting years:", years)
    make_static_plot(df, years, OUT_PNG)
    make_interactive_plot(df, years, OUT_HTML)
    print("Done. Open the HTML in a browser for interactive exploration.")

if __name__ == "__main__":
    main()