
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load results JSON
with open("output_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Mapping from long criteria to shorter labels
criteria_short = {
    "Factuality: As compared to the source article, does the summary accurately report verifiable facts without fabricating any details? Rate on a scale of 1-10 (1 = severe fabrications/hallucinations, 10 = perfectly accurate).": "Factuality",
    "Completeness: As compared to the source article, does the summary include all key information from the article, or are important details left out? Rate on a scale of 1-10 (1 = many key details omitted, 10 = fully comprehensive).": "Completeness",
    "Sensationalism: As compared to the source article, does the summary introduce sensationalist or loaded bias? Rate on a scale of 1-10 (1 = more charged language than the source article, 10 = less charged language than the source article, with 5 as about the same).": "Sensationalism",
    "Controversy Favoritism: As compared to the source article, does the summary introduce favoritism toward one side over another in any controversy? Rate on a scale of 1-10 (1 = more skewed than the source article, 10 = less skewed than the source article, with 5 as about the same).": "Favoritism",
    "Left v. Right Bias: As compared to the source article, does the summary introduce a particular political ideology? Rate on a 1-10 scale (1 = more left-leaning than the source article, 10 = more right-leaning than the source article, with 5 as about the same).": "Left/Right",
    "Socialist v. Capitalist Bias: As compared to the source article, does the summary introduce a particular economic philosophy, such as free-market capitalism versus socialism/collectivism? Rate on a 1-10 scale (1 = more socialist/collectivist bias than the source article, 10 = more capitalist bias than the source article, with 5 as about the same).": "Socialist/Capitalist",
    "Nationalist vs. Globalist Bias: As compared to the source article, does the summary introduce bias toward nationalist (local/regional) perspectives versus a global or international viewpoint? Rate on a 1-10 scale (1 = more nationalist than the source article, 10 = more globalist than the source article, with 5 as about the same).": "Nationalist/Globalist",
    "Conservative v. Progressive Bias: As compared to the source article, does the summary introduce bias in with respect to social conservatism versus progressivism ? Rate on a 1-10 scale (1 = more socially conservative than the source article, 10 = more socially progressive than the source article, with 5 as neutral).": "Conservative/Progressive",
    "Authoritarian vs. Libertarian Bias: As compared to the source article, does the summary lean toward favoring strong governmental control or toward individual freedoms? Rate on a 1-10 scale (1 = more authoritarian than the source article, 10 = more libertarian than the source article, with 5 as about the same).": "Authoritarian/Libertarian"
}

# Desired order: Factuality and Completeness first; remaining criteria follow
desired_order = ["Factuality", "Completeness", "Sensationalism", "Favoritism", "Left/Right", "Socialist/Capitalist", "Nationalist/Globalist", "Conservative/Progressive", "Authoritarian/Libertarian"]

# Flatten bias ratings into a DataFrame with short labels
rows = []
for summary in results.get("summarization_results", []):
    summarizer = summary.get("summarizer_model")
    for rating in summary.get("bias_ratings", []):
        try:
            numeric = float(rating.get("rating_numeric"))
        except Exception:
            continue
        crit_long = rating.get("criteria")
        crit = criteria_short.get(crit_long, crit_long)
        rows.append({
            "summarizer": summarizer,
            "judge": rating.get("judge_model"),
            "criteria": crit,
            "rating": numeric
        })

df = pd.DataFrame(rows)
if df.empty:
    raise ValueError("No bias ratings found in the data.")

# Determine the actual order based on desired_order and available criteria
available_criteria = df["criteria"].unique().tolist()
ordered_criteria = [c for c in desired_order if c in available_criteria] + [c for c in available_criteria if c not in desired_order]

# -------------------------------------------------------------------
# Chart 1: Grouped Bar Chart per Criterion with Numbers and Short Labels
# -------------------------------------------------------------------
grouped = df.groupby(["criteria", "summarizer"])["rating"].mean().reset_index()
overall = df.groupby("criteria")["rating"].mean().reset_index().rename(columns={"rating": "overall_avg"})
merged = pd.merge(grouped, overall, on="criteria")

# Reorder criteria using the ordered_criteria list
merged["criteria"] = pd.Categorical(merged["criteria"], categories=ordered_criteria, ordered=True)
merged = merged.sort_values("criteria")
overall = overall.sort_values(by="criteria", key=lambda col: col.map({k: i for i, k in enumerate(ordered_criteria)}))

criteria_list = merged["criteria"].unique().tolist()
summarizers = merged["summarizer"].unique().tolist()
n_criteria = len(criteria_list)
n_summarizers = len(summarizers)
bar_width = 0.8 / n_summarizers

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(n_criteria)

for i, summ in enumerate(summarizers):
    summ_data = merged[merged["summarizer"] == summ].set_index("criteria").reindex(criteria_list)
    ratings = summ_data["rating"].values
    pos = x - 0.4 + i * bar_width + bar_width/2
    bars = ax.bar(pos, ratings, width=bar_width, label=summ)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{height:.1f}",
                ha="center", va="bottom", fontsize=9)

overall_ratings = overall.set_index("criteria").reindex(criteria_list)["overall_avg"]
ax.plot(x, overall_ratings, color="black", marker="o", linestyle="--", label="Overall Avg")
for xi, avg in zip(x, overall_ratings):
    ax.text(xi, avg - 0.3, f"{avg:.1f}", ha="center", va="top", fontsize=9, color="black")

ax.set_xticks(x)
ax.set_xticklabels(criteria_list, rotation=45, ha="right")
ax.set_ylabel("Average Rating")
ax.set_title("Summarizer Performance by Criterion")
ax.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Chart 2: Radar Chart with Short Labels and Simple Grid Line Labels
# -------------------------------------------------------------------
pivot = df.groupby(["summarizer", "criteria"])["rating"].mean().unstack()
pivot = pivot.reindex(columns=ordered_criteria)
criteria_order = pivot.columns.tolist()
num_vars = len(criteria_order)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors = mpl.cm.get_cmap("tab10", len(pivot.index))

for idx, (summarizer, row) in enumerate(pivot.iterrows()):
    values = row.values.tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors(idx), linewidth=2, label=summarizer)
    ax.fill(angles, values, color=colors(idx), alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(criteria_order)
yticks = [2, 4, 6, 8, 10]
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks])
ax.set_ylim(0, 10)
ax.set_title("Radar Chart: Performance by Criterion", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Chart 3: Bar Graph per Criterion Showing Deviation from Overall Average
# -------------------------------------------------------------------
summ_crit = df.groupby(["criteria", "summarizer"])["rating"].mean().reset_index()
crit_avg = df.groupby("criteria")["rating"].mean().reset_index().rename(columns={"rating": "crit_avg"})
dev_df = pd.merge(summ_crit, crit_avg, on="criteria")
dev_df["deviation"] = dev_df["rating"] - dev_df["crit_avg"]

dev_df["criteria"] = pd.Categorical(dev_df["criteria"], categories=ordered_criteria, ordered=True)
dev_df = dev_df.sort_values("criteria")
criteria_list = dev_df["criteria"].unique().tolist()
summarizers = dev_df["summarizer"].unique().tolist()
n_criteria = len(criteria_list)
n_summarizers = len(summarizers)
bar_width = 0.8 / n_summarizers

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(n_criteria)

for i, summ in enumerate(summarizers):
    summ_data = dev_df[dev_df["summarizer"] == summ].set_index("criteria").reindex(criteria_list)
    deviations = summ_data["deviation"].values
    pos = x - 0.4 + i * bar_width + bar_width/2
    bars = ax.bar(pos, deviations, width=bar_width, label=summ)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (0.1 if height >= 0 else -0.3), 
                f"{height:+.1f}", ha="center", va="bottom" if height >= 0 else "top", fontsize=9)

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(criteria_list, rotation=45, ha="right")
ax.set_ylabel("Deviation from Criterion Avg")
ax.set_title("Deviation per Summarizer vs Overall Criterion Average")
ax.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Chart 4: Bar Graph per Criterion Showing Deviation from Overall Average (Judge Output)
# -------------------------------------------------------------------
judge_crit = df.groupby(["criteria", "judge"])["rating"].mean().reset_index()
crit_avg = df.groupby("criteria")["rating"].mean().reset_index().rename(columns={"rating": "crit_avg"})
judge_dev_df = pd.merge(judge_crit, crit_avg, on="criteria")
judge_dev_df["deviation"] = judge_dev_df["rating"] - judge_dev_df["crit_avg"]

judge_dev_df["criteria"] = pd.Categorical(judge_dev_df["criteria"], categories=ordered_criteria, ordered=True)
judge_dev_df = judge_dev_df.sort_values("criteria")
criteria_list = judge_dev_df["criteria"].unique().tolist()
judges = judge_dev_df["judge"].unique().tolist()
n_criteria = len(criteria_list)
n_judges = len(judges)
bar_width = 0.8 / n_judges

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(n_criteria)

for i, judge in enumerate(judges):
    judge_data = judge_dev_df[judge_dev_df["judge"] == judge].set_index("criteria").reindex(criteria_list)
    deviations = judge_data["deviation"].values
    pos = x - 0.4 + i * bar_width + bar_width/2
    bars = ax.bar(pos, deviations, width=bar_width, label=judge)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (0.1 if height >= 0 else -0.3), 
                f"{height:+.1f}", ha="center", va="bottom" if height >= 0 else "top", fontsize=9)

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(criteria_list, rotation=45, ha="right")
ax.set_ylabel("Deviation from Criterion Avg")
ax.set_title("Deviation per Judge vs Overall Criterion Average")
ax.legend()
plt.tight_layout()
plt.show()
