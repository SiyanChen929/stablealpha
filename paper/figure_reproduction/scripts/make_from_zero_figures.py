"""Publication figures assembled only from frozen, completed result tables.

No estimator refit, factor evaluation, or LLM request occurs here. Output is
editable SVG and font-embedded vector PDF, with a 300 dpi PNG preview.
"""
from pathlib import Path
import csv
import hashlib
import json
import os


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SOURCE = PAPER / "data/from_zero"
ANALYSIS = SOURCE / "evaluation/analysis"
OUT = PAPER / "output"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.hashsalt": "stablealpha-v19-publication", "hatch.linewidth": .45,
})
COLORS = {"price": "#0072B2", "wide": "#D55E00"}
LABELS = {"price": "Price-volume", "wide": "Expanded information"}
MARKERS = {"price": "o", "wide": "s"}
STYLES = {"price": "-", "wide": "--"}
SOURCES = {}

def load(relative):
    path = SOURCE / relative
    SOURCES[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    with path.open() as f:
        return list(csv.DictReader(f))

def val(row, key):
    return float(row[key]) if row[key] != "" else np.nan

def clean(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7E7E7", linewidth=.45)
    ax.tick_params(direction="out")
    ax.spines["left"].set_color("#505050")
    ax.spines["bottom"].set_color("#505050")

def title(ax, letter, name):
    ax.set_title(name, loc="left", pad=8)
    ax.text(-.14, 1.04, letter, transform=ax.transAxes, weight="bold", fontsize=10,
            va="bottom", ha="left")

def save(fig, stem):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300,
                    metadata={"Creator": "Matplotlib; StableAlpha manuscript figures"} if extension == "pdf" else None)
    svg = OUT / f"{stem}.svg"
    tree = ET.parse(svg)
    assert not tree.findall(".//{http://www.w3.org/2000/svg}image"), "Figure must remain vector"
    texts = tree.findall(".//{http://www.w3.org/2000/svg}text")
    assert texts, "SVG text must remain editable"
    visible = " ".join("".join(x.itertext()) for x in texts)
    for forbidden in ("104729", "130363", "155921", "181081", "NOT_VALIDATED", "C3", "CI05", "sha256"):
        assert forbidden not in visible, forbidden
    plt.close(fig)

curve = load("evaluation/analysis/FULL_BUDGET_CURVE_80.csv")
means = load("evaluation/analysis/CURVE_MEANS_20.csv")
raw = load("evaluation/analysis/RAW16_BASELINE_1.csv")[0]
intervals = load("evaluation/analysis/SEARCH_VS_RAW16_REPLICATE_INTERVALS.csv")
pairs = load("evaluation/analysis/SEARCH_VS_RAW16_PAIRED_16.csv")
costs = load("results/COST_BY_BATCH.csv")
assert len(curve) == 80 and len(means) == 20 and len(pairs) == 16

# Figure 1: discovery quality and downstream transmission over the same budget.
fig, axs = plt.subplots(2, 2, figsize=(18/2.54, 12.6/2.54))
fig.subplots_adjust(left=.095, right=.985, bottom=.10, top=.855, wspace=.28, hspace=.42)
specs = [
    ("frontier_best_ic", 1, "Training frontier", "Best IC"),
    ("member_median_train_ic", 1, "Selected library", "Median IC"),
    ("dev_rank_ic_mean", 1, "Prediction", "RankIC"),
    ("arithmetic_annual_return", 100, "Investment", "Net annual return (%)"),
]
for ax, (key, scale, label, ylabel), letter in zip(axs.flat, specs, "abcd"):
    clean(ax); title(ax, letter, label)
    ax.axvspan(0, 24, color="#F5F5F5", zorder=0)
    ax.axvline(24, color="#929292", linewidth=.7, linestyle=(0, (2.5, 2.5)), zorder=1)
    for arm in ("price", "wide"):
        for repeat in sorted({r["seed"] for r in curve}):
            rows = sorted([r for r in curve if r["information_set"] == arm and r["seed"] == repeat], key=lambda r:int(r["budget"]))
            ax.plot([int(r["budget"]) for r in rows], [val(r,key)*scale for r in rows],
                    color=COLORS[arm], alpha=.23, linewidth=.75, linestyle=STYLES[arm], zorder=2)
        rows = sorted([r for r in means if r["information_set"] == arm], key=lambda r:int(r["budget"]))
        ax.plot([int(r["budget"]) for r in rows], [val(r,key)*scale for r in rows],
                color=COLORS[arm], linewidth=1.65, linestyle=STYLES[arm], marker=MARKERS[arm],
                markersize=3.1, markeredgewidth=.5, zorder=4)
    if key in ("dev_rank_ic_mean", "arithmetic_annual_return"):
        baseline = val(raw, "raw16_rank_ic" if key == "dev_rank_ic_mean" else "raw16_net_return") * scale
        ax.axhline(baseline, color="#3B3B3B", linestyle=(0,(5,2,1,2)), linewidth=1, zorder=3)
    ax.set_xlim(0,36); ax.set_xticks([0,8,16,24,28,32,36]); ax.set_ylabel(ylabel)
    if scale == 1: ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    if letter in "cd": ax.set_xlabel("Candidate opportunities, q")
    else: ax.tick_params(labelbottom=False)
handles=[Line2D([0],[0],color=COLORS[a],ls=STYLES[a],marker=MARKERS[a],ms=3,lw=1.6,label=LABELS[a]) for a in ("price","wide")]
handles += [Line2D([0],[0],color="#3B3B3B",ls=(0,(5,2,1,2)),lw=1,label="16-field Ridge")]
fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.992),ncol=3,frameon=False,columnspacing=2,handlelength=2.7)
save(fig, "from_zero_budget_curve")

# Figure 2: equal-capacity comparison and its full observed token cost.
fig, axs=plt.subplots(1,3,figsize=(18/2.54,7.5/2.54),gridspec_kw={"width_ratios":[1,1,1.05]})
fig.subplots_adjust(left=.14,right=.985,bottom=.20,top=.87,wspace=.43)
conditions=[("price",24),("price",36),("wide",24),("wide",36)]
short=["Price-volume\nInitial", "Price-volume\nFinal", "Expanded\nInitial", "Expanded\nFinal"]
for ax,metric,scale,name,xlabel,letter in zip(axs[:2],
    ["dev_rank_ic_mean","arithmetic_annual_return"],[1000,100],
    ["Prediction difference","Return difference"],
    [r"RankIC difference ($10^{-3}$)","Return difference (pp/year)"],"ab"):
    title(ax,letter,name); ax.axvline(0,color="#656565",lw=.7,ls=(0,(3,3)))
    for i,(arm,q) in enumerate(conditions):
        y=3-i
        row=next(r for r in intervals if r["information_set"]==arm and int(r["budget"])==q and r["metric"]==metric)
        mu,lo,hi=[val(row,k)*scale for k in ("mean","seed_ci_low","seed_ci_high")]
        selected=[r for r in pairs if r["information_set"]==arm and int(r["budget"])==q]
        vals=np.array([val(r,metric+"_search_minus_raw16")*scale for r in selected])
        assert np.isclose(vals.mean(),mu,rtol=0,atol=1e-10)
        ax.scatter(vals, y+np.array([-.075,-.025,.025,.075]),s=13,marker=MARKERS[arm],
                   facecolor="white",edgecolor=COLORS[arm],linewidth=.7,zorder=3,alpha=.9)
        ax.errorbar(mu,y+.18,xerr=[[mu-lo],[hi-mu]],fmt="D",ms=3.7,color=COLORS[arm],
                    lw=1.05,capsize=2,capthick=.8,zorder=4)
    ax.set_yticks([3,2,1,0]); ax.set_yticklabels(short if letter=="a" else ["", "", "", ""])
    ax.set_ylim(-.35,3.65); ax.set_xlabel(xlabel,labelpad=6)
    ax.grid(axis="x",color="#E7E7E7",lw=.45); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
    ax.xaxis.set_major_locator(MultipleLocator(2))
ax=axs[2]; clean(ax); title(ax,"c","LLM usage")
for i,arm in enumerate(("price","wide")):
    sourcearm="price_volume" if arm=="price" else "expanded_admitted_fields"
    available=sorted({r["information_set"] for r in costs})
    if sourcearm not in available:
        sourcearm=next(a for a in available if ("price" in a)==(arm=="price"))
    init=[val(r,"total_tokens") for r in costs if r["information_set"]==sourcearm and int(r["budget"])==24]
    final=[val(r,"total_tokens") for r in costs if r["information_set"]==sourcearm and int(r["budget"])==36]
    assert len(init)==len(final)==4
    initialization=np.mean(init)/1000; continuation=(np.mean(final)-np.mean(init))/1000
    ax.bar(i,initialization,width=.56,color=COLORS[arm],alpha=.85,edgecolor="white",lw=.5)
    ax.bar(i,continuation,bottom=initialization,width=.56,color=COLORS[arm],alpha=.4,edgecolor="#444444",lw=.5,hatch="///")
    ax.text(i,initialization/2,f"{initialization:.1f}",ha="center",va="center",fontsize=7,color="white",weight="bold")
    ax.text(i,initialization+continuation/2,f"{continuation:.1f}",ha="center",va="center",fontsize=7,color="#292929")
    ax.text(i,initialization+continuation+3,f"{initialization+continuation:.1f}",ha="center",fontsize=7.5,weight="bold")
ax.set_ylim(0,145);ax.set_xlim(-.6,1.6);ax.set_xticks([0,1]);ax.set_xticklabels(["Price-volume","Expanded"])
ax.set_ylabel("Thousand tokens per search")
save(fig,"from_zero_capacity_cost")

captions={
 "from_zero_budget_curve": {
   "title": "Search quality and downstream outcomes from an empty learned-factor library",
   "caption": "Four independent search repetitions compare price-volume inputs with an expanded information set. Each path receives 24 initialization opportunities followed by 12 continuation opportunities; q counts both successful and unsuccessful proposals. Thin curves show repetitions and thick curves show their means. The vertical divider marks the end of initialization. Panels (a) and (b) report training-period factor quality; panels (c) and (d) report prediction RankIC and annualized arithmetic net return in 2020–2022. The horizontal reference is one Ridge model using 16 training-selected added fields. All models include the same Alpha158 base; empty learned-factor slots are zero-padded. Training IC is undefined at the empty-library origin and is therefore omitted. Net returns include 2 basis points per side under the common ten-session rebalancing rule.",
 },
 "from_zero_capacity_cost": {
   "title": "Matched-capacity performance and the cost of initialization",
   "caption": "Panels (a) and (b) compare the initialized (q = 24) and final (q = 36) learned-factor libraries with one fixed 16-field Ridge comparator. All use Alpha158 plus 16 additional columns. Small symbols denote four search repetitions; diamonds and whiskers show means and 95% Student-t intervals across those repetitions, conditional on the fixed comparator and evaluation period. Positive differences favour LLM search. Panel (c) reports mean observed token use per search, including initialization, continuation, cached tokens, and unsuccessful opportunities. A direct-field Ridge fit uses no LLM calls. Computational feature-screening cost is accounted for separately.",
 },
}
(OUT/"from_zero_figure_captions.json").write_text(json.dumps(captions,indent=2,ensure_ascii=False)+"\n")
(OUT/"from_zero_figure_sources.json").write_text(json.dumps({"sources":SOURCES,"dimensions_cm":{"from_zero_budget_curve":[18,12.6],"from_zero_capacity_cost":[18,7.5]},"software":{"matplotlib":matplotlib.__version__,"numpy":np.__version__},"data_recomputed":False,"note":"Plots are read-only assembly of frozen tables; means and interval coordinates are checked against saved analyses."},indent=2)+"\n")
print(json.dumps({"figures":[str(OUT/ (s+".pdf")) for s in captions],"sources":len(SOURCES)}))
