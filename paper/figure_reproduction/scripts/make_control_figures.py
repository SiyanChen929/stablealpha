"""Vector manuscript figures from completed matched-search and raw-input controls."""
from pathlib import Path
import csv, hashlib, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
LAB=PAPER
OUT=PAPER/"output"
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
 "font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
 "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,"xtick.major.width":.6,
 "ytick.major.width":.6,"xtick.major.size":3,"ytick.major.size":3,"svg.fonttype":"none",
 "pdf.fonttype":42,"ps.fonttype":42,"svg.hashsalt":"stablealpha-v19-publication","savefig.facecolor":"white"})
SOURCES={}
def load(path):
 SOURCES[str(path.relative_to(LAB))]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
 with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def title(ax,letter,name):
 ax.set_title(name,loc="left",pad=10)
 ax.text(-.14,1.04,letter,transform=ax.transAxes,fontsize=10,weight="bold",va="bottom")
def save(fig,name):
 for ext in ["svg","pdf","png"]:fig.savefig(OUT/f"{name}.{ext}",dpi=300)
 root=ET.parse(OUT/f"{name}.svg").getroot()
 assert not root.findall('.//{http://www.w3.org/2000/svg}image')
 texts=root.findall('.//{http://www.w3.org/2000/svg}text');assert texts
 visible=" ".join("".join(x.itertext()) for x in texts)
 for term in ["seed","C3","CI05","NOT_VALIDATED","sha256"]:assert term not in visible
 plt.close(fig)

src=PAPER/"data/continuation"
summary=load(src/"PRIMARY_Q8_SEED_SUMMARIES.csv")
repeats=load(src/"PRIMARY_Q8_SEED_CONTRASTS.csv")
policies=["fixed_mixture","repair_directed","fresh_seed","deterministic_genetic"]
labels=["Mixed operators","Repair-directed","Independent\ngeneration","Genetic search"]
colors=["#0072B2","#D55E00","#009E73","#8B6BAE"]
markers=["o","s","^","D"]
fig,axes=plt.subplots(1,3,figsize=(18/2.54,7.5/2.54))
fig.subplots_adjust(left=.19,right=.985,bottom=.21,top=.86,wspace=.29)
for ax,metric,scale,xlabel,label,letter in zip(axes,
 ["median_member_train_ic","dev_rank_ic_mean","arithmetic_annual_return"],
 [1000,1000,100],
 [r"IC difference ($10^{-3}$)",r"RankIC difference ($10^{-3}$)","Return difference (pp/year)"],
 ["Training","Prediction","Investment"],"abc"):
 title(ax,letter,label)
 ax.axvline(0,color="#626262",lw=.8,ls=(0,(3,3)),zorder=1)
 for i,(policy,c,marker) in enumerate(zip(policies,colors,markers)):
  y=3-i
  r=next(r for r in summary if r["left"]==policy and r["right"]=="stop" and r["metric"]==metric)
  assert r['seed_units']=='8' and r['states_averaged_per_seed']=='6'
  m,lo,hi=[float(r[k])*scale for k in ['mean','lower','upper']]
  rr=[r for r in repeats if r['left']==policy and r['right']=='stop' and r['metric']==metric]
  assert len(rr)==8
  dots=np.array([float(x['difference'])*scale for x in rr])
  assert np.isclose(dots.mean(),m,atol=1e-10)
  ax.scatter(dots,y+np.linspace(-.09,.09,8),s=11,marker=marker,facecolor="white",edgecolor=c,lw=.6,alpha=.5,zorder=2)
  ax.errorbar(m,y+.18,xerr=[[m-lo],[hi-m]],fmt=marker,ms=4,color=c,lw=1.25,capsize=2.2,zorder=3)
 ax.set_yticks([3,2,1,0]);ax.set_yticklabels(labels if letter=='a' else ['']*4)
 ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
 ax.set_ylim(-.3,3.65);ax.set_xlabel(xlabel,labelpad=7)
 ax.grid(axis='x',color='#E7E7E7',lw=.45);ax.set_axisbelow(True)
save(fig,"controlled_search_transmission")

data=load(PAPER/"data/common_inputs/effects.csv")
fig,axes=plt.subplots(1,2,figsize=(18/2.54,7.2/2.54))
fig.subplots_adjust(left=.15,right=.985,bottom=.23,top=.86,wspace=.28)
periods=["2016–2019","2020–2022","2016–2022"]
names=["2016–2019","2020–2022","Pooled\n2016–2022"]
for ax,metric,scale,label,xlabel,letter in zip(axes,["RankIC","Net return (pp/year)"],[1000,1],
 ["Prediction difference","Return difference"],
 [r"RankIC difference ($10^{-3}$)","Return difference (pp/year)"],"ab"):
 title(ax,letter,label)
 if letter=='b':ax.axvspan(-1,1,color="#F2F2F2",zorder=0)
 ax.axvline(0,color="#676767",lw=.8,ls=(0,(3,3)),zorder=1)
 for i,period in enumerate(periods):
  r=next(r for r in data if r['period']==period and r['metric']==metric)
  m,lo,hi=[float(r[k])*scale for k in ['estimate','ci_low','ci_high']]
  ax.errorbar(m,2-i,xerr=[[m-lo],[hi-m]],fmt='D' if i==2 else 'o',ms=5 if i==2 else 4,
              color='#0072B2' if i==2 else '#626262',lw=1.6 if i==2 else 1,capsize=3,zorder=3)
 ax.set_yticks([2,1,0]);ax.set_yticklabels(names if letter=='a' else ['']*3)
 ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.set_ylim(-.55,2.55)
 ax.set_xlabel(xlabel,labelpad=8);ax.grid(axis='x',color='#E4E4E4',lw=.45);ax.set_axisbelow(True)
 if letter=='b':ax.set_xlim(-1.3,1.1);ax.xaxis.set_major_locator(MultipleLocator(.5))
save(fig,"composition_beyond_raw_inputs")

captions={
 "controlled_search_transmission":{
  "title":"Better training scores transmit unevenly to predictions and returns",
  "caption":"Each search policy receives eight candidate opportunities from each of six common starting libraries in eight new search repetitions. Differences are measured against the unchanged starting library using a common Alpha158-plus-library Ridge model and portfolio rule. Small open symbols show repetition-level means across the six starting libraries; filled symbols and whiskers show the overall mean and its 95% Student-t interval across eight repetitions. Panel (a) uses the median IC among selected factors, panel (b) uses prediction RankIC, and panel (c) uses annualized arithmetic net return in 2020–2022. Positive values indicate improvement over the starting library. The intervals describe search variation over the fixed market evaluation period.",
 },
 "composition_beyond_raw_inputs":{
  "title":"Formula composition after conditioning on the same raw information",
  "caption":"Expanded-information formulas are compared with price-volume formulas after both Ridge models receive the same raw-input feature set. Points and whiskers show the paired difference and a 95% HAC interval by evaluation period; blue diamonds pool 2016–2022. Prediction differences use HAC lag one and daily net-return differences use lag five. The shaded region in panel (b) marks the prespecified ±1 percentage point annual-return margin. Both models follow the same ten-session rebalancing and trading-cost convention. The pooled estimate measures the incremental value of formula composition beyond direct access to the underlying information.",
 }
}
(OUT/'control_figure_captions.json').write_text(json.dumps(captions,indent=2,ensure_ascii=False)+'\n')
(OUT/'control_figure_sources.json').write_text(json.dumps({'sources':SOURCES,'dimensions_cm':{'controlled_search_transmission':[18,7.5],'composition_beyond_raw_inputs':[18,7.2]},'software':{'matplotlib':matplotlib.__version__},'scientific_computation_repeated':False},indent=2)+'\n')
print(json.dumps({'figures':[str(OUT/(k+'.pdf')) for k in captions]}))
