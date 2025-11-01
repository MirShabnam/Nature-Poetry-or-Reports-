#!/usr/bin/env python3
import os, re, math
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '..', 'data', 'nature_corpus.csv')
OUT  = os.path.join(HERE, '..', 'output')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# --- basic tokenization
def tokenize(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [w for w in t.split() if w]

df["tokens"] = df["text"].apply(tokenize)
df["token_count"] = df["tokens"].apply(len)
df["type_token_ratio"] = df["tokens"].apply(lambda toks: len(set(toks))/max(1,len(toks)))

# --- naive sentiment (tiny lexicon for demo)
POS = {"sing","sings","freedom","brave","bless","verses","scripture","companion","remembers","friend","silences"}
NEG = {"loss","decreased","declined","exceeded","dropped","bleaching","invasive"}
def sentiment_score(toks):
    p = sum(1 for w in toks if w in POS)
    n = sum(1 for w in toks if w in NEG)
    return (p - n) / max(1,len(toks))
df["sentiment"] = df["tokens"].apply(sentiment_score)

# --- charts: docs per type
ax = df["type"].value_counts().sort_index().plot(kind="bar", title="Documents per Type")
ax.set_xlabel("Type"); ax.set_ylabel("Count")
fig = ax.get_figure(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "docs_per_type.png")); plt.close(fig)

# --- average metrics by type
for col in ["token_count","type_token_ratio","sentiment"]:
    agg = df.groupby("type")[col].mean()
    ax = agg.plot(kind="bar", title=f"Average {col} by Type")
    ax.set_xlabel("Type"); ax.set_ylabel(col)
    fig = ax.get_figure(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"avg_{col}_by_type.png")); plt.close(fig)

# --- keyword extraction (very simple: type-specific top tokens)
from collections import Counter
def top_words(texts, stop=set()):
    c = Counter()
    for toks in texts:
        c.update([w for w in toks if w not in stop and not w.isdigit() and len(w)>2])
    return c.most_common(15)

STOP = {"the","and","with","for","into","that","this","are","was","were","has","have","had","but","not","now","one","two","three","four",
        "from","over","under","above","below","across","into","onto","off","you","your","their","our","its","his","her","they","them","who","which","what",
        "year","since","baseline","index","peak","mean","avg","per","site","sites"}

po = df[df["type"]=="poetry"]["tokens"].tolist()
reps = df[df["type"]=="report"]["tokens"].tolist()
top_po = top_words(po, STOP)
top_rep = top_words(reps, STOP)

def plot_top(items, title, fname):
    if not items: return
    words, counts = zip(*items)
    fig = plt.figure()
    plt.bar(range(len(words)), counts)
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.title(title); plt.tight_layout()
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

plot_top(top_po, "Top Keywords — Poetry", "top_keywords_poetry.png")
plot_top(top_rep, "Top Keywords — Reports", "top_keywords_reports.png")

# --- co-occurrence network (window = whole text for demo)
G = nx.Graph()
def add_edges(tokens):
    uniq = list(sorted(set([w for w in tokens if w not in STOP and len(w)>2])))
    for i in range(len(uniq)):
        for j in range(i+1,len(uniq)):
            a,b = uniq[i], uniq[j]
            if G.has_edge(a,b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a,b, weight=1)

df["tokens"].apply(add_edges)

pos = nx.spring_layout(G, seed=42)
plt.figure()
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, width=[G[u][v]["weight"] for u,v in G.edges()])
nx.draw_networkx_labels(G, pos, font_size=7)
plt.axis("off"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "cooccurrence_network.png")); plt.close()

# --- save summary CSV
summary = df[["id","type","author","year","token_count","type_token_ratio","sentiment"]]
summary.to_csv(os.path.join(OUT, "summary_metrics.csv"), index=False)

print("Analysis complete. Outputs written to output/.")
