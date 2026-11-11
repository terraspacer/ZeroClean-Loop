"""Dirty-only contextual search with leave-one-out conditional support."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import re
import warnings
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.metrics import normalized_mutual_info_score


def pattern(value):
    # Compress runs, preserving separators and case. Pattern rarity alone is not dirtiness.
    s = re.sub(r"[A-Z]+", "A", value)
    s = re.sub(r"[a-z]+", "a", s)
    return re.sub(r"\d+", "9", s)


class DirtyIndex:
    def __init__(self, table, dependencies=(), discover=True):
        self.table = table
        self.n, self.m = table.shape
        self.columns = table.columns
        self.data = np.asarray(table.rows, dtype=object)
        self.counts = [Counter(self.data[:,j]) for j in range(self.m)]
        self.patterns = [Counter(pattern(v) for v in self.data[:,j]) for j in range(self.m)]
        self.postings = [defaultdict(list) for _ in range(self.m)]
        for j in range(self.m):
            for i,v in enumerate(self.data[:,j]): self.postings[j][v].append(i)
        self.nmi = np.eye(self.m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for a in range(self.m):
                for b in range(a):
                    self.nmi[a,b] = self.nmi[b,a] = normalized_mutual_info_score(self.data[:,a], self.data[:,b])
        self.related = {j: sorted((k for k in range(self.m) if k != j),
                                  key=lambda k: -self.nmi[j,k])[:4] for j in range(self.m)}
        self.pairs = {}
        for j in range(self.m):
            for k in self.related[j]:
                self.pairs[j,k] = Counter(zip(self.data[:,k], self.data[:,j]))
        self.dependencies = []
        for dep in dependencies:
            try:
                ks = tuple(self.columns.index(c) for c in dep["determinants"])
                j = self.columns.index(dep["dependent"])
            except (KeyError, ValueError): continue
            if j in ks or not ks: continue
            self._add_dependency(ks, j, int(dep.get("min_support", 3)), float(dep.get("confidence", .9)), "llm")
        if discover:
            for j in range(self.m):
                for k in self.related[j]:
                    if len(self.counts[k]) > .7 * self.n: continue
                    groups = defaultdict(Counter)
                    for row in self.data:
                        if row[k] and row[j]: groups[(row[k],)][row[j]] += 1
                    repeated = [c for c in groups.values() if sum(c.values()) >= 3]
                    denom = sum(sum(c.values()) for c in repeated)
                    purity = sum(max(c.values()) for c in repeated) / max(denom, 1)
                    if denom >= self.n * .25 and purity >= .96:
                        self._add_dependency((k,),j,3,.9,"dirty_discovery")

    def _add_dependency(self, ks, j, support, confidence, origin):
        if any(d["keys"] == ks and d["target"] == j for d in self.dependencies): return
        groups = defaultdict(Counter)
        for row in self.data:
            if all(row[k] for k in ks) and row[j]:
                groups[tuple(row[k] for k in ks)][row[j]] += 1
        self.dependencies.append({"keys":ks,"target":j,"support":support,"confidence":confidence,
                                  "groups":groups,"origin":origin})

    def conditional(self, i, j, value, row=None):
        """Maximum reliable FD support, excluding this cell's original observation."""
        row = self.data[i] if row is None else row
        best, evidence = 0.0, []
        for dep in self.dependencies:
            if dep["target"] != j: continue
            key = tuple(row[k] for k in dep["keys"])
            count = dep["groups"].get(key, Counter()).copy()
            original_key = tuple(self.data[i,k] for k in dep["keys"])
            if key == original_key and self.data[i,j] in count:
                count[self.data[i,j]] -= 1
            total = sum(count.values())
            if total < dep["support"] or not count: continue
            mode, n = count.most_common(1)[0]
            purity = n / total
            if purity < dep["confidence"]: continue
            support = max(count.get(value,0),0) / total
            best = max(best,support)
            evidence.append({"determinants":[self.columns[k] for k in dep["keys"]],
                             "mode":mode,"support":n,"peers":total,"purity":purity})
        return best, evidence

    def fd_violation(self, i, j, row=None):
        row = self.data[i] if row is None else row
        _, evidence = self.conditional(i,j,row[j],row)
        if not evidence: return 0.0
        # Conflicting equally reliable dependencies abstain.
        modes = {e["mode"] for e in evidence}
        if len(modes) > 1: return 0.0
        return max(e["purity"] for e in evidence) if row[j] not in modes else 0.0

    def vicinity(self, i, j, value, row=None):
        row = self.data[i] if row is None else row
        score, weights = 0.0,0.0
        for k in self.related[j]:
            peers = self.counts[k].get(row[k],0) - int(row[k] == self.data[i,k])
            if peers < 2 or not row[k]: continue
            count = self.pairs[j,k].get((row[k],value),0)
            count -= int(row[k] == self.data[i,k] and value == self.data[i,j])
            w = float(self.nmi[j,k])
            score += w * math.log((max(count,0)+.5)/(peers+.5*max(len(self.counts[j]),1)))
            weights += w
        return score / weights if weights else 0.0

    def domain_support(self, i, j, value):
        count = self.counts[j].get(value,0) - int(value == self.data[i,j])
        pat = self.patterns[j].get(pattern(value),0) - int(pattern(value) == pattern(self.data[i,j]))
        return .5 * math.log1p(max(count,0)) + .5 * math.log1p(max(pat,0))

    def contextual_candidates(self, i, j, k=3):
        original = self.data[i,j]
        row = self.data[i]
        candidates = {}
        _, evidence = self.conditional(i,j,original)
        for e in evidence:
            if e["mode"] != original:
                candidates[e["mode"]] = {"value":e["mode"],"score":e["purity"],
                    "origin":"fd_consensus", "explanation":f"{e['support']}/{e['peers']} leave-one-out peers agree",
                    "context":e}
        # NMI tuple search: only observed context matches, excluding the query row.
        neighbours = defaultdict(float)
        for c in self.related[j]:
            if not row[c]: continue
            indices = self.postings[c].get(row[c],[])
            for r in indices[:1500]:
                if r != i: neighbours[r] += float(self.nmi[j,c])
        votes = defaultdict(float)
        for r,w in sorted(neighbours.items(), key=lambda x:-x[1])[:30]:
            if self.data[r,j]: votes[self.data[r,j]] += w
        total = sum(votes.values())
        for v,w in sorted(votes.items(),key=lambda x:-x[1])[:k]:
            if v == original: continue
            candidates.setdefault(v,{"value":v,"score":min(.85,w/max(total,1e-9)),"origin":"cts",
                                     "explanation":"NMI-weighted contextual tuple agreement"})
        if original and len(original) <= 300:
            for value,score,_ in process.extract(original,list(self.counts[j]),scorer=fuzz.ratio,limit=4,score_cutoff=75):
                if value == original or self.counts[j][value] < 2: continue
                candidates.setdefault(value,{"value":value,"score":.5*score/100,"origin":"lexical",
                                             "explanation":"Observed same-column lexical neighbour"})
        return sorted(candidates.values(),key=lambda x:(-x["score"],x["value"]))[:k]
