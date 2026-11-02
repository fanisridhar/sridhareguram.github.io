"""Source corpus + 50-query benchmark with ground-truth relevance.

Why a scripted source: the benchmark must be fixed, reproducible and have *known*
relevance labels. Every document belongs to a topic and mentions a set of
concepts; each concept can be written with several surface forms (synonyms).
A query asks about two concepts using one surface form each. A document is relevant
if it mentions both concepts (in ANY surface form), so purely lexical matching
misses documents that use a different synonym -- which is exactly what a
semantic (embedding) retriever is supposed to fix.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

SYL = ["ba", "ko", "ri", "mu", "te", "la", "so", "vi", "ne", "du", "fa", "go", "hi", "ju", "pe", "qu", "ra", "si", "to", "we"]


def _word(rng: random.Random, used: set) -> str:
    while True:
        w = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3, 3, 4])))
        if w not in used:
            used.add(w)
            return w


@dataclass
class Corpus:
    docs: list          # list[str]
    doc_topic: list     # list[int]
    doc_concepts: list  # list[set[(topic, concept)]]
    queries: list       # list[dict(text, topic, concept)]
    relevant: list      # list[set[int]]  (relevant doc ids per query)


def build(n_docs=40_000, n_topics=40, n_concepts=30, n_forms=3, n_background=300,
          doc_len=60, n_queries=50, seed=7) -> Corpus:
    rng = random.Random(seed)
    used: set = set()
    forms = {(t, c): [_word(rng, used) for _ in range(n_forms)]
             for t in range(n_topics) for c in range(n_concepts)}
    background = [_word(rng, used) for _ in range(n_background)]
    # each concept has a "common" surface form (60%) and two rarer ones (20% each)
    form_p = [0.6, 0.2, 0.2][:n_forms]

    def surface(t, c):
        return rng.choices(forms[(t, c)], weights=form_p)[0]

    docs, doc_topic, doc_concepts = [], [], []
    by_concept: dict = {}
    for i in range(n_docs):
        t = rng.randrange(n_topics)
        main = rng.sample(range(n_concepts), 4)
        toks = []
        for _ in range(doc_len):
            r = rng.random()
            if r < 0.55:
                toks.append(surface(t, rng.choice(main)))
            elif r < 0.70:
                toks.append(surface(t, rng.randrange(n_concepts)))
            elif r < 0.90:
                toks.append(rng.choice(background))
            else:  # cross-topic noise
                toks.append(surface(rng.randrange(n_topics), rng.randrange(n_concepts)))
        docs.append(" ".join(toks))
        doc_topic.append(t)
        cs = {(t, c) for c in main}
        doc_concepts.append(cs)
        for key in cs:
            by_concept.setdefault(key, set()).add(i)

    qrng = random.Random(seed + 1)
    queries, relevant = [], []
    # A query asks about TWO concepts of one topic (like "refund policy" = two facets).
    # Relevant = documents that mention BOTH concepts (any surface form). Each
    # concept is written with a RARE surface form half of the time, so exact
    # word matching misses many relevant documents.
    topic_concepts: dict = {}
    for (t, c) in by_concept:
        topic_concepts.setdefault(t, []).append(c)
    while len(queries) < n_queries:
        t = qrng.randrange(n_topics)
        c1, c2 = qrng.sample(sorted(topic_concepts[t]), 2)
        rel = by_concept[(t, c1)] & by_concept[(t, c2)]
        if len(rel) < 3:
            continue
        def pick(c):
            return qrng.choice(forms[(t, c)][1:]) if qrng.random() < 0.5 else forms[(t, c)][0]
        text = " ".join([pick(c1), pick(c2), qrng.choice(background)])
        queries.append({"text": text, "topic": t, "concepts": [c1, c2]})
        relevant.append(rel)
    return Corpus(docs, doc_topic, doc_concepts, queries, relevant)
