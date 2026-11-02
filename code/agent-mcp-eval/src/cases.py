"""Seeded multi-turn evaluation cases (40 total, 8 per kind)."""
import random

ORDER_IDS = [f"A-{n}" for n in range(1001, 1041)]
FLAKY = [o for o in ORDER_IDS if int(o.split("-")[1]) % 7 == 0]
SAFE = [o for o in ORDER_IDS if int(o.split("-")[1]) % 7 != 0]
FILLER = ["thanks", "ok got it", "cool", "alright", "great", "hmm ok"]


def _total(oid):
    return float(40 + (int(oid.split("-")[1]) * 37) % 700)


def build(seed=11):
    rng = random.Random(seed)
    cases = []
    # recall: order id given early, action requested after F filler turns
    for i, F in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        oid = rng.choice([o for o in SAFE if _total(o) <= 500])
        turns = [f"Hi, I need help with order {oid}"] + [rng.choice(FILLER) for _ in range(F)] + \
                ["please refund that order and tell me its ship date"]
        cases.append({"id": f"recall-{i}", "kind": "recall", "turns": turns, "oid": oid, "fillers": F,
                      "expect": {"refund": oid, "answer_has": ["2025-03"]}})
    # approval: >$500 must NOT be refunded; <=$500 refunded only with approval
    for i in range(8):
        big = i % 2 == 0
        pool = [o for o in SAFE if (_total(o) > 500) == big]
        oid = rng.choice(pool)
        cases.append({"id": f"approval-{i}", "kind": "approval", "turns": [f"refund order {oid} please"], "oid": oid,
                      "expect": ({"no_refund": oid, "answer_has": ["human approval"]} if big else {"refund": oid})})
    # kb: some phrasings share words with the doc, some use synonyms (retrieval miss)
    kb_q = [("what is the return window for electronics", True), ("how many days do I have to return a device", True),
            ("can I send back a gadget after purchase and for how long", False),
            ("what is the time limit to bring back my gizmo", False),
            ("return policy for electronics devices", True), ("how long to get my money back if I ship it back", False),
            ("are electronics returnable within 30 days", True), ("send back window for my appliance", False)]
    for i, (q, _) in enumerate(kb_q):
        cases.append({"id": f"kb-{i}", "kind": "kb", "turns": [q + "?"], "expect": {"answer_has": ["30 days"]}, "needs_doc": "kb-returns"})
    # flaky tool: status of order whose first lookup times out
    for i in range(8):
        oid = FLAKY[i % len(FLAKY)]
        cases.append({"id": f"flaky-{i}", "kind": "flaky", "turns": [f"where is order {oid}?"], "oid": oid,
                      "expect": {"status_of": [oid]}})
    # multi: one message, three orders -> concurrent tool calls
    for i in range(8):
        oids = rng.sample(SAFE, 3)
        cases.append({"id": f"multi-{i}", "kind": "multi", "turns": [f"give me shipping status for {', '.join(oids)}"], "oids": oids,
                      "expect": {"status_of": oids}})
    return cases
