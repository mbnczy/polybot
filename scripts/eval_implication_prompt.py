"""
scripts/eval_implication_prompt.py
──────────────────────────────────
Score the implication classifier on a labelled set of real market pairs, with
their resolution rules — before changing the prompt, and after.

    python scripts/eval_implication_prompt.py <tag> [repeats]

tests/fixtures/implication_eval.json: 59 pairs from the live window and from
pairs the model once got wrong (a player advancing ⊆ match completed, a
first-half spread ⊆ the full spread, US #1 ⊆ global #1, an exact score read as
Over), plus a held-out set of both-teams-to-score pairs the prompt never
mentions. Measured 2026-09-16 on sztaki_pipeline-gemma4:31b, three runs each:

    prompt                  false positives   true found   held-out FP / TP
    titles only             24/60  (40%)      71/72        —
    rules + counterexample   0/87  ( 0%)      90/90        0/27, 18/18

Calls the configured model; places nothing.
"""
import sys, json, inspect, logging, collections, time
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dotenv import load_dotenv
load_dotenv(os.environ.get("IMPLICATION_ENV_FILE", "/home/ubuntu/polybot-dev/cross-market/.env"))
import strategy.implication_mapper as im
from strategy.outcomes import market_outcomes
from concurrent.futures import ThreadPoolExecutor

D = os.environ.get("EVAL_OUT", "/tmp")
DATASET = os.path.join(REPO, "tests", "fixtures", "implication_eval.json")
tag, repeats = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
data = json.load(open(DATASET)); items, markets = data["items"], data["markets"]

warnings = collections.Counter()
class H(logging.Handler):
    def emit(self, r):
        msg = r.getMessage()
        for k in ("unparseable", "request failed", "rejected verdict", "refused"):
            if k in msg: warnings[k] += 1
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.INFO)

provider = im.resolve_provider(); model = im.resolve_model(provider); client = im.build_client(provider)
takes_rules = "rules" in inspect.signature(im.classify_one).parameters
from strategy.market_rules import rules_digest
rules = {cid: rules_digest(m.get("description")) for cid, m in markets.items()}

def run(item):
    ma, mb = markets[item["a"]], markets[item["b"]]
    c = im.Candidate(a_id=item["a"], b_id=item["b"], a_title=ma["question"], b_title=mb["question"],
                     overlap=0.8, same_event=True, event_id="e",
                     a_outcomes=market_outcomes(ma), b_outcomes=market_outcomes(mb))
    kw = {"rules": rules} if takes_rules else {}
    rel = im.classify_one(client, provider, model, c, **kw)
    if rel is None or rel.confidence < 0.90:
        return "NONE"
    return "A_IMPLIES_B" if rel.narrow == item["a"] else "B_IMPLIES_A"

jobs = [(i, it) for i, it in enumerate(items) for _ in range(repeats)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    preds = list(pool.map(lambda j: run(j[1]), jobs))
by_item = collections.defaultdict(list)
for (i, _), p in zip(jobs, preds): by_item[i].append(p)

per_class = collections.defaultdict(lambda: [0, 0])
fp = tp = fn = wrong_dir = 0
report = []
for i, it in enumerate(items):
    for p in by_item[i]:
        ok = p == it["expected"]
        per_class[it["cls"]][0] += ok; per_class[it["cls"]][1] += 1
        if it["expected"] == "NONE" and p != "NONE": fp += 1
        if it["expected"] != "NONE":
            if ok: tp += 1
            elif p == "NONE": fn += 1
            else: wrong_dir += 1
    report.append({**it, "preds": by_item[i]})
n_false = sum(repeats for it in items if it["expected"] == "NONE")
n_true = sum(repeats for it in items if it["expected"] != "NONE")
print(f"[{tag}] {len(jobs)} calls in {time.time()-t0:.0f}s on {provider.name}/{model} | rules in prompt: {takes_rules}")
print(f"  FALSE pairs accepted (false positives): {fp}/{n_false} ({fp/n_false*100:.0f}%)")
print(f"  TRUE pairs found in the right direction: {tp}/{n_true} ({tp/n_true*100:.0f}%)  missed {fn}  wrong direction {wrong_dir}")
for cls, (ok, n) in sorted(per_class.items()): print(f"    {ok:3}/{n:<3} {cls}")
hold = [r for r in report if r.get("holdout")]
if hold:
    hf = sum(1 for r in hold for p in r["preds"] if r["expected"] == "NONE" and p != "NONE")
    ht = sum(1 for r in hold for p in r["preds"] if r["expected"] != "NONE" and p == r["expected"])
    print(f"  HELD-OUT only: false positives {hf}/{sum(repeats for r in hold if r['expected']=='NONE')}, true found {ht}/{sum(repeats for r in hold if r['expected']!='NONE')}")
print("  warnings:", dict(warnings))
json.dump(report, open(f"{D}/result_{tag}.json", "w"), indent=1)
