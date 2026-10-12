#!/usr/bin/env python3
"""
The moon child: can a mind alone invent religion?

One thought experiment, three ways of raising the child, one model.

    1. human caretaker  a person raises the child. Language and culture come
                        free, science is scarce, the environment is lethal.
    2. perfect machine  a machine with no beliefs raises the child. It cannot
                        transmit a religion, but it answers every question
                        truthfully and keeps the habitat running.
    3. a dog            a dog is the child's only companion. Nothing is
                        explained, almost nothing can be transmitted, but the
                        child is not alone.

Two models:

    model_a   a probability chain, 200,000 Monte Carlo draws, run for three
              cognitive tiers (below human, human, above human). Asks: does this
              mind produce a supernatural concept, does it become a shared
              religion, does it become moralising gods?

    model_b   an agent-based simulation, 50 agents over 40 generations. Asks
              whether a population starting near zero belief converges on a
              shared one.

The parameters of model_a are priors written as (point, low, high), and they are
sampled. Their ranges are judgement calls anchored in the literature named below.

Two things are NOT sampled, and an audit showed they decide the headline:
  * the structural form of model_a (a product of independent terms, with
    hand-set coefficients such as 0.15 + 0.85 * language), and
  * eight constants in model_b (base_gen, decay, shock_gain, analytic_pull,
    trans, fidelity, fertility_bias, seed_belief). They were tuned until belief
    could appear at all. Moving each by -50% / +50% changes the caretaker's
    "believers" (published 41%) as follows: base_gen 1% / 98%, fidelity
    5% / 100%, decay 75% / 16%, analytic_pull 73% / 17%, shock_gain 21% / 59%.
    trans and fertility_bias barely matter: copying a random peer preserves the
    mean, so the "costly display" mechanism does not move the headline.
So read the numbers as the consequences of stated assumptions, not as estimates.
The ordering caretaker > machine > dog survives every change I tried. The
magnitudes do not.

Run:  python3 model.py            (~2 minutes; prints all tables, writes results.json)
      python3 model.py --quick    (smaller sample, ~15 seconds)

Reference: "The moon child: Can a mind alone invent religion?"
https://gauravsingh.net/posts/moon-child-religion/
"""
from __future__ import annotations

import argparse
import json
import numpy as np

SEED = 20261011

# ---------------------------------------------------------------------------
# Parameter anchors. Each prior is (point, low, high).
#
#   agency      Barrett 2000, "Exploring the natural foundations of religion":
#               hyperactive agency detection, pan-human.
#   tom         theory of mind able to represent an invisible mind. Boyer 2003,
#               "Religious thought and behaviour as by-products of brain
#               function".
#   teleology   Kelemen 2004, "Are Children Intuitive Theists?"; Evans 2001 on
#               creationist origin beliefs.
#   death       Bering 2002 on dead-agent cognition; Vail et al. 2010 and 2012
#               on terror management.
#   pattern     apophenia / meaning detection, part of the by-product cluster.
#   mci         Boyer 2001, minimally counterintuitive concepts are recalled
#               better than fully intuitive ones.
#   analytic    Zuckerman et al. 2013 and 2020: analytic cognitive style partly
#               mediates the negative intelligence-religiosity relation
#               (r about -0.20 to -0.24 across 83 studies).
#
# Environment priors:
#   language              linguistic scaffolding the caretaker supplies. Without
#                         language there is no channel for a shared belief
#                         system. A dog supplies essentially none.
#   science_available     whether an alternative explanatory account is to hand.
#   existential_insecurity  Norris & Inglehart 2011, Sacred and Secular: the
#                         theory of existential security. A lethal environment
#                         scores high; a habitat that never fails scores low.
# ---------------------------------------------------------------------------
PRIORS = {
    "below": {
        "agency":    (0.80, 0.60, 0.95),
        "tom":       (0.15, 0.05, 0.40),
        "teleology": (0.50, 0.30, 0.70),
        "death":     (0.30, 0.10, 0.60),
        "pattern":   (0.70, 0.50, 0.90),
        "mci":       (0.40, 0.20, 0.65),
        "analytic":  (0.10, 0.00, 0.25),
    },
    "human": {
        "agency":    (0.97, 0.90, 1.00),
        "tom":       (0.98, 0.92, 1.00),
        "teleology": (0.90, 0.70, 0.98),
        "death":     (0.98, 0.90, 1.00),
        "pattern":   (0.95, 0.85, 1.00),
        "mci":       (0.80, 0.60, 0.95),
        "analytic":  (0.25, 0.12, 0.40),
    },
    "above": {
        "agency":    (0.98, 0.92, 1.00),
        "tom":       (0.99, 0.95, 1.00),
        "teleology": (0.85, 0.60, 0.97),
        "death":     (1.00, 0.98, 1.00),
        "pattern":   (0.97, 0.88, 1.00),
        "mci":       (0.85, 0.65, 0.97),
        "analytic":  (0.78, 0.60, 0.95),
    },
}

# Belief is never transmitted by the caretaker in any scenario: the point of the
# exercise is that nobody brings a religion with them.
SCENARIOS = {
    "human caretaker": {
        "language":              (0.95, 0.85, 1.00),
        "science_available":     (0.20, 0.05, 0.45),
        "existential_insecurity": (0.90, 0.70, 1.00),
        "transmitted_religion":  0.0,
    },
    "perfect machine": {
        "language":              (1.00, 0.98, 1.00),
        "science_available":     1.0,
        "existential_insecurity": (0.25, 0.10, 0.45),
        "transmitted_religion":  0.0,
    },
    "a dog": {
        "language":              (0.05, 0.00, 0.15),
        "science_available":     0.0,
        "existential_insecurity": (0.85, 0.65, 1.00),
        "transmitted_religion":  0.0,
    },
}

MEGA = 1e6          # population at which moralising high gods are expected to appear
N_TRIALS = 200_000


def sample(triple, n, rng):
    """Bounded triangular sample. Accepts a scalar for a fixed value."""
    if isinstance(triple, (int, float)):
        return np.full(n, float(triple))
    point, lo, hi = triple
    if hi <= lo:
        return np.full(n, float(point))
    return np.clip(rng.triangular(lo, point, hi, n), 0.0, 1.0)


def model_a(tier, env, rng, n=N_TRIALS):
    """
    Probability chain:
      seed      does this mind produce a supernatural concept on its own?
      shared    does it become a shared religion?
      big       does it become moralising gods?

    Note the moralising-god step depends on the settlement's population, which
    is drawn from a growth process rather than fixed. An earlier version capped
    population at 59 people, which capped the result by construction and
    produced a misleadingly clean near-zero.
    """
    p = PRIORS[tier]
    agency, tom = sample(p["agency"], n, rng), sample(p["tom"], n, rng)
    teleology, death = sample(p["teleology"], n, rng), sample(p["death"], n, rng)
    pattern, mci = sample(p["pattern"], n, rng), sample(p["mci"], n, rng)
    analytic = sample(p["analytic"], n, rng)
    language = sample(env["language"], n, rng)
    science = sample(env["science_available"], n, rng)
    insecurity = sample(env["existential_insecurity"], n, rng)

    driver = 1 - (1 - teleology) * (1 - death) * (1 - pattern)
    seed = agency * tom * driver * (1 - analytic * 0.6) * (1 - 0.30 * science)
    seed = np.clip(seed * (1 + 0.4 * insecurity), 0.0, 1.0)

    # settlement history: growth rate and length both uncertain, and many small
    # colonies never boom
    gens = rng.integers(10, 400, n)
    growth = rng.uniform(1.05, 1.55, n)
    stall = rng.random(n) < 0.35
    pop = np.power(growth, gens) * 2.0
    pop = np.where(stall, np.minimum(pop, rng.uniform(50, 5000, n)), pop)
    pop = np.minimum(pop, 5e8)

    pop_factor = 1 - np.exp(-(np.minimum(pop, 200) - 1) / 8.0)
    lang_factor = 0.15 + 0.85 * language
    ritual = 0.5 + 0.4 * insecurity
    cred = 0.55 + 0.25 * mci
    shared = seed * pop_factor * lang_factor * (1 - (1 - ritual) * (1 - cred) * 0.6)
    shared = np.clip(shared, 0.0, 1.0)

    scale = 1 - np.exp(-np.maximum(pop - MEGA * 0.1, 0) / (MEGA * 2.0))
    scale = np.where(pop < MEGA * 0.1, 0.0, scale)
    big = np.clip(shared * scale * (1 - 0.35 * science), 0.0, 1.0)

    return {"private_belief": seed, "shared_religion": shared,
            "institutionalized": big, "pop": pop}


def model_b(tier, env, rng, n_agents=50, generations=40, reps=300, seed_belief=0.02):
    """
    Agent-based. Belief is a continuous trait with a homeostat:

      decay        forgetting, proportional to current belief
      generation   the by-product machinery runs continuously, so minds produce
                   supernatural ideas at a low rate all the time
      shock        existential events, with the gain scaled by how available
                   supernatural explanations already are (cultural scaffolding)
      analytic     reflection pulls belief down, more strongly for sharper minds
      trans        copying a model, weighted by that model's costly display,
                   scaled by whether language exists at all
      vertical     offspring inherit from a parent at high fidelity
      fertility    religious groups are pronatalist (Norenzayan et al. 2016)

    The population starts near zero belief, which is the actual scenario, and we
    measure whether it climbs.
    """
    p = PRIORS[tier]
    represent = p["agency"][0] * p["tom"][0]
    analytic = p["analytic"][0]
    mci = p["mci"][0]
    science = (env["science_available"] if isinstance(env["science_available"], (int, float))
               else env["science_available"][0])
    insecurity = env["existential_insecurity"][0]
    language = env["language"][0]

    explain_damp = 1 - 0.45 * science
    channel = 0.15 + 0.85 * language

    base_gen = 0.18 * explain_damp
    shock_rate = 0.15 * insecurity
    shock_gain = 0.30 * represent * explain_damp
    decay = 0.04
    analytic_pull = 0.12 * analytic
    trans = 0.30 * (0.5 + 0.5 * mci) * channel
    fidelity = 0.88 * channel + 0.05
    fertility_bias = 0.25

    prevalence, consensus, tips = [], 0, 0
    means = []
    for _ in range(reps):
        b = np.clip(np.full(n_agents, seed_belief) + rng.normal(0, 0.01, n_agents), 0, 1)
        for _g in range(generations):
            availability = float(b.mean())
            scaffold = 0.25 + 0.75 * availability

            b *= (1 - decay)
            b = b + base_gen * scaffold * represent * (1 - b)

            shock = rng.random(n_agents) < shock_rate
            b = b + shock * (1 - b) * shock_gain * scaffold
            b *= (1 - analytic_pull)

            model = rng.integers(0, n_agents, n_agents)
            display = np.clip(b[model] * (0.5 + 0.5 * mci), 0, 1)
            b = b + (b[model] - b) * trans * (0.4 + display)
            b = np.clip(b + rng.normal(0, 0.015, n_agents), 0, 1)

            fit = 1 + fertility_bias * b
            parents = rng.choice(n_agents, size=n_agents, p=fit / fit.sum())
            b = fidelity * b[parents] + (1 - fidelity) * rng.beta(1.5, 4, n_agents)
            b = np.clip(b + rng.normal(0, 0.02, n_agents), 0, 1)
        means.append(float(b.mean()))
        f = float((b > 0.5).mean())
        prevalence.append(f)
        if f >= 0.8:
            consensus += 1
        if b.mean() > 0.5:
            tips += 1
    return {
        "p_consensus": consensus / reps,
        "p_tipped": tips / reps,
        "mean_prevalence": float(np.mean(prevalence)),
        # mean belief averaged over ALL replicates. An earlier version reported
        # only the last replicate's mean here, which is one noisy draw.
        "equilibrium_belief": float(np.mean(means)),
        "belief_sd": float(np.std(means)),
    }


def sensitivity(base_env, n=60_000):
    """Move one prior at a time across its range and see what the answer does."""
    tier = "human"
    base = model_a(tier, base_env, np.random.default_rng(11), n)["shared_religion"].mean()
    rows = []

    for key, (point, lo, hi) in PRIORS[tier].items():
        for target in (lo, hi):
            saved = PRIORS[tier][key]
            PRIORS[tier][key] = (target, target, target)
            v = model_a(tier, base_env, np.random.default_rng(11), n)["shared_religion"].mean()
            PRIORS[tier][key] = saved
            rows.append((f"{key} = {target:.2f}", v, v - base))

    for key in ("language", "science_available", "existential_insecurity"):
        point, lo, hi = base_env[key]
        for target in (lo, hi):
            env = dict(base_env)
            env[key] = (target, target, target)
            v = model_a(tier, env, np.random.default_rng(11), n)["shared_religion"].mean()
            rows.append((f"{key} = {target:.2f}", v, v - base))

    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return base, rows


def tipping(base_env, reps=200):
    """How does the answer change with the rate of existential shocks?"""
    out = {}
    for multiplier in (0.02, 0.05, 0.09, 0.15, 0.25):
        insecurity = multiplier / 0.15
        env = dict(base_env)
        env["existential_insecurity"] = (insecurity, insecurity, insecurity)
        r = model_b("human", env, np.random.default_rng(3), reps=reps)
        out[f"{multiplier:.2f}"] = {"p_tipped": r["p_tipped"],
                                   "equilibrium_belief": r["equilibrium_belief"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="smaller sample, faster")
    args = ap.parse_args()

    global N_TRIALS
    reps = 60
    if args.quick:
        N_TRIALS = 20_000
        reps = 60

    results = {"model_a": {}, "model_b": {}, "sensitivity": {}, "tipping": {}}

    for name, env in SCENARIOS.items():
        print("=" * 78)
        print(name.upper())
        print("=" * 78)
        print(f"{'mind':>12} | {'private':>8} | {'shared':>18} | {'big gods':>18}")
        results["model_a"][name], results["model_b"][name] = {}, {}
        for tier in ("below", "human", "above"):
            r = model_a(tier, env, np.random.default_rng(SEED))
            row = {
                "private": float(r["private_belief"].mean()),
                "shared": float(r["shared_religion"].mean()),
                "shared_ci": [float(np.percentile(r["shared_religion"], 5)),
                              float(np.percentile(r["shared_religion"], 95))],
                "big": float(r["institutionalized"].mean()),
                "big_ci": [float(np.percentile(r["institutionalized"], 5)),
                           float(np.percentile(r["institutionalized"], 95))],
                "pop_reaches_mega": float((r["pop"] >= MEGA).mean()),
            }
            results["model_a"][name][tier] = row
            print(f"{tier:>12} | {row['private']:>7.1%} | {row['shared']:>6.1%} "
                  f"[{row['shared_ci'][0]:.0%}-{row['shared_ci'][1]:.0%}] | "
                  f"{row['big']:>6.1%} [{row['big_ci'][0]:.1%}-{row['big_ci'][1]:.1%}]")
        b = model_b("human", env, np.random.default_rng(7), reps=reps * 5)
        results["model_b"][name] = b
        print(f"  agent model, human mind: believers {b['mean_prevalence']:.1%}, "
              f"mean belief {b['equilibrium_belief']:.2f}, "
              f"swings to majority {b['p_tipped']:.1%}")
        print()

    base, rows = sensitivity(SCENARIOS["human caretaker"])
    results["sensitivity"] = {"base": base,
                              "rows": [{"label": l, "value": v, "delta": d} for l, v, d in rows]}
    print("=" * 78)
    print(f"SENSITIVITY  human caretaker, baseline shared religion = {base:.1%}")
    print("=" * 78)
    for l, v, d in rows[:10]:
        print(f"  {l:>32} : {v:>6.1%}  ({d:+.1%})")

    # 200 reps to match the figures published in the essay
    t = tipping(SCENARIOS["human caretaker"], reps=200)
    results["tipping"] = t
    print()
    print("=" * 78)
    print("TIPPING  human caretaker, human mind")
    print("=" * 78)
    print(f"{'shocks/generation':>18} | {'swings to majority':>18} | {'mean belief':>11}")
    for k, v in t.items():
        print(f"{k:>18} | {v['p_tipped']:>17.1%} | {v['equilibrium_belief']:>11.2f}")

    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
