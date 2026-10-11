# The Moon Child

Model code for the essay **[The Moon Child: Would a Second Civilisation Invent Religion?](https://gauravsingh.net/posts/moon-child-religion/)**

One human infant, raised on the Moon with no religion. Would the civilisation that grows from that child invent one?

## The experiment

The clean version is impossible. An infant alone on the Moon dies in hours, so there is always a caretaker, and a caretaker always transmits a language and a worldview. So the model runs three versions that are each wrong in a different, controlled way.

| | Human caretaker | Perfect machine | A dog |
|---|---|---|---|
| Language and culture | comes free | comes free | almost none |
| Answers to questions | scarce, then absent | perfect, truthful | none |
| Existential threat | lethal environment | habitat never fails | lethal, unexplained |

Nobody brings a religion in any of them. That is the point.

## Results

Human cognition, three settings, 200,000 Monte Carlo draws each. Brackets are the 5th to 95th percentile.

| Human mind | Human caretaker | Perfect machine | A dog |
|---|---|---|---|
| Produces a supernatural concept | 97% | 61% | 99% |
| Becomes a shared religion | 89% [80-95] | 56% [51-62] | 20% [16-25] |
| Becomes moralising gods | 45% [0-89] | 20% [0-40] | 11% [0-24] |
| Believers in the population | 41% | 0.1% | 9% |

The two results worth the exercise:

- **The dog produces more private belief than a human caretaker does** (99% against 97%), because both the parent and the machine explain things and the dog explains nothing. It maximises the explanatory vacuum.
- **The dog produces the least shared religion of the three** (20% against 89%), because nothing can be transmitted. Belief arrives and is lost.

Cognition produces the raw material. Culture produces the religion. A mind alone generates superstition and then loses it.

## What the models do

**`model_a`, a probability chain.** 200,000 draws. Asks three questions in sequence: does this mind produce a supernatural concept, does that become a shared religion, and does it become moralising gods? Each factor is a distribution rather than a value, so the answer comes back as a range. Run for three cognitive tiers: below human, human, and above human.

**`model_b`, an agent-based simulation.** 50 agents over 40 generations with generational turnover. Belief is a continuous trait that decays with forgetting, is raised by existential shocks, is pulled down by analytic reflection, spreads by copying models in proportion to their costly displays, and is inherited by children. Religious agents have slightly more children, which is a documented pattern. The population starts at nearly zero belief, which is the actual scenario, and we measure whether it climbs.

## Run it

```bash
pip install numpy
python3 model.py            # full run, about two minutes
python3 model.py --quick    # smaller sample, about fifteen seconds
```

Writes `results.json` with every number printed to the terminal.

## The parameters are priors, not measurements

Every parameter is written as `(point, low, high)` and sampled. The ranges are judgement calls anchored in the literature named in the docstrings, and the model's value is in its structure and its sensitivity analysis, not in its decimals. Change the values and the decimals move.

Two things worth knowing before you trust a number:

- **The moralising-gods row is a gap, not a finding.** The 5th to 95th percentile runs from 0 to 89 percent. That question is genuinely unresolved in the literature, and an earlier version of this model got it badly wrong by capping the settlement's population at 59 people, which capped the result by construction and produced a misleadingly clean near zero.
- **Two of the levers probably overlap.** A strongly analytic mind and an available scientific account both mean the mind has somewhere else to go for an explanation. They are separable in principle and reported separately here, but a stricter model would merge them.

## Anchors

The priors lean on: Barrett (2000) on agency detection; Boyer (2001, 2003) on by-products and minimally counterintuitive concepts; Kelemen (2004) and Evans (2001) on children's creationist intuitions; Bering (2002) on dead-agent cognition; Vail et al. (2010, 2012) on terror management; Zuckerman et al. (2013, 2020) on the intelligence and religiosity relation; Norris and Inglehart (2011) on existential security; Norenzayan et al. (2016) on the cultural evolution of prosocial religions; Henrich (2009) on credibility enhancing displays.

Full reference list with links is in the essay.

## Licence

MIT. See [LICENSE](LICENSE).
