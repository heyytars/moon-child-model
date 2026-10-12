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
| Produces a supernatural concept | 96% | 61% | 99% |
| Becomes a shared religion | 89% [80-95] | 56% [51-62] | 20% [16-25] |
| Becomes moralising gods | 45% [0-89] | 20% [0-40] | 11% [0-24] |
| Believers in the population | 41% | 0.1% | 9% |

What the table shows, and what it does not (details in "What a stress test found" below):

- **Dog, shared religion: 20% against 89%.** This is the size of the consequence of giving the dog almost no language. It is an input coming back out, not a discovery that culture is needed.
- **Dog, private belief: 99% against 96%.** Not a finding. The gap is under three points, comes from giving the dog no science, and reverses if the dog gets the caretaker's 20%.
- **The ordering caretaker, machine, dog** is the part that survives every change tried.

That cognition supplies the raw material and culture supplies the religion is the standard position of the cognitive science of religion. It rests on the literature cited in the essay, not on this table.

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

The parameters of the probability chain are written as `(point, low, high)` and sampled. The ranges are judgement calls anchored in the literature named in the docstrings. **That is not true of everything.** The structure of the chain (a product of independent terms, with hand-set coefficients like `0.15 + 0.85 * language`) and eight constants in the agent model are fixed by hand and never sampled. The agent-model constants were tuned until belief could appear at all.

Read every number as the consequence of a stated assumption, not as an estimate.

## What a stress test found

The numbers below were produced by importing this file unchanged and attacking it. Seeds and Monte Carlo noise are not the problem (standard deviation 0.0001 across seeds). The problem is what the outputs are made of.

| Claim in the essay | What the stress test found |
|---|---|
| Dog: 20% shared religion, against 89% for a human caretaker | This is the language prior. Inside the dog arm, language alone explains 90% of the variance. Give the dog the caretaker's language prior and it scores 91%. Give the caretaker the dog's and it scores 19%. |
| The dog produces more private belief than a parent (99% against 96%) | The gap reverses (to -0.7 points) if the dog is given the same 20% science as the caretaker. 77% of the dog's draws hit the ceiling at 100%, and the unclipped mean is 1.04, so the number is not a probability. |
| Ordering: caretaker above machine above dog | Survives three functional forms and priors stretched 3x. |
| Size of the gaps | Does not survive. The dog is 22% of the caretaker under a product form and 75% under an additive one. At 3x wider priors the caretaker and the machine bands overlap. |
| Teleology, death awareness and pattern seeking do not matter | Built in. They feed an "any one of these will do" term that sits at 0.9996, so nothing can move it. |
| Agent model: "switch, not a dial" | A threshold artifact. Across a twelvefold change in shocks, mean belief moves from 0.45 to 0.53. The 0.5 line cuts through that range. With zero shocks, mean belief is still 0.44. |
| Agent model: caretaker believers 41% | 26% with a 5th to 95th percentile of 5 to 68 once the priors are propagated. Moving `base_gen` or `fidelity` by 50% moves it from about 1% to about 100%. |

One bug was fixed after this audit: `equilibrium_belief` reported only the last replicate's mean, and is now the mean over all replicates. The headline numbers are identical before and after. The docstring and this README also no longer claim every parameter is sampled, which was false.

Two things worth knowing before you trust a number:

- **The moralising-gods row is a gap, not a finding.** The 5th to 95th percentile runs from 0 to 89 percent. That question is genuinely unresolved in the literature, and an earlier version of this model got it badly wrong by capping the settlement's population at 59 people, which capped the result by construction and produced a misleadingly clean near zero. The model also bakes in "scale comes first," which two analyses support but which is not settled.
- **Two of the levers probably overlap.** A strongly analytic mind and an available scientific account both mean the mind has somewhere else to go for an explanation. They are separable in principle and reported separately here, but a stricter model would merge them.

## Anchors

The priors lean on: Barrett (2000) on agency detection; Boyer (2001, 2003) on by-products and minimally counterintuitive concepts; Kelemen (2004) and Evans (2001) on children's creationist intuitions; Bering (2002) on dead-agent cognition; Vail et al. (2010, 2012) on terror management; Zuckerman et al. (2013, 2020) on the intelligence and religiosity relation; Norris and Inglehart (2011) on existential security; Norenzayan et al. (2016) on the cultural evolution of prosocial religions; Henrich (2009) on credibility enhancing displays.

Full reference list with links is in the essay.

## Licence

MIT. See [LICENSE](LICENSE).
