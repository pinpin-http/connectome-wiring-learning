# Extending the study

The preprint leaves seven open questions, listed here by how directly they bear on its main claim. Each entry says
what is missing, where the code is, and what the computation costs. Costs are on one RTX 3090 (24 GB) unless stated.
The measured unit times are: one visual-task run of one graph ~15 min (2 tasks x 6 learners x 6,000 trials), one
tuning batch of the equal-tuning controls ~4.6 min, one main run ~17 min.

Before comparing anything, read the GPU note in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Runs are not
bit-identical across card models, so every run in one comparison must use the same card, and a new experiment
reruns its own reference instead of reusing the published numbers.

## 1. Bracket the equal-budget tuning optimum

Every one of the 21 graphs selected the same corner of the 9-point tuning grid (gamma 0.063, eta 0.003, tau 1;
`configs/revision48/campaign.yaml`, `tuning.trials`). The lead after tuning is therefore not measured at each
graph's best setting.

Add trials beyond the corner (for example gamma 0.016, eta 0.01, tau 0.5 and their combinations), select again per
graph, rerun the tuned condition, and report the four contrasts G with the same bootstrap. Declare the new grid and
the reading before running. Do not edit the frozen config; write a new one.

Code: `scripts/revision48/r48.py` (`prepare`, `tune`, `select`, `run`), `scripts/revision48/campaign.py`
(queue), `src/flymario/revision48/analysis.py` (contrasts and bootstrap). `r48.py` reads the config path from
`CFG_P`; point it at the new config. `configs/revision48/graph_manifest.json` lists the 21 graphs with their
SHA-256; the nulls are rebuilt with `make nulls` (family S) and `make nulls FAMILY=E`.

Cost with 6 new trials (2 batches per graph): 42 batches x 4.6 min + 21 main runs x 17 min, about 9-10 h.

## 2. Correct the sign of the ON pathway

The eye injects a current that rises with luminance into L1 and L5 (`src/flymario/env/vision.py`, `ON_TYPES`). In
the fly, light hyperpolarizes every lamina cell. L1 is glutamatergic in MaleCNS (1,776 of 1,776) and the model makes
glutamate inhibitory (`src/flymario/connectome/malecns.py`, `FAST_SIGN`), so Mi1 and Tm3 receive the opposite sign
from the animal's. The comparison between graphs is unaffected, since every graph gets the same input, but no result
says anything about ON processing.

The fix is either to give lamina cells a tonic current that light reduces, or to treat the L1 to Mi1 synapse as a
sign-inverting relay. Both change every experiment's input, so every experiment has to be rerun.

Cost: the whole study, about 20 h (visual tasks, 81 graphs) + 40 h (equal-tuning controls) + 34 h (Mario) + a few
hours for the frozen-network experiments, about 100 h.

## 3. A null that keeps the wiring between cell types

Families S and E keep degrees, signs, weights, depth, laterality and one strength per neuron, but not which cell types
connect to which. The advantage could come from type-level wiring, which repeats across flies, or from this
individual's detail.

Rewire edges only within each (source type, target type) class, check the preserved properties, and run the visual
tasks on 40 such nulls.

Code: `src/flymario/connectome/randomize.py` (`_rewire_class` already rewires within a class; the existing families
define the classes in `depth_stratified*`), checks in `scripts/diagnostics/null_validate.py`, tests in
`tests/unit/`, visual tasks with `make study-s` (see the README).

Cost: about 10 h (40 graphs x 15 min) plus the null builds on CPU (~4 min each).

## 4. A readout learned by reward

Only the synapses onto the output neurons learn, and a fixed projection maps the output neurons to the eight actions.
The wiring decides which of these synapses exist and sets their initial weights. The advantage might depend on these
wiring-constrained synapses, or it might survive with every connectome weight frozen and a linear readout learned by
reward, the classical reservoir setting. This control was planned as an option of the equal-tuning controls and never
implemented (`configs/revision48/campaign.yaml`, `conditions.readout`).

Freeze every connectome weight, train a linear readout of the output neurons' activity with the same reward signal,
and compare the connectome with the same nulls.

Code: learning rule in `src/flymario/training/rules.py` and `plastic.py`; imposed-action learning in
`src/flymario/experience_commune/learn.py`.

Cost: the same order as one condition of the equal-tuning controls, 21 runs x ~17 min, about 6 h.

## 5. Replicate on a second connectome

The study uses one connectome from one individual. FlyWire 783 (female brain) is the natural replication.

Build the graph, the two null families and the eye map for FlyWire, then rerun the visual tasks.

Code: `make download-all` fetches the tables; `src/flymario/connectome/flywire.py` is the adapter. No reported
experiment used it, and it has not been validated like the MaleCNS build (`scripts/check_connectome.py`). Expect
work on the eye map (`vision.py` reads lamina positions), because FlyWire covers the brain without the ventral cord.

Cost: engineering first, then about 20 h for 81 graphs.

## 6. Measure the mechanism

The preprint shows that the connectome learns faster but does not say why. The connectome is the least active graph
at the common setting; activity at the tuned setting is not reported; saturation and sensitivity of the output
neurons are not measured during learning.

From existing runs, report the output activity and saturation at the tuned setting, and the distribution of weight
updates per synapse. Then test one candidate directly, for example by matching activity neuron by neuron rather than
on average.

Code: each run of the equal-tuning controls writes `diagnostics.npz` next to `run.json`
(`artifacts/revision48/runs/...`; not committed, regenerated by `r48.py run`).

Cost: analysis of existing outputs needs only a CPU; each new experiment takes a few hours.

## 7. Mario with memory between decisions

The network is reset before every decision, so it cannot integrate motion over time. No network, connectome or
null, learns to play.

Keep the network state between decisions (or add an explicit memory), check that a network can then learn the jump
timing, and only then compare wirings.

Code: `src/flymario/training/mario.py`, `mario_sweep.py`, `src/flymario/env/`; play harness in
`scripts/gates/play_harness_v2.py`. Requires your own ROM ([ROM.md](ROM.md)).

Cost: the Mario mission took 34 GPU hours; a memory variant should be validated on one graph before any comparison.
