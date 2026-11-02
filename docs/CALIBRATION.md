# Calibration — what is constrained, what is assumed, what is free

**Date: 2026-09-22. Status: method and parameter audit. No frozen value is
changed by this document.**

Cited by `env/vision.py` and `models/spiking_reservoir.py` since the first
commit; it did not exist until now.

---

## 0. The principle this document enforces

> Validate the model with realistic inputs. Do not retune the model so it passes
> the benchmark.

Nothing in the selection path may read a reward, a score, a completion rate or a
Mario outcome. Real inputs characterise the regime a parameter set produces;
they are not a target to tune against.

Two consequences, both binding:

- **The regional spread is a diagnostic, not an objective.** Driving every
  region to the same rate is achievable by raising the gain until they all
  saturate, which is the trap `JOURNAL` recorded at Gate B and Gate E
  reproduced: spread falls monotonically with gain because saturation flattens
  ratios. A flat spread bought that way means less, not more.
- **If several values remain defensible, they stay as a sensitivity analysis.**
  Collapsing them to one because it performs better downstream is the failure
  this document exists to prevent.

---

## 1. Why the old calibration stimulus was replaced

Every operating point frozen before today was measured against
`calibrate.probe_current`: a uniform amplitude on every lamina neuron, held
still. Gate E measured what that costs (`JOURNAL`, 2026-09-22c):

| | probe | the real interface |
|---|---|---|
| cells driven | **7,114** — both eyes | **3,575** — right eye only |
| amplitude | 2.00 uniform | ~1.00 mean, 1.34 p99 |
| total current | 14,228 | ~3,575 — **0.25×** |
| temporal structure | none, held still | changes every frame |

`probe_current` selects `lamina_types` without filtering by side, so **it drives
the contralateral eye the visual interface never touches.**

⚠️ **A claim in an earlier draft of this section was wrong and is corrected in
§2b.** It said `encode` is mostly a temporal-derivative detector, reasoning from
the gains: `on_gain = 14.0` against `static_gain = 2.0`. Measured on the bank,
the static term carries **83–100 %** of the injected current. The temporal gain
is seven times larger, but the frame-to-frame luminance difference is far
smaller than the luminance level it multiplies. Held-still calibration therefore
measures the *dominant* term, not the small one.

The probe is still unrepresentative — wrong population, wrong eye, wrong
amplitude — and Gate E showed the held-still sweep reports a regional spread of
9.8× at gain 12 where a moving sequence gives 13.8×. But the reason is not that
motion dominates the input.

**Replacement.** The calibration input is now the chain the model actually runs
on — `real frames → vision.encode → connectome` — drawn from
`env/stimulus_bank.py`.

---

## 2. The input bank

Ten sequences of consecutive decisions, fixed before anything was measured,
chosen for coverage of the input distribution rather than for the responses they
produce. Content hash recorded with every run.

| sequence | state | why it is in the bank |
|---|---|---|
| `1-1_open` | Level1-1 | bright sky over ground, steady forward scrolling |
| `1-1_obstacles` | Level1-1 | pipes and blocks entering frame; occlusion, edges |
| `1-1_jumping` | Level1-1 | vertical optic flow; sky and ground exchange position |
| `1-1_still` | Level1-1 | no self-motion — isolates the static term |
| `1-1_backward` | Level1-1 | reverse flow; the derivative changes sign |
| `1-4_castle` | Level1-4 | dark background, bright foreground: inverted statistics |
| `2-1_clouds` | Level2-1 | bright uniform sky, weak edges |
| `3-1_night` | Level3-1 | night palette, a third luminance regime |
| `4-1`, `8-1` | Level4-1, 8-1 | further geometry and density |

Measured coverage, mean frame luminance and frame-to-frame change (HUD excluded):

| | mean luminance | frame-to-frame Δ |
|---|---:|---:|
| `3-1_night` | 0.128 | 0.050 |
| `1-4_castle` | 0.324 | **0.083** |
| `8-1` | 0.464 | 0.039 |
| `1-1_open` | 0.534 | 0.023 |
| `2-1_clouds` | **0.564** | 0.034 |
| `1-1_still` | 0.533 | **0.000** |

A 4.4× range in mean luminance and a full range in temporal change, including a
zero-motion control. `1-1_backward` initially recorded Δ = 0.0001 — the camera
does not scroll back at a level start, so it was a second still condition in
disguise. It now starts 120 decisions in and gives Δ = 0.011.

### 2b. What the connectome actually receives

Luminance describes Mario. It does not describe the input: the network never
sees a frame, it sees `I_t = vision.encode(frame_t)` over 3,575 lamina cells.
Measured on both banks, 16 decisions each (`stimulus_bank.input_statistics`):

| | calibration bank | held-out bank |
|---|---|---|
| cells driven, mean | 2,279 – 3,575 | 2,067 – 3,575 |
| **sparsity** | **0.000 – 0.363** | 0.000 – 0.422 |
| mean amplitude | 1.00 – 2.35 | 1.06 – 1.99 |
| total current | 3,576 – 5,342 | 3,572 – 4,333 |
| **static share** | **0.67 – 1.00** | **0.83 – 1.00** |
| temporal share | 0.00 – 0.33 | 0.00 – 0.17 |
| between-sequence CV, total current | 0.113 | 0.071 |

Three findings, none of them about Mario:

1. **The interface is a luminance-level detector, not a motion detector.** The
   static term dominates everywhere, reaching 100 % on the still conditions and
   never falling below 67 %. The ON/OFF temporal channels it is built around
   contribute a minority of the current in every sequence. Whether that is
   acceptable is a design question; that it is the case was not known.
2. **The input is dense, not sparse.** In every bright level all 3,575 cells
   are driven on every frame, because `on_now * static_gain` is non-zero
   wherever there is light. Sparsity rises to 0.25–0.42 only in dark levels, and
   for a mechanical reason: the ON channel goes quiet in the dark while the OFF
   channel, computed as `(1 − off_now)`, goes to its maximum.
3. **The banks cover comparable input space.** Cells driven, amplitude and total
   current overlap closely between them, and the held-out bank is slightly
   *tighter* — it lacks a sequence as dark as `3-1_night`. That is a stated
   limitation of the held-out set, not something to correct by swapping a
   sequence in after seeing this table.

---

## 3. Parameter audit against the literature

### 3.1 The reference model

[Shiu et al., *A Drosophila computational brain model reveals sensorimotor
processing*, Nature 634:210–219 (2024)](https://www.nature.com/articles/s41586-024-07763-9)
— whole-brain FlyWire LIF, ~125k neurons, code at
[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model).
It is the closest published analogue of what this project runs, so its parameter
table is the natural yardstick.

| parameter | Shiu et al. | provenance |
|---|---|---|
| resting potential | −52 mV | prior *Drosophila* electrophysiology |
| reset potential | −52 mV | same |
| firing threshold | −45 mV | same — **7 mV above rest** |
| membrane resistance | 10 kΩ·cm² | same |
| membrane capacitance | 2 µF·cm⁻² | same |
| **membrane time constant** | **20 ms** | derived, R×C |
| synaptic decay | 5 ms | published |
| refractory period | 2.2 ms | published |
| synaptic delay | 1.8 ms | published |
| **W_syn, weight per synapse** | **0.275 mV** | **free parameter**, chosen so sugar GRNs at 100 Hz give ~80 % of maximal MN9 firing |

**The single most useful fact in this table is the last row.** Even the canonical
whole-brain model has no biologically-derived synaptic weight. It is fitted — but
fitted to a *physiological* criterion, a measured input-rate-to-motor-output
relationship, not to a behavioural outcome. That is the standard to meet.

### 3.2 Our parameters, in five categories

Provenance and justification for each. The categories matter more than the
values: three of these five have no biological content at all, and two of those
three sit between the measured anatomy and everything we then attribute to it.

#### (a) Measured biologically / constrained by the literature

| parameter | value | provenance |
|---|---|---|
| membrane time constant | `tau_ms = 20.0` | R·C from *Drosophila* electrophysiology, via the reference. **Matches exactly.** |
| refractory period | `refractory_ms = 2.0` | published 2.2 ms. Ours is 10 % shorter and, at `dt = 3.33`, rounds to one tick either way |
| excitatory / inhibitory sign | `NT_SIGN` | measured transmitter identity per neuron, ground truth preferred over prediction. Glutamate inhibitory (GluCl-α), which is fly-specific and easy to get backwards |
| connectivity | the graph itself | measured synapse counts, no threshold in the primary condition |

#### (b) Approximately transferred — needs adaptation, does not carry over intact

| parameter | status |
|---|---|
| threshold−rest gap | The reference's 7 mV is real; our `threshold = 1.0, v_rest = 0.0` is a normalised stand-in. **No voltage transfers**, only the notion of a gap |
| `w_syn` ↔ `gain` correspondence | §3.3. Equates one narrow property — coincidence needed to fire — between models differing in filtering, delay and weight transform. **A sanity check, not a conversion** |
| activity target | §3.4. No transferable value exists; whole-brain firing rates are not characterised in *Drosophila* at this scale |

#### (c) Simulation choices — implementation, not biology

| parameter | value | note |
|---|---|---|
| time step | `dt_ms = 3.33` | sets every tick↔Hz conversion in the project. The reference integrates far finer |
| ticks per decision | 16 | measured at Gate D as what the signal needs, on *this* model |
| warm-up | 150 ticks | Gate C. Load-bearing: the network ignites slowly, and measuring inside the transient reports it as seed variance |
| backend | scipy CSC, float32 | numerical, not modelling |

#### (d) Free parameters — no sufficiently constrained biological value

| parameter | value | why free |
|---|---|---|
| **`gain`** | 12.0 | the analogue of the reference's `w_syn`, which the reference itself fitted. Ours was fitted to an activity target |
| `noise_std` | 0.02 | **no counterpart at all** — the reference has zero basal firing and no noise term |
| `bias` | 0.0 | currently inert |
| activity target | 7 % | see §3.4 |

#### (e) Engineering transformations — they change the relation between anatomy and dynamics

**This is the category that matters most, and the one nothing had documented.**

| transformation | what it does | what the reference does |
|---|---|---|
| **`log1p` on weights** | compresses synapse counts 1…20,000 into 0.69…9.9, so a 1,000-synapse connection is ~10× a 1-synapse one rather than 1,000× | uses synapse count **linearly** |
| **`normalize_in_degree`** | every neuron's inputs sum to unit absolute mass, so each fires on the same *fraction* of its input | **no normalisation**; each fires on the same *absolute* synapse count |
| **global `gain`** | one scalar multiplies every recurrent input | one scalar `w_syn`, but applied before any normalisation |
| **no synaptic filtering** | a spike delivers its full weight in one tick | 5 ms exponential synapse |
| **no transmission delay** | instantaneous | 1.8 ms |

Model B in `models/reference_lif.py` exists to measure what (e) costs or buys.
Its results are in §4.

### 3.3 Why `gain` is not W_syn — a threshold correspondence, not a conversion

The reference: a neuron fires when accumulated input crosses 7 mV, each synapse
contributing 0.275 mV, so **≈ 25 simultaneous synapses** fire a neuron from
rest — *regardless of how many inputs it has*.

Ours: `normalize_in_degree` scales every row to unit absolute mass, so a neuron
fires when **1/gain of its input mass** is simultaneously active — *the same
fraction for every neuron*.

These can agree for exactly one in-degree. Matching them neuron by neuron on
MaleCNS gives a distribution, and its width is the finding.

⚠️ **Read the table below as a sanity check, not as a conversion.** It equates
one quantity only — how much simultaneous input it takes to cross threshold from
rest — between two models that differ in almost everything else: ours has no
synaptic filtering, no transmission delay, log1p weights and row normalisation.
A number from this table is **not** a gain to adopt, and the phrase "equivalent
gain" is shorthand for "the gain at which one narrow property would match", not
a claim of equivalence. **No operating point may be selected from it.**

| | in-strength (synapses) | fraction the reference needs | equivalent gain |
|---|---:|---:|---:|
| p5 | 64 | 40 % | 2.5 |
| p25 | 198 | 12.9 % | 7.8 |
| **median** | **350** | **7.3 %** | **13.8** |
| p75 | 685 | 3.7 % | 26.9 |
| p95 | 2,690 | 0.9 % | 105.7 |
| max | 119,936 | 0.02 % | 4,703 |

Ours, at `gain = 12.0`, requires **8.3 % from every neuron**.

**So the frozen gain sits close to the reference's median requirement (7.3 %),
which is reassuring and nothing more** — it was chosen to hit an activity target,
not derived, and a single coincident quantity between two different dynamical
systems is weak evidence. What the table does show robustly is the *shape* of
the disagreement: under row normalisation the least-connected neurons fire far
more easily than the reference would allow, and the hubs far less.

By region, the equivalent gain spans **3.5 to 85.6** — a 24× range:

| region | median in-strength | equivalent gain | vs our uniform 8.3 % |
|---|---:|---:|---|
| sensory | 88 | 3.5 | ours fires it 3.4× more easily |
| optic | 274 | 10.8 | ours ≈ 0.9× (closest) |
| projection | 737 | 29.0 | ours makes it 2.4× harder |
| central | 698 | 27.4 | ours makes it 2.3× harder |
| motor readout | 2,180 | 85.6 | ours makes it 7.1× harder |

**No single global gain can satisfy this.** That is a structural consequence of
row-normalisation, independent of any activity target.

⚠️ **But it does not explain the observed spread, and predicts its opposite.**
Normalisation makes the central brain and readout *harder* to fire, yet Gate E
measures central at 14.8 % and optic at 1.5 %. Something else dominates —
recurrence is the obvious candidate, the central brain being densely recurrent
where the optic lobe is largely feedforward. **This is a hypothesis, not a
result. It is testable and has not been tested.**

### 3.4 The activity target has no biological anchor

`dt_ms = 3.33`, so a population rate converts directly:

| rate per tick | per-neuron mean |
|---:|---:|
| 7.0 % — the frozen target | **21.0 Hz** |
| 5.0 % — Gate E, measured in-loop | 15.0 Hz |
| 2.0 % | 6.0 Hz |
| **1.33 %** | **4.0 Hz** |

⚠️ **The 4 Hz row is not a measurement and must not become a target.** It comes
from a secondary source reporting an estimate derived from *metabolic
constraints*, not from whole-brain electrophysiology, which does not exist for
*Drosophila* at this scale. It is quoted here for one purpose only: to show that
the frozen target sits well above any published figure. **Replacing 7 % with
1.33 % would substitute one arbitrary number for another** and is explicitly not
what this document recommends.

On that estimate the frozen target is roughly **5× higher**, and it
was never derived from biology: `JOURNAL` records it moving 4.5 % → 8 % → 7 %,
each time for a methodological reason internal to this project — the readout was
inaudible, then the measurement window was wrong.

**This does not mean 7 % is wrong.** Our units are normalised, our operator is
log1p-transformed and row-normalised, and our network has no synaptic filtering,
so a rate is not directly comparable to a biological one either. It means the
target is **a free modelling parameter presented as a constraint**, and should
be labelled as such wherever it appears.

---

## 4. What our own transformations do

`log1p` and `normalize_in_degree` are applied together, so neither Model A alone
nor Model B can separate them — Model B changes both *and* adds synaptic
filtering and a delay. `scripts/gates/compare_dynamics.py` holds the LIF fixed
and varies only the transforms:

| | weights | in-degree normalisation |
|---|---|---|
| `A_log_norm` | log1p | yes — **what the project runs** |
| `A_lin_norm` | linear | yes — isolates `log1p` |
| `A_log_raw` | log1p | no — isolates the normalisation |
| `A_lin_raw` | linear | no — neither |

The free scalar is a **coincidence fraction**, not a gain: a raw operator has
row sums orders of magnitude larger than a normalised one, so the same `gain`
means different things in each cell. The fraction is the share of a median
neuron's input mass that must be simultaneously active to reach threshold — the
same physical statement everywhere. The project's gain 12.0 is fraction 0.083.

### 4.1 What `log1p` does to the weights — measured, structural

| | linear | log1p |
|---|---:|---:|
| p99 / p50 | 23.0× | 3.5× |
| max / p50 | 1,295× | 7.2× |
| share of a neuron's input mass held by its strongest 1 % of inputs | 13.4 % | 2.8 % |
| a 1,000-synapse connection is worth | 1,000 × a 1-synapse one | **10 ×** |

`log1p` compresses the dynamic range by roughly 180×. Whether a thousandfold
anatomical difference should survive into the dynamics is a modelling question
this project has never posed; it answered it implicitly, in one line.

### 4.2 What each transform costs — ⚠️ provisional

Fraction 0.083, calibration bank, 4-decision sequences, 2 seeds, chance 0.100.
**A smoke test, not an established result.** A fuller sweep across three
fractions and both banks is the confirmation.

| variant | sensory info at readout | active readout units | history ratio |
|---|---:|---:|---:|
| `A_log_norm` — the project | **0.250** | 154 | 0.71× |
| `A_lin_norm` | 0.300 | 878 | 0.88× |
| `A_log_raw` | **0.700** | 1,462 | **1.91×** |
| `A_lin_raw` | 0.600 | 1,617 | 1.24× |
| `B_reference` (w_syn 0.275) | 0.650 | 794 | 0.59× |

Two readings, both to be confirmed:

1. **The normalisation is the costly transform, not `log1p`.** Removing `log1p`
   moves decoding 0.250 → 0.300. Removing the normalisation moves it
   0.250 → 0.700, and multiplies the active readout population by ten. §4.1
   explains why `log1p` matters so little *here*: the normalisation that follows
   rescales every row anyway, so most of what `log1p` did is absorbed.
2. **Only the un-normalised variant approaches history dependence.** 1.91×
   against a 2× threshold, where every normalised variant sits below 0.9×. If
   this holds, `normalize_in_degree` is a substantial part of why Gate E and G6
   both found the state carrying nothing about its past.

A plausible mechanism, **untested**: normalisation erases in-degree differences,
so a hub integrating thousands of synapses is held to the same fractional
threshold as a neuron with sixty. The heterogeneity that would let different
neurons play different roles is flattened away.

### 4.3 Model B, at the reference's own published value

At `w_syn = 0.275 mV` — the value Shiu et al. actually fitted — Model B on this
graph produces a global rate near 1.5 %, which at `dt = 3.33 ms` is ≈4.6 Hz, and
an audible readout. It is neither silent nor saturated. Its free scalar is also
markedly **flatter** than ours: between `w_syn` 3 and 100, a 33× change, the
global rate moves only 7.0 % → 9.3 %.

Against that, Model B showed a regional spread of 74.7× and a coefficient of
variation of 1.50, both far worse than any Model A variant. **Neither model wins
here.** Model B transforms the anatomy less and carries more information; Model A
is more uniform and more stable. That trade-off is the object of study.

---

## 5. Recommendation, before any protocol change

1. **Fix the instrument, not the number.** `probe_current` drives neurons the
   interface never touches. Replace it as the primary calibration reference with
   the bank in §2. This is a correction to a measuring device and does not by
   itself re-freeze anything.
2. **Do not re-freeze a single gain yet.** The §3.3 range is wide and defensible
   across its width; the §3.4 target has no biological anchor. Carry a declared
   set — the reference p25, median and p75 are the natural candidates — as a
   sensitivity axis, per §30.
3. **Re-express the activity target honestly.** Either anchor it to a cited
   firing rate and state the unit conversion, or label it an engineering target
   in every document that quotes it. It is currently the latter presented as the
   former.
4. **Close the three divergences, or declare them.** Absent synaptic delay,
   absent synaptic time constant, and `log1p` on the weights each separate this
   model from its published reference. Each is defensible; none is currently
   written down.
5. **Test the recurrence hypothesis in §3.3** before attributing the regional
   spread to anything. The normalisation argument predicts the opposite sign.
6. **Report conclusions across the range, not at a point.** If the structural
   results hold across the defensible gain range, that is far stronger than any
   single calibrated value. If they do not, §10 of the owner's brief applies:
   **that is a methodological result to keep, not a problem to correct.**

---

## 5b. The ten questions, answered

Status is explicit. Half of these are settled by structural analysis that no
amount of compute changes; the other half wait on the sweep in §4.2.

| # | question | answer | status |
|---|---|---|---|
| 1 | What is actually constrained by biology? | Membrane time constant (20 ms, exact match), refractory period (2.0 vs 2.2 ms), transmitter sign per neuron, and the connectivity itself. **Four things.** | ✅ settled, §3.2a |
| 2 | What comes from Shiu or other references? | The voltage scale (−52/−45 mV), synaptic time constant (5 ms), transmission delay (1.8 ms), refractory period. All inherited from prior *Drosophila* electrophysiology via the reference. | ✅ settled, §3.1 |
| 3 | What is only approximately transferable? | The threshold gap (real in mV, a normalised stand-in here), the `w_syn`↔`gain` correspondence, and any activity target. **None of these carries a number across.** | ✅ settled, §3.2b |
| 4 | Which parameters remain free? | `gain`, `noise_std` (no counterpart in the reference at all), `bias`, and the activity target. | ✅ settled, §3.2d |
| 5 | What does `normalize_in_degree` change? | Structurally: it imposes one coincidence fraction on every neuron where the reference imposes an absolute synapse count, and the per-region equivalents span 24×. Empirically: it appears to be the costly transform — decoding 0.700 → 0.250 and history 1.91× → 0.71× when it is applied. | 🔶 structural settled (§3.3); empirical **provisional** (§4.2) |
| 6 | What does `log1p` change? | Compresses the dynamic range ~180×; a 1,000-synapse connection becomes worth 10 single synapses rather than 1,000. But with normalisation downstream, most of that is absorbed: removing it moves decoding only 0.250 → 0.300. | 🔶 structural settled (§4.1); empirical provisional |
| 7 | What difference does reference-style dynamics make? | At the reference's own published `w_syn`, Model B is alive at ≈4.6 Hz with an audible readout, carries more sensory information, and is far less sensitive to its free scalar — but is much less uniform (spread 74.7×) and much less stable (CV 1.50). | 🔶 provisional, §4.3 |
| 8 | Are the conclusions stable across both banks? | **Not yet measured.** The held-out bank exists and is built; the cross-bank comparison is the sweep now running. | ⬜ pending |
| 9 | Is there a non-pathological, reproducible parameter range? | Partially. Model A is alive across at least gain 12–48 under real input; Model B across `w_syn` 0.275–100. Whether either range is *reproducible on the held-out bank* is question 8. | 🔶 partial |
| 10 | Do the structural conclusions change with the dynamics? | **This is the one that matters and it is not yet answered.** §4.2 shows the readout's information content changing by 2.8× with a transform choice alone. Whether the *connectome-versus-null* conclusions move with it has not been tested — it needs the null models run under each variant. | ⬜ pending, and the highest-value remaining measurement |

**Question 10 is the study's exposure.** If a topology comparison comes out
differently under `A_log_norm` than under `A_log_raw`, then every structural
result this project produces is conditional on a transformation that was never
justified in writing. That is worth knowing before any learning experiment, and
it is cheap relative to what follows it.

---

## 5c. What to run next, in order

Everything below is measurement. None of it changes a frozen value, and none of
it may be used to select one. All of it was written on a machine too slow to
finish the sweeps, so the numbers in §4.2 are smoke tests awaiting confirmation.

```bash
# 1. THE EXPOSURE TEST -- the highest-value measurement outstanding.
#    Does the connectome-minus-null contrast survive the transform choice?
#    A sign change here conditions every structural result the project has.
.venv/bin/python scripts/gates/transform_exposure.py \
    --null-seeds 0 1 2 --length 8 --warmup 100 --seeds 5

# 2. CONFIRM THE 2x2. §4.2 is one fraction, one bank, two seeds.
.venv/bin/python scripts/gates/compare_dynamics.py \
    --fractions 0.25 0.15 0.083 0.04 0.02 \
    --wsyn 0.275 1.0 3.0 10.0 30.0 \
    --length 12 --warmup 150 --seeds 10 \
    --banks calibration heldout

# 3. CROSS-BANK CHECK (question 8). Included in step 2 -- read the two bank
#    rows for each configuration and compare. Do NOT revise a choice because
#    the held-out bank disagrees; report the disagreement.

# 4. Gate F, once the readout category is declared (below).
.venv/bin/python scripts/gates/gate_f_interface.py --steps 480 --episodes 8
```

**Running order matters.** Step 1 first: if the contrast reverses with the
transform, steps 2–4 are measuring a system whose structural claims are already
known to be conditional, and that changes what they are for.

### The one decision that is not mine

§11.2 requires the study to **declare** which of four categories the readout is,
and freeze it before confirmatory comparisons:

| category | what it means here |
|---|---|
| `fixed` | a frozen random projection; all adaptation happens elsewhere |
| `minimally_trainable` | a linear decoder, small and regularised |
| `fully_trainable` | an unconstrained decoder — risks the §11.2 bypass, where the decoder compensates for the network and the topology stops mattering |
| `part_of_the_learning_intervention` | the readout *is* where plasticity lives, and its placement is the experiment |

Three readouts exist in the code (`RandomReadout`, `MotorPopulationReadout`, and
the Gate D linear decoder) and nothing declares which the study uses. It cannot
be inferred. Gate F fails on this and will keep failing until
`readout.category` is set in `configs/base.yaml`.

§11.3 is also unimplemented: a fixed random readout must not be one arbitrary
draw for the whole study — same realisation across matched arms within a block,
different across blocks. `RandomReadout(seed=0)` is one draw.

---

## 6. Sources

- Shiu et al., *A Drosophila computational brain model reveals sensorimotor processing*, Nature 634:210–219 (2024) — https://www.nature.com/articles/s41586-024-07763-9
- Model code and parameter file — https://github.com/philshiu/Drosophila_brain_model
- Preprint with the full methods — https://www.biorxiv.org/content/10.1101/2023.05.02.539144
- Lappalainen et al., *Connectome-constrained networks predict neural activity across the fly visual system*, Nature (2024) — https://www.nature.com/articles/s41586-024-07939-3
- Suárez et al., *Connectome-based reservoir computing with the conn2res toolbox*, Nat Commun 15, 656 (2024) — https://www.nature.com/articles/s41467-024-44900-4

**Verification note.** The Shiu parameter table was read from the paper's methods
and cross-checked against the values in the repository's model file; they agree.
The ~4 Hz whole-brain figure is an estimate from metabolic constraints reported
in a secondary source, not a direct measurement, and is used here only to show
that the frozen target sits well above any published estimate — not as a value to
calibrate to.
