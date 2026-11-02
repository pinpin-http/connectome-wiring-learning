# What Wiring Determines
## Biological Architecture, Learning, and Control in a Whole-Brain Connectome

**French working title:** *Ce que le câblage détermine — Architecture biologique, apprentissage et contrôle dans un connectome cérébral complet*

**Document type:** Master experimental protocol and manuscript scaffold  
**Status:** Living pre-confirmatory protocol  
**Intended use:** research design, implementation guidance, preregistration preparation, reproducibility record, manuscript scaffold, and video-production backbone  
**Primary benchmark family:** closed-loop sensorimotor control, including a deterministic videogame speedrun benchmark  
**Primary anatomical substrate:** a whole-brain Drosophila connectome, with exact release and derived graph hashes to be frozen before confirmatory experiments

---

# 0.0 Authority

**This document is the single source of truth for the study.** It supersedes
everything written before it.

Retired on 2026-09-21, recoverable from Git history at `8c67292`:
`PROTOCOL.md` revision 2, `PROTOCOL2.md`, `PROTOCOL3.md`, `GATES.md`,
`HANDOFF.md`, `STATUS.md`. None of them carries authority any more, and no
decision may cite them. Where they contained measurements rather than
decisions, those measurements live on in `JOURNAL.md`, which stays
append-only and is not retired.

Existing code and data are treated as **unvalidated** until re-examined
against this document. `docs/INVENTORY.md` records what survived that
re-examination, mapped to the section of this protocol that now governs it.
Anything not in that inventory is provisional regardless of how it was
described before.

Gaps in this document are deliberate. The exploratory phase exists so that
open questions can be answered by measurement rather than by guessing in
advance. Filling a gap is a decision, and §33 governs how it is recorded.

---

# 0. Purpose of this document

This document defines the scientific foundation of the project. Its role is to remain stable even if implementation details, software backends, numerical parameters, or individual mechanistic hypotheses change during the exploratory phase.

The study is not built around demonstrating that a connectome can complete a videogame. The videogame is an experimental instrument. The scientific subject is the relationship between measured biological wiring, network dynamics, learning, and closed-loop control.

The central objective is to determine which properties of learning and control are genuinely constrained by biological network architecture once alternative explanations are controlled for.

The protocol therefore separates five layers that must never be conflated:

1. **Measured anatomy** — neurons, connections, annotations, and spatial organization derived from connectomic data.
2. **Model dynamics** — the mathematical rules used to turn static anatomy into an executable dynamical system.
3. **Interfaces** — engineered mappings between the external environment and the neural model.
4. **Learning** — the mechanism and location through which adaptation is allowed to occur.
5. **Task** — the external benchmark used to probe the resulting system.

Any scientific claim must identify which layer it concerns.

This document is intentionally detailed on experimental logic and intentionally flexible on some exact equations, estimators, thresholds, and implementation choices. Those details must be chosen only after literature review and exploratory validation justify them.

---

# 1. Scientific title and central question

## 1.1 Working title

> **What Wiring Determines**  
> *Biological Architecture, Learning, and Control in a Whole-Brain Connectome*

The title is deliberately independent of Super Mario Bros., reservoir computing, a specific neuronal equation, or a single graph statistic. The project should remain scientifically coherent even if the strongest result eventually concerns another architectural property.

## 1.2 Central research question

> **Which properties of learning and control are genuinely constrained by the architecture of a biological connectome?**

A more operational formulation is:

> **When measured biological wiring is used as the architecture of an adaptive controller, what differences in information flow, learning dynamics, robustness, plasticity, and closed-loop behavior remain after appropriate structural and modelling controls are introduced?**

The study does not assume that the biological graph is superior.

Possible outcomes include:

- biological wiring improves some properties;
- biological wiring impairs some properties;
- biological wiring changes learning speed but not final performance;
- biological wiring changes robustness but not nominal performance;
- the apparent biological effect disappears when a confound is controlled;
- interface choices dominate topology;
- plasticity placement matters independently of global topology;
- no measurable effect survives strong controls.

All of these are scientifically meaningful if the experiment can distinguish them cleanly.

---

# 2. Scientific scope and claim discipline

## 2.1 What the study is allowed to claim

Depending on the final evidence, the study may support claims about:

- the computational consequences of a measured connectome-derived topology;
- differences between biological and controlled alternative graphs;
- how network organization affects propagation and temporal processing;
- how architecture interacts with learning;
- how the location of plasticity affects adaptation;
- how robust learned behavior is to perturbation;
- how sensitive conclusions are to modelling and interface choices;
- whether a candidate anatomical property helps explain an observed effect.

## 2.2 What the study must not claim without additional evidence

The study must not automatically claim that:

- a living fly understands the benchmark;
- the simulation reproduces the biological brain in full physiological detail;
- synapse counts are exact physiological synaptic efficacies;
- the chosen neural dynamics are uniquely correct;
- the artificial sensory interface reconstructs real fly vision;
- the artificial motor interface reconstructs real fly motor control;
- a result observed in one benchmark generalizes to cognition broadly;
- a negative result demonstrates that biological wiring has no computational value;
- a positive result demonstrates that evolution optimized the connectome for the tested artificial task.

## 2.3 Required language throughout the project

Preferred terminology:

- **connectome-derived architecture**
- **measured anatomical graph**
- **connectome-constrained model**
- **engineered sensory interface**
- **engineered motor interface**
- **modelled neural dynamics**
- **closed-loop benchmark**
- **biological topology**
- **control graph / null model**

Avoid using “digital fly brain” as a scientific description unless immediately qualified.

---

# 3. Scientific motivation

Connectomics provides increasingly complete maps of nervous systems, but a wiring diagram alone does not specify behavior. Static connectivity must be combined with assumptions about neuronal dynamics, sensory drive, learning, and readout before it can produce actions.

Recent work has shown several relevant directions:

- whole-brain Drosophila connectomes enable network-wide structural analyses;
- simple dynamical models placed on connectome-derived networks can produce interpretable sensorimotor responses;
- connectomes can be used as fixed reservoirs or architectural priors for machine-learning tasks;
- the conclusions of connectome-vs-random comparisons depend strongly on the chosen null model and on modelling assumptions;
- anatomical organization, hierarchy, modularity, spatial structure, and plasticity localization are plausible sources of computational bias.

The unresolved issue is therefore not simply whether a biological connectome “works” as a neural network. The stronger question is which aspects of its organization change what can be learned, how it is learned, and how stable that behavior remains under perturbation.

This project addresses that question through a combination of structural analysis, controlled dynamical experiments, learning experiments, closed-loop control, null models, perturbations, and mechanistic follow-up analyses.

---

# 4. High-level study architecture

The project is organized into four scientific axes.

The first three are confirmatory candidates. The fourth remains mechanistic and exploratory until a specific hypothesis has enough support to be frozen.

## Axis I — Structural controls and information flow

**Question:** When a connectome is randomized while preserving selected statistics, which functional properties of the original organization disappear?

This axis establishes whether the controls themselves alter properties that are directly relevant to sensorimotor computation.

It is the methodological foundation of the study.

## Axis II — Architecture and learning

**Question:** Does biological wiring affect learning speed, final performance, stability, generalization, or robustness once appropriate structural controls are used?

This axis studies learning trajectories, not only endpoint performance.

## Axis III — Localization of plasticity

**Question:** Does the anatomical location of allowed plasticity influence learning once the amount and opportunity for plasticity are controlled as fairly as possible?

This axis tests whether adaptation depends only on the quantity of modifiable parameters or also on where they are placed in the network.

## Axis IV — Mechanistic explanation

**Question:** Which structural or dynamical properties best explain the effects discovered in Axes I–III?

Candidate mechanisms may include, but are not limited to:

- sensorimotor path structure;
- depth;
- modularity;
- recurrence;
- motifs;
- rich-club structure;
- centrality;
- spatial organization;
- excitation/inhibition organization;
- propagation delays;
- controllability or observability-like properties;
- activity distribution;
- memory timescales.

No single mechanism is declared primary before exploratory evidence supports it.

---

# 5. Core hypotheses

The project should maintain a distinction between broad hypotheses and exact statistical hypotheses.

The broad hypotheses below define the conceptual study. Exact estimands and decision thresholds are frozen later.

## H1 — Structural nulls are not functionally neutral

A control graph preserving simple local statistics may nevertheless alter functionally relevant organization such as hierarchy, path structure, modularity, or sensorimotor routing.

**Interpretation:** a topology comparison is only meaningful if the control preserves properties that should not be part of the tested contrast.

## H2 — Biological wiring may affect learning trajectory independently of final performance

Two architectures may reach similar performance while differing in learning speed, stability, variance, or the route taken through parameter space.

## H3 — Biological wiring may affect robustness differently from nominal performance

Architecture may have little effect under nominal conditions but substantial effects under perturbation, distribution shift, noise, lesions, or altered sensory conditions.

## H4 — Plasticity placement may be functionally consequential

The location of plastic synapses may affect learning even when the amount of plasticity is held constant and major accessibility confounds are controlled.

## H5 — Modelling choices may be comparable in magnitude to topology effects

Sensory mapping, neural dynamics, normalization, calibration, temporal discretization, and readout design may substantially change results.

This is not treated as an inconvenience. Quantifying this sensitivity is part of the scientific contribution.

---

# 6. Experimental philosophy

## 6.1 No result is required to favor biology

The study must not be tuned to make the connectome outperform controls.

A valid study can conclude that:

- topology has no detectable effect;
- the main effect comes from path length;
- the effect comes from the interface rather than the connectome;
- the biological arrangement is less efficient for the artificial benchmark;
- plasticity placed artificially near the output outperforms biological placement;
- the model is too sensitive to assumptions to support a strong topology claim.

## 6.2 Separate discovery from confirmation

All experiments belong to one of three phases:

### Exploratory
Used to understand feasibility, discover confounds, choose defensible parameters, and formulate candidate mechanisms.

### Pilot
Used to estimate variance, computational cost, learning behavior, valid operating ranges, and statistical power requirements.

Pilot results do not enter confirmatory estimates.

### Confirmatory
Runs only after the protocol, parameters, primary comparisons, outcome definitions, and analysis plan are frozen.

No primary hypothesis may be rewritten after confirmatory data are inspected.

## 6.3 Negative results remain results

The analysis and manuscript structure must allow a clean null or negative finding to be reported without changing the scientific question.

---

# 7. Anatomical data and graph construction

## 7.1 Source dataset

The study uses a whole-brain Drosophila connectome as the primary anatomical substrate.

Before the confirmatory phase, the following must be frozen:

- dataset name;
- release/version;
- source URLs or repositories;
- file hashes;
- neuron inventory;
- connectivity inventory;
- annotation files;
- transmitter information source;
- coordinate system and units;
- preprocessing version;
- code commit used to build the graph.

A second connectome may be used as a replication dataset if technically and scientifically appropriate, but it is not allowed to silently replace the primary dataset after results are known.

## 7.2 Preservation of neuronal identity

Unless a dedicated experiment explicitly tests aggregation, the primary analysis should preserve individual neurons rather than collapse them into cell-type averages or communities.

Neuron reduction, filtering, or aggregation must be scientifically justified and declared as a separate condition or sensitivity analysis.

## 7.3 Sparse representation

The graph should be stored using sparse representations appropriate for its size.

Sparse storage is a computational representation choice and must not be confused with biological pruning.

The complete node index must remain stable across compared graph conditions unless node removal is itself the experimental intervention.

## 7.4 Anatomical versus effective graph

Maintain two distinct concepts:

### Anatomical graph
The graph directly derived from the connectomic data.

### Effective simulation operator
The graph after modelling assumptions are applied, for example:

- weight transformation;
- transmitter sign assumptions;
- normalization;
- omission of unsupported mechanisms;
- delays or other dynamical parameters.

Every transformation from anatomical graph to effective operator must be reproducible and logged.

## 7.5 Acceptance tests

At minimum, automated tests should verify:

- stable node identity;
- edge direction convention;
- correct aggregation of duplicate records if present;
- conservation of relevant counts;
- transmitter handling;
- treatment of missing annotations;
- sparse/dense agreement on small fixtures;
- deterministic graph derivation from a fixed input and configuration;
- invariants required by each null model.

No scientific experiment begins on a graph that has not passed the acceptance suite.

---

# 8. Null models and structural controls

## 8.1 Rationale

A null model defines the scientific contrast.

If the null accidentally destroys hierarchy, spatial organization, path structure, sign organization, or another functionally relevant property, the experiment no longer isolates the intended variable.

The null-model design is therefore a primary research object, not a preprocessing detail.

## 8.2 Control ladder

The study should use a hierarchy of controls rather than a single “random graph.”

Possible classes include:

1. **Original connectome** — measured architecture.
2. **Strongly constrained rewiring** — preserves selected local and global properties judged necessary for a fair contrast.
3. **Degree-preserving rewiring** — preserves in/out degree and other declared quantities but may destroy mesoscale organization.
4. **Weight or sign controls** — preserve topology while altering selected edge attributes.
5. **More weakly constrained random graphs** — useful for context, not necessarily as primary controls.
6. **Conventional artificial architectures** — engineering baselines such as recurrent or feed-forward models with a comparable task interface and training budget.

The exact primary null must be selected after structural diagnostics establish what each candidate control changes.

## 8.3 Null validation

Every control ensemble must be validated independently of task performance.

Validation should include relevant invariants and mixing diagnostics where applicable.

The project must never select a null realization because it produces a convenient behavioral result.

## 8.4 Structural audit

Before learning experiments, compare the connectome and null ensembles on a broad structural panel.

Potential measures include:

- degree distributions;
- weighted degree / strength;
- path-length distributions;
- sensory-to-output reachability;
- modularity;
- clustering;
- motifs;
- rich-club organization;
- spatial length statistics;
- community structure;
- sign/transmitter composition;
- reciprocity;
- spectral properties;
- centrality distributions.

Not every metric must become a primary endpoint. The objective is to understand what each null preserves and destroys.

---

# 9. Neural dynamics

## 9.1 Model boundary

The connectome does not specify complete neural dynamics.

A dynamical model is required to make the graph executable.

The selected primary dynamics should be:

- computationally feasible;
- transparent;
- reproducible;
- sufficiently expressive for the tested questions;
- compatible with the available anatomical annotations;
- clearly separated from measured biology.

A simple spiking model is a reasonable candidate, but the protocol does not require one exact equation before validation.

## 9.2 Required documentation

The final frozen model must document:

- state variables;
- update rules;
- temporal discretization;
- thresholds;
- reset/refractory behavior;
- noise;
- external drive;
- recurrent scaling;
- normalization;
- weight transformation;
- transmitter/sign treatment;
- initialization;
- reset behavior between trials;
- simulation clock relative to the environment clock.

## 9.3 Dynamical validation before behavior

Before training any policy, run controlled dynamical probes.

At minimum, establish:

- that sensory stimulation enters the intended population;
- that activity can propagate to the intended output population;
- that the network is not trivially silent or saturated;
- that compared conditions operate in interpretable regimes;
- whether the current state depends measurably on previous input history;
- how activity changes after removing or changing sensory drive;
- whether observed timing is compatible with the chosen control frequency.

## 9.4 Calibration

Calibration is part of the model and must not be hidden.

Calibration stimuli should resemble the real operating regime sufficiently to make the resulting operating point meaningful.

Potential calibration targets may include:

- global activity;
- regional activity;
- active-neuron fraction;
- propagation success;
- stability;
- variance;
- motor-readout observability.

A single global firing-rate target must not be assumed sufficient unless diagnostics support it.

## 9.5 Rate matching and shared-parameter comparisons

If activity matching is used between graph conditions, it changes the scientific question.

Therefore two conceptually distinct comparisons may be useful:

- **matched-operating-regime comparison:** asks whether architecture matters when gross activity is controlled;
- **shared-parameter comparison:** asks how architecture changes dynamics under the same physical/model parameters.

Neither should silently substitute for the other.

---

# 10. Sensory interface

## 10.1 Role

The sensory interface translates the task environment into stimulation of selected neural populations.

It is an engineered component and may strongly influence results.

## 10.2 Requirements

The primary sensory mapping must be:

- deterministic unless noise is explicitly part of the experiment;
- documented;
- reproducible;
- fixed before confirmatory runs;
- identical across graph conditions in the same comparison;
- biologically motivated where possible without claiming full physiological realism.

## 10.3 Sensory sensitivity analyses

At least one alternative sensory mapping should be predeclared as a sensitivity analysis if feasible.

Examples may involve changes to:

- retinotopy;
- spatial orientation;
- temporal filtering;
- ON/OFF processing;
- visual field coverage;
- mapping density;
- selected sensory populations.

The goal is to determine whether a topology result depends on one arbitrary interface choice.

## 10.4 Information leakage

Task-state variables used for scoring, termination, debugging, or reward must not leak into the policy input unless declared.

For the videogame benchmark, direct RAM information may be used for evaluation or reward bookkeeping while policy observations remain pixel-derived, provided this separation is explicit.

---

# 11. Motor interface and readout

## 11.1 Role

The motor interface translates activity in selected output populations into task actions.

## 11.2 Primary principles

The motor interface should not accidentally erase the architectural differences the study is trying to measure.

If a highly trainable decoder is used, it may compensate for poor internal representations and obscure the contribution of the network.

Therefore the study should explicitly decide whether the readout is:

- fixed;
- minimally trainable;
- fully trainable;
- part of the learning intervention.

This choice is scientifically consequential and must be frozen before confirmatory comparisons.

## 11.3 Random readout handling

If a random fixed readout is used, it should not be represented by one arbitrary draw for the entire study.

A stronger design treats readout initialization as part of the experimental block:

- same readout realization across matched arms within a block;
- different readout realizations across blocks.

This prevents conclusions from depending on a single lucky or unlucky projection.

---

# 12. Learning framework

## 12.1 Purpose

The learning mechanism must allow the study to ask whether architecture and plasticity location matter.

The protocol does not precommit to one precise rule before exploratory validation.

## 12.2 Requirements for the primary learning rule

The final learning mechanism should satisfy as many of the following as possible:

- compatible with selected plastic synapse subsets;
- scalable to the graph size;
- reproducible;
- interpretable;
- stable enough for paired comparisons;
- applicable identically across the relevant arms;
- capable of learning a trivial control task before being used in the main benchmark.

A local reward-modulated rule is one candidate because it permits direct control over where plasticity is allowed, but its theoretical interpretation must be stated accurately.

## 12.3 Plasticity budget

When comparing locations of plasticity, conditions should use a matched budget according to a predeclared definition.

Potential matching dimensions include:

- number of modifiable synapses;
- parameter count;
- pre/post degree;
- edge-weight distribution;
- sign;
- network depth;
- activity exposure;
- accessibility to sensory information;
- accessibility to the output;
- expected eligibility or update opportunity.

The exact matching rule should be determined during exploratory analysis and frozen before confirmation.

## 12.4 Learning sanity gate

Before the main benchmark, the learning rule must demonstrate non-trivial learning on a simple controlled task.

Failure at this stage is a learning-rule feasibility result and must not be attributed to connectome architecture.

---

# 13. Experimental tasks and benchmark hierarchy

## 13.1 Why multiple task levels are necessary

One environment cannot establish a general architecture effect.

The study therefore uses a hierarchy of tasks with increasing ecological and computational complexity.

## 13.2 Level A — Controlled diagnostic tasks

These are simple tasks designed to isolate specific functions.

Candidate families include:

- stimulus discrimination;
- delayed response;
- temporal integration;
- cue association;
- simple target localization;
- obstacle response;
- sequence dependence;
- controlled perturbation recovery.

Their purpose is diagnosis rather than spectacle.

A diagnostic task should have meaningful floor and ceiling behavior and should discriminate a functioning network from trivial or graph-free controls.

## 13.3 Level B — Complex closed-loop benchmark

The primary complex benchmark is **Super Mario Bros.**, used as a deterministic closed-loop control environment.

The benchmark is useful because it combines:

- visual input;
- continuous temporal pressure;
- sequential decisions;
- action timing;
- state dependence;
- recovery from small deviations;
- deterministic replay;
- an external expert/TAS reference;
- intuitive visual interpretation.

Mario is the instrument, not the scientific thesis.

## 13.4 Speedrun framing

The videogame benchmark should reward both successful completion and efficiency.

A reference run may provide a normalized timing scale, but the report should distinguish an empirical TAS reference from a mathematically proven global optimum unless such proof exists.

The final metric must avoid conditioning only on successful episodes.

It should be reported alongside interpretable components such as:

- completion rate;
- progress on failure;
- timing among completions;
- action stability;
- recovery behavior.

The exact primary score is frozen after the pilot.

## 13.5 Level C — Generalization

At least one generalization axis is required.

Possible options include:

- additional game levels;
- modified visual conditions;
- altered initial states;
- mild physical/task perturbations;
- a non-game control environment;
- a different sensorimotor task.

The goal is to establish whether a finding belongs to the architecture or only to one benchmark instance.

---

# 14. Study I — Structural controls and functional consequences

## 14.1 Main question

> **What does each null model actually remove, and which removed properties affect sensorimotor computation?**

## 14.2 Stage 1 — Structural comparison

Characterize the connectome and each control ensemble independently of task performance.

The output of this stage is a map of preserved and destroyed properties.

## 14.3 Stage 2 — Controlled propagation experiments

Use simple sensory perturbations and record responses across relevant regions and output populations.

Potential outcomes include:

- propagation latency;
- response amplitude;
- temporal spread;
- fraction of output population reached;
- persistence;
- dependence on history;
- region-by-region response structure.

These experiments should be interpretable without learning.

## 14.4 Stage 3 — Candidate mechanism intervention

If a structural property emerges as a plausible explanation, construct an intervention that changes this property as selectively as feasible while preserving other relevant quantities.

This stage should attempt to move beyond correlation.

For example, if sensorimotor depth emerges as important, the intervention should alter depth while avoiding unnecessary changes in graph size, degree, sign, or weight statistics.

The exact mechanism is not frozen until Stage 1–2 results justify it.

## 14.5 Study I output

A successful Study I should answer:

1. Which null models are suitable for later learning comparisons?
2. Which architectural properties are confounded by common randomization procedures?
3. Which structural differences produce measurable dynamical consequences?
4. Which candidate mechanisms deserve causal follow-up?

---

# 15. Study II — Architecture and learning

## 15.1 Main question

> **After appropriate controls are established, how does architecture affect the process and outcome of learning?**

## 15.2 Compared conditions

At minimum, Study II should include:

- connectome-derived architecture;
- the strongest justified primary null;
- a conventional degree-preserving or literature-comparable null;
- at least one conventional artificial-network baseline where computationally feasible.

Exact baseline architecture should be selected before the confirmatory freeze.

## 15.3 Learning outcomes

Learning should be evaluated as a trajectory rather than a single final number.

Primary or secondary outcome families may include:

- area under the learning curve;
- final performance;
- episodes or interactions required to reach a defined threshold;
- variance across seeds/blocks;
- stability after reaching competence;
- catastrophic regression;
- action consistency;
- completion and timing metrics;
- recovery after perturbation;
- generalization to held-out scenarios.

The exact primary estimand is selected after pilot behavior is observed, then frozen.

## 15.4 Avoid overfitting the analysis model

No single functional form should be assumed to describe all learning curves unless pilot evidence strongly supports it.

Parametric curve fits may be used as secondary summaries, with fit quality reported.

## 15.5 Interpretation

Potential interpretations include:

- same endpoint, different learning efficiency;
- different endpoint, similar learning speed;
- similar nominal performance, different robustness;
- no difference under strong controls;
- differences only under weaker nulls;
- differences dominated by modelling assumptions rather than topology.

Each interpretation should remain valid regardless of whether the connectome is better or worse.

---

# 16. Study III — Localization of plasticity

## 16.1 Main question

> **Does where learning is permitted to modify the network matter independently of how much plasticity is available?**

## 16.2 Biological reference region

A biologically identified learning-related pathway may serve as the reference plasticity site.

The exact anatomical definition must be based on published annotations and the selected connectome release.

## 16.3 Controls

Plasticity-location controls should progressively address alternative explanations.

Possible conditions include:

- biological plasticity site;
- anatomically or functionally matched alternative site;
- depth-matched site;
- degree/weight/sign matched site;
- random plastic synapses;
- engineered terminal/readout plasticity.

Not every condition must remain primary. The final hierarchy is determined during the pilot and frozen before confirmation.

## 16.4 Plasticity opportunity

Matching only parameter count may be insufficient.

Before interpreting plasticity-location effects, quantify whether compared synapses actually receive comparable opportunities to change.

Potential diagnostics include:

- pre/post activity;
- eligibility magnitude;
- access to sensory information;
- access to output pathways;
- update frequency;
- saturation or clipping frequency.

If one site is almost never active, poorer learning cannot be attributed simply to anatomy.

## 16.5 Interpretation

A result favoring the biological site may support a claim that anatomical placement carries functional information beyond the matched controls.

A result favoring an artificial output-near site may show that engineering efficiency differs from biological organization.

A null result may indicate that location is not load-bearing under the tested model and task.

---

# 17. Axis IV — Mechanistic analysis

This section is explicitly exploratory unless promoted to confirmatory status before the final freeze.

The purpose is to explain, not merely describe, the effects found in Studies I–III.

Candidate analyses may include:

- structural regression;
- targeted graph intervention;
- lesion analysis;
- region removal;
- pathway-specific perturbation;
- activity mediation analysis;
- temporal response analysis;
- representation analysis;
- dimensionality analysis;
- graph-theoretic decomposition;
- comparison across connectome releases or sexes/species where justified.

A candidate mechanism should ideally satisfy three levels of evidence:

1. **Association:** the property differs between conditions and correlates with the outcome.
2. **Specificity:** the relationship persists after obvious alternative structural explanations are controlled.
3. **Intervention:** selectively changing the property predictably changes the outcome.

Only the third supports a strong causal interpretation.

---

# 18. Robustness and perturbation experiments

Robustness is treated as a distinct scientific property.

Possible perturbations include:

- sensory noise;
- occlusion;
- altered visual contrast;
- timing perturbations;
- initial-state perturbations;
- action noise;
- neuron lesions;
- edge lesions;
- regional silencing;
- transmitter/sign perturbations;
- weight noise;
- delayed observations;
- changed control frequency.

Perturbations should be predefined before confirmatory robustness tests.

For lesions, report both absolute performance and change from the unperturbed condition.

Do not claim “robustness” from one perturbation family alone without qualification.

---

# 19. Experimental blocks and randomization

## 19.1 Block principle

The independent experimental unit should reflect the sources of random variation that matter scientifically.

A block may include:

- topology/null realization;
- learning seed;
- readout seed;
- neural-noise seed family;
- evaluation scenario bank.

Matched conditions within the same block should share the relevant random factors.

## 19.2 Why paired blocks are preferred

Pairing reduces variance and prevents a topology comparison from being confounded by unrelated initialization differences.

## 19.3 Seed policy

All seeds must be generated and frozen before confirmatory runs.

The seed registry should identify the purpose of every random stream rather than use one global seed for all operations.

---

# 20. Pilot phase

## 20.1 Pilot objectives

The pilot is allowed to determine:

- whether the learning rule works;
- whether the benchmark has floor and ceiling;
- reasonable operating ranges;
- computational throughput;
- variance between runs;
- appropriate episode/training budgets;
- feasible number of confirmatory blocks;
- whether candidate outcome metrics behave sensibly;
- whether planned controls are sufficiently discriminative;
- whether a null-model ensemble mixes or samples adequately;
- whether additional confounds must be controlled.

## 20.2 Pilot exclusions

Pilot observations are not pooled with confirmatory estimates.

Pilot runs must be labeled explicitly in all files and plots.

## 20.3 Freeze after pilot

After the pilot, freeze:

- primary research questions;
- primary dataset release;
- graph derivation;
- primary null;
- primary neural model;
- primary sensory interface;
- primary motor interface;
- learning rule;
- plasticity sets;
- training budget;
- primary benchmark configuration;
- primary outcomes;
- statistical analysis;
- seed list;
- exclusion criteria;
- confirmatory block count;
- stopping rule.

---

# 21. Confirmatory phase

The confirmatory campaign begins only after the freeze manifest is committed and archived.

Requirements:

- no optional stopping;
- no dropping inconvenient arms after results are observed;
- no changing primary outcomes after inspection;
- no selecting seeds based on performance;
- no silently changing graph preprocessing;
- no changing model calibration based on confirmatory task performance;
- deviations must be logged before re-running affected analyses.

If the planned sample size becomes computationally impossible, the study must be labeled exploratory or the scope must be formally amended before inspecting the final comparison.

---

# 22. Statistical analysis framework

## 22.1 General principles

The analysis should emphasize:

- paired comparisons;
- effect sizes;
- uncertainty intervals;
- full distributions;
- per-block results;
- learning curves;
- robustness of conclusions to reasonable analysis choices.

P-values may be reported where appropriate but should not replace effect size and uncertainty.

## 22.2 Primary estimands

Each confirmatory study should have one clearly defined primary estimand.

Examples of possible estimand families:

- paired difference in learning-curve AUC;
- paired difference in final score;
- paired difference in a propagation metric;
- paired difference between two plasticity locations;
- paired change in performance under perturbation.

The exact definition is frozen after the pilot.

## 22.3 Confidence intervals

Use a method appropriate to the block structure and outcome distribution.

Bootstrap procedures, hierarchical models, or classical paired estimators may be considered.

The method must resample or model the correct experimental unit rather than treating individual game frames as independent observations.

## 22.4 Practical significance

A statistical difference should not automatically be interpreted as practically meaningful.

The protocol should define a practical-effect criterion in units tied to the benchmark or a meaningful baseline gap.

This threshold is chosen before confirmatory results are inspected.

## 22.5 Multiplicity

Limit the number of primary hypotheses.

Secondary analyses remain labeled secondary even if they are visually striking.

Exploratory findings may motivate a future study but must not be retroactively presented as preregistered primary outcomes.

## 22.6 Missing or failed runs

Predefine handling of:

- crashes;
- corrupted emulator states;
- numerical instability;
- invalid null generation;
- failed calibration;
- learning divergence;
- hardware interruption.

Exclusion must depend on technical validity, not outcome direction.

---

# 23. Compute and performance engineering

Performance engineering is part of feasibility but must not silently change the scientific model.

Benchmark separately:

- graph construction;
- neural tick latency;
- sensory encoding;
- motor decoding;
- emulator step;
- training update;
- logging overhead;
- total closed-loop decision latency.

Report:

- hardware;
- software versions;
- CPU/GPU backend;
- sparse library/backend;
- precision;
- batch/parallelization settings.

An optimized backend must be numerically validated against a trusted reference implementation on controlled fixtures before use in confirmatory experiments.

If a reduced graph or thresholded graph is required for feasibility, declare it before confirmation and retain the full graph as a sensitivity or replication where possible.

---

# 24. Reproducibility and provenance

This project should be reproducible from source data to final figure.

## 24.1 Every run gets a manifest

Each experimental run should record at least:

- run ID;
- timestamp;
- study/phase;
- commit SHA;
- dirty-worktree status;
- machine identifier;
- operating system;
- Python/runtime versions;
- major package versions;
- accelerator/backend;
- dataset release;
- source-data hashes;
- graph-derived file hashes;
- configuration file;
- seed registry;
- topology condition;
- learning condition;
- benchmark state/version;
- runtime;
- output artifact paths.

## 24.2 Generated results

Scientific figures and tables should be generated from machine-readable experiment artifacts rather than manually typed values.

Preferred outputs:

- JSON/JSONL for detailed logs;
- Parquet/CSV for tabular analysis;
- versioned configuration files;
- machine-readable summaries;
- vector figures where possible.

## 24.3 Immutable confirmatory manifest

Before confirmation, archive one manifest containing every frozen primary choice.

Possible archive locations:

- Git tag/release;
- Zenodo/OSF preregistration;
- institutional repository;
- immutable commit plus external timestamped archive.

---

# 25. Daily scientific journal

The existing append-only journal becomes an official part of the methodology.

## 25.1 Purpose

The journal records the evolution of understanding and prevents retrospective rewriting of the project history.

A corrected belief should not erase the previous belief.

Instead:

1. record the earlier interpretation;
2. record the new evidence;
3. mark the previous interpretation as refuted or superseded;
4. document what changed in the protocol or code.

## 25.2 Status vocabulary

Use a stable vocabulary such as:

- **Established** — directly measured and reproduced sufficiently for current use.
- **Indication** — observed but confounded, underpowered, or insufficiently reproduced.
- **Planned** — not yet measured.
- **Refuted** — contradicted by later evidence.
- **Superseded** — method or interpretation replaced for documented reasons.

## 25.3 Minimum daily entry

Each active development/research day should record:

### Goal
What question was addressed?

### Changes
What code, data, model, or configuration changed?

### Experiments run
Which run IDs and commands were executed?

### Observations
What was measured?

### Interpretation
What does the evidence currently suggest?

### Confidence
Established / indication / planned / refuted.

### Problems discovered
Bugs, confounds, invalid assumptions, performance limits.

### Decision
What changed in the protocol or next-step plan?

### Evidence
Links to artifacts, plots, logs, videos, commits.

### Next action
What experiment follows and why?

## 25.4 Journal discipline

The journal must never contain invented measurements or inferred results presented as executed experiments.

If a number comes from another machine, agent, or unverified run, mark its provenance.

---

# 26. Video as a scientific trace

The project already produces a visual artifact for each development stage. This practice should be formalized.

The videos serve two roles:

1. **Research trace:** demonstrate that a system behavior or implementation milestone existed at a known commit/configuration.
2. **Communication asset:** provide material for the final explanatory video.

A video is supporting evidence, not a substitute for machine-readable measurements.

## 26.1 Per-stage video package

For each major gate or development phase, store:

- video file;
- short title;
- date;
- commit SHA;
- config/run IDs shown;
- scientific question illustrated;
- claim demonstrated;
- claim explicitly not demonstrated;
- related plot or table;
- narration notes;
- optional raw capture.

## 26.2 Video naming convention

Example:

```text
videos/
  01_graph_integrity/
  02_null_models/
  03_dynamics/
  04_closed_loop/
  05_learning_sanity/
  06_study1/
  07_study2/
  08_study3/
  09_robustness/
  final/
```

Each folder should contain a metadata file linking the video to the experiment artifacts.

## 26.3 Scientific visuals to capture during development

Prioritize visuals that can later become figures or video sequences:

- connectome visualization;
- sensory-to-output pathways;
- null-model structural comparison;
- activity propagation after a controlled stimulus;
- regional activity over time;
- closed-loop Mario control;
- learning curve evolution;
- same benchmark across different graph conditions;
- plasticity locations highlighted anatomically;
- lesion/perturbation comparison;
- generalization runs;
- failure cases.

The final video should not rely only on successful cherry-picked runs.

---

# 27. Final explanatory video structure

The final video should mirror the scientific paper rather than follow implementation chronology.

## Chapter 1 — The question

Introduce the connectome and the central research problem:

> What does biological wiring actually determine about learning and control?

## Chapter 2 — Static anatomy versus executable model

Explain the difference between connectome, dynamics, interface, and learning.

## Chapter 3 — The control problem

Show why comparing against “a random network” is insufficient and why null models matter.

## Chapter 4 — From pixels to actions

Show the closed-loop architecture and Mario as the experimental instrument.

## Chapter 5 — Study I

Show how different control graphs alter information flow and the candidate structural explanation.

## Chapter 6 — Study II

Show learning trajectories, not only final runs.

## Chapter 7 — Study III

Show anatomical locations of plasticity and their behavioral consequences.

## Chapter 8 — Robustness and failure

Show what happens when the system is perturbed and include negative/failure cases.

## Chapter 9 — What can actually be concluded

Separate findings about topology, dynamics, plasticity, and the artificial interface.

## Chapter 10 — Limits and next experiments

State what remains unknown.

---

# 28. Figure plan for the scientific report

The final manuscript should aim for figures that explain mechanisms rather than decorate results.

## Figure 1 — Study overview

A single schematic:

```text
Environment
   ↓
Sensory interface
   ↓
Connectome-derived dynamical network
   ↓
Motor interface
   ↓
Action
   ↺
```

With separate labels identifying measured and engineered components.

## Figure 2 — Anatomical graph and control ensembles

Show what each control preserves and destroys.

Include key structural diagnostics.

## Figure 3 — Controlled signal propagation

Stimulus-to-readout response across graph conditions.

## Figure 4 — Learning dynamics

Full learning curves with uncertainty across blocks.

## Figure 5 — Robustness/generalization

Nominal and perturbed performance across conditions.

## Figure 6 — Plasticity placement

Anatomical visualization plus learning outcomes for plasticity-site conditions.

## Figure 7 — Mechanistic intervention

Show the strongest candidate mechanism and a targeted intervention if one is validated.

## Supplementary figures

- calibration sweeps;
- structural distributions;
- null mixing diagnostics;
- per-block results;
- sensitivity analyses;
- runtime benchmarks;
- failure cases;
- alternate interfaces;
- alternate connectome replication if available.

---

# 29. Gate structure

Gates exist to prevent expensive experiments from being run on invalid foundations.

The exact naming may evolve, but the logical order should remain.

## Gate A — Data and graph integrity

Pass only if graph derivation is reproducible and all acceptance tests succeed.

## Gate B — Null validity

Pass only if control graphs preserve required invariants and the randomization/sampling procedure is valid enough for the intended inference.

## Gate C — Dynamical viability

Pass only if the model has a usable non-trivial operating regime and controlled sensory inputs produce interpretable network responses.

## Gate D — Temporal coherence

Pass only if the neural simulation timescale and environment decision timescale allow the intended signals to reach the readout.

## Gate E — In-loop calibration

Pass only if calibration remains valid under realistic sensory drive and closed-loop operation.

## Gate F — Interface integrity

Pass only if sensory and motor interfaces do not contain unintended task information or bypass the network.

## Gate G — Learning sanity

Pass only if the selected learning mechanism solves a simple task under controlled conditions.

## Gate H — Benchmark discrimination

Pass only if the benchmark can distinguish meaningful competence from trivial controls and has neither a universal floor nor ceiling.

## Gate I — Compute feasibility

Pass only if the pilot and confirmatory campaigns can be executed with a declared resource budget.

## Gate J — Reproducibility

Pass only if independent reruns from frozen inputs/configuration reproduce the relevant result within expected stochastic variation.

Only then begin the confirmatory campaign.

---

# 30. Sensitivity analyses

Sensitivity analyses quantify how dependent the conclusions are on modelling choices.

Potential axes:

- alternative weight transformation;
- alternative normalization;
- alternate activity calibration;
- alternate sensory mapping;
- alternate readout mapping;
- alternate neural clock;
- alternate connectome thresholding;
- alternate transmitter treatment;
- alternate noise level;
- alternate benchmark instance;
- alternate connectome dataset.

Sensitivity analyses should not become an unbounded search for a favorable result.

A small predefined set is preferred.

---

# 31. Replication strategy

A result becomes substantially stronger if reproduced under a second independent axis.

Possible replication levels:

1. different seeds / null realizations;
2. different benchmark instances;
3. different connectome release or sex;
4. different but conceptually similar neural dynamics;
5. independent code path for a key analysis;
6. external reproduction by another machine or contributor.

The minimum replication requirement for each claim should be defined before manuscript submission.

---

# 32. Failure modes and interpretation rules

## Case A — Nothing learns

Interpretation: the learning framework or interface is not yet a valid test of topology.

Do not conclude that the connectome cannot learn.

## Case B — All conditions perform perfectly

Interpretation: benchmark too easy or output interface too powerful.

The comparison is uninformative.

## Case C — All conditions fail

Interpretation: benchmark, model, learning rule, or interface lacks feasibility.

No equivalence claim is allowed.

## Case D — Weak null beats the connectome, strong null does not

Interpretation: the weak null probably changed an additional functionally relevant property.

Investigate what differs structurally/dynamically.

## Case E — Connectome beats all controls

Interpretation: report the effect, then test whether it survives interface, calibration, seed, and modelling sensitivities.

Do not immediately attribute it to biological optimality.

## Case F — Connectome and strong null are equivalent

Interpretation: the preserved statistics may capture the relevant architectural contribution, or the tested task/model may not depend on the remaining organization.

This is a meaningful result.

## Case G — Plasticity site effect disappears after activity matching

Interpretation: apparent anatomical specialization was mediated by plasticity opportunity rather than location alone.

## Case H — Interface choice reverses the conclusion

Interpretation: the topology effect is conditional on the interface and should not be presented as intrinsic.

This may itself be a valuable methods result.

---

# 33. Deviation policy

Any deviation after protocol freeze must be recorded in a dedicated deviation log.

For each deviation record:

- date;
- original specification;
- reason for change;
- whether confirmatory data had already been inspected;
- affected runs;
- whether runs must be repeated;
- whether the changed analysis is confirmatory or exploratory.

No silent protocol drift.

---

# 34. Software-engineering requirements

The codebase should make invalid comparisons difficult.

Recommended properties:

- one configuration system;
- typed or validated configs;
- explicit dataset adapters;
- explicit graph derivation stage;
- deterministic seeds;
- test suite for graph invariants;
- separate modules for anatomy, dynamics, interfaces, learning, tasks, evaluation;
- experiment registry;
- automatic run manifests;
- automatic figure generation;
- CI without proprietary ROM data;
- benchmark commands that report hardware/backend;
- clear separation between exploratory scripts and confirmatory pipeline.

No scientific result should require manually editing code constants between runs without recording the change.

---

# 35. Data management and legal constraints

## 35.1 Connectome data

Respect original dataset licenses and citation requirements.

Do not redistribute files that prohibit redistribution.

## 35.2 Game data

Do not distribute proprietary ROM content.

The repository should contain only legal integration metadata, code, hashes, and instructions requiring the researcher to supply an independently obtained compatible ROM.

## 35.3 Experiment outputs

Derived numerical results, code, figures, and run metadata should be published openly where licensing permits.

---

# 36. Ethics and scientific communication

The project does not involve a living experimental animal if it operates exclusively on published connectomic data and simulations.

Nevertheless, the final manuscript should include an ethics/data statement describing:

- data provenance;
- whether any original biological experiment was performed;
- licensing;
- computational resource use;
- conflicts of interest;
- use of AI coding/research assistance where disclosure is required.

Communication material must avoid overstating consciousness, agency, or biological fidelity.

---

# 37. Manuscript structure

The final paper should be written progressively during the project rather than reconstructed at the end.

## Abstract

Structure:

1. problem;
2. gap;
3. approach;
4. strongest result(s);
5. interpretation;
6. limits.

Do not write numerical claims until confirmatory results exist.

## 1. Introduction

### 1.1 Connectomes as measured architectures
### 1.2 From wiring to computation
### 1.3 The problem of model assumptions and null networks
### 1.4 Learning in fixed biological architectures
### 1.5 Research questions and contributions

## 2. Related Work

### 2.1 Drosophila whole-brain connectomics
### 2.2 Connectome-constrained dynamical models
### 2.3 Connectome-based reservoir computing
### 2.4 Graph null models in neuroscience
### 2.5 Bio-inspired and connectome-informed controllers
### 2.6 Local plasticity and mushroom-body learning
### 2.7 Closed-loop control benchmarks

## 3. Data and Experimental Framework

### 3.1 Connectome dataset
### 3.2 Graph derivation
### 3.3 Neural dynamics
### 3.4 Sensory interface
### 3.5 Motor interface
### 3.6 Learning mechanism
### 3.7 Benchmarks
### 3.8 Experimental blocks and randomization
### 3.9 Evaluation and statistics
### 3.10 Reproducibility

## 4. Study I — Structural Controls and Functional Consequences

### 4.1 Null-model construction
### 4.2 Structural audit
### 4.3 Controlled propagation
### 4.4 Mechanistic intervention
### 4.5 Study I interpretation

## 5. Study II — Architecture and Learning

### 5.1 Learning conditions
### 5.2 Learning trajectories
### 5.3 Final performance
### 5.4 Robustness and generalization
### 5.5 Study II interpretation

## 6. Study III — Localization of Plasticity

### 6.1 Plasticity-site definitions
### 6.2 Matching and controls
### 6.3 Learning comparison
### 6.4 Plasticity opportunity
### 6.5 Study III interpretation

## 7. Mechanistic Analyses

Use only analyses justified by results from the primary studies.

## 8. Sensitivity and Replication

### 8.1 Model assumptions
### 8.2 Interface assumptions
### 8.3 Alternative graph construction
### 8.4 Secondary connectome or benchmark

## 9. Discussion

### 9.1 What the architecture appears to constrain
### 9.2 What is explained by simpler graph properties
### 9.3 What depends on modelling choices
### 9.4 Biological interpretation
### 9.5 Implications for AI and neuromorphic design
### 9.6 What the videogame benchmark does and does not establish

## 10. Limitations

Mandatory categories:

- connectome incompleteness / reconstruction uncertainty;
- static anatomy versus dynamic physiology;
- synapse-count weight approximation;
- neurotransmitter/receptor simplification;
- absence or simplification of neuromodulation;
- engineered sensory interface;
- engineered motor interface;
- artificial reward;
- benchmark specificity;
- compute-driven simplifications;
- null-model limitations;
- statistical power;
- generalization limits.

## 11. Conclusion

Answer the central research question using only the claims supported by the final evidence.

## Supplementary Methods

Include exhaustive implementation details that would interrupt the main narrative.

## Supplementary Results

Include full null distributions, per-block data, sensitivity analyses, calibration results, and failed/negative diagnostics.

---

# 38. Results registry

Maintain a machine-readable and human-readable registry of every claim intended for the paper.

Suggested fields:

| Field | Meaning |
|---|---|
| claim_id | stable identifier |
| claim | exact sentence candidate |
| study | I / II / III / exploratory |
| status | planned / indication / established / refuted |
| estimand | what was measured |
| run_ids | supporting experiments |
| figure | linked figure |
| commit | code version |
| config | experiment config |
| evidence_level | exploratory / pilot / confirmatory / replication |
| limitations | direct caveats |
| video_clip | related visual artifact |

A manuscript sentence that cannot be linked to evidence should not be presented as a result.

---

# 39. Video registry

Maintain a parallel registry for visual evidence.

Suggested fields:

| Field | Meaning |
|---|---|
| clip_id | stable identifier |
| stage | development/scientific stage |
| question | question illustrated |
| run_id | source run |
| commit | code state |
| config | configuration |
| start/end | useful time range |
| visible metric | what viewers can see |
| valid claim | what the clip demonstrates |
| invalid claim | what it does not demonstrate |
| final_video_section | intended destination |

This avoids recreating visual material months later and prevents clips from losing provenance.

---

# 40. Daily journal template

```markdown
## YYYY-MM-DD — Short title

### Objective
What scientific or engineering question was addressed?

### Changes
- Code:
- Data:
- Configuration:
- Protocol:

### Experiments
| Run ID | Condition | Purpose | Status |
|---|---|---|---|

### Measurements
Report only executed measurements.

### Interpretation
What does the evidence currently support?

### Status changes
- ✅ Established:
- 🔶 Indication:
- ⬜ Planned:
- ❌ Refuted:

### Confounds / bugs
What could invalidate the current interpretation?

### Decision
What was changed and why?

### Artifacts
- logs:
- plots:
- video:
- commit:

### Next experiment
State the next question, not only the next coding task.
```

---

# 41. Experiment manifest template

```yaml
experiment:
  id:
  date:
  phase: exploratory | pilot | confirmatory | replication
  study:
  question:

code:
  commit:
  dirty: false

machine:
  id:
  os:
  cpu:
  gpu:
  backend:

anatomy:
  dataset:
  release:
  source_hashes: {}
  graph_hash:
  graph_config:

model:
  dynamics_config:
  sensory_config:
  motor_config:
  learning_config:

condition:
  topology:
  null_realisation:
  plasticity_site:

randomness:
  topology_seed:
  learning_seed:
  readout_seed:
  noise_seed:
  evaluation_bank:

benchmark:
  environment:
  version:
  state_hash:
  observation_mode:
  action_space:

outputs:
  telemetry:
  summary:
  figures:
  video:

notes:
```

---

# 42. Protocol-freeze checklist

Before the first confirmatory run, verify that the following are frozen and archived.

## Scientific

- [ ] Central question
- [ ] Primary hypothesis for each study
- [ ] Primary estimand for each study
- [ ] Interpretation rules

## Data

- [ ] Connectome release
- [ ] Source hashes
- [ ] Graph derivation
- [ ] Annotation versions

## Model

- [ ] Dynamics
- [ ] Calibration procedure
- [ ] Sensory interface
- [ ] Motor interface
- [ ] Learning rule
- [ ] Plasticity definitions

## Controls

- [ ] Primary null
- [ ] Null validation criteria
- [ ] Baselines

## Benchmark

- [ ] Environment version
- [ ] ROM hash where applicable
- [ ] Savestates
- [ ] Action space
- [ ] Observation rules
- [ ] Reward
- [ ] Termination rules
- [ ] Evaluation scenarios

## Statistics

- [ ] Experimental unit
- [ ] Block design
- [ ] Seeds
- [ ] Sample size
- [ ] Primary analysis
- [ ] Confidence interval method
- [ ] Practical-effect criterion
- [ ] Multiplicity policy
- [ ] Exclusion rules

## Reproducibility

- [ ] Commit tagged
- [ ] Environment lockfile
- [ ] Manifest archived
- [ ] Pilot excluded
- [ ] Deviation log created
- [ ] Figure scripts tested

---

# 43. Development priorities

The development roadmap should follow scientific dependency rather than visual appeal.

## Phase 1 — Foundation

- validate dataset and graph;
- automate provenance;
- freeze anatomical representation conventions;
- establish tests.

## Phase 2 — Controls

- build and validate null ensembles;
- measure what they preserve/destroy;
- produce structural report and visual demo.

## Phase 3 — Dynamics

- validate operating regimes;
- validate propagation/history;
- establish temporal coherence;
- produce dynamical report and visual demo.

## Phase 4 — Closed loop

- validate sensory interface;
- validate motor interface;
- validate benchmark determinism;
- produce reproducible closed-loop demo.

## Phase 5 — Learning sanity

- implement learning;
- solve controlled task;
- verify updates and logging;
- produce learning demo.

## Phase 6 — Pilot

- run small matched blocks;
- estimate variance and compute;
- inspect metric behavior;
- finalize confirmatory plan.

## Phase 7 — Freeze

- archive protocol;
- freeze seeds/configs;
- tag release.

## Phase 8 — Confirmatory Study I

- structural and functional control comparison.

## Phase 9 — Confirmatory Study II

- learning comparison.

## Phase 10 — Confirmatory Study III

- plasticity-location comparison.

## Phase 11 — Mechanistic and sensitivity analyses

Only after primary studies.

## Phase 12 — Replication

Independent benchmark/connectome/model axis where feasible.

## Phase 13 — Manuscript and final video

Generate both from the same evidence registry.

---

# 44. Criteria for a scientifically useful final study

The project should be considered scientifically successful if it achieves all of the following, regardless of whether the connectome performs well:

1. A reproducible whole-connectome-derived controller is built.
2. The boundary between anatomy and modelling assumptions is explicit.
3. Appropriate null models are constructed and independently validated.
4. The effect of structural controls on dynamics is quantified.
5. Learning comparisons use paired and reproducible experimental blocks.
6. At least one learning outcome is evaluated beyond final performance alone.
7. Plasticity placement is tested with serious controls for opportunity/accessibility.
8. Robustness or generalization is evaluated separately from nominal performance.
9. Modelling/interface sensitivities are quantified.
10. Primary and exploratory analyses are clearly separated.
11. Every claim is linked to machine-readable evidence and code provenance.
12. Negative and null findings are retained.
13. The final video accurately reflects the scientific evidence rather than only the most impressive demonstration.
14. An independent researcher could reproduce the main figures without manually reconstructing undocumented steps.

---

# 45. Decision rule for future discoveries

This project is expected to discover unexpected structural or dynamical properties.

The protocol therefore distinguishes **socle changes** from **mechanistic discoveries**.

A new finding does **not** require changing the study foundation if it can be expressed as:

> “We discovered that property X may explain the architecture effect; we will test X in the mechanistic-analysis layer.”

A foundation change is required only if the finding invalidates one of the following:

- the dataset representation;
- the comparison itself;
- the benchmark’s ability to measure the intended phenomenon;
- the learning mechanism’s validity;
- the independence of the experimental unit;
- the integrity of the confirmatory analysis.

This is the main safeguard against repeatedly redesigning the project around every interesting observation.

---

# 46. Current working contribution statement

Until results exist, the project should describe its intended contribution cautiously:

> **This study investigates how measured biological wiring constrains neural dynamics, learning, and closed-loop control. It combines whole-brain connectomic architecture with controlled network null models, adaptive learning, plasticity-location experiments, and a deterministic sensorimotor benchmark. Rather than assuming an advantage for biological topology, the study aims to determine which observed effects survive structural, dynamical, interface, and learning controls.**

This statement is broad enough to survive negative findings and specific enough to define a coherent research contribution.

---

# 47. Reference literature to anchor the study

The final literature review should be expanded systematically, but the project is already anchored by several directly relevant lines of work.

## Whole-brain connectome and network structure

- Dorkenwald, S. et al. Whole-brain Drosophila connectome work and FlyWire resource.
- Lin, A. et al. *Network statistics of the whole-brain connectome of Drosophila*. Nature (2024). DOI: 10.1038/s41586-024-07968-y.

## Connectome-derived dynamics

- Shiu, P. K. et al. *A Drosophila computational brain model reveals sensorimotor processing*. Nature 634, 210–219 (2024). DOI: 10.1038/s41586-024-07763-9.

This work demonstrates that a simple dynamical model constrained by Drosophila connectivity and neurotransmitter identity can be used to investigate brain-wide sensory processing, while also making clear that the dynamical model is an additional modelling layer placed on the connectome.

## Connectome-based computation

- Suárez, L. E. et al. *Connectome-based reservoir computing with the conn2res toolbox*. Nature Communications 15, 656 (2024). DOI: 10.1038/s41467-024-44900-4.

This work provides a useful conceptual and methodological precedent for studying how network architecture shapes computational properties while treating the connectome as a fixed recurrent substrate.

## Additional literature categories to review before freeze

The agent responsible for the literature review should search systematically for:

- Drosophila connectome null models;
- hierarchy-preserving or spatially constrained graph nulls;
- connectome-informed artificial neural networks;
- biologically constrained reservoir computing;
- reward-modulated local plasticity;
- mushroom-body learning models;
- graph rewiring and network controllability;
- robustness and lesion studies in connectome-derived models;
- embodied Drosophila simulation;
- neuromorphic implementations of connectome-scale networks;
- recent 2025–2026 connectome-informed controllers;
- best practices in computational neuroscience reproducibility and preregistration.

The literature review must distinguish peer-reviewed publications, preprints, software papers, and hobby/demo repositories.

---

# 48. Final principle

The project should always be able to answer four separate questions:

1. **What came from the measured anatomy?**
2. **What came from our dynamical assumptions?**
3. **What came from the engineered interface and learning rule?**
4. **What did the experiment actually demonstrate?**

If those four answers remain separable, the study can produce a meaningful scientific result even when the result is negative, surprising, or different from the mechanism initially expected.

That separation is the permanent foundation of **What Wiring Determines**.
