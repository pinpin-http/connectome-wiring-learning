"""RETINO_ALIGN_V1 (docs/retino_align/MISSION.md): does the spatial alignment between the eye's signals and the
connectome make spatial information easier to read out with few examples?

The network and its nulls stay frozen; only an external linear decoder learns. Modules:

    retina     columns of the injected lamina cells and the column permutations (mission §6)
    stimuli    the oriented-grating generator with two independent labels (mission §7)
    simulate   the frozen LIF network, batch-invariant keyed noise, readout spike counts (mission §8)
    decode     multiclass ridge with the normalised-loss convention (mission §9)
    analysis   M, L, D, G, S and the paired, graph-block bootstrap (mission §11-§12)
"""
