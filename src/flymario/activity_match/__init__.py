"""ACTIVITY_MATCH_V1 (docs/activity_match/MISSION.md): does the connectome's orientation-decoding deficit of
RETINO_ALIGN_V1 persist when the connectome and its nulls run at comparable mean output activity?

The network stays the frozen RETINO_ALIGN_V1 simulator (flymario.retino_align.simulate); the only intervention is a
positive scalar gamma on the global synaptic gain, `gain = gamma * dynamics.gain`, which multiplies W @ s and
nothing else (not the input current, the noise, the threshold or the time constants). Modules:

    calibrate  label-free activity metrics, validity rules, common range, targets, bounded gamma search, selection
    analysis   M per graph and condition, G_c, R, T_real, T_nulls and their paired graph-block bootstrap
"""
