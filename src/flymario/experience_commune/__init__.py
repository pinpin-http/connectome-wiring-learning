"""EXPERIENCE_COMMUNE_V1 (docs/experience_commune/MISSION.md): does the connectome keep its learning advantage on
the two Study S tasks when every graph receives the same images, actions and rewards?

    policy    the target policy (softmax / tau mixed with an epsilon-uniform floor), the behaviour distributions of
              modes N (own policy) and U (independent uniform controller), and the importance-corrected
              score-function signal
    learn     the batched plastic learner (Study S's three-factor rule with the corrected signal), greedy
              evaluation on keyed noise, checkpoints and resume
"""
