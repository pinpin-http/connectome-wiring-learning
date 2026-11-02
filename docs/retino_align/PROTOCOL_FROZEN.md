# RETINO_ALIGN_V1: frozen protocol (version 1)

Internal, traceable pre-specification (mission §13.2), recorded before any analysed graph (the connectome or
`nullin:0-39`) was simulated on the task. It is not a registration in an independent public registry. Its sha256
and the commit that contains it are written in `docs/retino_align/RUN_LOG.md` at the freeze; any later change is a
new protocol version with its own justification and list of affected results (mission §13.3).

Authority: `docs/retino_align/MISSION.md`. Configuration: `configs/retino_align/protocol.yaml`
(`protocol_status: frozen`). Code: `src/flymario/retino_align/`, `scripts/retino_align/`.

## 1. Hypothesis and primary contrast

In this model, a spatial permutation of the eye's input lowers the few-example decodability of orientation more in
the measured connectome than in family-c nulls.

- M(g, c, t): test accuracy averaged uniformly over the 5 budgets, the 2 dynamic seeds and the 10 subset draws.
- L(g, t) = M(g, ordered, t) − mean over the 3 permutations of M(g, p, t).
- **Primary contrast: D_orientation = L(0, orientation) − mean over the 40 nulls of L(g, orientation)**, in
  accuracy points. D > 0: the connectome loses more.
- Secondary: G = M(0, ordered, orientation) − mean_g M(g, ordered, orientation); D_brightness (same as D for the
  brightness task); S = D_orientation − D_brightness; D per permutation, per seed and per budget; activity
  diagnostics; descriptive ranks.

## 2. Margin of practical importance

**δ = 0.05 (5 accuracy points on D)**, the mission's proposal, accepted. Stated now: on the pilot graphs the
ordered orientation score M is ≈ 0.34, so a permutation can lower it by about 0.09 at most (to chance); δ is
therefore a large share of the available range, and an "equivalence at δ" reading would exclude only large
differential effects. Reading of the 95 % interval: the table of mission §12.4, implemented in
`analysis.reading` and tested.

## 3. Graphs

- g = 0: `data/connectome/malecns.npz` (MaleCNS v1.0), sha256 `6befb67a0278568316c26d32c3f2cadf76fbb8cc35d93137819c86f804a1df9f`.
- g = 1..40: `nullin:0` … `nullin:39`, family c (`depth_stratified_lateral_instrength`), **all 40** (the seeds of
  Study S; a deterministic rule that does not depend on any score). K = 40. Rebuilt on this server; each file's
  sha256 equals the one built on the owner's PC (`artifacts/retino_align_prep/nullin_hashes_{local,server}.json`).
  Validated (mission §5.1): `artifacts/retino_align/graph_validation/graphs.json`.
- `nullin:100`, `nullin:101`: pilot and development only, never analysed.
- Conclusions are relative to family c only.

## 4. Retina conditions

- ordered; perm1, perm2, perm3 = whole-column permutations within the 886-column signature group (seeds 31, 32,
  33), the 6 atypical columns fixed (99.1 % of cells permutable; 98.9 % moved). The same three permutations for every
  graph. Applied after the encoder and before injection, to train, validation and test alike. Files and sha256:
  §11.
- Static eye: reset, then two identical frames (change term zero; ON = 2·L, OFF = 2·(1 − L)); float frames.

## 5. Stimuli

- Static sinusoidal gratings, raised-cosine envelope (plateau 60 px, edge 92 px) centred on the eye's sampling
  centroid ± 8 px; orientation = direction of the stripes (0°, 45°, 90°, 135°; pixel space, y up); period U(24, 40) px;
  phase U(0, 2π); **amplitude U(0.15, 0.20)** (pilot P1); pixel noise U(−0.02, 0.02); pattern centred on the eye's
  sample points before the brightness is added; brightness = centre (0.35, 0.45, 0.55, 0.65) + U(−0.065, 0.065)
  (adjacent classes overlap by 0.03; Bayes accuracy from the true eye mean ≈ 0.83). Max luminance 0.935: no saturation.
- Partitions: train pool 1,024 (64 per orientation × brightness cell), validation 512 (32), test 2,560 (160); seeds
  5005, 6006, 7007; every image has its own latent draw; unique ids, digests and latents checked. The test is from
  the same generator: no out-of-distribution claim.

## 6. Network and features

- Frozen MaleCNS LIF network, anatomical weights (lin_raw: synapse count × sign), gain 0.18452, noise SD 0.02 per
  neuron and tick, threshold 1, decay 0.8466 (τ 20 ms, dt 3.33 ms), one refractory tick, 16 ticks from rest per
  image, input held constant. No plasticity. Operator sha256 checked on the GPU before and after every lot.
- Noise: one generator per (dynamic seed 1 or 2, image id), shared by every graph and condition.
- Features: spike counts of the 2,227 readout neurons (`datasets.motor_index`, ascending node order) before any
  action projection; same neurons and order for every graph; uint8 (exact, max 8).
- Lots: 41 graphs × 4 conditions × 2 seeds = 328, each with all 4,096 images.

## 7. Decoders

- Multiclass ridge on one-hot targets, objective (1/n)‖XW + b − Y‖² + λ‖W‖², intercept not penalised, argmax.
- Standardisation on the training subset only (SD 0 → 1).
- λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}; chosen per task, budget and fit on the 512 validation images; ties to the
  largest λ; no refit on train + validation. The validation images are a common resource of every fit.
- Budgets 8, 16, 32, 64, 128 per class (2, 4, 8, 16, 32 per cell); 10 draws (seeds 8101-8110), nested, the same
  subsets for both tasks and every graph and condition.
- Saved per fit: W, b, the subset's mean and SD, the validation scores of every λ, the λ chosen; per test example
  the prediction and the score margin; confusion matrices.

## 8. Uncertainty

- 95 % percentile interval of a paired bootstrap, 5,000 replicates, seed 4242: test images resampled with
  replacement within each of the 16 cells (the same multiset for every graph, condition and task); nulls resampled
  as whole graphs; M, L, D, G, S recomputed from saved per-example predictions without refitting.
- Scope: conditional on the connectome, the train and validation banks, the three permutations, the two dynamic seeds
  and the subsets used. It is not a population interval over flies, permutations or retrainings, and it is
  approximate.
- Sensitivity to permutations, seeds and budgets: reported separately (point estimates), secondary.
- Rank of the connectome's L among the 41 graphs: **descriptive only**. No calibrated Monte-Carlo test is claimed:
  the exchangeability of the connectome with family-c draws is not established.

## 9. Controls (checked before the freeze; repeated on the main inputs before the campaign's results are read)

- Permutation: bijection, exact inverse, types and sides kept, ON/OFF of a column together, value multisets kept
  globally and per type, three distinct permutations.
- Direct ridge on the input vectors, ordered vs permuted: identical scores within 1e-8 and identical predictions.
- Pixel orientation-energy estimate (no label, no latent): ≥ 0.90 accuracy required.
- Datasets: balance of the 16 cells in every partition and budget, nesting, no shared image or latent.
- Non-spatial references (declared secondary, PILOT.md P2): global descriptors and the sorted input values per cell
  type. These carry some orientation information (the eye's lattice is sampled anisotropically) that a permutation
  cannot change; they are reported as the arrangement-blind reference, not as an exclusion criterion.

## 10. Technical exclusion, retries, budget, interruptions

- Technical failure of a lot = CUDA error, out of memory or I/O error: up to 2 attempts, logged in
  `logs/failures.json`. An operator change, a count above 8 or a readout mismatch stops the campaign (scientific
  error, not retried). No lot, graph, seed or condition is excluded or rerun for its activity or score.
- Order: graphs in a fixed random order (seed 20261019), conditions rotated, seeds alternated.
- Measured budget: ≈ 8-10 s per lot (≈ 1 h for 328), decoders ≈ 15-30 s per graph × condition (≈ 1 h with 2-3
  workers), analysis minutes. The full target plan fits; no reduction of mission §14.3 is used.
- No new lot after the final reserve (T0 + 66 h = 2026-10-22T10:40:12Z, enforced by the orchestrator). An
  interruption leaves complete lots in place; incomplete ones are recomputed; the status file lists expected, done,
  failed and missing lots; incomplete analyses are labelled.

## 11. Files, hashes, commit and environment

From `artifacts/retino_align/main_v1/manifest.json`, written at the freeze.

| item | value |
|---|---|
| run | `artifacts/retino_align/main_v1/` (manifest created 2026-10-19T18:13:08+00:00) |
| code commit | `d16f46cad1739413d2afaf680b4cb017cd5f73fe` (src, scripts, configs clean: dirty = False) |
| `configs/retino_align/protocol.yaml` sha256 | `0e554d7b231d6ada9d34feaa8c4f86ae4861b81267da8c58df6578d21516040a` |
| `simulate.py` / `decode.py` / `stimuli.py` sha256 | `1c88146e2268f265…` / `e58db921566fc94a…` / `d2f30ff730e619a6…` |
| inputs (`inputs/inputs.npz`: X, uids, labels, latents) sha256 | `a035ef6de0990a021d467c0427db1b8efe7c79551d58e841ef480bceaa6baf5a` |
| permutations file sha256 | `506cc61f4e4f84e74d5ff2d642d24e6843514382dfa8a57f06d560d0e776989b` |
| perm1 (seed 31) cell map sha256 | `7cc8dd5ca935c82777af649b47ed85f547e59ebf6da37ca6291eac304e88b9a6`; moved 99.13 % of cells, mean displacement 14.68 columns, neighbour pairs kept 0.94 % |
| perm2 (seed 32) cell map sha256 | `beab562aa1e40efd909f5446aea4c8f89d62e4949b4aa1bc71110e4ea9edaa6b`; moved 99.13 % of cells, mean displacement 14.28 columns, neighbour pairs kept 0.78 % |
| perm3 (seed 33) cell map sha256 | `d9e5e0da6f91bc3169b5f0325a15b9442e79c7ef966a044670d88ac2446a5ade`; moved 99.13 % of cells, mean displacement 14.63 columns, neighbour pairs kept 0.82 % |
| retina digest (injected cells, columns) | `9955446c01ed4157a697bed9d511a92d50641ae8a9960fe5f0c0f8f1c6c7a372` |
| readout | 2227 neurons; node indices sha256 `95e80de210dadcc7bc9b76d419af5d5844f66af7379d8711248c2d094ac37b26`; body ids `0e3492fee2a8664e57a22d1cdf4e858d64f70040134cb4cbdcb6165f1c399bab` |
| images | train 1024, val 512, test 2560 |
| environment | Python 3.11.17, numpy 2.1.3, scipy 1.17.1, torch 2.9.1+cu128 (CUDA 12.8), NVIDIA GeForce RTX 3090 |
| lots | 328 in the declared order (manifest `lots`) |

Graph files (family c, `artifacts/cache/nulls/malecns_6befb67a0278_depth_stratified_lateral_instrength_s<seed>.npz`), sha256:

- `nullin:0` `cb8eeb7b70d716a31580ce667161a56c5dfd1de49744d182ae768d54c4b3ac33`
- `nullin:1` `2291f353e8ec1e3c384b780ce54adc62dcc04892d5bd56c82bee128773c01d2e`
- `nullin:2` `eb623d2f71092685eeef854d057885d590d6f7f3ae6e16ca6bf25503126224fa`
- `nullin:3` `f9fe02d63b56a1dbd61ae97732df912ba0d0177748959c3a42fc22621a68a40b`
- `nullin:4` `62d2875f797977a320c1e4b5b848a7c2e7262541186bdc1e04b7eba28a4eef55`
- `nullin:5` `371494f5cd8eb9a3b9122d4f701e71bf86a2673265f09ca1cf384cd6b38cc629`
- `nullin:6` `6ec6ed2e9b3777f3725f029fd30f8dc0ea1d683de3ce28c34391dcaf836ac4ab`
- `nullin:7` `6ac23e292971e0e781fb3c789394a935523e382dd09ea51a345ea38f84d91476`
- `nullin:8` `532a9b3202216679036159febc96eac65edde3b241a037e9d3c967d8ee623d51`
- `nullin:9` `30c84df3acf7685537215b3c6e312bd823aa289aa222596cbc5e830c6d8ffb9b`
- `nullin:10` `bf16a81d89ca711ce06e4fcfdcbf54077e55264790f247e65829ffcce250f8d7`
- `nullin:11` `50bf0a01a41e1de33730c08a335365e6eb1a5274135e91a417e34a7767c6dc2d`
- `nullin:12` `be1aee37916c5a84cd4df852fc7388192e8d2f8f21c30f3be2287798cdb50cff`
- `nullin:13` `c3aa0e03a6e05111b21417e0b1a7c0dda40165c6c92742c501aba427080d4607`
- `nullin:14` `8ba38ea38e43f9d38cb7be45267442a1cbb8931359583280a6b6fcfdc86e9c82`
- `nullin:15` `255aa446cdbe8aaad12838974eeaf8ff4c994fe16ccd7d9dccf326c650648fbb`
- `nullin:16` `6a2089101ba773d181b77064e67d522dce03366a41dc1f23748a807f5c886079`
- `nullin:17` `c696c837a93c60a0b87141c8855d5232775f795f82f593e80cf4461ff180fdb2`
- `nullin:18` `0a2b21580ad76ec36e2f92db805c7aad28bd5d58e27696ea9d15840c79f114dd`
- `nullin:19` `b525a375e12a646adb06f84c1936e1e7d088f845edac04d059b4fab3f43d6d6e`
- `nullin:20` `8c7f0d916d23e81ba10831ef5a42f1201df2df38bf2b235de4fae3555aaf1311`
- `nullin:21` `aa0706248367f78423328c69775cb734fb681bcf82877fefe4f223a1c3174c2b`
- `nullin:22` `6634a66180cae98bc92113b16de967b20c9518dab6a5d7b98932236db92f2469`
- `nullin:23` `0e54ee5daea8c80470134ea1a74b5df1e0fdf2790344b275d2efb08278baa5a4`
- `nullin:24` `c19880e35f50bde161a5c1684f7b6cfca116a4297622518773347354891c08eb`
- `nullin:25` `49f7212f42e0d7c937c24f398493e702fa662c5f6d499d9b18bd52ffeb659b36`
- `nullin:26` `a8a99433db342c4cd482b8634bcb6f3d95a19408c4471ecf43a4aa0cf7ec7dd3`
- `nullin:27` `4915cd61efa4d312216ec48cba3725d6cb921bb7d7ab122e589b9e6e15580f5b`
- `nullin:28` `96678318a78d2555d869b272d4660fb30542860093b1f6aeb53dc2c2db793ecc`
- `nullin:29` `ca99b7cf6567b64b72aeb34f8cffdd3f9562ddcf74acbf4edd24805939617908`
- `nullin:30` `704466ea78766c524b173be8676fd5a2dc07cb798ee4f3720358788b0c1cccea`
- `nullin:31` `74411c3f6a912ac329d66dc1df88262698c879b3c4394ad28ae6a4fddfb226f6`
- `nullin:32` `df6e6ea61c50d8d24a3bae43d2b4e89d47de267dafb9f006409c2776296bd4ca`
- `nullin:33` `cda8bd20e4da25543543e206eddc1628baf12086f48d25f985accd7098cdc7d5`
- `nullin:34` `92dda5ada5e90482c0d97f3f8f877f488a53a22292976ec3105ce793cec3766f`
- `nullin:35` `4ba166f6064d7f1101c260fbd5c1e9ac03e57fc26e0df6384349ca1d58e23cd9`
- `nullin:36` `feff1bc65ce80f66b3acdef8df5f1394b1a96793e8e51fbd6f2c45b404b16265`
- `nullin:37` `2fa03f952a2f0bd929e6f7fe40559a46352ca927e16243f001d5c47e0bbd9c6c`
- `nullin:38` `b68e37f80912ef032b03bcbdfe4d62cf6eabdf43bb55a77e56e7ca98e1b3a5a1`
- `nullin:39` `0b04db0c999aed9415265fd6f83506fe5067ae1181f0fa8f78088eae04439a25`
