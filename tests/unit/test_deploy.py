"""deployment/flydeploy.py: configuration, deny lists, experiment resolution, inventory
destinations (no SSH, no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment"))

import flydeploy as fd


def test_config_file_then_environment(tmp_path, monkeypatch):
    f = tmp_path / "deploy.env"
    f.write_text('# comment\nREMOTE_HOST=ssh9.vast.ai\nREMOTE_PORT=41000\nLOCAL_ROM="roms/My ROM.nes"\n')
    monkeypatch.delenv("REMOTE_HOST", raising=False)
    monkeypatch.setenv("REMOTE_PORT", "42000")
    cfg = fd.load_config(f)
    assert cfg["REMOTE_HOST"] == "ssh9.vast.ai"
    assert cfg["REMOTE_PORT"] == "42000"                    # the environment wins
    assert cfg["LOCAL_ROM"] == "roms/My ROM.nes"            # quotes removed
    assert cfg["REMOTE_DIR"] == fd.DEFAULTS["REMOTE_DIR"]


@pytest.mark.parametrize("path", ["roms/Super Mario Bros. (World).nes", "roms/x.zip", "deployment/deploy.env",
                                  "home/.ssh/id_ed25519", "a/b/server.pem", ".claude/settings.json",
                                  "gpucore.87935", "x/aws_credentials.json"])
def test_sensitive_files_are_refused(path):
    assert fd.denied(path, fd.SENSITIVE)
    assert fd.denied(path)


@pytest.mark.parametrize("path", ["src/flymario/paths.py", "deployment/flydeploy.py", "data/connectome/.gitkeep",
                                  "integrations/SuperMarioBros-Nes/Level1-1.state", "docs/ROM.md"])
def test_ordinary_tracked_files_are_not_sensitive(path):
    assert not fd.denied(path, fd.SENSITIVE)


def test_experiments_resolve_commands_outputs_and_trainer_names():
    exps = fd.experiments()
    _, cmd, out = fd.resolve_experiment(exps["probe-report"], ["suite=unseen", "rule=hold6", "stream=expert"])
    assert cmd[-6:] == ["--suite", "unseen", "--rule", "hold6", "--stream", "expert"]
    assert out == ["artifacts/runs/frozen_probe/reportv2x-unseen-hold6-20261009.json"]
    p, cmd, out = fd.resolve_experiment(exps["stageD"], ["graph=nulllat:1"])
    assert p["runname"] == "mario_stageD_nulllat1" and out == ["artifacts/runs/checkpoints/mario_stageD_nulllat1.pt"]
    assert "--graph" in cmd and cmd[cmd.index("--graph") + 1] == "nulllat:1"
    p, _, _ = fd.resolve_experiment(exps["stageD-eval"], ["graph=null:0", "ckpt=x.pt"])
    assert p["evalname"] == "mario_stageD_eval_null0_frozeneval"
    _, cmd, out = fd.resolve_experiment(exps["cs-expert"], ["graph=nulllat:0"])
    assert out[0].endswith("common_stream_expert_nulllat0.pt") and cmd[-2:] == ["--stream", "expert"]
    with pytest.raises(SystemExit):
        fd.resolve_experiment(exps["cs-expert"], ["graph=null:7"], "cs-expert")


def test_every_registered_experiment_is_declared_and_resolves():
    for name, e in fd.experiments().items():
        assert e.get("declared_in"), name
        fd.resolve_experiment(e, [], name)


def test_rocm_reference_files_never_land_on_new_output_paths(tmp_path):
    for rel in ("artifacts/runs/checkpoints/common_stream_connectome.pt", "data/connectome/malecns.npz",
                "artifacts/runs/harness_v2/starts-20261009.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    items = {it["path"]: it for it in fd.expand_spec(tmp_path)}
    assert items["artifacts/runs/checkpoints/common_stream_connectome.pt"]["dest"] == \
        "artifacts/rocm_reference/checkpoints/common_stream_connectome.pt"
    assert items["data/connectome/malecns.npz"]["dest"] == "data/connectome/malecns.npz"
    assert items["artifacts/runs/harness_v2/starts-20261009.json"]["dest"] == "artifacts/runs/harness_v2/starts-20261009.json"


def test_committed_manifest_matches_the_spec_and_has_checksums():
    man = json.loads(fd.MANIFEST.read_text())
    required = [it for it in man["items"] if it["set"] in ("core", "reference")]
    assert required and all(it["sha256"] and it["size"] for it in required)
    graph = next(it for it in man["items"] if it["path"] == "data/connectome/malecns.npz")
    nulls = [it for it in man["items"] if "depth_stratified_lateral" in it["path"]]
    assert all(graph["sha256"][:12] in it["path"] for it in nulls)       # null caches are keyed to the graph


def _fake_run(tmp_path, monkeypatch, experiment, params, log):
    monkeypatch.setattr(fd, "RUNS_DIR", tmp_path)
    rd = tmp_path / f"{experiment}-x"
    rd.mkdir()
    (rd / "manifest.json").write_text(json.dumps({"experiment": experiment, "params": params, "outputs": []}))
    (rd / "status.json").write_text(json.dumps({"state": "finished", "exit_code": 0, "elapsed_s": 3600}))
    (rd / "log.txt").write_text(log)
    return rd


def test_progress_of_a_common_stream_run(tmp_path, monkeypatch):
    rd = _fake_run(tmp_path, monkeypatch, "cs-expert", {"graph": "connectome"},
                   "expert stream: 4384 frames, label on 0.205, blocked among label-on 0.136\n"
                   "connectome {'frames': 0, 'auc': [0.3, 0.4, 0.5]}\n"
                   "connectome {'frames': 1500, 'auc': [0.81, 0.82, 0.83]}\n")
    p = fd.run_progress(rd)
    assert p["frames"] == "1500/4000" and p["test_auc_per_readout"] == "0.81, 0.82, 0.83"
    assert p["stream_check"].endswith("0.136")


def test_progress_of_a_frozen_play_run(tmp_path, monkeypatch):
    rd = _fake_run(tmp_path, monkeypatch, "probe-play-training", {"graph": "connectome"},
                   "calibration {}\nLevel1-1 delay  0 (65s): cs_ro11=312/312\nLevel1-1 delay  1 (60s): cs_ro11=400/350\n")
    p = fd.run_progress(rd)
    assert p["starts"] == "2/64" and p["last_start"].startswith("Level1-1 delay  1")
