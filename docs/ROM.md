# Supplying the Super Mario Bros. ROM

The ROM is **not** in this repository and never will be: it is Nintendo's
copyrighted work. `roms/` and `*.nes` are in `.gitignore`.

You need a dump of a cartridge you own.

1. Put the file at `roms/SuperMarioBros.nes`.
2. Check it matches the revision stable-retro's integration expects:

   ```bash
   sha1sum roms/SuperMarioBros.nes
   # must equal: facee9c577a5262dbe33ac4930bb0b58c8c037f7
   ```

   That is the US/JP World revision. Other dumps have different RAM layouts and
   the addresses in `integrations/SuperMarioBros-Nes/data.json` will read garbage.

3. Import it:

   ```bash
   .venv/bin/python -m retro.import roms/
   ```

4. Verify:

   ```bash
   make play
   ```

Everything except `make play` — including `make selftest`, `make calibrate` and
the whole test suite — runs without the ROM, because the synthetic probe frames
in `smbfly.brain.calibrate` stand in for gameplay.

---

## Revision matters — verified failure

A PAL dump was tested here and **rejected**:

```
Super Mario Bros. (Europe).nes
  sha1  b4140688f9a5b3434276c5a7394b56022436e57a
  want  facee9c577a5262dbe33ac4930bb0b58c8c037f7
```

`python -m retro.import roms/` reports `Imported 0 games`, and `retro.make`
then raises `FileNotFoundError: Game not found`. stable-retro matches the ROM
against `rom.sha`, so a different revision is simply not seen.

Even if it were accepted, PAL would break the project downstream:

- PAL runs at **50 Hz**, NTSC at 60 Hz, so every frame-timing figure shifts.
- The shipped savestates (`Level1-1.state` … `Level8-1.state`) were recorded on
  the NTSC ROM and will not load.
- Essentially every SMB TAS on TASVideos is NTSC, so the deterministic-replay
  gate would fail for reasons unrelated to any model.

You need the **US / World** revision. Verify before anything else:

```bash
sha1sum roms/*.nes    # must be facee9c577a5262dbe33ac4930bb0b58c8c037f7
```

---

## Two gotchas, both verified here

**1. The sha1 you compute will not match `rom.sha`.** stable-retro hashes the
ROM *without* its 16-byte iNES header:

```
Super Mario Bros. (World).nes
  sha1 of whole file      33d23c2f2cfa4c9efec87f7bc1321ce3ce6c89bd   ← looks wrong
  sha1 of body (skip 16)  facee9c577a5262dbe33ac4930bb0b58c8c037f7   ← matches
```

**2. `retro.import` does not put the ROM where our integration looks.** It
copies into the packaged directory (`site-packages/.../SuperMarioBros-Nes-v0/`),
while our extended integration lives in `integrations/SuperMarioBros-Nes/`.
Without a link you get `FileNotFoundError: No romfiles found`.

`make import-rom` handles both: it unzips, checks the headerless sha1, imports,
and symlinks. Verified working — 700 frames of `RIGHT+B` at 2737 FPS, Mario
reaching x=312 before dying on the first obstacle.
