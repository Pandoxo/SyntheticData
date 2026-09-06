# Synthetic trash dataset

Renders COCO-annotated images of litter using BlenderProc: trash models are dropped
with physics into marked areas, photographed under randomised HDRIs, focal lengths and
camera poses, and written out with instance masks and bounding boxes.

Two variants share the same pipeline:

| | script | config | environment |
|---|---|---|---|
| plane | `generate.py` | `config.yaml` | a bare ground slab — fast, for checking assets and labels |
| scene | `generate_scene.py` | `config_scene.yaml` | a real `.blend` environment with marked spawn zones |

## Quick start

```bash
# 1. Check the asset catalogue: native sizes and the scale factor each implies.
.venv/bin/blenderproc run inspect_blend.py -- --assets

# 2. Render one scene on the bare plane and look at the result.
.venv/bin/blenderproc run generate.py -- --config config.yaml --scene-index 0
cd output/plane/scene_00000 && ../../../.venv/bin/blenderproc vis coco -i 0 -c coco_annotations.json -b .

# 3. Render a dataset: one Blender process per scene, in parallel, then merge.
.venv/bin/python run_batch.py --config config.yaml --jobs 4 --quiet
```

Output lands in `output/<name>/scene_XXXXX/` (images, per-scene `coco_annotations.json`,
`scene_meta.json` with the HDRI/lights/camera parameters used), plus a merged
`output/<name>/coco_annotations.json` covering the whole dataset.

Every flag is spelled out under [Running the scripts](#running-the-scripts); splitting a
big run across GPUs or machines is under
[Rendering on several machines or GPUs](#rendering-on-several-machines-or-gpus).

## Running the scripts

Everything that touches Blender is launched through `blenderproc run`, which re-executes
the script inside Blender's own Python:

```bash
.venv/bin/blenderproc run [blenderproc flags] <script.py> -- [script flags]
```

The bare `--` marks where BlenderProc's own flags stop and the script's begin.
BlenderProc forwards anything it does not recognise to the script, so leaving `--` out
usually still works — but a flag it *does* recognise (`--temp-dir`,
`--custom-blender-path`, `--keep-temp-dir`, …) is consumed before the script ever sees
it. Keep the separator.

`run_batch.py` is the exception — it runs under the venv's plain Python and spawns
`blenderproc run` itself, once per scene.

Paths inside the YAML configs are resolved against the project root, not the working
directory, so the commands below work from anywhere. `~` is expanded.

### `generate.py` / `generate_scene.py` — render one scene

```bash
.venv/bin/blenderproc run generate.py -- --config config.yaml --scene-index 0
.venv/bin/blenderproc run generate_scene.py -- --config config_scene.yaml --scene-index 3
```

| flag | default | meaning |
|---|---|---|
| `--config PATH` | `config.yaml` (`config_scene.yaml` for the scene variant) | which YAML to read; everything randomised lives there |
| `--scene-index N` | `0` | which scene to render. Also **drives the seed** — `seed = config seed + N`, so the same index always rebuilds the same scene |
| `--output-dir DIR` | `<output_dir>/scene_<N:05d>` | where this scene's images, `coco_annotations.json` and `scene_meta.json` go |
| `--debug` | off | listen for a `debugpy` client on port 5678 and block until one attaches |

A scene can legitimately produce nothing and still exit 0: if every object bounces out
of its spawn zone, or no camera pose passes validation, the script logs the reason and
skips the render.

### `run_batch.py` — render a dataset

```bash
.venv/bin/python run_batch.py --config config.yaml --jobs 4 --quiet
.venv/bin/python run_batch.py --config config_scene.yaml --script generate_scene.py --jobs 2
```

| flag | default | meaning |
|---|---|---|
| `--config PATH` | `config.yaml` | passed through to each scene process |
| `--script NAME` | `generate.py` | `generate.py` (bare plane) or `generate_scene.py` (`.blend` environments) |
| `--jobs N` | `1` | scenes rendered concurrently; each is a full Blender process |
| `--scenes N` | `num_scenes` from the config | how many scenes to render |
| `--start N` | `0` | first scene index — the sharding knob, see below |
| `--resume` | off | skip scenes that already have a `coco_annotations.json` |
| `--merge-only` | off | skip rendering; just merge the per-scene COCO files that already exist |
| `--quiet` | off | capture Blender's output and print it only for scenes that failed |

Each scene writes its own COCO file, which is what keeps parallel workers from racing
each other; the merge step at the end renumbers image and annotation ids and rewrites
`file_name` relative to the dataset root. Exit status is non-zero if any scene failed.

### `inspect_blend.py` — check the catalogue or a scene

```bash
.venv/bin/blenderproc run inspect_blend.py -- --assets
.venv/bin/blenderproc run inspect_blend.py -- --scene scenes/factory.blend
```

| flag | default | meaning |
|---|---|---|
| `--assets` | off | for every catalogued asset: class, mesh count, native bounding box, and the scale factor `longest_m` implies |
| `--scene PATH` | — | list a scene's objects with their sizes and flag which ones are spawn zones |
| `--config PATH` | `config_scene.yaml` | note the default differs from the generators — pass `--config config.yaml` to inspect the plane variant's catalogue |
| `--pattern REGEX` | `scene.spawn_zone_pattern` | override what counts as a spawn zone |

At least one of `--assets` / `--scene` is required.

### `usdz_to_blend.py` — turn USD models into catalogue assets

```bash
.venv/bin/blenderproc run usdz_to_blend.py -- --input ~/downloads/models --asset
.venv/bin/blenderproc run usdz_to_blend.py -- --input can.usdz --asset --dry-run
```

| flag | default | meaning |
|---|---|---|
| `--input PATH…` | required | files and/or directories; directories are searched recursively for `--pattern` |
| `--output-dir DIR` | `resources/objects` | already in `model_dirs`, so converted models are picked up automatically |
| `--pattern GLOB` | `*.usdz` | used when `--input` names a directory; `'*.usd*'` sweeps every USD flavour |
| `--asset` | off | **the flag for catalogue models** — shorthand for `--geometry-only --join --center` |
| `--geometry-only` | off | drop cameras and lights, keep meshes and materials |
| `--join` | off | join every mesh into one object, so a bottle's cap and label do not each get their own COCO box |
| `--center` | off | origin at the bounding-box centre, model sat on the world origin |
| `--scale F` | `1.0` | Blender ignores the stage's `metersPerUnit`, so a model authored in centimetres imports 100× too big — `--scale 0.01` fixes those. Irrelevant for catalogue assets, which `longest_m` rescales anyway |
| `--flat` | off | write `<output-dir>/<name>.blend` instead of `<output-dir>/<name>/<name>.blend` (the layout `sdlib/assets.py` reads the asset name from) |
| `--compress` | off | smaller on disk, slower to load |
| `--overwrite` | off | reconvert inputs whose `.blend` already exists |
| `--dry-run` | off | list what would be converted and where, then stop |

A bad model is reported and skipped rather than killing the batch; the exit status is
non-zero if any file failed.

### Looking at the results

```bash
cd output/plane/scene_00000
../../../.venv/bin/blenderproc vis coco -i 0 -c coco_annotations.json -b . --save
```

`-i` picks the image index, `-c` the annotation file, `-b` the folder holding it and the
images, `--save` writes `<base_path>/coco_annotated_<i>.png`. Be aware that the
BlenderProc script calls `im.show()` unconditionally, `--save` or not: on a headless
machine it writes the PNG and then hangs trying to open a viewer, so interrupt it once
the file exists.

The merged dataset-level file works too — its `file_name`s are relative to
`output/<name>/`, so run it with `-b output/plane`.

### BlenderProc's own flags

These go **before** the `--`, and matter mostly when deploying to a machine you do not
own:

| flag | meaning |
|---|---|
| `--custom-blender-path PATH` | use an existing Blender install instead of BlenderProc's own. It must be Blender 4.2.x — 4.2.1 is what this project is verified against |
| `--blender-install-path PATH` | where BlenderProc downloads Blender to. The default is `/home_local/<user>/blender`, which does not exist on most machines; it then falls back to `~/blender` and says so |
| `--temp-dir DIR` | scratch space, default `/dev/shm` if present, else `/tmp`. Point it at a real disk on nodes with a small tmpfs |
| `--keep-temp-dir` | leave the temp directory behind for debugging |
| `--reinstall-blender` | delete and re-download Blender |

### Other entry points

`make_example_scene.py` takes no arguments and rebuilds `resources/example_scene.blend`.
`quickstart.py` is a BlenderProc smoke test that blocks on `debugpy` port 5678 until a
debugger attaches — it is not part of the dataset pipeline.

## Rendering on several machines or GPUs

The unit of work is the **scene index**. `seed = config seed + scene_index` drives every
random choice in a scene, so an index is a complete, self-contained description of what
to render: two workers given disjoint index ranges produce disjoint, non-overlapping
scenes with no coordination, no shared state and no locking. Sharding is therefore just
arithmetic on `--start` and `--scenes`.

Measured cost of one scene on an RTX 3060 + Ryzen 5 2600, at this repo's defaults
(640×640, 128 samples, 6 views): **22 s wall, 1.9 GB peak RSS, ~6 cores busy**. Scale
those numbers to your own hardware before sizing a run — they are the budget per
concurrent job, not per machine.

### One machine

```bash
.venv/bin/python run_batch.py --config config.yaml --scenes 200 --jobs 4 --quiet
```

`--jobs` is bounded by RAM (~2 GB per job) and by VRAM if you render on the GPU. Note
that BlenderProc renders on the GPU **and** the CPU by default — the log lines
`Device … of type OPTIX found and used` and `Device … of type CPU found and used` say
what it picked — so concurrent jobs contend for cores as well as for the GPU. The
measurement above averaged ~6 busy cores over the whole 22 s (single-threaded while
loading assets and simulating physics, all threads during the render), so on a 12-thread
box 2–4 jobs saturate it and more mostly queue.

### Several GPUs in one machine

Cycles honours `CUDA_VISIBLE_DEVICES`, so pinning a worker to a GPU needs no code
change. One `run_batch.py` per GPU, each with its own slice of the index space:

```bash
# 4 GPUs x 250 scenes = scenes 0..999
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python run_batch.py \
      --config config.yaml --start $((gpu * 250)) --scenes 250 --jobs 2 --quiet &
done
wait
.venv/bin/python run_batch.py --config config.yaml --merge-only
```

Sanity-check the pinning in the log: a worker that sees no usable GPU prints
`Using only the CPU for rendering` and quietly renders far slower rather than failing.

The final `--merge-only` is deliberate — every worker's own merge pass at exit sees only
the scenes it finished, so one merge after all of them have exited is what produces a
complete dataset-level `coco_annotations.json`.

### Several machines

Give each machine a disjoint range, render to **local** disk, then collect and merge
once:

```bash
# on each node: its own rank, 4 nodes x 250 scenes = scenes 0..999
NODE=0
.venv/bin/python run_batch.py --config config.yaml \
    --start $((NODE * 250)) --scenes 250 --jobs 4 --quiet

# on the collector, once every node is done
rsync -a node1:/path/output/plane/  output/plane/
rsync -a node2:/path/output/plane/  output/plane/
rsync -a node3:/path/output/plane/  output/plane/
.venv/bin/python run_batch.py --config config.yaml --merge-only
```

`scene_XXXXX` directory names carry the index, so shards from different machines drop
into one tree without collisions as long as their ranges do not overlap. Rendering
straight onto a network share works but is usually the bottleneck — each scene writes
its PNGs, masks and JSON at the end of a 20-second job.

Under Slurm the same shape becomes an array job, one task per index block:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
PER_TASK=100
.venv/bin/python run_batch.py --config config.yaml \
    --start $((SLURM_ARRAY_TASK_ID * PER_TASK)) --scenes $PER_TASK --jobs 2 --quiet
```

### What every node needs

A node renders from local files only; nothing is fetched at run time. Each one needs the
repo and a venv holding `blenderproc` (2.8.0 here), `pyyaml` and `numpy` — the driver
scripts import nothing else outside Blender — plus:

| resource | where it comes from | if it is missing |
|---|---|---|
| Blender 4.2.1 | downloaded by BlenderProc on first run, or `--custom-blender-path` | first run on each node pays the download; pre-seed it in the image |
| every directory in `objects.model_dirs` | `resources/objects` is in the repo; `~/blenderkit_data/models` is not | **hard failure** — `FileNotFoundError` before anything renders |
| every directory in `hdri.folders` | `resources/haven/hdris` is in the repo; `~/blenderkit_data/hdrs` is not | only a warning, and the run continues with whatever it found |

That last asymmetry is the one to watch on a cluster: the HDRI is picked by index from
the sorted list of files found, so a node with a different set of HDRIs renders the same
scene index under a **different sky**. Sync the HDRI folders, or trim `hdri.folders` to
the ones that live in the repo, before splitting a run across machines.

### Reproducibility and reruns

Same index + same config + same assets + same BlenderProc/Blender version ⇒ same scene:
same models, same drop physics, same camera poses, same HDRI. Pixels can still differ
slightly between unlike GPUs, since the denoiser and Cycles' device backends are not
bit-identical; the annotations are unaffected.

Reruns are cheap to make safe. `--resume` skips any scene that already has a
`coco_annotations.json`, so a node that died halfway through can be restarted with the
same command:

```bash
.venv/bin/python run_batch.py --config config.yaml --start 250 --scenes 250 --resume --quiet
```

Failed scenes are listed at the end of the run with their exit codes, and `run_batch.py`
itself exits non-zero if any scene failed — enough for a scheduler to retry the task.

## Using your own scenes

BlenderProc ships no scenes of its own — only loaders for datasets you download
separately, all of which are indoor. Outdoor `.blend` environments (BlenderKit or your
own) are loaded directly instead.

To mark where trash may appear:

1. Open the scene `.blend` in Blender.
2. `Add > Mesh > Plane`, then scale and move it so it lies flat **on** the ground
   exactly where you want objects — a 2×2 m patch of pavement, the gutter by a kerb.
3. Name it starting with `spawn_` (`spawn_pavement`, `spawn_gutter`, …). Add as many
   as you like, then save.
4. Verify: `blenderproc run inspect_blend.py -- --scene path/to/scene.blend`
5. Add the file to `scene.files` in `config_scene.yaml`.

Zones are never rendered and never annotated. An invisible box collider is built with
its top face on the zone surface, so objects land on the zone regardless of what the
scene's real ground mesh looks like. Keep zones flat and roughly axis-aligned; for
sloped ground set `physics.collider: mesh`.

`resources/example_scene.blend` (rebuild with `blenderproc run make_example_scene.py`)
is a minimal already-marked-up scene showing the convention.

## Assets

The catalogue is built from two directories, both listed under `objects.model_dirs`:

| directory | what it is |
|---|---|
| `resources/objects/<name>/<name>.blend` | 140 models converted from the `.usdz` originals in `resources/trash3D-…/`, one item per file, geometry joined and textures packed |
| `~/blenderkit_data/models` | the original BlenderKit downloads |

`resources/objects_needs_split/` holds 45 further models that each contain **several**
items — a set of plates, three bottles, a pile of knives. They are deliberately outside
`model_dirs`: `sdlib/assets.py` joins every mesh in a file into one object, so each of
those would produce one COCO box spanning every item and one rigid body that falls as a
lump. Split them in Blender to promote them; see that directory's README.

`resources/assets/` is a flat archive of the same models as saved by Blender 5.2. The
pipeline cannot read it — BlenderProc runs Blender 4.2.1, which rejects a 5.x file
outright — so nothing points at it.

### Adding or correcting assets

`assets.yaml` maps each model file to one of 19 classes and to its real-world size.
Every asset is uniformly rescaled on load so its longest axis matches `longest_m`, which
cancels out the models' inconsistent unit conventions — one ships in centimetres,
another at 1/1000 scale. Run `inspect_blend.py -- --assets` after editing to check the
implied scale factors, and correct any `longest_m` that clearly does not match the item
the model depicts.

`match:` is a case-insensitive substring of the `.blend` path. For `resources/objects`
write it as `/<name>/` — the slashes keep each rule matching exactly one model and stop
`/knife/` from also swallowing `pocket_knife`. First rule that matches wins.

Entries marked `CHECK:` are the ones where the model alone did not settle the size:
`beer_`, `cigarette`, `fbx`, and the three container-plus-detached-lid models
(`cc0_tin_can_4_open`, `cc0_potato_chips_tube_2_open`,
`scan_chinese_steamer_freepolyorg`) whose bounding box spans both pieces.

Three single-item models sit under `ignore:` rather than in the catalogue:
`giggle_pacifier` and `covid_19_oral_vaccine_inhaler_prototype` are a large ground plane
with a tiny prop on it (the plane is what would get annotated), and
`controller_3d_scan_test` is an unidentifiable scrap of a test scan.

### From `.usdz`

The catalogue only globs `.blend`, so USD models (Apple Quick Look, Reality Composer,
Polycam scans, Sketchfab USD exports) have to be converted first:

```bash
# One file, or a whole directory tree, into resources/objects/<name>/<name>.blend
.venv/bin/blenderproc run usdz_to_blend.py -- --input ~/downloads/models --asset
```

`--asset` is the flag to use for catalogue models: it keeps geometry only, joins every
mesh into one object (otherwise a bottle's cap and label each become their own COCO
box), and centres the result. Without it the conversion is faithful — cameras, lights
and hierarchy are preserved. The
[full flag list](#usdz_to_blendpy--turn-usd-models-into-catalogue-assets) is above.

Textures are packed into each `.blend`, so the output is self-contained: a `.usdz` is a
zip and its images exist only inside it. The default output directory is
`resources/objects`, which is already in `objects.model_dirs`, so a converted model only
needs a `match: /<name>/` rule in `assets.yaml`; check it with
`inspect_blend.py -- --assets`.

Convert rather than saving from Blender by hand: BlenderProc runs Blender 4.2.1, which
cannot open a file written by Blender 5.x at all — it rejects it as "not a blend file".

## Layout

```
generate.py            plane variant entry point
generate_scene.py      scene variant entry point
run_batch.py           renders many scenes (one process each) and merges the COCO files
inspect_blend.py       --assets / --scene <file>: inspect the catalogue or a scene
usdz_to_blend.py       converts .usdz/.usd models into catalogue-ready .blend files
make_example_scene.py  rebuilds resources/example_scene.blend
assets.yaml            class map + real-world sizes
sdlib/
  assets.py            catalogue, loading, size normalisation
  environment.py       HDRI, lights, spawn zones, colliders, drop physics
  capture.py           renderer setup, camera framing, annotation writing
  pipeline.py          per-scene orchestration shared by both entry points
  cli.py               argument parsing and start-up
  config.py            YAML loading and path resolution
resources/
  objects/             catalogue models, one item per .blend
  objects_needs_split/ models holding several items each - not in model_dirs
  assets/              flat Blender 5.2 archive, unreadable by the pipeline
  trash3D-.../         the .usdz originals everything above was converted from
  haven/hdris/         HDRIs
```

## Notes

- One Blender process per scene is deliberate. Blender never frees appended materials,
  images and meshes, so looping scenes in-process grows memory until it dies.
- Segmentation is switched on immediately before rendering, not at start-up:
  `enable_segmentation_output` stamps a `pass_index` onto objects *at call time*, and
  anything created later is silently treated as background.
- Focal length is keyframed per frame. `add_camera_pose` only keyframes position and
  rotation, so setting `camera.data.lens` alone would apply the last sampled value to
  every frame in the render.
- Camera distance is derived from the subject's real size and the current focal length
  rather than sampled from a metre range, so a long lens backs off and a wide lens
  moves in.
- Annotations below `annotations.min_visible_pixels` are dropped after rendering. The
  instance segmap already accounts for occlusion, so this removes the sliver boxes left
  by objects hidden behind scene geometry.
- Blender's USD importer converts a Y-up stage to Z-up, but ignores `metersPerUnit`, so
  a `.usdz` authored in centimetres imports 100× too large. `usdz_to_blend.py --scale`
  corrects it; for catalogue assets it does not matter, since `longest_m` rescales them.
- `import_cameras=False`/`import_lights=False` do not skip those USD prims — Blender
  still imports them as empties to keep the hierarchy, which is why `--geometry-only`
  deletes non-mesh objects itself.
- BlenderKit assets carry per-asset licences (royalty-free vs CC-BY). Check them before
  publishing a dataset built from them.
