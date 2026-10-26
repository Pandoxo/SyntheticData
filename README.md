# Synthetic litter dataset for a trash-collecting robot

## Overview — read this first

We are building a small robot that drives along sidewalks, streets, lawns and park
paths, spots litter with a camera mounted **about 10 cm above the ground**, and picks
it up with a robotic arm that has a second camera of its own. The robot needs a
small, fast neural network that finds and outlines every piece of trash in the image
(*instance segmentation*): bottles, cans, cardboard, cups, wrappers, cigarette
butts and so on.

Training such a network needs tens or hundreds of thousands of labelled photos taken
from exactly those unusual viewpoints. Photographing and hand-labelling that many is
not realistic, so this repository **generates the photos instead**: it builds small
3D worlds in Blender, drops 3D models of litter into them with real physics, and
renders pictures from the robot's point of view — together with perfect labels,
because in a synthetic world we know exactly which pixel belongs to which object.

Every rendered picture comes with

* the **colour image** (what the camera sees),
* a **depth map** (distance of every pixel, in millimetres),
* an **instance mask** (which pixels belong to which piece of trash),
* a **semantic mask** (which class every pixel belongs to),
* **COCO annotations** (boxes + masks, the standard format), convertible to
  **YOLO** for training,
* and a record of every random choice that produced it (scene, lights, camera,
  models, licences).

The worlds are deliberately varied, because a network only learns what it has seen:

* **10 classes of litter** from ~1000 3D models (scanned and modelled), randomly
  crushed, bent, dirtied and half-buried;
* **8 kinds of places** — sidewalk, street gutter, lawn, park path, forest floor,
  gravel, plaza, plain ground — built from hundreds of photo-scanned ground
  materials, with real 3D grass, weeds, stones, bins, hydrants, lamps, trees and
  building walls; plus corners of a few hand-made 3D scenes;
* **hundreds of real sky/lighting captures** (sun, overcast, dusk, night with street
  lights) and shadows of trees or buildings that are outside the picture;
* **three cameras**: the robot's own low front camera (40% of images), the arm
  camera looking down at an item (30%), and a person-height camera (30%) for extra
  variety — each with randomised lens, height and tilt.

Rendering runs on rented NVIDIA GPUs (NVIDIA Brev); results go straight to Google
Drive. A pipeline test on 4× RTX 6000 Ada is summarised in
[`docs/test_report/report.pdf`](docs/test_report/report.pdf) — render time, memory and
cost per number of GPUs.

**Where it stands:** the generator, the asset downloaders, the render-farm scripts
and the QA tools work end to end. What is still open is listed under
[Known issues and next steps](#known-issues-and-next-steps) — most importantly,
validating on *real* photos from the robot's camera, which is the only test that
counts.

---

## Contents

- [Quick start](#quick-start)
- [How a scene is made](#how-a-scene-is-made)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Assets: where every model and texture comes from](#assets-where-every-model-and-texture-comes-from)
- [The litter catalogue](#the-litter-catalogue)
- [Output format](#output-format)
- [Checking the data](#checking-the-data)
- [Rendering at scale on NVIDIA Brev](#rendering-at-scale-on-nvidia-brev)
- [Google Drive](#google-drive)
- [Training data export](#training-data-export)
- [Known issues and next steps](#known-issues-and-next-steps)
- [Implementation notes (gotchas)](#implementation-notes-gotchas)
- [Licences](#licences)

---

## Quick start

Everything that touches Blender runs through [BlenderProc](https://github.com/DLR-RM/BlenderProc)
2.8, which downloads and drives its own Blender **4.2.1** (it cannot open files saved
by Blender 5).

```bash
# 0. Python environment (once)
python3 -m venv .venv
.venv/bin/pip install -r brev/requirements.txt

# 1. Download the environment assets (ground materials, HDRIs, plants, props).
#    `dev` is ~2 GB of small textures for a laptop; `full` (~30 GB) is for render nodes.
.venv/bin/python fetch_assets.py --profile dev

# 2. (optional) Download litter models from Objaverse - ~800 more models.
.venv/bin/python fetch_objaverse.py --max-per-class 150

# 3. Render one scene and look at it.
.venv/bin/blenderproc run generate.py -- --config config.yaml --scene-index 0
.venv/bin/python qa_report.py output/robot_v1/scene_00000

# 4. Render a dataset on this machine: one Blender process per scene, then merge.
.venv/bin/python run_batch.py --config config.yaml --scenes 50 --jobs 2 --quiet
```

Useful while developing:

```bash
# force a world type / biome
.venv/bin/blenderproc run generate.py -- --scene-index 3 --env procedural --biome lawn
.venv/bin/blenderproc run generate.py -- --scene-index 3 --env scene
# thumbnails of every catalogue model, to catch wrong classes/sizes/textures
.venv/bin/blenderproc run catalog_sheet.py -- --out qa/catalog
```

The bare `--` separates BlenderProc's own flags from the script's. Keep it: a flag
BlenderProc recognises (`--temp-dir`, `--custom-blender-path`, …) is otherwise
consumed before the script sees it. Paths in the configs are resolved against the
project root, so commands work from any directory.

---

## How a scene is made

One scene = one Blender process = one random world photographed from 12 viewpoints.
The scene index is also the random seed: the same index always rebuilds the same scene,
which is what lets many machines render disjoint index ranges with no coordination.

`sdlib/pipeline.py` runs these stages in this order:

| stage | what happens | code |
|---|---|---|
| environment | pick a world type: a **procedural** patch of ground (80%) or a corner of a real `.blend` **scene** (20%) | `procedural.py`, `scenes.py` |
| lighting | random HDRI (sky + light), random rotation and strength; street lights at night | `environment.py` |
| litter | draw 3–9 models (class-balanced), scale to real size, maybe crush/bend/flatten, maybe add dirt; drop them with rigid-body physics | `assets.py`, `realism.py` |
| after physics | half-bury items in soft ground, scatter stones/weeds/branches around them, maybe a shadow-casting canopy | `procedural.py`, `scatter.py` |
| cameras | 12 viewpoints, each from one of three rigs, validated by ray casts (target visible, camera not inside geometry) | `cameras.py` |
| after cameras | grow 3D grass, kept clear of the litter and of where the robot stands | `procedural.py` |
| render | Cycles, OptiX denoising; colour + depth + segmentation | `capture.py` |
| write | drop barely-visible annotations, write images, masks, depth, COCO, metadata | `capture.py` |

### Procedural worlds (`sdlib/procedural.py`)

The problem they solve: a camera 10 cm above a flat textured plane sees one texture
repeating to the horizon and nothing else. So each world is assembled from parts:

* **Layout** — parallel strips of ground, each a solid box with its own PBR material:
  `building | verge | sidewalk | curb | road | curb | sidewalk | building` for urban
  biomes, `lawn | path | lawn` for parks, a single surface for forest/gravel/plaza.
  The strips are also the physics floor; the gutter along the curb is a spawn zone.
* **Materials** — chosen by keyword from ~800 photo-scanned ground textures (asphalt,
  paving, cobbles, grass, soil, gravel, leaves, …), mapped at real-world tile size,
  with a low-frequency colour variation so a 2 m tile does not visibly repeat.
* **Vegetation** — grass clumps instanced by geometry nodes (tens of thousands per
  scene for the memory of one), mowed / normal / long.
* **Clutter** — stones, weeds, dandelions, branches, roots near the litter. Never
  annotated: these are the hard negatives that teach "a stone is not a bottle".
* **Props** — bins, hydrants, lamps, benches, barriers, utility boxes, tyres, trees.
* **Facades** — textured walls that fill the horizon in urban biomes.

### Real scenes (`sdlib/scenes.py`)

Downloaded `.blend` environments (`scenes/*.blend`). `prepare_scenes.py` finds flat,
upward-facing ground automatically (plus any hand-made `spawn_*` planes) and records
which objects lie within 25 m of each patch, in `<scene>.blend.zones.json`. The
generator then loads only those objects — a 1.4 GB forest scene that needed 13 GB of
RAM loaded whole now renders comfortably. Run it again after editing a scene:

```bash
.venv/bin/blenderproc run prepare_scenes.py -- --config config.yaml
```

### Cameras (`sdlib/cameras.py`)

| rig | share | height | tilt | horizontal FOV | notes |
|---|---|---|---|---|---|
| `robot` | 40% | 6–16 cm above the ground it stands on | 3° up … 28° down | 65–105° | target 0.25–4 m away (log-uniform), anywhere across the frame |
| `arm` | 30% | 15–60 cm | 45–90° down | 60–90° | aimed at one item, any roll |
| `general` | 30% | person height | 15–75° | 20–85 mm lens | the original "photographer" sampler |

6% of frames are deliberately aimed away from all litter (empty hard negatives).
When the real robot's cameras are known, set their FOV ranges to match.

---

## Repository layout

```
generate.py            render one scene (procedural or real-scene world)
generate_scene.py      same, forced to real scenes
run_batch.py           render many scenes, one process each, then merge the COCO files
prepare_scenes.py      find spawn patches + crop lists in scenes/*.blend
fetch_assets.py        download ground materials, HDRIs, plants, props (Poly Haven, ambientCG)
fetch_objaverse.py     download litter models from Objaverse into catalog/objaverse.yaml
catalog_sheet.py       render a thumbnail of every catalogue model + flag suspects
qa_report.py           contact sheets (RGB | masks | depth) and statistics of a dataset
benchmark_report.py    timing / memory / cost numbers and plots from a render run
ship_dataset.py        pack finished scenes into tar shards and upload to Google Drive
export_yolo.py         convert a dataset into Ultralytics YOLO-seg format
inspect_blend.py       print sizes of catalogue models or the objects of a scene
usdz_to_blend.py       convert .usdz/.usd models into catalogue-ready .blend files

config.yaml            every randomisation range of the dataset
asset_sources.yaml     what fetch_assets.py downloads
assets.yaml            class list + hand-curated catalogue rules for resources/objects
catalog/               more catalogue files: polyhaven.yaml, local.yaml, objaverse.yaml

sdlib/
  pipeline.py          per-scene orchestration (stages above)
  procedural.py        procedural worlds: layouts, biomes, props, clutter, grass
  scenes.py            real .blend scenes: auto spawn zones, cropping
  library.py           asset index, PBR material builder, Poly Haven model variants
  scatter.py           geometry-nodes instancing and discrete placement
  realism.py           deformation, grime, burial, shadow canopy
  cameras.py           robot / arm / general rigs, ray-cast validation
  assets.py            litter catalogue, loading, size normalisation
  capture.py           renderer settings, annotation filtering, all writers
  environment.py       HDRI, lights, spawn zones, colliders, drop physics
  cli.py, config.py    start-up and config loading

brev/                  render-farm scripts (see "Rendering at scale")
docs/test_report/      LaTeX report of the first pipeline test
resources/             hand-collected models and HDRIs (git LFS)
scenes/                real .blend environments (git LFS) + their .zones.json
assets/                downloaded by fetch_*.py - not in git
output/                rendered datasets - not in git
```

---

## Configuration

`config.yaml` is the single source of truth; nothing random is hard-coded. The
sections in pipeline order:

| section | controls |
|---|---|
| top level | output folder, scenes, views per scene, resolution (960×720), samples (64), denoiser |
| `environments` | procedural vs real-scene weight |
| `procedural` | biome weights, ground extent, spawn area, props/clutter/grass density, canopy, how many model files per scene |
| `scene` | which `.blend` scenes, crop radius, automatic spawn-zone search |
| `hdri`, `lights` | lighting |
| `objects` | catalogue files, model folders, **class weights**, items per scene |
| `realism` | deformation and grime probabilities and strengths |
| `physics` | drop simulation |
| `camera` | rig shares and every rig's ranges |
| `annotations` | minimum visible pixels, empty-frame policy |

For a quick look at a change, copy the config and lower `resolution`,
`render_samples` and `views_per_scene`.

---

## Assets: where every model and texture comes from

| what | source | licence | how it gets here |
|---|---|---|---|
| ground materials (~240 + ~600) | Poly Haven, ambientCG | CC0 | `fetch_assets.py` |
| facade materials (~250) | Poly Haven, ambientCG | CC0 | `fetch_assets.py` |
| HDRIs (~680) | Poly Haven | CC0 | `fetch_assets.py` |
| grass, plants, stones, props, trees (~80 models) | Poly Haven | CC0 | `fetch_assets.py` |
| litter, single items (14) | Poly Haven | CC0 | `fetch_assets.py` → `catalog/polyhaven.yaml` |
| litter (~800) | Objaverse (Sketchfab uploads) | CC-BY 4.0 / CC0 | `fetch_objaverse.py` → `catalog/objaverse.yaml` |
| litter (140 + 17) | earlier hand collection (`resources/objects`, BlenderKit) | mixed, see below | git LFS / `~/blenderkit_data` |
| real scenes (5) | BlenderKit + own | per asset | git LFS |

`assets/index.json` lists which downloaded ids belong to which group;
`assets/manifest.jsonl` records source, id, licence and authors of every downloaded
asset. `asset_sources.yaml` decides what is fetched and has a small `dev` profile.

Some Poly Haven `.blend` files are saved by a newer Blender than 4.2 and cannot be
opened; litter models therefore also get a glTF copy, scenery models are skipped.

---

## The litter catalogue

Ten classes, chosen to line up with the supercategories of the real-photo dataset
[TACO](http://tacodataset.org) so a model trained here can be validated on it:

| id | class | examples |
|---|---|---|
| 1 | `bottle_plastic` | PET bottles, detergent bottles, jugs |
| 2 | `bottle_glass` | beer/wine bottles, jars |
| 3 | `can` | drink cans, tins, aerosols |
| 4 | `paper` | paper, cardboard, cartons, paper bags, packs |
| 5 | `cup` | disposable cups |
| 6 | `plastic_film` | bags, wrappers, chip bags, sachets, foil |
| 7 | `cigarette` | butts |
| 8 | `container` | tubs, takeaway boxes, styrofoam |
| 9 | `small_item` | caps, lids, straws, pens, matches |
| 10 | `other` | everything else people drop |

Catalogue files (`objects.catalogs`; the first defines the classes):

* `assets.yaml` — hand-curated rules for `resources/objects` and BlenderKit models:
  `match:` is a substring of the model path, `/<name>/` matches exactly one model;
* `catalog/polyhaven.yaml`, `catalog/local.yaml` — explicit `glob:` / `path:` entries;
* `catalog/objaverse.yaml` — generated; reject bad models by adding their uid to
  `catalog/objaverse_rejects.txt` and rerunning `fetch_objaverse.py`.

Every rule gives `longest_m`, the item's real longest dimension. Models arrive in mm,
cm, m or arbitrary units; each is rescaled on load so its longest axis matches, then
jittered a few percent. Scale is therefore a usable cue for the detector.

Classes are drawn by `objects.class_weights` first and a model within the class
second, so the class mix does not follow whichever class has the most downloads.

**Always look at new models** before a big render:
`blenderproc run catalog_sheet.py -- --source objaverse --out qa/objaverse`. The
sheet shows every model next to a 10 cm red cube, with its class and size, and frames
in red the ones with stray geometry or missing textures (`suspects.json`).

`resources/objects_needs_split/` holds 46 files with several items each (a set of
plates, three bottles). They are kept out of the catalogue on purpose — joined into
one object they would get one box spanning every item; split them in Blender first.

---

## Output format

```
output/<dataset>/
  scene_00042/
    images/000000.png        RGB
    depth/000000.png         uint16 depth along the optical axis in mm; 0 = sky / > 65.5 m
    instance/000000.png      uint16; 0 = not litter, k = k-th litter object (stable across frames)
    semantic/000000.png      uint8 class id per pixel (0 = background)
    coco_annotations.json    COCO instances with RLE masks
    scene_meta.json          world, biome, materials, HDRI, every object (+ licence),
                             every camera (rig, height, tilt, FOV, K, pose), timings
  coco_annotations.json      all scenes merged (run_batch.py --merge-only)
  config.yaml, run_info.json the config and machine of the run
  qa/                        contact sheets and stats
```

Depth and masks are rendered, i.e. exact. Sensor-like depth noise (stereo holes,
noise growing with distance²) belongs in training-time augmentation, once the robot's
depth camera is known.

---

## Checking the data

```bash
.venv/bin/python qa_report.py output/robot_v1 --count 48           # random frames
.venv/bin/python qa_report.py output/robot_v1 --rig robot          # only the robot camera
```

Writes `qa/sheet_XX.jpg` (rows of RGB | masks with boxes and class names | depth) and
`qa/stats.json` (frames per rig and biome, annotations per class, empty frames, box
sizes). Read the sheets before trusting a new config — every systematic bug so far
(black frames, mismatched masks, magenta missing textures, stray geometry) was
obvious at a glance and invisible in the numbers.

---

## Rendering at scale on NVIDIA Brev

The unit of work is the scene index (= seed), so splitting a run is arithmetic: give
every machine and every GPU a disjoint index range. `brev/` automates it.

```bash
brev login                                    # opens a browser
brev search gpu --gpu-name RTX --sort price   # pick an instance type
brev create trash-1 --type massedcompute_RTX6000Adax4
brev refresh                                  # adds `ssh trash-1` to ~/.ssh/config

brev/push.sh trash-1                          # upload code + models, install, fetch assets
brev/launch.sh --scenes 2000 trash-1 trash-2  # split scenes across instances, start in tmux
ssh trash-1 -t tmux attach -t render          # watch (detach: Ctrl-b d)
brev/pull.sh trash-1                          # optional: rsync results here instead of Drive
brev delete trash-1                           # stop paying
```

| script | runs | does |
|---|---|---|
| `push.sh <inst>…` | locally | rsyncs code, scenes and hand-collected models (~4.5 GB) and the rclone config, then runs `setup.sh` |
| `setup.sh` | instance | system libraries, venv, rclone, Blender; downloads the full asset set and Objaverse models on the data-centre link; prepares missing scene zones |
| `launch.sh [opts] <inst>…` | locally | splits `[--start, --start+--scenes)` across instances, starts `render.sh` in tmux |
| `render.sh [opts]` | instance | one `run_batch.py --resume` per GPU; ships finished shards to Drive every 10 min; at the end merges, writes QA, ships the rest; logs GPU/RAM utilisation |
| `pull.sh <inst>…` | locally | rsyncs `scene_*` back and merges |

`render.sh` resumes: rerun the same `launch.sh` after an interruption and only the
missing scenes are rendered. Measured throughput and cost per GPU count are in the
[test report](docs/test_report/report.pdf); `benchmark_report.py` recomputes them
from any run.

---

## Google Drive

Rendered data goes to Google Drive through [rclone](https://rclone.org) (remote name
`gdrive`), into `gdrive:syntheticData/datasets/<dataset>/`:

```
shards/scenes_00000-00099.tar   100 scenes per tar (Drive is slow with many small files)
meta/<host>/                    merged COCO, config, run info, QA sheets of each machine
```

Get a dataset back:

```bash
rclone copy gdrive:syntheticData/datasets/robot_v1/shards ./robot_v1_shards -P
mkdir robot_v1 && for t in robot_v1_shards/*.tar; do tar xf "$t" -C robot_v1; done
.venv/bin/python run_batch.py --merge-only --config <config with output_dir: robot_v1>
```

Set-up: `rclone config create gdrive drive scope=drive` on your machine; `push.sh`
copies `~/.config/rclone/rclone.conf` to the instances. **Use your own Google OAuth
client id** (`client_id`/`client_secret` in that config): rclone's shared one is being
retired in 2026 and is already rate-limited (`403 Quota exceeded`) — see
<https://rclone.org/drive/#making-your-own-client-id>.

---

## Training data export

```bash
.venv/bin/python export_yolo.py output/robot_v1 --out exports/robot_v1_yolo
yolo segment train data=exports/robot_v1_yolo/data.yaml model=yolo11n-seg.pt imgsz=640
```

Polygons are traced from the rendered instance masks; the train/val split is by
scene, never by frame. `--rig robot --rig arm` exports only the robot's own cameras.

Suggested training recipe (from the literature on synthetic-to-real transfer):
pre-train on the synthetic set, fine-tune on a few hundred labelled real images from
the robot's camera, and measure only on real images.

---

## Known issues and next steps

* **Validate on real images.** Collect a few hundred photos from the robot's camera
  height on real sidewalks/lawns and label them (SAM can pre-label); add TACO for
  extra variety. Synthetic validation scores say little about real performance.
* **Match the real cameras.** Set `camera.robot.hfov_deg` / `camera.arm.hfov_deg`,
  resolution and (if stereo depth is used) a depth-noise augmentation to the actual
  sensors.
* **Catalogue quality.** Review the Objaverse sheets and fill
  `catalog/objaverse_rejects.txt`; small classes (`cup`, `cigarette`, `container`)
  still need more models — Sketchfab and BlenderKit searches are the next sources.
* **`scenes/warta.blend` is saved by Blender 5** and skipped; re-save it from Blender
  4.2 (or replace it with `warta.blend1`). Same for `resources/assets/`.
* **Grass hides litter**: on lawns ~half the frames lose their target to the grass
  grown after the camera check and are dropped. Fine for realism, wasteful for
  render time — a post-grass visibility check would recover them.
* `resources/objects_needs_split/` — 46 multi-item models wait to be split.
* Two FBX cans from `resources/objects_extra` have no texture file and render plain
  white.

---

## Implementation notes (gotchas)

* **One Blender process per scene.** Blender never frees appended data; looping
  scenes in one process grows memory until it dies.
* **Segmentation is enabled right before rendering**: `enable_segmentation_output`
  stamps a `pass_index` onto objects at call time; anything created later is
  silently background.
* **BlenderProc's physics invalidates Python references to collections** (the
  collection survives). `ModelLibrary` therefore looks its collection up by name.
* **BlenderProc's `load_blend` purges new orphan data blocks**, which includes a
  collection hanging off the scene root; library collections carry a fake user.
* **No BlenderProc BVH trees.** They copy every triangle into Python lists (4–7 GB
  with a lawn of weeds). Camera checks use Blender's own `scene.ray_cast`.
* **Load few model files per scene.** Loading every tree for every scene cost
  minutes and up to 13 GB; `procedural.models_per_scene` caps it, unused LODs are
  purged before the simulation, and props use their lowest LOD.
* **Focal length is keyframed per frame** — `add_camera_pose` only keyframes the
  pose, so a plain `lens` assignment would apply the last value to every frame.
* **A scene directory is cleared before writing** — the COCO writer appends, the
  dense maps overwrite; a rerun into a used folder would pair images with the wrong
  masks.
* **Missing textures are unplugged** (`assets.fix_missing_images`) instead of
  rendering Blender's magenta, which a detector would happily learn as "litter".
* **Blender 4.2 cannot read files saved by Blender 5** ("not a blend file"); this
  hits some Poly Haven downloads, `warta.blend` and `resources/assets/`.

---

## Licences

* Poly Haven and ambientCG assets: CC0.
* Objaverse models: kept only if CC-BY 4.0 or CC0 (no NC, no SA). **CC-BY requires
  attribution** wherever the dataset is published: author and URL of every model are
  in `catalog/objaverse.yaml` and per object in each `scene_meta.json`.
* BlenderKit models: royalty-free or CC0 per asset; check before publishing.
* If the trained model ships in a commercial product, note that Ultralytics YOLO is
  AGPL-3.0 (an enterprise licence or an Apache-licensed model such as RF-DETR avoids
  that).
