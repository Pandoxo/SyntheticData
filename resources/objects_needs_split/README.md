# Models that hold more than one item

These 45 models each contain several *separate* items — five peppers, three bottles, a
pile of knives. They are converted here **unjoined**: every part is still its own mesh
object, centred as a group, textures packed.

They are kept out of `resources/objects/` on purpose. `sdlib/assets.py` joins every
mesh in a file into one object on load, so each of these would become a single COCO box
spanning every item, and a single rigid body that falls as one lump.

Do **not** add this directory to `objects.model_dirs` — the catalogue globs
`**/*.blend` recursively and would pick these up as-is.

## Fixing one

1. Open the `.blend`, `Object > Select All`, then split what belongs apart.
2. Save each real item as `resources/objects/<name>/<name>.blend` with one mesh in it.
3. Add a `match: /<name>/` rule for each new name in `assets.yaml`, then check with
   `.venv/bin/blenderproc run inspect_blend.py -- --assets`.

## Contents

Every model below was checked by eye from a rendered turnaround, not just by measuring
part separation — a stack of touching books reads as one blob to an automatic count.

| model | what is in it |
|---|---|
| `5_bell_peppers_capsicum_low_poly_vegetable` | 5 peppers, touching |
| `alchemy_potions_pack_mid_poly` | 6 potion bottles |
| `book_stack` | a stack of ~8 books |
| `bottles_download_like_please` | several bottles, far from the origin |
| `bread_group_3_photoscaned_breads` | 3 loaves |
| `business_card_holder` | holder plus loose cards |
| `can_of_beans` | 2 cans |
| `cans_dirty_and_crumpled_3_piece_set` | 3 cans |
| `cardboard_boxes_pack_tlbjdbova_high` | boxes nested in the same place |
| `cardboard_boxes_pack_uewlfg2va_high` | boxes nested in the same place |
| `cc0_cutlery` | fork, knife, spoon |
| `cc0_paper` | 3 sheets |
| `cc0_pencil_holder` | holder plus a dozen pencils |
| `cc0_planks_broken` | 5 planks |
| `ceramic_plate_set` | 4 plates |
| `chips_packet` | 4 packets |
| `clothespins` | 3 pins |
| `dirty_plastic_bottles_trash_3_piece_set` | 3 bottles |
| `fbx(1)` | 4 food containers |
| `fbx(3)` | 3 popcorn boxes |
| `fbx(4)` | a saw and several blades |
| `free_practice_models_3d_modeling_for_beginners` | glass, traffic cone, headphones |
| `game_ready_free_chocolates` | 6 bars |
| `game_ready_free_paper_goods` | coffee cup and a plate |
| `hand_soap_bottle` | 3 dispensers |
| `knife_collection` | many knives in a pile |
| `low_poly_cup_silverware_set` | cup plus cutlery |
| `low_poly_soap_collection` | many soap bars |
| `mango_cut_half_mango_fruit` | whole fruit plus a cut half |
| `metal_cutlery` | fork, spoon, knife |
| `milks_and_some_juices` | 5 cartons |
| `movie_theater_props_hotdog_drink_popcorn` | hotdog, drink, popcorn |
| `nuts_and_bolts` | ~6 fasteners |
| `orange_cut_half_orange_fruit` | whole fruit plus a cut half |
| `paper_debris` | 5 scraps |
| `poly_world_toon_projectiles` | 9 projectiles |
| `rotten_dirty_organic_waste` | 3 pieces |
| `stylized_chocolate_bar` | 3 bars |
| `tableware` | bowl, plate, cup and cutlery |
| `three_sizes_of_hex_screws_3d_animation_models` | many screws |
| `tin_cans` | 5 cans |
| `water_bottles` | 2 bottles |
| `wine_bottle_low_to_high` | 3 bottles side by side |
| `wine_bottles` | 2 bottles |
| `wood_spoons` | 3 spoons |

`spin_v_pack` used to be listed here. It is one hand spinner despite the name, so it
was re-converted with `--asset` into `resources/objects/` and catalogued as a toy.

## Single items that are still not usable

These three are in `resources/objects/` but listed under `ignore:` in `assets.yaml`
rather than catalogued. They hold one item each, so they are not a splitting job:

- `giggle_pacifier`, `covid_19_oral_vaccine_inhaler_prototype` — the model is a large
  ground plane with a tiny prop on it. The plane is what would get annotated. Delete
  the plane in Blender and they become usable.
- `controller_3d_scan_test` — an unidentifiable scrap from a test scan.
