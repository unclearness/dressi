# Dataset auto-fetch at CMake configure time

## Goal

`data/` is `.gitignore`'d (the assets are large / license-encumbered), but a
fresh checkout had no way to obtain them — the example paths (`data/bunny`,
`--mesh=...Avocado...`) were dangling, and the working copy had accumulated a
1.2 GB `glTF-Sample-Models` repo zip + full extraction just to use one 8 MB
model. Make the datasets download and place themselves at build time.

## What was done

- Added `cmake/FetchAssets.cmake` (included from the top `CMakeLists.txt` when
  `DRESSI_BUILD_TESTS OR DRESSI_BUILD_EXAMPLES`). Gated by
  `option(DRESSI_FETCH_DATA ON)`. Each asset is guarded by a sentinel file so
  it is a no-op once present:
  - **bunny** — `file(DOWNLOAD)` of `kunzhou.net/tex-models/bunny.zip`
    (SHA256-pinned) → `file(ARCHIVE_EXTRACT)` into `data/bunny/` (the zip holds
    the 3 files at its root; example default dir).
  - **Avocado** — per the user's choice, a full `FetchContent` git clone of
    `KhronosGroup/glTF-Sample-Models` (main, shallow, ~1.2 GB, one-time), then
    `file(COPY)` of just `2.0/Avocado/glTF` into `data/Avocado/` so runtime
    paths stay uniform under `data/`. Guarded on `data/Avocado/glTF/Avocado.gltf`
    so the huge clone happens at most once per machine.
  - **suburban_garden_4k.exr** — `file(DOWNLOAD)` from Poly Haven
    (SHA256-pinned) into `data/`, for the upcoming PBS shading env map.
  - Downloads stage into `${CMAKE_BINARY_DIR}/_assets`, are hash-verified,
    then `COPY_FILE`'d into `data/` — an interrupted/corrupt download never
    lands behind the sentinel.
- Docs: README gained a "Datasets" table + `-DDRESSI_FETCH_DATA=OFF` note and
  the Avocado `--mesh=data/Avocado/glTF/Avocado.gltf` invocation; CLAUDE.md
  Build & test section notes the fetch and the new Avocado path.
- Populated `data/Avocado/glTF/` from the user's existing extraction so the
  first configure skips the giant clone.

## Pinned hashes / sizes (measured from the local copies)

- bunny.zip  SHA256 `a1fb80b8…6bc756f` (1.1 MB → 3 files, ~3 MB)
- suburban_garden_4k.exr  SHA256 `fe13daff…8f2171c`, 97,113,268 B
  (polyhaven HEAD Content-Length matches exactly)
- glTF-Sample-Models main tip at time of writing:
  `d7a3cc8e51d7c573771ae77a57f16b0662a905c6`

## Verification

- `file(DOWNLOAD)`+`EXPECTED_HASH`+`ARCHIVE_EXTRACT` of bunny.zip run in a
  throwaway dir: status 0, hash matched, 3 files extracted at root.
- Poly Haven EXR URL `HEAD` → 200, Content-Length == local size.
- Full `cmake --preset msvc` reconfigure with all sentinels present:
  Configuring/Generating done, no downloads, no errors.

## Open issues

- The redundant `data/glTF-Sample-Models-main.zip` (1.2 GB) and its
  `data/glTF-Sample-Models-main/` extraction are now obsolete (superseded by
  `data/Avocado/`). Left in place pending user confirmation to delete.
- glTF Avocado is pinned to branch `main`, not a commit. The repo is archived
  (stable), so `main` is effectively immutable; pin to the commit above if
  stricter reproducibility is wanted.
