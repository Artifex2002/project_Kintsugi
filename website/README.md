# Research showcase

The complete, buildless website source lives in [`../docs/`](../docs/):

- `index.html`: the research story and experiment explorer.
- `styles.css`: responsive ivory, charcoal, and gold visual design.
- `app.js`: method tabs, recorded optimization replay, image selection, and action comparison.
- `assets/`: conceptual hero artwork and actual saved LIBERO observations.
- `data/run-snapshot.json`: a dated, sanitized snapshot of completed evaluations.
- `.nojekyll`: serves the static files directly with GitHub Pages.

The website uses ordinary relative asset paths and works at the repository's
GitHub Pages subpath. No Node installation, model inference, account system,
API key, or build step is needed to serve it.

## GitHub Pages

In repository **Settings → Pages**, select **Deploy from a branch**, then
**main** and **/docs**, and save. Future pushes that change `docs/` are published
automatically by GitHub Pages.

Expected public address: <https://artifex2002.github.io/project_Kintsugi/>.

## Local preview

From the repository root:

```sh
python3 -m http.server 4173 --directory docs
```

Open <http://localhost:4173/>. The initial explorer view is an intentional empty
state. Select **Recorded snapshot** to inspect the saved partial run.

## Refresh experiment data

```sh
python3 website/sync-run.py --run /path/to/experiment/run
```

This imports only completed evaluation records, copies their observation
images, and excludes machine paths, credentials, environment details, raw
console logs, and private model files. Review and commit the resulting changes
under `docs/` to update the published snapshot. The hosted website does not
connect to a running local experiment.

The source run directory is expected to contain `metadata.json`, `images.json`,
`evaluations.jsonl`, and image files referenced by the image manifest.

## Research interpretation

Scores are negative hybrid losses combining arm cosine distance and weighted
action error. Higher values mean closer target-action proximity; they do not
measure physical task failure. The current experiment predicts one action on
saved frames with a fixed robot state. A partial run, a completed optimization,
and a paired simulator rollout are distinct artifacts. The site labels pending
results and the planned robot-video slot accordingly.

## Artwork

`docs/assets/kintsugi-arm.png` is conceptual artwork created using the built-in
image-generation tool. It is labeled on the page and is not experimental
evidence. The production prompt is in [`art-direction.md`](art-direction.md).
The remaining scene images are saved simulation observations from this project.
