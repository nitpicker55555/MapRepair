# Interactive viz — TextWorld + MANGO graph viewer

A standalone, single-page web visualization showing how LLM-MapRepair
operates on real game data. Renders three states side-by-side for each
game:

  ① **Ground truth** — the navigation graph as the game world actually
     defines it.
  ② **With noise** (TextWorld) / **LLM-built** (MANGO) — the same
     graph after exp16-style mango_like noise has been injected, or
     the actual LLM-emitted map from exp14. Conflicts surface and bad
     edges are highlighted by colour.
  ③ **After MapRepair** — heuristic_remove repair applied; removed
     edges shown as faded dashed ghosts.

## What's in the dataset

| Dataset | Games | Source |
|---------|-------|--------|
| TextWorld | tw_00 — tw_09 (5-11 rooms each) | `results/exp23/games/` |
| MANGO | zork1, cutthroat, advent, night, zork2, ludicorp | `data_fixed/` (GT) + `results/exp14/gpt-4.1/` (LLM-built) |

The cutthroat case is the F3 (hierarchical-name) showcase — 5 different
"back alley (...)" rooms whose LLM-built version produces 17 non-correct
edges; the viewer makes the failure pattern obvious. zork2 / advent
show F6 (vertical inverse-direction) and F1 (magic-words) respectively.

## How to run

The viewer is pure HTML+JS but uses `fetch()` to load JSON, so you
need any local HTTP server:

```bash
cd viz
python3 -m http.server 8765
# then visit http://localhost:8765/
```

## Deep links

The URL hash format is `#<dataset>/<game>[/<state>]`. Examples:

```
#textworld/tw_03                       (defaults to repaired view)
#textworld/tw_03/noised                (show noise injections + conflicts)
#mango/cutthroat/llm_built             (show the F3 hierarchical-name failure)
#mango/zork1/repaired                  (show clean repair on a small map)
```

Three states are available:

- `gt`        ground truth
- `noised`    (TextWorld) after noise injection
- `llm_built` (MANGO) actual LLM-emitted edges
- `repaired`  after heuristic_remove

## Re-exporting data

If you change the noise regime, the model, or the included games,
re-run the exporter:

```bash
PYTHONPATH=src python3 -m viz.export_data
```

This regenerates `viz/data/<dataset>/<game>.json` and
`viz/data/games_index.json`. The viewer needs no rebuild — refresh
the page.

## What each node and edge colour means

Nodes:

- **Solid blue border**: real GT room.
- **Purple border**: hallucinated room (LLM invented; not in GT).

Edges:

- **Grey** — GT edge (also present in the LLM-emitted map).
- **Green** — correctly predicted by the LLM.
- **Red** — spurious (LLM emitted an edge with no GT counterpart).
- **Orange** — same room-pair but wrong direction.
- **Olive** — same (src, direction) but wrong destination (F3).
- **Purple** — edge to / from a hallucinated room (F4).
- **Dashed grey ghost** — edge removed by MapRepair (only shown in
  the "After MapRepair" view).
- **Blue** — edge whose direction was modified by MapRepair.

## Files

```
viz/
├── README.md           — this file
├── index.html          — the page
├── viewer.js           — cytoscape rendering + UI
├── style.css           — styling
├── export_data.py      — Python data exporter (run once)
└── data/
    ├── games_index.json
    ├── textworld/<game>.json
    └── mango/<game>.json
```

The viewer relies on these CDN libraries (no build step required):

- cytoscape.js 3.30.x
- cytoscape-cola (force-directed layout)
- cytoscape-dagre (hierarchical layout)
- IBM Plex Sans + Mono fonts

## License

Same as the rest of the LLM-MapRepair repository.
