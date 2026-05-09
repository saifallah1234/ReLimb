"""visualize_stgcn_graph.py

Visualize the ST-GCN joint graph used by `src.models.model_stgcn.GaitSTGCN`.

What this shows
---------------
- The *compact* 13-joint index space that actually goes into the model
  (because the dataset packs `GAIT_LANDMARK_INDICES` densely).
- The edges (connections) used to build the adjacency matrix.

Outputs
-------
- A simple node+edge plot (matplotlib)
- Optionally, writes a PNG to disk.

Usage
-----
    python -m src.utils.visualize_stgcn_graph

Optional:
    python -m src.utils.visualize_stgcn_graph --save data/output/stgcn_graph.png

Notes
-----
This is intended as a *connectivity sanity check*, not a motion visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _connections_compact_from_model() -> tuple[list[tuple[int, int]], list[int]]:
    """Recreate the exact connection list used inside `GaitSTGCN`.

    Returns
    -------
    connections_compact:
        List of edges in compact [0..12] joint index space.
    gait_landmarks:
        The MediaPipe landmark indices in compact order.
    """

    from src.models.model_lstm import GAIT_LANDMARK_INDICES

    gait_landmarks = list(GAIT_LANDMARK_INDICES)
    mp_to_compact = {mp_idx: i for i, mp_idx in enumerate(gait_landmarks)}

    raw_connections_mp = [
        (11, 23),
        (12, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
        (27, 29),
        (29, 31),
        (28, 30),
        (30, 32),
        (23, 24),
    ]

    connections_compact: list[tuple[int, int]] = []
    for a, b in raw_connections_mp:
        if a in mp_to_compact and b in mp_to_compact:
            connections_compact.append((mp_to_compact[a], mp_to_compact[b]))

    return connections_compact, gait_landmarks


def _nice_name_for_mp(mp_idx: int) -> str:
    names = {
        0: "nose",
        11: "L-shoulder",
        12: "R-shoulder",
        23: "L-hip",
        24: "R-hip",
        25: "L-knee",
        26: "R-knee",
        27: "L-ankle",
        28: "R-ankle",
        29: "L-heel",
        30: "R-heel",
        31: "L-foot",
        32: "R-foot",
    }
    return names.get(mp_idx, f"mp{mp_idx}")


def _layout_positions(gait_landmarks: list[int]) -> np.ndarray:
    """A simple fixed 2D layout that looks like a stick figure.

    Returns array of shape (V, 2).
    """

    # Compact index -> MediaPipe index
    compact_to_mp = {i: mp for i, mp in enumerate(gait_landmarks)}

    # baseline coordinates in a canonical body layout
    mp_xy = {
        0: (0.0, 2.2),
        11: (-0.6, 1.6),
        12: (0.6, 1.6),
        23: (-0.4, 1.0),
        24: (0.4, 1.0),
        25: (-0.5, 0.4),
        26: (0.5, 0.4),
        27: (-0.55, -0.2),
        28: (0.55, -0.2),
        29: (-0.65, -0.6),
        30: (0.65, -0.6),
        31: (-0.45, -0.95),
        32: (0.45, -0.95),
    }

    V = len(gait_landmarks)
    xy = np.zeros((V, 2), dtype=np.float32)
    for i in range(V):
        mp = compact_to_mp[i]
        xy[i] = mp_xy.get(mp, (0.0, 0.0))

    return xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    connections, gait_landmarks = _connections_compact_from_model()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for this visualization. "
            "Install it with: pip install matplotlib"
        ) from e

    xy = _layout_positions(gait_landmarks)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    # edges
    for a, b in connections:
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color="black",
            linewidth=2,
            alpha=0.8,
            zorder=1,
        )

    # nodes
    ax.scatter(xy[:, 0], xy[:, 1], s=220, color="#4C78A8", zorder=2)

    # labels: compact index + MP name
    for i, mp_idx in enumerate(gait_landmarks):
        ax.text(
            xy[i, 0],
            xy[i, 1] + 0.06,
            f"{i}: {_nice_name_for_mp(mp_idx)}\n(mp={mp_idx})",
            ha="center",
            va="bottom",
            fontsize=9,
            zorder=3,
        )

    ax.set_title("ST-GCN Graph (compact 13-joint order used by the model)")
    ax.set_aspect("equal")
    ax.axis("off")

    # show mapping summary in console
    print("Compact joint order used by dataset/model:")
    for i, mp_idx in enumerate(gait_landmarks):
        print(f"  {i:2d} -> mp {mp_idx:2d} ({_nice_name_for_mp(mp_idx)})")

    print("\nEdges (compact indices):")
    for a, b in connections:
        print(f"  {a} -- {b}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"\nSaved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
