"""
Convergent Evolution — two rhodopsins, side by side, spinning in place.

The story in one loop: bacteriorhodopsin (archaeal, type-1) and bovine
rhodopsin (animal, type-2) share the same 7-transmembrane retinal-binding
architecture and the same job — catching a photon — with no detectable
sequence homology between them. That is exactly what the detector finds.

No PyMOL, no conda, no Biopython. PDB files are plain text, so the parser
below is ~15 lines of stdlib. You need matplotlib, numpy, and ffmpeg.

    pip install matplotlib numpy

Run:
    python make_convergence_video.py

Writes convergence.mp4 (8s, 720p). Takes about a minute.

If your network blocks files.rcsb.org, download 1C3W.pdb and 1F88.pdb by
hand from rcsb.org, drop them next to this script, and it will use those.
"""

import os
import subprocess
import urllib.request

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Type-1 (microbial) and type-2 (animal) rhodopsins.
# Same fold, same function, unrelated sequences.
STRUCTURES = [
    ("1C3W", "A", "#19b8eb", (-16, 0, 0)),  # bacteriorhodopsin, cyan
    ("1F88", "A", "#6b2eff", ( 16, 0, 0)),  # bovine rhodopsin, indigo
]

N_FRAMES = 240              # 240 frames @ 30fps = 8s, one full revolution
FIGSIZE = (12, 5)           # 2.4:1 — crops cleanly to the 600x240 modal cover
FRAME_DIR = "_frames"
OUT = "convergence.mp4"


def fetch_pdb(pdb_id):
    """Grab the PDB file, or reuse a local copy if it's already here."""
    path = f"{pdb_id}.pdb"
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"  downloading {pdb_id}...")
        urllib.request.urlretrieve(url, path)
    return path


def parse_ca(path, chain):
    """Pull CA coordinates off one chain. PDB is fixed-column plain text."""
    coords = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            coords.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
    if not coords:
        raise ValueError(f"No CA atoms found in {path} chain {chain}")
    return np.array(coords)


def smooth(p, passes=2):
    """Chaikin subdivision, so the backbone reads as a tube not a polyline."""
    for _ in range(passes):
        out = [p[0]]
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            out.append(0.75 * a + 0.25 * b)
            out.append(0.25 * a + 0.75 * b)
        out.append(p[-1])
        p = np.array(out)
    return p


def upright(ca):
    """Rotate so the principal axis points along z — bundles stand upright
    regardless of how they happened to be deposited."""
    _, _, vt = np.linalg.svd(ca - ca.mean(0))
    axis = vt[0]
    z = np.array([0, 0, 1.0])
    v = np.cross(axis, z)
    s, c = np.linalg.norm(v), float(np.dot(axis, z))
    if s < 1e-6:
        return ca
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    return ca @ R.T


def rotate_z(p, deg):
    """Spin about the molecule's own axis — helices run roughly along z."""
    t = np.deg2rad(deg)
    m = np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t),  np.cos(t), 0],
        [0, 0, 1],
    ])
    return p @ m.T


def load():
    out = []
    for pdb_id, chain, colour, offset in STRUCTURES:
        ca = parse_ca(fetch_pdb(pdb_id), chain)
        ca = upright(ca - ca.mean(0))
        ca = ca - ca.mean(0)
        out.append({
            "coords": smooth(ca),
            "colour": colour,
            "offset": np.array(offset, dtype=float),
        })
        print(f"  {pdb_id}: {len(ca)} residues")
    return out


def draw_frame(ax, mols, angle):
    ax.clear()
    ax.set_facecolor("#0e0e0e")
    for m in mols:
        p = rotate_z(m["coords"], angle) + m["offset"]
        segs = np.stack([p[:-1], p[1:]], 1)
        ax.add_collection3d(
            Line3DCollection(segs, linewidths=4.0, colors=m["colour"], alpha=0.92)
        )
    ax.set_xlim(-34, 34)
    ax.set_ylim(-17, 17)
    ax.set_zlim(-28, 28)
    ax.set_axis_off()
    ax.view_init(elev=8, azim=0)
    try:
        ax.set_box_aspect((68, 34, 56))
    except Exception:
        pass


def main():
    print("Loading structures...")
    mols = load()

    os.makedirs(FRAME_DIR, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE, facecolor="#0e0e0e")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    print(f"Rendering {N_FRAMES} frames...")
    step = 360.0 / N_FRAMES
    for i in range(N_FRAMES):
        draw_frame(ax, mols, i * step)
        fig.savefig(
            os.path.join(FRAME_DIR, f"f{i:04d}.png"),
            dpi=90, facecolor="#0e0e0e",
        )
        if i % 40 == 0:
            print(f"  frame {i}/{N_FRAMES}")
    plt.close(fig)

    print("Encoding...")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", os.path.join(FRAME_DIR, "f%04d.png"),
        "-c:v", "libx264", "-crf", "32", "-preset", "veryslow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", "scale=720:-2",
        OUT,
    ], check=True)

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
