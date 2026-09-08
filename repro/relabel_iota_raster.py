#!/usr/bin/env python3
"""
Raster label surgery: replace ONLY the flow-bias axis title "Bayes directional
bias m_hat" -> "Bayes directional bias $\\widehat{\\iota}$" on two heatmaps,
WITHOUT touching a single data/colormap/colorbar pixel.

Why raster surgery (not re-render): the paper heatmaps were produced by the
trainer DeepSarsaQRunner_REGIME.py at a config we cannot reproduce byte-exactly
(different figsize/base_state than the standalone viz), so any re-render would
change the data. Overwriting just the title text in the bottom/left MARGIN is
the only method that guarantees the plotted values are identical.

The replacement label is rendered by matplotlib itself (same DejaVu Sans font +
mathtext engine as the original), scaled to the original title's pixel size, and
pasted centered in the whited-out title box.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PAPER = Path("/Users/felipemoret/Desktop/extended_first_abstract_RLMM")
OUT = Path("/private/tmp/claude-501/-Users-felipemoret-Desktop-extended-first-abstract-RLMM/"
           "a518939f-1993-4689-a288-5dac028de335/scratchpad")
OUT.mkdir(parents=True, exist_ok=True)
NEW = r"Bayes directional bias $\widehat{\iota}$"


def render_text(text, rotation, target_px, axis):
    """Render `text` (mathtext ok) on transparent bg, tight-cropped, then scale so
    that its extent along `axis` ('h'=height, 'w'=width) equals target_px."""
    fig = plt.figure(figsize=(12, 3), dpi=300)
    fig.patch.set_alpha(0.0)
    t = fig.text(0.5, 0.5, text, ha="center", va="center", rotation=rotation,
                 fontsize=40, color="black")
    fig.canvas.draw()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    # tight crop to non-transparent
    alpha = np.asarray(im)[:, :, 3]
    ys, xs = np.where(alpha > 10)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = im.size
    scale = target_px / (h if axis == "h" else w)
    im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    return im


def surgery(fname, box, rotation, axis, pad=6):
    """box = (col0,row0,col1,row1) of the ORIGINAL title text."""
    c0, r0, c1, r1 = box
    img = Image.open(PAPER / fname).convert("RGB")
    # white-out the title box (+pad), staying in the margin
    wb = Image.new("RGB", (c1 - c0 + 2 * pad, r1 - r0 + 2 * pad), (255, 255, 255))
    img.paste(wb, (c0 - pad, r0 - pad))
    # target size = original title extent along its long axis
    target = (r1 - r0) if axis == "h" else (c1 - c0)   # x-title: match height; y-title(rot): match width
    lbl = render_text(NEW, rotation, target, axis)
    cx = (c0 + c1) // 2
    cy = (r0 + r1) // 2
    img.paste(lbl, (cx - lbl.size[0] // 2, cy - lbl.size[1] // 2), lbl)
    out = OUT / fname
    img.save(out)
    print(f"  wrote {out}  (label box {box}, new-label size {lbl.size})")


# p_buy_mhat_vs_inv: X-title (horizontal) at rows 1246..1300, cols 458..1243
surgery("p_buy_mhat_vs_inv.png", (458, 1246, 1243, 1300), rotation=0, axis="h")
# bayes_m_vs_spread: Y-title (rotated 90) at cols 65..119, rows 232..1017
surgery("bayes_m_vs_spread.png", (65, 232, 119, 1017), rotation=90, axis="w")
print("done -> scratch (verify before copying over the paper figures)")
