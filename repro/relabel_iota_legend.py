#!/usr/bin/env python3
"""
Raster surgery on bayes_single_estimation_example.png: replace only the legend
math symbols  m -> iota  and  \\widehat{m} -> \\widehat{\\iota}  in panel 1's
legend ("true $m$" / "Bayes $\\widehat{m}$"). Every curve/axis pixel is kept.

Only the two small symbol glyphs (top-right, above the panel border and clear of
the curves) are whited out and replaced with matplotlib-rendered iota glyphs.
"""
from pathlib import Path
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PAPER = Path("/Users/felipemoret/Desktop/extended_first_abstract_RLMM")
OUT = Path("/private/tmp/claude-501/-Users-felipemoret-Desktop-extended-first-abstract-RLMM/"
           "a518939f-1993-4689-a288-5dac028de335/scratchpad")
FN = "bayes_single_estimation_example.png"


def render_glyph(text, target_h):
    fig = plt.figure(figsize=(3, 3), dpi=300)
    fig.patch.set_alpha(0.0)
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=60, color="black")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    al = np.asarray(im)[:, :, 3]
    ys, xs = np.where(al > 10)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = im.size
    s = target_h / h
    return im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)


img = Image.open(PAPER / FN).convert("RGB")

def paste_centered(box, cx, cy, glyph):
    """white out box, paste glyph centered at (cx,cy)."""
    img.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255)),
              (box[0], box[1]))
    img.paste(glyph, (cx - glyph.size[0] // 2, cy - glyph.size[1] // 2), glyph)

# --- line 1: "m" -> iota. original m spans ~925-943 (center ~935), rows 30-42.
paste_centered((921, 27, 948, 44), cx=935, cy=36, glyph=render_glyph(r"$\iota$", target_h=16))

# --- line 2: "\widehat{m}" -> "\widehat{\iota}". m-hat ~937-963 (center ~950), rows 48-72.
paste_centered((933, 46, 968, 74), cx=950, cy=60, glyph=render_glyph(r"$\widehat{\iota}$", target_h=24))

out = OUT / FN
img.save(out)
print(f"wrote {out}")
