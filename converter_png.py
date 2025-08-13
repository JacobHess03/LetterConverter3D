#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
png2svg_bezier.py
Converte PNG bianco/nero in SVG con bordi LISCI (curve Bézier cubiche).

Uso:
  python png2svg_bezier.py input.png [output.svg]
Opzioni utili (vedi costanti default sotto):
  --threshold   Soglia binaria (0..255, default 128)
  --invert      Se il foreground è bianco su sfondo nero
  --blur        Kernel Gauss (3,5,7...), default 5, 0 per disattivare
  --step        Passo di ricampionamento lungo il contorno (px), default 2.0
  --smooth      Fattore di morbidezza Catmull–Rom→Bézier (0.1..1.0), default 1.0
  --min-area    Filtra piccoli rumori (px^2), default 10
  --scale       Scala coordinate SVG (float), default 1.0

Dipendenze:
  pip install opencv-python numpy
"""
import argparse
import os
import math
import numpy as np
import cv2

def convert_alpha_to_white(img):
    # Se l'immagine ha 4 canali (RGBA)
    if img.shape[2] == 4:
        # Separa i canali BGR e il canale alfa
        bgr = img[:, :, 0:3]
        alpha = img[:, :, 3]

        # Crea uno sfondo bianco a 3 canali
        white_bg = np.full(bgr.shape, 255, dtype=np.uint8)

        # Incolla l'immagine BGR sopra lo sfondo bianco usando il canale alfa come maschera
        # Il canale alfa viene espanso a 3 canali per l'operazione di copia
        alpha_3_channels = cv2.merge([alpha, alpha, alpha])
        result = np.where(alpha_3_channels == 255, bgr, white_bg)
        
        # Ritorna l'immagine in scala di grigi
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        # Se non c'è canale alfa, restituisci l'immagine originale in scala di grigi
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarize(img_gray, threshold=128, invert=False, blur=5):
    if blur and blur >= 3 and blur % 2 == 1:
        img_gray = cv2.GaussianBlur(img_gray, (blur, blur), 0)
    # foreground = scuro per default
    if invert:
        _, bw = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return bw

def find_contours(bw, min_area=10):
    # Contorni + buchi
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    result = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= float(min_area):
            result.append(cnt.squeeze(1))  # Nx2
    return result

def resample_closed_contour(points, step=2.0):
    """Ricampiona un contorno chiuso a passo (lunghezza arco) ~ costante."""
    pts = points.astype(np.float64)
    # assicurati che sia chiuso
    if not (np.allclose(pts[0], pts[-1])):
        pts = np.vstack([pts, pts[0]])
    # lunghezze cumulative
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    L = float(seg.sum())
    if L == 0:
        return pts[:1]
    n_samples = max(int(math.ceil(L / max(step, 1e-6))), 8)
    d_targets = np.linspace(0.0, L, n_samples, endpoint=False)
    res = []
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    j = 0
    for d in d_targets:
        while j < len(seg)-1 and cum[j+1] < d:
            j += 1
        # interp tra pts[j] e pts[j+1]
        denom = seg[j] if seg[j] > 1e-9 else 1.0
        t = (d - cum[j]) / denom
        p = (1.0 - t) * pts[j] + t * pts[j+1]
        res.append(p)
    res = np.array(res)
    return res

def catmull_rom_to_beziers(closed_pts, smooth=1.0):
    """
    Converte una lista di punti chiusi in segmenti Bézier cubici.
    smooth scala la tangente Catmull–Rom (1.0 = standard, <1 più teso).
    Ritorna lista di tuple (B0, B1, B2, B3) per comandi 'C' consecutivi.
    """
    P = np.asarray(closed_pts, dtype=np.float64)
    n = len(P)
    if n < 4:
        # fallback: chiudi con straight cubic
        if n == 3:
            P = np.vstack([P, P[0]])
        elif n == 2:
            P = np.vstack([P, P[0], P[1], P[0]])
        else:
            return []
        n = len(P)
    beziers = []
    s = (1.0/6.0) * float(smooth)
    for i in range(n):
        P0 = P[(i-1) % n]
        P1 = P[i % n]
        P2 = P[(i+1) % n]
        P3 = P[(i+2) % n]
        B0 = P1
        B1 = P1 + (P2 - P0) * s
        B2 = P2 - (P3 - P1) * s
        B3 = P2
        beziers.append((B0, B1, B2, B3))
    return beziers

def beziers_to_svg_path(beziers, scale=1.0):
    if not beziers:
        return ""
    # parti da B0 del primo segmento
    (B0, B1, B2, B3) = beziers[0]
    path = [f"M {B0[0]*scale:.3f} {B0[1]*scale:.3f}"]
    for (b0, b1, b2, b3) in beziers:
        path.append(
            "C {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                b1[0]*scale, b1[1]*scale, b2[0]*scale, b2[1]*scale, b3[0]*scale, b3[1]*scale
            )
        )
    path.append("Z")
    return " ".join(path)

def save_svg(paths, width, height, out_file, scale=1.0):
    w = width * scale
    h = height * scale
    d_all = " ".join(paths)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}" height="{h:.2f}"
      viewBox="0 0 {width:.2f} {height:.2f}" preserveAspectRatio="xMidYMid meet">
  <g fill="#000" stroke="none" fill-rule="evenodd">
    <path d="{d_all}"/>
  </g>
</svg>
'''
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(svg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--threshold", type=int, default=128)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--blur", type=int, default=15)
    ap.add_argument("--step", type=float, default=8.0, help="px tra punti campione")
    ap.add_argument("--smooth", type=float, default=1.5, help="0.1..1.5 circa")
    ap.add_argument("--min-area", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + ".svg"

    # Carica l'immagine, includendo il canale alfa
    img_full = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img_full is None:
        raise SystemExit(f"Impossibile aprire: {args.input}")

    # Pulisci l'immagine con trasparenza
    img_gray = convert_alpha_to_white(img_full)
    height, width = img_gray.shape

    bw = binarize(img_gray, threshold=args.threshold, invert=args.invert, blur=args.blur)
    contours = find_contours(bw, min_area=args.min_area)

    paths = []

    for cnt in contours:
        # 1) ricampionamento per togliere "gradini" a pixel
        pts = resample_closed_contour(cnt, step=args.step)
        # 2) conversione a Bézier lisce (Catmull–Rom → cubic)
        beziers = catmull_rom_to_beziers(pts, smooth=args.smooth)
        # 3) path SVG
        d = beziers_to_svg_path(beziers, scale=args.scale)
        if d:
            paths.append(d)

    if not paths:
        raise SystemExit("Nessun contorno valido trovato. Prova --invert o regola --threshold.")

    save_svg(paths, width, height, out, scale=args.scale)
    print(f"SVG scritto in: {out}")

if __name__ == "__main__":
    main()