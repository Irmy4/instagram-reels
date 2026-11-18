#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orman Yeşili ASCII Göz Animasyonu  -  TEK DOSYA (reels77.py)
------------------------------------------------------------
Kurulum    : pip install numpy opencv-python
Çalıştırma : python reels77.py              -> tam ekran pencere (Esc / Q ile çıkış)
             python reels77.py --pencere    -> 1280x720 normal pencere
             python reels77.py --hucre 8    -> daha ince harfler (varsayılan 10)
             python reels77.py --terminal   -> terminalde çalışır (OpenCV gerekmez)
             python reels77.py --kalp 15 --dongu 30   -> kalp 15. sn'de çıkar, döngü 30 sn
Kalp : sona doğru gözbebeğinin yerinde pembe/kırmızı, atan bir kalp belirir.
"""

import argparse
import math
import os
import shutil
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ----------------------------------------------------------------------------
# Renk paleti (RGB)
# ----------------------------------------------------------------------------
def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)

BLACK  = hex2rgb("#000000")
FOREST = hex2rgb("#2E8B57")   # orman yeşili
JADE   = hex2rgb("#00A36C")   # canlı derin yeşil
MUTED  = hex2rgb("#4F7942")   # grimsi / dumanlı yeşil
SLATE  = hex2rgb("#708090")   # metalik gri
PUPIL  = np.array([5, 16, 11], dtype=np.float32)
SPEC   = np.array([185, 205, 195], dtype=np.float32)  # gri-yeşil parlama

RAMP = np.array(list(" .,:;-=+*#%@"))
LUM_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ----------------------------------------------------------------------------
def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def lerp(a, b, t):
    return a + (b - a) * t

def unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)

LIGHT1 = unit([-0.35, -0.50, 0.80])
HALF1  = unit(LIGHT1 + np.array([0, 0, 1], dtype=np.float32))
LIGHT2 = unit([0.50, 0.45, 0.75])
HALF2  = unit(LIGHT2 + np.array([0, 0, 1], dtype=np.float32))


# ----------------------------------------------------------------------------
# Kare üretimi
# ----------------------------------------------------------------------------
def gaze(t):
    """Gerçek göz gibi: sabit bakış + hızlı sıçrama (sakkad) + mikro titreme."""
    seg = int(t // 1.7)
    ph = t - seg * 1.7

    def tg(i):
        return 0.40 * math.sin(i * 7.13 + 0.5), 0.20 * math.sin(i * 3.77 + 2.0)

    a, b = tg(seg - 1), tg(seg)
    k = float(smooth(0.0, 0.14, ph))
    yaw = a[0] + (b[0] - a[0]) * k + 0.012 * math.sin(9 * t)
    pitch = a[1] + (b[1] - a[1]) * k + 0.010 * math.sin(11 * t + 1)
    return yaw, pitch


def render(t, W, H, noise, heart_t=25.0, cycle=40.0):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    # --- Zaman tabanlı kontrol değerleri -----------------------------------
    intro = float(smooth(0.0, 4.0, t))
    pulse = max(0.0, math.sin(0.23 * t)) ** 6
    yaw, pitch = gaze(t)
    yaw += 1.6 * (1 - intro) ** 2                      # girişte dönerek gelir

    # Kalp zamanı yaklaşırken bakış ortaya yerleşir
    settle = float(smooth(heart_t - 4.0, heart_t, t))
    yaw *= (1 - settle)
    pitch *= (1 - settle)
    hk = float(smooth(heart_t, heart_t + 3.0, t))
    if cycle > 0:
        hk *= 1.0 - float(smooth(cycle - 2.5, cycle - 0.4, t))

    zoom = ((0.15 + 0.85 * intro)
            * (1 + 0.04 * math.sin(0.7 * t) + 0.10 * pulse) * (1 + 0.08 * hk))

    # Göz kırpma
    openness = 1.0
    tb = t % 6.1
    if t > 3.5 and tb < 0.30:
        openness = max(0.0, 1.0 - math.sin(math.pi * tb / 0.30))
    openness = min(openness, 0.15 + 0.85 * float(smooth(0.3, 3.2, t)))

    # --- Koordinatlar ------------------------------------------------------
    R = min(W / 2 * 0.88, H * 0.74) * zoom
    u = (xs - (W - 1) / 2) / R
    v = ((ys - (H - 1) / 2) * 2.0) / R

    amp = 0.010 + 0.006 * math.sin(0.9 * t) + 0.05 * (1 - intro)
    u2 = u + amp * np.sin(v * 10 + t * 3.0)
    v2 = v + amp * np.sin(u * 10 + t * 2.4)

    r2 = u2 ** 2 + v2 ** 2
    on = r2 < 1.0
    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    N = np.stack([u2, v2, nz], axis=-1)

    # 3D rotasyon
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    Nl = N @ (Ry @ Rx)
    theta = np.arccos(np.clip(Nl[..., 2], -1, 1))
    phi = np.arctan2(Nl[..., 1], Nl[..., 0])

    # --- Göz kapakları -----------------------------------------------------
    un = np.clip(np.abs(u2), 0, 1)
    shape = (1 - un ** 2.2) ** 0.8
    up = (0.62 + 0.18 * hk) * openness * shape * (1 + 0.25 * math.sin(-pitch)) + 1e-3
    lo = (0.46 + 0.12 * hk) * openness * shape + 1e-3
    visible = on & (v2 >= -up) & (v2 <= lo)

    # --- İris ----------------------------------------------------------------
    pupil_base = 0.20 + 0.04 * math.sin(1.7 * t) + 0.03 * math.sin(0.6 * t) + 0.04 * hk
    pr = pupil_base + 0.010 * np.sin(phi * 9 + t * 0.5)       # organik gözbebeği kenarı
    ir = 0.60 + 0.008 * np.sin(phi * 5)
    k2 = np.clip((theta - pr) / (ir - pr), 0, 1)
    k = k2[..., None]

    fib1 = 0.5 + 0.5 * np.sin(phi * 18 + 2.5 * np.sin(phi * 3 + t * 0.3) + theta * 6)
    fib2 = 0.5 + 0.5 * np.sin(phi * 41 - theta * 14)
    fib3 = 0.5 + 0.5 * np.sin(phi * 90 + 4 * np.sin(phi * 4) + theta * 8)
    tex2 = 0.45 * fib1 + 0.30 * fib2 + 0.25 * fib3
    crypt = smooth(0.7, 0.95, np.sin(phi * 7 + theta * 20) * np.sin(phi * 11 - theta * 13 + 2.0))
    cmask = smooth(0.15, 0.35, k2) * (1 - smooth(0.7, 0.9, k2))
    tex = (tex2 * (1 - 0.55 * crypt * cmask))[..., None]       # kriptalar (koyu lekeler)

    iris = lerp(MUTED, FOREST, tex)
    c0 = (0.30 + 0.05 * np.sin(phi * 12))[..., None]           # dalgalı kolarette halkası
    coll = np.exp(-(((k - c0) / 0.13) ** 2))
    iris = lerp(iris, JADE, coll * 0.85 * (0.5 + 0.5 * tex))
    iris = lerp(iris, SLATE, 0.45 * k ** 2 * (1 - tex))
    fringe = np.exp(-(((k - 0.04) / 0.05) ** 2))               # gözbebeği çevresi pigment
    shade = ((1 - 0.9 * smooth(0.82, 1.0, k)) * (0.55 + 0.45 * tex)
             * (0.35 + 0.65 * smooth(0.0, 0.15, k)) * (1 - 0.5 * fringe))
    iris = iris * shade

    # --- Sklera ---------------------------------------------------------------
    sclera = np.array([98, 118, 108], dtype=np.float32) + np.zeros_like(iris)
    vein = (np.clip(1 - np.abs(np.sin(phi * 9 + 3 * np.sin(theta * 7 + 0.5))) * 7, 0, 1)
            + 0.6 * np.clip(1 - np.abs(np.sin(phi * 17 + 2 * np.sin(theta * 11))) * 9, 0, 1))
    vein = np.clip(vein, 0, 1) * smooth(0.9, 1.4, theta)
    sclera = lerp(sclera, FOREST * 0.8, vein[..., None] * 0.55)
    sclera = sclera * (1 - 0.5 * np.exp(-(((theta - ir) / 0.04) ** 2)))[..., None]   # limbal halka
    sclera = sclera * (0.72 + 0.28 * (1 - un ** 2))[..., None]

    im = (1 - smooth(ir - 0.012, ir + 0.012, theta))[..., None]
    pm = (1 - smooth(pr - 0.010, pr + 0.010, theta))[..., None]
    eye = lerp(sclera, iris, im)
    eye = lerp(eye, PUPIL, pm)

    # Işıklandırma
    diff = np.clip(N @ LIGHT1, 0, 1)[..., None]
    limb = (0.35 + 0.65 * nz ** 0.5)[..., None]
    eye = eye * limb * (0.45 + 0.70 * diff)
    sh_lid = (0.35 + 0.65 * smooth(0.0, 0.22, v2 + up)) * (0.6 + 0.4 * smooth(0.0, 0.18, lo - v2))
    eye = eye * sh_lid[..., None]

    # Islak kornea: pencere yansıması (sabit) + parlamalar
    s1 = (np.clip(N @ HALF1, 0, 1) ** 60)[..., None]
    s2 = (np.clip(N @ HALF2, 0, 1) ** 120)[..., None]
    cornea = theta < (ir + 0.15)
    wr = smooth(0.115, 0.085, np.abs(u2 + 0.30)) * smooth(0.085, 0.06, np.abs(v2 + 0.30))
    bar = 1 - 0.9 * (smooth(0.012, 0.006, np.abs(u2 + 0.30)) + smooth(0.012, 0.006, np.abs(v2 + 0.30)))
    wr = (np.clip(wr * bar, 0, 1) * cornea)[..., None]
    eye = eye + SPEC * (s1 * 0.6 + s2 * 0.4 + wr * 1.1)

    # --- Çevre: deri, kapak kıvrımı, kaş ---------------------------------------
    rd = np.sqrt((u2 / 1.6) ** 2 + (v2 / 1.2) ** 2)
    fade = 1 - smooth(1.5, 2.3, rd)
    skin = lerp(MUTED, SLATE, 0.3) * (0.30 * np.exp(-rd ** 2 * 0.7) * fade)[..., None] * 1.5
    skin = skin + MUTED * ((noise > 0.88) * 0.12 * fade)[..., None]            # gözenekler
    crease = np.exp(-(((v2 + up + 0.2 * shape) / 0.03) ** 2)) * (un < 0.95) * (openness > 0.1)
    crease2 = 0.5 * np.exp(-(((v2 - lo - 0.12 * shape) / 0.03) ** 2)) * (un < 0.9)
    skin = skin + MUTED * (0.55 * crease + crease2)[..., None]
    vb = -1.08 - 0.10 * (1 - u2 ** 2)
    thick = np.clip(0.075 * (1 - 0.55 * np.abs(u2)), 0, None)
    brow = (np.abs(v2 - vb) < thick) & (np.abs(u2) < 1.15) & (np.sin(u2 * 60 + v2 * 14) > -0.2)
    skin = np.where(brow[..., None], MUTED * 0.95, skin)

    col = np.where(visible[..., None], eye, skin)

    # --- Kalp (pembe / kırmızı) ------------------------------------------------
    hm = np.zeros((H, W), dtype=bool)
    heart_depth = np.zeros((H, W), dtype=np.float32)
    if hk > 0.002:
        u0, v0 = sy * cp, -sp                              # gözbebeğinin ekrandaki merkezi
        ph = (t * 1.3) % 1.0
        beat = math.exp(-8 * ph) + 0.6 * math.exp(-8 * ((ph - 0.28) % 1.0))   # lub-dub
        s_h = 0.34 * (0.4 + 0.6 * hk) * (1 + 0.12 * beat)
        hx = (u2 - u0) / s_h
        hy = -(v2 - v0) / s_h + 0.25

        # kalbin arkası kararır (gözbebeği genişler)
        dd = np.sqrt(hx ** 2 + (hy - 0.25) ** 2)
        dark = 1 - 0.93 * hk * (1 - smooth(1.35, 1.9, dd))
        col = col * np.where(visible, dark, 1.0)[..., None]

        # klasik kalp eğrisi: x^2 + (y - |x|^(2/3))^2 = 1  (belirgin çatlak + sivri uç)
        g = hx ** 2 + (hy - np.abs(hx) ** (2.0 / 3.0)) ** 2
        inside = g <= 1.0
        depth = np.clip(1 - g, 0, 1)

        RED = np.array([205, 14, 46], dtype=np.float32)
        PINK = np.array([255, 105, 180], dtype=np.float32)
        HILITE = np.array([255, 215, 232], dtype=np.float32)
        base = lerp(RED, PINK, (smooth(-1.0, 1.5, hy) * 0.85)[..., None])
        base = base * (0.6 + 0.4 * smooth(0.0, 0.5, depth))[..., None]
        hl = np.exp(-(((hx + 0.45) / 0.30) ** 2 + ((hy - 1.0) / 0.22) ** 2))[..., None]
        base = lerp(base, HILITE, hl * 0.85)
        base = base * (0.85 + 0.18 * beat)

        halo = (np.exp(-np.clip(g - 1, 0, None) * 5) * (~inside) * (g < 2.2))[..., None]
        col = col + halo * np.array([190, 25, 70], dtype=np.float32) * 0.6 * hk * visible[..., None]

        hm = inside & visible & (hk > 0.05)
        col = np.where(hm[..., None], lerp(col, base, hk), col)
        heart_depth = depth

    # Kapak kenarları ve kirpikler
    edge = (np.abs(v2 + up) < 0.035) | (np.abs(v2 - lo) < 0.030)
    edge &= (un < 1.0) & (r2 < 1.15)
    col = np.where(edge[..., None], MUTED * 0.9, col)

    ub = u2 + 0.55 * (v2 + up) * np.sign(u2)                  # dışa doğru yelpaze
    lash_len = 0.20 * shape * (0.55 + 0.45 * np.sin(ub * 13) ** 2)
    lash = ((v2 < -up) & (v2 > -up - lash_len) & (un < 0.97)
            & (np.sin(ub * 62) > 0.55) & (openness > 0.15))
    col = np.where(lash[..., None], MUTED * 0.95, col)

    # --- Belirme efekti + parlaklık --------------------------------------------
    show = noise < (intro * 1.3 - 0.15)
    col = col * (0.3 + 0.7 * intro)
    col = np.where(show[..., None], col, 0)
    col = np.clip(col, 0, 255)

    # --- Karakter seçimi -------------------------------------------------------
    lum = (col @ LUM_W) / 255.0
    idx = np.clip((lum ** 0.8) * 1.6 * (len(RAMP) - 1), 0, len(RAMP) - 1).astype(int)
    chars = RAMP[idx]
    chars = np.where(lum < 0.035, " ", chars)
    chars = np.where(brow & ~visible & show & (lum >= 0.035), '"', chars)
    lash_ch = np.where(np.abs(u2) < 0.22, "|", np.where(u2 > 0, "/", "\\"))
    chars = np.where(lash & show, lash_ch, chars)
    chars = np.where(edge & show & (lum >= 0.035), "=", chars)
    chars = np.where(hm & show, np.where(heart_depth > 0.18, "@", "#"), chars)

    disp = np.clip(col * 1.25, 0, 255).astype(np.uint8)
    return chars, disp


def zaman(t, args):
    """Döngü varsa zamanı döngü uzunluğuna göre sarar."""
    return (t % args.dongu if args.dongu > 0 else t,)


# ----------------------------------------------------------------------------
# ANSI çıktısı
# ----------------------------------------------------------------------------
def to_ansi(chars, rgb):
    H, W = chars.shape
    q = ((rgb // 8) * 8).tolist()
    ch = chars.tolist()
    lines = []
    for y in range(H):
        out, prev = [], None
        for x in range(W):
            c = ch[y][x]
            if c == " ":
                out.append(" ")
                continue
            col = q[y][x]
            if col != prev:
                out.append(f"\x1b[38;2;{col[0]};{col[1]};{col[2]}m")
                prev = col
            out.append(c)
        lines.append("".join(out))
    return "\x1b[0m\n".join(lines)



def main_terminal(args):
    if os.name == "nt":
        os.system("")  # Windows'ta ANSI desteğini etkinleştirir

    rng = np.random.default_rng(7)
    size, noise = None, None
    frame_dt = 1.0 / max(args.fps, 1)
    out = sys.stdout

    # alternatif ekran, imleci gizle, siyah arka plan
    out.write("\x1b[?1049h\x1b[?25l\x1b[48;2;0;0;0m\x1b[2J")
    out.flush()
    start = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            t = t0 - start
            if args.duration and t > args.duration:
                break

            cols, rows = shutil.get_terminal_size((80, 24))
            W, H = max(cols, 20), max(rows - 1, 10)
            if (W, H) != size:
                size = (W, H)
                noise = rng.random((H, W)).astype(np.float32)
                out.write("\x1b[48;2;0;0;0m\x1b[2J")

            chars, rgb = render(*zaman(t, args), W, H, noise, args.kalp, args.dongu)
            out.write("\x1b[H" + to_ansi(chars, rgb))
            out.flush()

            sleep = frame_dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        out.flush()



def ekran_boyutu():
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def harf_atlasi(cw, ch):
    """Her karakter için cw x ch piksellik gri tonlu bitmap üretir."""
    karakterler = list(RAMP) + ["|", "=", "/", "\\", '"']
    atlas = np.zeros((256, ch, cw), dtype=np.uint8)   # indeks = ASCII kodu
    scale = ch / 28.0
    for c in karakterler:
        if c == " ":
            continue
        img = np.zeros((ch, cw), dtype=np.uint8)
        (tw, th), base = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max((cw - tw) // 2, 0)
        y = (ch + th) // 2
        cv2.putText(img, c, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
        atlas[ord(c)] = img
    return atlas


def kare_olustur(chars, rgb, atlas, cw, ch):
    H, W = chars.shape
    kodlar = np.ascontiguousarray(chars).astype("U1").view(np.uint32).reshape(H, W)
    glif = atlas[kodlar]                                   # (H, W, ch, cw)
    img = (glif[..., None].astype(np.uint16) * rgb[:, :, None, None, :]) // 255
    img = img.astype(np.uint8).transpose(0, 2, 1, 3, 4).reshape(H * ch, W * cw, 3)
    return img[..., ::-1]                                  # RGB -> BGR (OpenCV)


def main_pencere(args):
    if cv2 is None:
        print("OpenCV kurulu değil. Kur:  python -m pip install opencv-python")
        print("Ya da terminal sürümünü çalıştır:  python reels77.py --terminal")
        return

    cw = max(args.hucre, 4)
    ch = cw * 2
    sw, sh = (1280, 720) if args.pencere else ekran_boyutu()

    W, H = sw // cw, sh // ch
    # Izgara ekrana tam oturmuyorsa, kalan boşluk siyah kenar olarak iki yana eşit dağıtılır (ortalama)
    off_x = (sw - W * cw) // 2
    off_y = (sh - H * ch) // 2

    atlas = harf_atlasi(cw, ch)
    noise = np.random.default_rng(7).random((H, W)).astype(np.float32)

    ad = "reels77"
    cv2.namedWindow(ad, cv2.WINDOW_NORMAL)
    if args.pencere:
        cv2.resizeWindow(ad, sw, sh)
    else:
        cv2.setWindowProperty(ad, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(ad, lambda *a: None)

    tuval = np.zeros((sh, sw, 3), dtype=np.uint8)
    dt = 1.0 / max(args.fps, 1)
    basla = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        chars, rgb = render(*zaman(t0 - basla, args), W, H, noise, args.kalp, args.dongu)
        tuval[off_y:off_y + H * ch, off_x:off_x + W * cw] = kare_olustur(chars, rgb, atlas, cw, ch)
        cv2.imshow(ad, tuval)

        bekle = max(1, int((dt - (time.perf_counter() - t0)) * 1000))
        tus = cv2.waitKey(bekle) & 0xFF
        if tus in (27, ord("q"), ord("Q")):
            break
        if cv2.getWindowProperty(ad, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="ASCII orman yeşili göz animasyonu")
    ap.add_argument("--terminal", action="store_true", help="pencere yerine terminalde çalıştır")
    ap.add_argument("--pencere", action="store_true", help="tam ekran yerine 1280x720 pencere")
    ap.add_argument("--hucre", type=int, default=10, help="hücre genişliği (piksel), yükseklik 2 katıdır")
    ap.add_argument("--fps", type=float, default=None, help="saniyedeki kare sayısı")
    ap.add_argument("--duration", type=float, default=0, help="terminal modunda süre (0 = sonsuz)")
    ap.add_argument("--kalp", type=float, default=25.0, help="kalbin belirmeye başladığı saniye")
    ap.add_argument("--dongu", type=float, default=40.0, help="animasyon döngü süresi (sn); 0 = döngü yok, kalp kalır")
    args = ap.parse_args()
    if args.dongu > 0:
        args.kalp = min(args.kalp, args.dongu - 6)
    if args.terminal:
        args.fps = args.fps or 24
        main_terminal(args)
    else:
        args.fps = args.fps or 30
        main_pencere(args)


if __name__ == "__main__":
    main()