"""On-screen HUD drawing (pure OpenCV: no extra dependencies). Sizes scale with frame height."""
import math
import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

# BGR colors
OK = (130, 225, 110)       # recognized
WARN = (70, 165, 255)      # unknown
IDLE = (215, 215, 215)     # still scanning
WALK = (255, 170, 70)
STAND = (170, 170, 170)
INK = (247, 247, 247)
DIM = (175, 175, 175)
PANEL = (28, 24, 22)
LIVE = (80, 80, 240)


def scale_of(img):
    return max(0.6, img.shape[0] / 720.0)


def _size(s, size, thick=1, font=FONT):
    (w, h), base = cv2.getTextSize(s, font, size, thick)
    return w, h, base


def text(img, s, org, size, color=INK, thick=1, font=FONT, shadow=False):
    if shadow:
        cv2.putText(img, s, org, font, size, (0, 0, 0), thick + 2, AA)
    cv2.putText(img, s, org, font, size, color, thick, AA)


def panel(img, x1, y1, x2, y2, color=PANEL, alpha=0.62, radius=10):
    """Translucent rounded rectangle."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    r = max(0, min(int(radius), w // 2, h // 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    blend = cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0)
    roi[mask > 0] = blend[mask > 0]


def brackets(img, x, y, w, h, color, thick, length):
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thick, AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thick, AA)


def _dim(color, k=0.45):
    return tuple(int(c * k) for c in color)


# ------------------------------------------------------------------ faces
def draw_face(img, box, state, label, score=None, motion=None, top_safe=0):
    """state: 'ok' (recognized), 'unknown', or 'scanning'."""
    H, W = img.shape[:2]
    s = scale_of(img)
    x, y, w, h = (int(v) for v in box)
    color = {"ok": OK, "unknown": WARN}.get(state, IDLE)

    cv2.rectangle(img, (x, y), (x + w, y + h), _dim(color), 1, AA)
    brackets(img, x, y, w, h, color, max(2, int(2.5 * s)), int(max(14 * s, min(w, h) * 0.22)))

    pad, gap = int(9 * s), int(6 * s)
    name_sz, small = 0.72 * s, 0.44 * s
    dot_r = int(5 * s)
    nw, nh, _ = _size(label, name_sz, 1, FONT_B)
    row_items = 0
    chip_txt = motion.upper() if motion and motion != "detecting" else None
    if chip_txt:
        cw, ch, _ = _size(chip_txt, small, 1)
        chip_w, chip_h = cw + int(14 * s), ch + int(9 * s)
    else:
        chip_w = chip_h = 0
    show_bar = score is not None and state == "ok"
    num_txt = f"{score:.2f}" if show_bar else ""
    bar_w = int(54 * s)
    num_w = _size(num_txt, small, 1)[0] if show_bar else 0
    row_w = chip_w + ((gap if chip_w else 0) + bar_w + gap + num_w if show_bar else 0)
    has_row = chip_w > 0 or show_bar
    row_h = max(chip_h, int(12 * s)) if has_row else 0

    head_w = dot_r * 2 + gap + nw
    card_w = max(head_w, row_w) + 2 * pad
    card_h = pad + nh + (gap + row_h if has_row else 0) + pad

    cx = min(max(4, x), max(4, W - card_w - 4))
    cy = y - int(8 * s) - card_h                   # preferred: above the face
    if cy < top_safe + 4:                          # face near the top edge -> put it below
        cy = y + h + int(8 * s)
        if cy + card_h > H - 4:                    # no room below either -> inside the box
            cy = y + int(8 * s)
    cy = int(min(max(top_safe + 4, cy), max(top_safe + 4, H - card_h - 4)))

    panel(img, cx, cy, cx + card_w, cy + card_h, PANEL, 0.68, int(10 * s))
    ty = cy + pad + nh
    cv2.circle(img, (cx + pad + dot_r, ty - nh // 2), dot_r, color, -1, AA)
    text(img, label, (cx + pad + dot_r * 2 + gap, ty), name_sz, INK, 1, FONT_B)

    if has_row:
        ry = ty + gap + int(4 * s)
        rx = cx + pad
        if chip_w:
            mcolor = WALK if motion == "walking" else STAND
            panel(img, rx, ry, rx + chip_w, ry + chip_h, mcolor, 0.9, int(chip_h / 2))
            text(img, chip_txt, (rx + int(7 * s), ry + chip_h - int(5 * s)), small, (20, 20, 20), 1)
            rx += chip_w + gap
        if show_bar:
            by = ry + chip_h // 2 - int(3 * s) if chip_h else ry
            bh = max(4, int(6 * s))
            cv2.rectangle(img, (rx, by), (rx + bar_w, by + bh), (70, 70, 70), -1)
            fill = int(bar_w * max(0.0, min(1.0, score)))
            cv2.rectangle(img, (rx, by), (rx + fill, by + bh), color, -1)
            text(img, num_txt, (rx + bar_w + gap, by + bh + int(1 * s)), small, DIM, 1)


# ------------------------------------------------------------------ HUD
def topbar_height(img):
    return int(44 * scale_of(img))


def draw_topbar(img, fps, enrolled, faces_in_view, present=None):
    H, W = img.shape[:2]
    s = scale_of(img)
    tb = topbar_height(img)
    panel(img, 0, 0, W, tb, PANEL, 0.6, 0)
    pulse = 0.55 + 0.45 * math.sin(time.time() * 4)
    cv2.circle(img, (int(24 * s), tb // 2), int(6 * s), tuple(int(c * pulse) for c in LIVE), -1, AA)
    text(img, "LOCALFACE", (int(42 * s), tb // 2 + int(6 * s)), 0.62 * s, INK, 1, FONT_B)
    lw = _size("LOCALFACE", 0.62 * s, 1, FONT_B)[0]
    text(img, "LIVE", (int(42 * s) + lw + int(12 * s), tb // 2 + int(5 * s)), 0.45 * s, LIVE, 1)
    right = f"{faces_in_view} in view    {enrolled} enrolled    {fps:.0f} FPS    {time.strftime('%H:%M')}"
    if present is not None:
        right = f"Present today {present[0]}/{present[1]}    " + right
    rw = _size(right, 0.5 * s, 1)[0]
    text(img, right, (W - rw - int(18 * s), tb // 2 + int(5 * s)), 0.5 * s, DIM, 1)


def draw_feed(img, events, now=None, max_age=25, max_rows=4):
    """events: iterable of (timestamp, name, motion). Newest at the bottom."""
    now = now or time.time()
    rows = [e for e in events if now - e[0] <= max_age][-max_rows:]
    if not rows:
        return
    H, W = img.shape[:2]
    s = scale_of(img)
    lines = [f"{time.strftime('%H:%M:%S', time.localtime(ts))}   {name}   {motion}"
             for ts, name, motion in rows]
    sz = 0.5 * s
    lh = int(24 * s)
    tw = max(_size(l, sz, 1)[0] for l in lines + ["RECENT"])
    pad = int(12 * s)
    pw, ph = tw + 2 * pad, pad * 2 + lh * (len(lines) + 1) - int(6 * s)
    x, y = int(14 * s), H - ph - int(14 * s)
    panel(img, x, y, x + pw, y + ph, PANEL, 0.62, int(10 * s))
    text(img, "RECENT", (x + pad, y + pad + int(12 * s)), 0.4 * s, DIM, 1)
    for i, l in enumerate(lines):
        age = now - rows[i][0]
        k = 1.0 if age < max_age * 0.6 else max(0.45, 1 - (age - max_age * 0.6) / (max_age * 0.4))
        text(img, l, (x + pad, y + pad + int(12 * s) + lh * (i + 1)), sz,
             tuple(int(c * k) for c in OK), 1)


def draw_hint(img, msg):
    H, W = img.shape[:2]
    s = scale_of(img)
    sz = 0.42 * s
    tw, th, _ = _size(msg, sz, 1)
    pad = int(8 * s)
    x, y = W - tw - 2 * pad - int(14 * s), H - th - 2 * pad - int(14 * s)
    panel(img, x, y, x + tw + 2 * pad, y + th + 2 * pad, PANEL, 0.5, int(8 * s))
    text(img, msg, (x + pad, y + pad + th), sz, DIM, 1)


def draw_empty_notice(img, msg):
    H, W = img.shape[:2]
    s = scale_of(img)
    tw, th, _ = _size(msg, 0.6 * s, 1)
    pad = int(12 * s)
    x = (W - tw) // 2 - pad
    panel(img, x, H // 2 - th - pad, x + tw + 2 * pad, H // 2 + pad, PANEL, 0.6, int(10 * s))
    text(img, msg, (x + pad, H // 2), 0.6 * s, WARN, 1)


# ------------------------------------------------------------------ enrollment
def draw_enroll(img, pose_text, step, steps, done, total, face_box, message, flash_age):
    H, W = img.shape[:2]
    s = scale_of(img)
    good = face_box is not None and message is None
    col = OK if good else IDLE

    # oval guide
    cv2.ellipse(img, (W // 2, int(H * 0.50)), (int(H * 0.25), int(H * 0.34)), 0, 0, 360, col,
                max(2, int(2 * s)), AA)
    if face_box is not None:
        x, y, w, h = (int(v) for v in face_box)
        brackets(img, x, y, w, h, col, max(2, int(2.5 * s)), int(max(14 * s, min(w, h) * 0.22)))

    # instruction card
    title_sz, sub_sz = 0.95 * s, 0.5 * s
    tw = _size(pose_text, title_sz, 1, FONT_B)[0]
    sub = f"Step {step} of {steps}"
    sw = _size(sub, sub_sz, 1)[0]
    pad = int(16 * s)
    cw = max(tw, sw) + 2 * pad
    ch = int(78 * s)
    cx = (W - cw) // 2
    panel(img, cx, int(18 * s), cx + cw, int(18 * s) + ch, PANEL, 0.7, int(14 * s))
    text(img, sub, ((W - sw) // 2, int(18 * s) + int(26 * s)), sub_sz, DIM, 1)
    text(img, pose_text, ((W - tw) // 2, int(18 * s) + int(62 * s)), title_sz, INK, 1, FONT_B)

    # message under the oval
    if message:
        mw, mh, _ = _size(message, 0.7 * s, 1)
        mx, my = (W - mw) // 2, int(H * 0.50 + H * 0.34) + int(34 * s)
        panel(img, mx - pad, my - mh - int(10 * s), mx + mw + pad, my + int(12 * s), PANEL, 0.7, int(10 * s))
        text(img, message, (mx, my), 0.7 * s, WARN, 1)

    # progress bar
    bw, bh = int(W * 0.4), max(8, int(10 * s))
    bx, by = (W - bw) // 2, H - int(46 * s)
    panel(img, bx, by, bx + bw, by + bh, (70, 70, 70), 0.9, bh // 2)
    fill = int(bw * done / max(1, total))
    if fill > 2:
        panel(img, bx, by, bx + fill, by + bh, OK, 1.0, bh // 2)
    lab = f"{done}/{total} captured    F fullscreen    Q cancel"
    lw = _size(lab, 0.45 * s, 1)[0]
    text(img, lab, ((W - lw) // 2, by + bh + int(22 * s)), 0.45 * s, DIM, 1)

    # capture flash
    if flash_age < 0.25:
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), OK, int(10 * s * (1 - flash_age / 0.25)) + 1)


# ------------------------------------------------------------------ sign-up kiosk
def draw_signup_name(img, buf, message, cursor_on):
    H, W = img.shape[:2]
    s = scale_of(img)
    dark = np.zeros_like(img)
    img[:] = cv2.addWeighted(img, 0.35, dark, 0.65, 0)
    pw, ph = int(W * 0.72), int(300 * s)
    x, y = (W - pw) // 2, (H - ph) // 2
    panel(img, x, y, x + pw, y + ph, PANEL, 0.85, int(16 * s))
    pad = int(28 * s)
    text(img, "Sign up for attendance", (x + pad, y + int(54 * s)), 0.95 * s, INK, 1, FONT_B)
    text(img, "Signing up saves a numeric face template (not a photo) on this laptop to take attendance.",
         (x + pad, y + int(88 * s)), 0.5 * s, DIM, 1)
    text(img, "Only sign up if you agree. Ask an officer any time to have your data deleted.",
         (x + pad, y + int(112 * s)), 0.5 * s, DIM, 1)
    text(img, "Type your name, then press Enter", (x + pad, y + int(156 * s)), 0.6 * s, INK, 1)
    bx1, by1, bx2, by2 = x + pad, y + int(172 * s), x + pw - pad, y + int(222 * s)
    panel(img, bx1, by1, bx2, by2, (60, 56, 52), 0.95, int(10 * s))
    shown = buf + ("|" if cursor_on else " ")
    text(img, shown, (bx1 + int(14 * s), by2 - int(14 * s)), 0.9 * s, OK, 1, FONT_B)
    if message:
        text(img, message, (x + pad, y + int(250 * s)), 0.55 * s, WARN, 1)
    hint = "Enter = continue     Esc = finish"
    hw = _size(hint, 0.45 * s, 1)[0]
    text(img, hint, (x + pw - pad - hw, y + ph - int(16 * s)), 0.45 * s, DIM, 1)


def draw_welcome(img, name):
    H, W = img.shape[:2]
    s = scale_of(img)
    title = f"Welcome, {name}!"
    sub = "You're enrolled. Next person, please."
    tw = _size(title, 1.3 * s, 2, FONT_B)[0]
    sw = _size(sub, 0.65 * s, 1)[0]
    pw = max(tw, sw) + int(80 * s)
    ph = int(150 * s)
    x, y = (W - pw) // 2, (H - ph) // 2
    panel(img, x, y, x + pw, y + ph, PANEL, 0.85, int(16 * s))
    text(img, title, ((W - tw) // 2, y + int(70 * s)), 1.3 * s, OK, 2, FONT_B)
    text(img, sub, ((W - sw) // 2, y + int(115 * s)), 0.65 * s, DIM, 1)
