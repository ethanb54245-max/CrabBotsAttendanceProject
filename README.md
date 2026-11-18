# LocalFace

Local, consent-based face recognition for a laptop webcam. Think "smart doorbell" or
"attendance check-in", not "city surveillance network."

- **Live recognition** – the webcam identifies *enrolled* people whether they are walking
  past or standing still, and labels each as `walking` or `standing`.
- **Photo recognition** – give it a clear photo and it tells you which enrolled person
  (if any) is in it.
- **Runs entirely offline** on CPU (OpenCV YuNet detector + SFace embeddings). No cloud, no LTE, no accounts.

## How it differs from a Flock-style system

| | Flock-style ALPR | LocalFace |
|---|---|---|
| Target | every passing vehicle | only people who enrolled and consented |
| Unknown subjects | captured and logged | shown as `unknown` on screen, **never stored** |
| Storage | cloud database | one local SQLite file you control |
| Sharing | networked / third parties | none; no network code at all after model download |

## Install

Requires Python 3.9+ and a webcam.

```bash
git clone <your-repo-url> localface && cd localface
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py                       # one-time, ~40 MB
```

## Use

```bash
# 1. Enroll someone (asks for consent; guided capture of 10 samples across 5 head poses)
python -m localface enroll "Ethan Badoud"

#    ...or enroll from clear, single-face photos
python -m localface enroll "Ethan Badoud" --photo "Ethan Badoud.jpg"

# 2. Live recognition
python -m localface watch
python -m localface watch --fullscreen
python -m localface watch --headless --cooldown 60      # no window, log once a minute per person

# 3. Who is in this photo?
python -m localface identify "Ethan Badoud.jpg" --annotate result.jpg
python -m localface identify member_photos

# Admin
python -m localface list
python -m localface events
python -m localface remove "Ethan Badoud"
python -m localface purge                                # wipe everything
```

Data lives in `./data/localface.db` (override with `LOCALFACE_DATA=/some/dir`).

## Running it for a club (attendance)

**1. Get everyone enrolled (pick whichever is easiest):**

```bash
# A. Sign-up kiosk: leave a laptop at the door. Each person types their name and follows
#    a ~6 second on-screen capture. No photos needed, no officer involved. (Esc to finish.)
python -m localface signup --fullscreen

# B. Bulk import from photos people already gave you (ID/yearbook-style, clear, one face each).
#    Name files "Ethan Badoud.jpg" (or "Ethan Badoud 2.jpg"), or use one folder per person.
python -m localface enroll-folder member_photos
```

The kiosk refuses a name that is already enrolled and refuses a face that is already enrolled
under another name, so nobody can be double-entered or check in as someone else.

**2. Each meeting,** leave the camera running and view the results:

```bash
python -m localface watch --fullscreen        # top bar shows "Present today 7/12"
python -m localface attendance                # today: who is present / absent
python -m localface attendance --date 2026-11-17
python -m localface attendance --summary      # grid of every member x every meeting
python -m localface attendance --csv attendance.csv
```

Attendance is a person's first and last sighting per calendar day, so Monday and Tuesday
are tracked separately without any setup.

**Minors and schools.** If members are under 18, get permission from your advisor, the school
and parents/guardians *before* enrolling anyone, and check your school or district policy:
some states and districts restrict or ban facial recognition in schools. Keep the data on one
club laptop, don't share it, and `purge` at the end of the season.

## Window controls

| Key | Action |
|---|---|
| `F` or double-click | toggle fullscreen (also `--fullscreen` to start that way) |
| `H` | hide/show the HUD (top bar, recent list, hints) |
| `Q` / `Esc` | quit |

The window is freely resizable and keeps its aspect ratio.

## Tuning and accuracy

- **Enroll across head angles.** The guided enrollment captures straight, left, right, chin-up and
  chin-down samples. This matters most for faces high or low in the frame, which are seen at an angle.
  Run `enroll` again for the same name to *add* more samples (e.g. in different lighting or with/without glasses).
- Faces cut off by the frame edge are handled by padding the image before detection.
  `--det-size` (default 960) trades speed for finding smaller/farther faces; `--det-score` (default 0.6)
  lowers/raises how readily blurry or partial faces are accepted.
- A single strong match identifies someone immediately; weaker matches need two agreeing frames.
  Once identified, the identity sticks to that person's track through brief blur or head turns.

- `--threshold` (default `0.363` cosine similarity): raise for fewer false matches, lower for fewer misses.
- Photo matching works best with front-facing, well-lit, unobstructed faces.
- Walking vs standing is judged from how far the face moves/changes size over ~1 second;
  adjust the constants at the top of `localface/tracking.py`.
- There is **no liveness check**: a printed photo or a phone screen held to the camera can
  fool it. Don't use this for access control or security decisions.
- Face recognition error rates vary across lighting, camera quality, and demographic groups.
  Treat a match as a hint, never as proof of identity.

## Responsible use

Face templates are biometric data. Several jurisdictions regulate collecting them
(for example Illinois BIPA, Texas CUBI, Washington's biometric law, and the EU GDPR), often
requiring written notice and consent. Only enroll people who agreed, post a notice if the
camera covers a space others use, and delete data on request (`remove` / `purge`).
Do not use this to identify, follow, or profile people who haven't opted in.

## Tests

```bash
python tests/test_core.py     # storage, matching, tracking, identity, UI drawing (no models needed)
python tests/test_club.py     # kiosk typing, folder import, attendance reports, sign-up loop
```

## License

MIT
