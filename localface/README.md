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
# 1. Enroll someone (asks for consent; captures 5 webcam samples)
python -m localface enroll "Alice"

#    ...or enroll from clear, single-face photos
python -m localface enroll "Alice" --photo alice1.jpg alice2.jpg

# 2. Live recognition (press q to quit)
python -m localface watch
python -m localface watch --headless --cooldown 60      # no window, log once a minute per person

# 3. Who is in this photo?
python -m localface identify group_photo.jpg --annotate out.jpg
python -m localface identify ./photos_folder

# Admin
python -m localface list
python -m localface events
python -m localface remove "Alice"
python -m localface purge                                # wipe everything
```

Data lives in `./data/localface.db` (override with `LOCALFACE_DATA=/some/dir`).

## Tuning and accuracy

- `--threshold` (default `0.363` cosine similarity): raise for fewer false matches, lower for fewer misses.
- Enroll in the lighting you'll deploy in, with a few slightly different head angles.
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
python tests/test_core.py     # storage, matching, walking/standing logic (no models needed)
```

## License

MIT
