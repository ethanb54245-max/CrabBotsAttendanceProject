# LocalFace

Local, consent-based face recognition for a laptop webcam, run from a friendly app in your browser.
Think "smart doorbell" or "club check-in", not "city surveillance network."

- **Recognize**
- **Live camera.** Click *Start live camera* and the video appears right on the page. Enrolled people are boxed and
  named as they walk by or stand still, with a walking/standing tag and a match bar, and each check-in is added to
  *Just checked in* and to attendance. People you add in the app are picked up within a couple of seconds, no
  restart. *Fullscreen* (or double-click the video) fills the screen, and the on-video buttons let you exit or stop.
  Untick *Record attendance* for a demo that shows names without saving anything. Your browser asks for camera
  permission the first time.
- **Who is this?** Drop a photo, click to choose one, or paste with Ctrl+V. Every face is boxed and named with a
  match score. A face that isn't recognized gets a *Name this person* form, so you can add them on the spot and
  see them recognized immediately.
- **Sign up with the camera.** The person types their name, agrees, and sees themselves live while a few prompts
  (straight, left, right, chin up, chin down) guide a ~6 second capture. A name that is already taken, or a face
  already enrolled under another name, is refused. The live camera pauses while someone signs up, then resumes.
- **Open in its own window** is the same live recognition in a separate fullscreen window (also `python -m localface
  watch`). Keys there: `F` or double-click = fullscreen, `H` = hide the overlay, `Q` / `Esc` = quit.
  A webcam can only be used by one thing at a time, so starting one camera feature stops the others.

**People**
- See everyone enrolled and remove anyone.
- **Add a person** from one clear, front-facing photo.
- **Import many photos** at once. Name files like `Ethan Badoud.jpg` (a trailing number such as
  `Ethan Badoud 2.jpg` is ignored, so one person can have several photos), check the names, tick the consent
  box, and import.
- **Delete all face data and attendance** in one click (you must type DELETE to confirm).
- **Admin PIN** (recommended): protects the two destructive actions above. See below.

**Attendance**
- Today's check-ins, a grid of everyone across every meeting, and **Download CSV**.
- Attendance is each person's first and last sighting per calendar day, so a Monday and Tuesday club are
  tracked separately with no setup. Leave the live camera running during meetings.

Uploaded photos are processed in memory and never written to disk. Only a numeric face template is stored, and
only for people enrolled with consent. The app only accepts connections from this computer.

## Admin PIN

On the **People** tab, click **Set a PIN** (4 to 64 characters). After that, **Remove** and **Delete all** ask for the
PIN first. Recognizing, signing up, importing, attendance and the CSV download stay open to everyone.

- Entering the PIN unlocks for 10 minutes of inactivity, or until you click **Lock now**.
- Five wrong guesses in a row lock the PIN prompt for a minute.
- Change or remove the PIN from the same card (both need the current PIN).
- The PIN is stored only as a salted hash in `data/localface.db`. **Delete all** does not remove it.
- The command-line `remove` and `purge` ask for the PIN too (or read it from a `LOCALFACE_PIN` environment variable
  for scripts).
- **Forgot it?** Run `python -m localface reset-pin` on the club computer, then set a new one in the app.

The PIN protects the app and the command-line commands. It can't stop someone who can open the computer's files
(or run `reset-pin`), so keep the laptop's own login secure too.

## Command line (optional)

Everything in the app is also available as commands, and `python -m localface` with no command opens the app.

```bash
python -m localface                                    # open the app (same as `app`)
python -m localface app --port 9000 --no-browser

python -m localface enroll "Ethan Badoud"              # guided webcam capture (asks for consent)
python -m localface enroll "Ethan Badoud" --photo "Ethan Badoud.jpg"
python -m localface enroll-folder member_photos        # bulk import from a folder of photos
python -m localface signup --fullscreen                # self-service kiosk: people type their name and enroll

python -m localface watch --fullscreen                 # live recognition window
python -m localface watch --headless --cooldown 60     # no window, log once a minute per person

python -m localface identify "Ethan Badoud.jpg" --annotate result.jpg
python -m localface identify member_photos             # a whole folder

python -m localface attendance                         # today: present / absent
python -m localface attendance --date 2026-11-17
python -m localface attendance --summary               # grid of every member x every meeting
python -m localface attendance --csv attendance.csv

python -m localface list | events | remove "Ethan Badoud" | purge   # remove/purge ask for the admin PIN if one is set
python -m localface reset-pin                          # clear a forgotten admin PIN
```

Data lives in `./data/localface.db` (override with `LOCALFACE_DATA=/some/dir`). The app, the command line
and the live camera all share it. If a command fails on a file name with spaces, put the name in quotes.

## Tuning and accuracy

- **Enroll across head angles.** Camera sign-up captures five poses. This matters most for faces high or low in
  the frame, which are seen at an angle. Enrolling the same name again from the command line adds more samples.
- Faces cut off by the frame edge are handled by padding the image before detection.
  `--det-size` (default 960) trades speed for finding smaller/farther faces; `--det-score` (default 0.6)
  lowers/raises how readily blurry or partial faces are accepted.
- A single strong match identifies someone immediately; weaker matches need two agreeing frames. Once
  identified, the identity sticks to that person's track through brief blur or head turns.
- `--threshold` (default `0.363` cosine similarity): raise for fewer false matches, lower for fewer misses.
  A score around 0.5 or higher is a solid match.
- Photo matching works best with front-facing, well-lit, unobstructed faces.
- There is **no liveness check**: a printed photo or a phone screen held to the camera can fool it. Don't use
  this for access control or security decisions.
- Face recognition error rates vary across lighting, camera quality, and demographic groups. Treat a match as a
  hint, never as proof of identity.

## Responsible use

Face templates are biometric data. Several jurisdictions regulate collecting them (for example Illinois BIPA,
Texas CUBI, Washington's biometric law, and the EU GDPR), often requiring written notice and consent. Only enroll
people who agreed, post a notice if the camera covers a space others use, and delete data on request (remove
them on the People tab, or `remove` / `purge`). Do not use this to identify, follow, or profile people who
haven't opted in.

**Minors and schools.** If members are under 18, get permission from your advisor, the school and
parents/guardians *before* enrolling anyone, and check your school or district policy: some states and districts
restrict or ban facial recognition in schools. Keep the data on one club laptop, don't share it, and delete it at
the end of the season.

**Sharing this code.** `.gitignore` keeps `data/`, `member_photos/`, `models/`, attendance CSV exports and loose
photos out of Git. The website's "upload files" button ignores `.gitignore`, so if you upload that way, leave those
out yourself. Never upload photos, your `data` folder or an attendance export to a public repo.

## Tests

```bash
python tests/test_core.py     # storage, matching, tracking, identity, UI drawing (no models needed)
python tests/test_club.py     # kiosk typing, folder import, attendance reports, sign-up loop
python tests/test_engine_camera.py  # face-coordinate math, camera reader, command-line flows (no hardware needed)
python tests/test_server_process.py # starts the real app process, port fallback, clean Ctrl+C
python tests/test_live.py     # live recognition: shared recognizer logic and the in-page camera endpoints
python tests/test_admin.py    # admin PIN: hashing, lockout, unlock tokens, protected actions, CLI prompts
python tests/test_demo.py     # the app's server: security guards, identify, enroll, camera sign-up,
                              # attendance history, CSV, delete-all, `python -m localface` opens the app
```

## License

MIT
