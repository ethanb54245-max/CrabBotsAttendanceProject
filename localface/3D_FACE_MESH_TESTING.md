# Experimental 3D pose assistance

This experiment adds MediaPipe Face Landmarker to LocalFace's live recognition flow. The facial
landmarks are processed in the background; **no dots or mesh are drawn on the camera preview**.

## Setup

1. Use 64-bit Python 3.10–3.12 for the pinned MediaPipe release.
2. From the project folder, run:

   ```bash
   python -m pip install -r requirements.txt
   python scripts/download_face_mesh_model.py
   ```

   The model is downloaded once and stored locally at `models/face_landmarker.task`.
3. Start LocalFace, open the Recognize tab, leave **Use experimental 3D pose assistance** checked,
   and start the live camera. Uncheck it and restart the camera to compare with the original
   YuNet + SFace flow.

## What the experiment changes

YuNet and SFace still perform face detection and identity matching. MediaPipe estimates facial
landmarks from the same RGB camera frame. The app associates each mesh with a detected face and
calculates a simple nose-to-eye-center offset as a rough head-turn signal. If that signal crosses
a conservative experimental threshold, LocalFace temporarily withholds the identity and shows
**Face camera** until the face is more frontal. It does not create a 3D identity template or add a
new identity score. This is intended to reduce questionable matches at strong angles, not to
claim better recognition from side views.

The yaw estimate is still a model estimate, not a calibrated physical measurement. It may be
inaccurate at some angles or lighting conditions, so compare both modes with consenting participants
and keep a non-face check-in option available. Do not use a face match as the sole proof of attendance.

A normal laptop RGB webcam does not measure physical depth. MediaPipe estimates 3D landmarks from
2D images; its z values are relative model depth. Landmark processing stays in memory and the
project does not save the mesh as a new identity record. If the option is unchecked, MediaPipe is
not initialized and the original recognition path is used. The first model download requires
internet access; inference runs locally afterward.
