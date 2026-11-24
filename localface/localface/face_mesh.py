"""Optional MediaPipe Face Landmarker integration for experimental pose assistance.

The landmarks and facial transformation matrices are estimates from an RGB image;
they are not measurements from a physical depth camera.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def _mesh_box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _associate_mesh_indices(faces, face_meshes, frame_shape):
    if not faces or not face_meshes:
        return [None] * len(faces)
    height, width = frame_shape[:2]
    usable = [(i, points) for i, points in enumerate(face_meshes) if len(points) > 263]
    boxes = [(i, points, _mesh_box(points)) for i, points in usable]
    output = []
    for face in faces:
        fx1, fy1 = face.x / width, face.y / height
        fx2, fy2 = (face.x + face.w) / width, (face.y + face.h) / height
        best_iou, best_index = 0.0, None
        for index, _points, (mx1, my1, mx2, my2) in boxes:
            ix1, iy1 = max(fx1, mx1), max(fy1, my1)
            ix2, iy2 = min(fx2, mx2), min(fy2, my2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = max(1e-8, (fx2-fx1)*(fy2-fy1) + (mx2-mx1)*(my2-my1) - inter)
            iou = inter / union
            if iou > best_iou:
                best_iou, best_index = iou, index
        output.append(best_index if best_iou >= 0.12 else None)
    return output


def _yaw_degrees(matrix):
    """Extract absolute yaw from MediaPipe's estimated 4x4 face transform matrix."""
    if matrix is None:
        return None
    try:
        import numpy as np
        m = np.asarray(matrix, dtype=float)
        if m.size != 16:
            return None
        rotation = m.reshape(4, 4)[:3, :3]
        if not np.isfinite(rotation).all():
            return None
        # Remove scale/shear before extracting Euler yaw.
        u, _s, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        yaw = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
        return abs(float(np.degrees(yaw)))
    except (ValueError, TypeError, ImportError):
        return None


def associate_pose_yaws(faces, face_meshes, face_matrices, frame_shape):
    """Return estimated absolute yaw in degrees for each YuNet face, when available."""
    indices = _associate_mesh_indices(faces, face_meshes, frame_shape)
    output = []
    for index in indices:
        matrix = face_matrices[index] if index is not None and index < len(face_matrices) else None
        output.append(_yaw_degrees(matrix))
    return output


def associate_pose_offsets(faces, face_meshes, frame_shape):
    """Legacy nose/eye offset helper retained for backwards-compatible tests/tools."""
    indices = _associate_mesh_indices(faces, face_meshes, frame_shape)
    output = []
    for index in indices:
        if index is None:
            output.append(None)
            continue
        points = face_meshes[index]
        eye_left, eye_right, nose = points[33][0], points[263][0], points[1][0]
        eye_span = abs(eye_right - eye_left)
        output.append(abs(nose - (eye_left + eye_right) / 2) / eye_span if eye_span > 1e-5 else None)
    return output


class FaceMeshEstimator:
    def __init__(self, num_faces=5):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "MediaPipe face model is missing. Run: python scripts/download_face_mesh_model.py"
            )
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Install project requirements with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = __import__("mediapipe")

    def process(self, bgr_image):
        """Return normalized landmark arrays, preserving the previous public API."""
        meshes, _matrices = self.process_with_matrices(bgr_image)
        return meshes

    def process_with_matrices(self, bgr_image):
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(mp_image)
        meshes = []
        for face in result.face_landmarks:
            meshes.append([[float(p.x), float(p.y), float(p.z)] for p in face])
        raw_matrices = result.facial_transformation_matrixes
        matrices = [np.asarray(m, dtype=float).reshape(4, 4).tolist()
                    for m in raw_matrices] if raw_matrices is not None else []
        return meshes, matrices

    def close(self):
        self._landmarker.close()
