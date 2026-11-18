"""Download the two OpenCV Zoo ONNX models LocalFace needs (~40 MB total)."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
dest = Path(__file__).resolve().parent.parent / "models"
dest.mkdir(exist_ok=True)

for name, url in MODELS.items():
    out = dest / name
    if out.exists() and out.stat().st_size > 100_000:
        print(f"ok      {name}")
        continue
    print(f"fetching {name} ...")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:  # noqa: BLE001
        print(f"failed: {e}\nDownload it manually from:\n  {url}\nand place it in {dest}")
        sys.exit(1)
    print(f"saved   {out} ({out.stat().st_size // 1024} KB)")
