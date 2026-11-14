"""เทียบผล detection ของ API กับผลใน Colab (samples/det_expected.json)

กล่องสองชุด "ตรงกัน" เมื่อ label เดียวกัน และ IoU >= 0.9
รัน:  python check_det.py            (API ต้องรันอยู่ที่ port 8001)
"""
import json
import sys

import requests

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
IOU_MIN = 0.9


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def match(expected, got):
    """จับคู่กล่อง expected กับ got แบบ greedy; คืนจำนวนกล่องที่จับคู่ได้"""
    used, hit = set(), 0
    for e in expected:
        best, best_j = 0, None
        for j, g in enumerate(got):
            if j in used or g["label"] != e["label"]:
                continue
            v = iou(e["box"], g["box"])
            if v > best:
                best, best_j = v, j
        if best >= IOU_MIN:
            used.add(best_j)
            hit += 1
    return hit


expected = json.load(open("samples/det_expected.json", encoding="utf-8"))
all_ok = True
for e in expected:
    with open(f"samples/det/{e['file']}", "rb") as f:
        r = requests.post(f"{API}/detect", params={"conf": 0.25},
                          files={"file": (e["file"], f, "image/jpeg")}, timeout=60)
    r.raise_for_status()
    got = r.json()["detections"]
    hit = match(e["detections"], got)
    ok = hit == len(e["detections"]) == len(got)
    all_ok &= ok
    print(f"{'✅' if ok else '❌'} {e['file']}  colab={len(e['detections'])} กล่อง  "
          f"api={len(got)} กล่อง  ตรงกัน={hit}  ({r.json()['latency_ms']} ms)")

print("\nผลรวม:", "ตรงกันทุกรูป ✅" if all_ok else "มีรูปที่ไม่ตรง ❌ — ตรวจ imgsz / conf / เวอร์ชัน ultralytics")
sys.exit(0 if all_ok else 1)
