"""BD532 Lab 2.1 — Corn Leaf Disease API (ฉบับสมบูรณ์)

รัน:  uvicorn leaf_app:app --reload --port 8000
"""
import io
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # ห้าม ultralytics pip install เองตอนรันจริง

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# ---------- 1) โหลดโมเดลครั้งเดียวตอน start server ----------
DEFAULT_MODEL = "models/leaf_best.pt"                    # โมเดล B (เทรนด้วย noisy input)
MODEL_PATH = os.getenv("LEAF_MODEL") or DEFAULT_MODEL      # LEAF_MODEL=models/leaf_clean.pt → โมเดล A
META = json.loads(Path("models/leaf_metadata.json").read_text(encoding="utf-8"))
if MODEL_PATH != DEFAULT_MODEL:
    META = {**META, "model_version": Path(MODEL_PATH).stem, "training_noise": None}
model = YOLO(MODEL_PATH, task="classify")
model_lock = threading.Lock()          # โมเดลตัวเดียวถูกหลาย request ใช้พร้อมกัน → ล็อกไว้
LOW_CONF = META["low_confidence_below"]  # prob ต่ำกว่านี้ → แนะนำให้ผู้ใช้ถ่ายใหม่
N_CLASSES = len(META["names"])

MAX_BYTES = 5 * 1024 * 1024            # 5 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="BD532 Corn Leaf Disease API",
    description="YOLO26n-cls ทำนายโรคใบข้าวโพด 4 คลาส (เทรนด้วยภาพสะอาด + noisy input)",
    version=META["model_version"],
)


# ---------- 2) สัญญา (contract) ของ response ----------
class TopK(BaseModel):
    label: str
    prob: float


class LeafPrediction(BaseModel):
    label: str
    prob: float
    low_confidence: bool        # True → ขอให้ผู้ใช้ถ่ายใหม่ให้ชัดขึ้น
    top_k: list[TopK]
    model_version: str
    latency_ms: float


# ---------- 3) helper ----------
def read_image(upload: UploadFile) -> Image.Image:
    """ตรวจไฟล์ที่อัปโหลด แล้วแปลงเป็น PIL RGB"""
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"รองรับเฉพาะ {sorted(ALLOWED_TYPES)} แต่ได้ {upload.content_type}")
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "ไฟล์ใหญ่เกิน 5 MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:   # ไฟล์เสีย / ไม่ใช่รูป / format ที่ไม่รองรับ
        raise HTTPException(400, "อ่านไฟล์รูปไม่ได้ (ไฟล์เสียหรือไม่ใช่รูป)")


# ---------- 4) endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return META


@app.post("/predict", response_model=LeafPrediction)
def predict(file: UploadFile = File(..., description="ไฟล์รูป jpg/png/webp"),
            k: int = Query(3, ge=1, le=N_CLASSES, description="จำนวนคำตอบอันดับต้น ๆ")):
    img = read_image(file)

    t0 = time.perf_counter()
    with model_lock:
        r = model.predict(img, imgsz=META["imgsz"], verbose=False)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = r.probs.data.tolist()
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    top_k = [TopK(label=r.names[i], prob=round(probs[i], 4)) for i in order]

    return LeafPrediction(
        label=top_k[0].label,
        prob=top_k[0].prob,
        low_confidence=top_k[0].prob < LOW_CONF,
        top_k=top_k,
        model_version=META["model_version"],
        latency_ms=round(latency_ms, 1),
    )
