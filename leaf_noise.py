# จำลองภาพใบไม้ที่ถ่ายด้วยมือถือในแปลงจริง ใช้ทั้งตอนเทรน ตอนวัดผล และตอนทดสอบ API
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def add_noise(img: Image.Image, strength: float, rng: np.random.Generator) -> Image.Image:
    # strength 0 = ภาพเดิม, 1 = ภาพแย่มาก · rng = np.random.default_rng(seed) เพื่อให้ทำซ้ำได้
    img = img.convert("RGB")
    if strength <= 0:
        return img
    img = ImageEnhance.Brightness(img).enhance(max(0.1, 1 + strength * rng.uniform(-0.55, 0.45)))  # มืด / จ้า
    img = ImageEnhance.Contrast(img).enhance(max(0.1, 1 - strength * rng.uniform(0.0, 0.5)))      # หมอก / ย้อนแสง
    img = img.filter(ImageFilter.GaussianBlur(strength * rng.uniform(0.5, 2.5)))          # โฟกัสพลาด / มือสั่น
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0, strength * rng.uniform(10, 40), a.shape)                         # noise จากเซนเซอร์
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(5, int(90 - strength * rng.uniform(40, 75))))  # บีบอัดตอนส่งผ่านแชต
    buf.seek(0)
    return Image.open(buf).convert("RGB")
