#!/usr/bin/env bash
# Step 2.1-B2: ทดสอบ POST /predict ทั้ง 7 กรณี (API ต้องรันอยู่ที่ port 8000)
API=localhost:8000

echo "1) ภาพสะอาด"
curl -s -w " [%{http_code}]\n" -X POST $API/predict -F "file=@samples/leaf/common_rust_clean.jpg"

echo "2) ภาพ noisy ขอครบ 4 คลาส (k=4)"
curl -s -w " [%{http_code}]\n" -X POST "$API/predict?k=4" -F "file=@samples/leaf/gray_leaf_spot_noisy.jpg"

echo "3) ไฟล์ที่ไม่ใช่รูป"
curl -s -w " [%{http_code}]\n" -X POST $API/predict -F "file=@requirements.txt"

echo "4) ไฟล์ข้อความแต่หลอกว่าเป็น jpeg"
curl -s -w " [%{http_code}]\n" -X POST $API/predict -F "file=@requirements.txt;type=image/jpeg"

echo "5) ไฟล์ใหญ่เกิน 5 MB"
head -c 6000000 /dev/urandom > big.jpg
curl -s -w " [%{http_code}]\n" -X POST $API/predict -F "file=@big.jpg"; rm big.jpg

echo "6) k เกินจำนวนคลาส (k=5)"
curl -s -w " [%{http_code}]\n" -X POST "$API/predict?k=5" -F "file=@samples/leaf/healthy_clean.jpg"

echo "7) ไม่แนบไฟล์"
curl -s -w " [%{http_code}]\n" -X POST $API/predict
