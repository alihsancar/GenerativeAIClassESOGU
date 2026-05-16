@echo off
echo.
echo  NöroVizyon - Beyin Tümörü MR Analiz Sistemi
echo  =============================================
echo.

REM Eksik paketleri kur
echo [1/2] Gerekli paketler kontrol ediliyor...
pip install fastapi uvicorn[standard] python-multipart reportlab -q

echo.
echo [2/2] Sunucu başlatılıyor...
echo  Tarayıcıda açın: http://localhost:8000
echo  Durdurmak için:  CTRL+C
echo.

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
