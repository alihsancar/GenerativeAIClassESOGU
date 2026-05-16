# Beyin Tümörü MR Sınıflandırması — ViT Ensemble

> Vision Transformer (ViT) mimarisi kullanarak MR görüntülerinden 4 sınıflı beyin tümörü sınıflandırması.  
> Ahmet Melih Gümüş · Ali İhsan Sancar — Eskişehir Osmangazi Üniversitesi, Bilgisayar Mühendisliği

---

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Ensemble Doğruluk | **%95.88** |
| F1 Skoru (weighted) | **%95.81** |
| ROC-AUC | **%99.31** |
| En iyi tek model (ViT-L/16@224) | %96.19 |

---

## Yöntem

- **4-Model Ensemble:** ViT-L/16@224 · ViT-B/16@384 · ViT-L/16@384 · ViT-L/16+GRU@384  
- **Kayıp Fonksiyonu:** Sınıf ağırlıklı Focal Loss (glioma α=2.5) + Label Smoothing ε=0.1  
- **Test-Time Augmentation:** 8 farklı dönüşüm  
- **XAI:** Attention Rollout + beyin maskesi  
- **Veri:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7.023 görüntü, 4 sınıf  

---

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADIN/brain-tumor-vit.git
cd brain-tumor-vit
pip install -r requirements.txt
```

Model ağırlıklarını (`*.pth`) `weights/` klasörüne koy.

---

## Web Arayüzü (Hekim Destek Sistemi)

```bash
python -m uvicorn app:app --reload --port 8000
```

Tarayıcıda `http://localhost:8000` adresini aç.

**Özellikler:**
- MR görüntüsü yükle → otomatik tahmin
- Attention Rollout ısı haritası
- PDF rapor indirme

---

## Proje Yapısı

```
brain-tumor-vit/
├── app.py                  # FastAPI web uygulaması
├── config/
│   └── settings.py         # Tüm sabitler
├── models/
│   ├── focal_loss.py       # FocalLoss sınıfı
│   ├── vit_gru.py          # ViT+GRU hibrit model
│   └── loader.py           # Model yükleme
├── preprocessing/
│   ├── transforms.py       # Auto-crop, TTA, augmentation
│   └── dataset.py          # Dataset sınıfları
├── inference/
│   ├── predictor.py        # Tahmin motoru
│   └── attention_rollout.py# XAI görselleştirme
├── utils/
│   ├── training.py         # Eğitim döngüsü
│   └── seed.py
├── templates/
│   └── index.html          # Web arayüzü
├── weights/                # .pth dosyaları (git e eklenmez)
└── requirements.txt
```

---

## Sınıf Başına Performans

| Sınıf | Recall | AUC |
|-------|--------|-----|
| Glioma | %85.2 | 0.974 |
| Meningioma | %99.0 | 0.998 |
| Tümörsüz | %100.0 | 1.000 |
| Hipofiz | %99.2 | 1.000 |
