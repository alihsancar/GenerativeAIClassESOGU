"""
FastAPI ana uygulama — Brain Tumor ViT Web Sitesi
Çalıştırma: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import base64
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # GUI gerektirmez
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import (
    INFERENCE_MODE, CLASSES, IDX2CLS, NUM_CLASSES, DEVICE
)
from models.loader import load_model_for_web
from preprocessing.transforms import get_val_transform, auto_crop_brain
from inference.predictor import predict_single
from inference.attention_rollout import rollout_for_web

# ── PDF için reportlab ────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm as rl_cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  reportlab bulunamadı → pip install reportlab")

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Brain Tumor ViT", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ── Model startup'ta bir kez yüklenir ─────────────────────────────────────────
print(f"🔄 Model yükleniyor: {INFERENCE_MODE} ...")
_model_obj, _mode_key = load_model_for_web(INFERENCE_MODE)

# Rollout için output_attentions=True ile yeniden yükle
from models.loader import load_vit
_rollout_model = load_vit("l384", output_attentions=True)

_val_tf = get_val_transform(384)
print(f"✅ Model hazır — cihaz: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

LABEL_INFO = {
    "glioma": {
        "tr": "Glioma",
        "desc": "Beynin veya omuriliğin glial hücrelerinden kaynaklanan tümör türüdür. "
                "Sınırları belirsiz olabilir ve farklı kötücüllük derecelerine sahiptir.",
        "color": "#E63946",
        "risk": "Yüksek",
    },
    "meningioma": {
        "tr": "Meningioma",
        "desc": "Beyin ve omuriliği çevreleyen zarlardan (meninksler) kaynaklanan, "
                "genellikle iyi huylu tümördür.",
        "color": "#457B9D",
        "risk": "Orta",
    },
    "notumor": {
        "tr": "Tümör Yok",
        "desc": "Görüntüde anlamlı bir tümör kitlesi tespit edilmemiştir.",
        "color": "#2A9D8F",
        "risk": "Düşük",
    },
    "pituitary": {
        "tr": "Hipofiz Tümörü",
        "desc": "Hipofiz bezinden kaynaklanan, sella turcica bölgesinde "
                "yerleşim gösteren tümördür. Genellikle iyi huyludur.",
        "color": "#E9C46A",
        "risk": "Orta",
    },
}


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def ndarray_to_b64(arr: np.ndarray, cmap_name: str = None) -> str:
    """numpy [0,1] array → base64 PNG"""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.axis("off")
    if cmap_name:
        ax.imshow(arr, cmap=cmap_name, vmin=0, vmax=1)
    else:
        ax.imshow(arr.clip(0, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    MR görüntüsü alır → tahmin + Attention Rollout → JSON döner.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Lütfen geçerli bir görüntü dosyası yükleyin.")

    try:
        data    = await file.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")

        # ── Tahmin ──
        result = predict_single(
            _model_obj, pil_img, size=384, is_gru=False, use_tta=True
        )

        # ── Attention Rollout ──
        rollout = rollout_for_web(_rollout_model, pil_img, _val_tf)

        # ── Görselleri base64'e çevir ──
        cropped   = auto_crop_brain(pil_img.convert("RGB"))
        orig_b64  = pil_to_b64(cropped.resize((384, 384)))
        amap_b64  = ndarray_to_b64(rollout["amap"], cmap_name="jet")
        over_b64  = ndarray_to_b64(rollout["overlay"])

        label    = result["label"]
        info     = LABEL_INFO[label]

        return JSONResponse({
            "label"      : label,
            "label_tr"   : info["tr"],
            "confidence" : round(result["confidence"] * 100, 2),
            "probs"      : {k: round(v * 100, 2) for k, v in result["probs"].items()},
            "color"      : info["color"],
            "risk"       : info["risk"],
            "desc"       : info["desc"],
            "orig_b64"   : orig_b64,
            "amap_b64"   : amap_b64,
            "over_b64"   : over_b64,
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"İşlem hatası: {str(e)}")


@app.post("/report")
async def report(file: UploadFile = File(...)):
    """
    Tahmin + Rollout → PDF rapor döner.
    """
    if not PDF_AVAILABLE:
        raise HTTPException(501, "reportlab kurulu değil: pip install reportlab")

    data    = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")

    result  = predict_single(_model_obj, pil_img, size=384, is_gru=False, use_tta=True)
    rollout = rollout_for_web(_rollout_model, pil_img, _val_tf)

    label  = result["label"]
    info   = LABEL_INFO[label]
    now    = datetime.now().strftime("%d.%m.%Y %H:%M")

    pdf_buf = _build_pdf(pil_img, rollout, result, info, now, file.filename)
    return StreamingResponse(
        io.BytesIO(pdf_buf),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="BrainTumor_Rapor_{now[:10]}.pdf"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# PDF Oluşturucu
# ─────────────────────────────────────────────────────────────────────────────

def _register_turkish_fonts():
    """
    Sisteme göre Türkçe karakterleri destekleyen TTF font arar ve kaydeder.
    Bulamazsa None döner (fallback ASCII modu devreye girer).
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import platform

    candidates = []
    system = platform.system()

    if system == "Windows":
        winfonts = Path("C:/Windows/Fonts")
        candidates = [
            (winfonts / "arial.ttf",   winfonts / "arialbd.ttf"),
            (winfonts / "calibri.ttf", winfonts / "calibrib.ttf"),
            (winfonts / "tahoma.ttf",  winfonts / "tahomabd.ttf"),
            (winfonts / "verdana.ttf", winfonts / "verdanab.ttf"),
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            (Path("/Library/Fonts/Arial.ttf"),       Path("/Library/Fonts/Arial Bold.ttf")),
            (Path("/System/Library/Fonts/Helvetica.ttc"), Path("/System/Library/Fonts/Helvetica.ttc")),
        ]
    else:  # Linux
        candidates = [
            (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
             Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
            (Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
             Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")),
            (Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
             Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf")),
        ]

    for regular, bold in candidates:
        if regular.exists():
            try:
                pdfmetrics.registerFont(TTFont("TR_Regular", str(regular)))
                if bold.exists():
                    pdfmetrics.registerFont(TTFont("TR_Bold", str(bold)))
                else:
                    pdfmetrics.registerFont(TTFont("TR_Bold", str(regular)))
                return "TR_Regular", "TR_Bold"
            except Exception:
                continue

    return None, None   # font bulunamadı


def _tr(text: str) -> str:
    """Türkçe karakterleri ASCII eşdeğerleriyle değiştirir (font fallback için)."""
    table = str.maketrans("şğüçöıŞĞÜÇÖİ", "sgucoisGUCOI")
    return text.translate(table)


def _build_pdf(pil_img, rollout, result, info, now, filename) -> bytes:
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*rl_cm, rightMargin=2*rl_cm,
                                topMargin=2*rl_cm,  bottomMargin=2*rl_cm)
    styles = getSampleStyleSheet()
    W      = A4[0] - 4 * rl_cm   # kullanılabilir genişlik

    # ── Font seçimi ────────────────────────────────────────────────────────────
    fn_reg, fn_bold = _register_turkish_fonts()
    if fn_reg:
        # TTF bulundu → Türkçe karakterler direkt çalışır
        def t(s): return s
    else:
        # TTF bulunamadı → ASCII fallback
        fn_reg  = "Helvetica"
        fn_bold = "Helvetica-Bold"
        def t(s): return _tr(s)

    def style(name, **kw):
        kw.setdefault("fontName", fn_reg)
        s = ParagraphStyle(name, parent=styles["Normal"], **kw)
        return s

    title_style   = style("T", fontSize=18, fontName=fn_bold,
                           textColor=colors.HexColor("#1D3557"), spaceAfter=4, alignment=TA_CENTER)
    sub_style     = style("S", fontSize=10, textColor=colors.grey,
                           spaceAfter=12, alignment=TA_CENTER)
    section_style = style("SEC", fontSize=12, fontName=fn_bold,
                           textColor=colors.HexColor("#1D3557"), spaceBefore=14, spaceAfter=6)
    body_style    = style("B", fontSize=10, leading=15, spaceAfter=8)
    warn_style    = style("W", fontSize=10, leading=14, spaceAfter=8,
                           textColor=colors.HexColor("#888888"), borderPadding=6)

    label_color = colors.HexColor(info["color"])

    # ── Görselleri hazırla ──
    def pil_to_rli(img_pil, w_cm):
        b = io.BytesIO(); img_pil.save(b, "PNG"); b.seek(0)
        return RLImage(b, width=w_cm*rl_cm, height=w_cm*rl_cm)

    orig_pil = auto_crop_brain(pil_img).resize((384, 384))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(rollout["amap"], cmap="jet", vmin=0, vmax=1)
    ax.axis("off")
    amap_buf = io.BytesIO(); fig.savefig(amap_buf, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig); amap_buf.seek(0)
    amap_pil = Image.open(amap_buf)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(rollout["overlay"].clip(0, 1))
    ax.axis("off")
    over_buf = io.BytesIO(); fig.savefig(over_buf, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig); over_buf.seek(0)
    over_pil = Image.open(over_buf)

    # ── Olasılık bar grafiği ──
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=120)
    cls_labels = [LABEL_INFO[c]["tr"] for c in CLASSES]
    values     = [result["probs"][c] * 100 for c in CLASSES]
    bar_colors = [LABEL_INFO[c]["color"] for c in CLASSES]
    bars = ax.barh(cls_labels, values, color=bar_colors, edgecolor="white", height=0.5)
    for bar, v in zip(bars, values):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                f"%{v:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, 110); ax.set_xlabel(t("Olasilik (%)"))
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title(t("Sinif Olasiliklari"), fontsize=11, fontweight="bold")
    plt.tight_layout()
    prob_buf = io.BytesIO(); fig.savefig(prob_buf, format="PNG", bbox_inches="tight")
    plt.close(fig); prob_buf.seek(0)
    prob_pil = Image.open(prob_buf)

    img_w = 5.5   # cm

    story = [
        Paragraph(t("Beyin Tumoru MR Analiz Raporu"), title_style),
        Paragraph(t(f"Olusturulma: {now}  |  Dosya: {filename}"), sub_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1D3557"), spaceAfter=14),

        Paragraph(t("Tani Sonucu"), section_style),
        Table(
            [[Paragraph(f"<b>{t(info['tr']).upper()}</b>", style("LBL", fontSize=16,
                        fontName=fn_bold, textColor=colors.white, alignment=TA_CENTER)),
              Paragraph(f"<b>{t('Guven')}: %{result['confidence']*100:.1f}</b><br/>"
                        f"{t('Risk Seviyesi')}: {t(info['risk'])}", style("CONF", fontSize=11,
                        fontName=fn_bold, leading=16))]],
            colWidths=[W * 0.38, W * 0.62],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), label_color),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F1FAEE")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ccc")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ])
        ),
        Spacer(1, 10),
        Paragraph(t(info["desc"]), body_style),

        Paragraph(t("MR Goruntu Analizi"), section_style),
        Table(
            [[pil_to_rli(orig_pil, img_w), pil_to_rli(amap_pil, img_w), pil_to_rli(over_pil, img_w)],
             [Paragraph(t("Orijinal MR"), style("CAP", alignment=TA_CENTER, fontSize=9)),
              Paragraph("Attention Rollout", style("CAP", alignment=TA_CENTER, fontSize=9)),
              Paragraph("Overlay", style("CAP", alignment=TA_CENTER, fontSize=9))]],
            colWidths=[W/3, W/3, W/3],
            style=TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ])
        ),
        Spacer(1, 10),

        Paragraph(t("Sinif Olasiliklari"), section_style),
        Table(
            [[RLImage(prob_buf, width=W*0.72, height=W*0.36)]],
            colWidths=[W],
            style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])
        ),
        Spacer(1, 14),

        Paragraph(t("Model Bilgisi"), section_style),
        Table(
            [["Model",            "ViT-L/16 @ 384 (Vision Transformer Large)"],
             [t("Dogruluk"),      "%96.00 (Test Seti)"],
             ["XAI " + t("Yontemi"), "Attention Rollout (discard=0.95)"],
             [t("Veri Kumesi"),   t("Kaggle Brain Tumor MRI Dataset (7.023 goruntu)")],
             [t("Siniflar"),      t("Glioma, Meningioma, Tumorsuz, Hipofiz")]],
            colWidths=[W * 0.32, W * 0.68],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1FAEE")),
                ("FONTNAME", (0, 0), (0, -1), fn_bold),
                ("FONTNAME", (1, 0), (1, -1), fn_reg),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccc")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#eee")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        ),
        Spacer(1, 16),
        HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=8),
        Paragraph(
            t("Bu rapor yalnizca arastirma amaclıdır ve klinik tani yerine gecmez. "
              "Kesin tani icin uzman bir radyolog veya norolog tarafindan degerlendirilmelidir."),
            warn_style
        ),
    ]

    doc.build(story)
    return buf.getvalue()
