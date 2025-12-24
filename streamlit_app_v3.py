import re
import pickle
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Şikayet Analiz Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# DARK MODE CSS - Daha Okunabilir Renkler
# =========================================================
st.markdown("""
<style>
    /* Ana tema - Dark Mode (Daha Açık) */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0;
    }
    
    /* Input alanları - NORMAL BOYUT */
    .stTextInput > div > div > input {
        background-color: #2a2a3e;
        color: #ffffff;
        border: 2px solid #4a5568;
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #2a2a3e;
        color: #ffffff;
        border: 2px solid #4a5568;
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
        outline: none;
    }
    
    /* Butonlar - BÜYÜK */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-size: 1.3rem;
        font-weight: 700;
        width: 100%;
        height: 60px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.7);
    }
    
    /* Metrikler - BÜYÜK ve OKUNABİLİR */
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8b8d1;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1.3rem;
    }
    
    /* Sonuçlar bölümünde genel yazı boyutu */
    .element-container {
        font-size: 1.1rem;
    }
    
    /* Başlıklar - BÜYÜK ve OKUNABİLİR */
    h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
    }
    
    /* Sekmeler - BÜYÜK VE YUKARI SĞDIR */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.5rem 0;
        margin-top: -1rem;
        border-bottom: 3px solid #667eea;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1.2rem 2.5rem;
        font-size: 1.8rem;
        font-weight: 700;
        color: #b8b8d1;
        background: transparent;
        border: none;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
        min-width: 250px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(102, 126, 234, 0.2);
        color: #ffffff;
        font-size: 2rem;
        border-bottom: 4px solid #667eea;
    }
    
    .stTabs [aria-selected="false"] {
        color: #b8b8d1;
    }
    
    h2 {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h4 {
        color: #ffffff !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Label'lar - BÜYÜK */
    label {
        color: #ffffff !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Tablo */
    .dataframe {
        background-color: #2a2a3e;
        color: #ffffff;
        font-size: 1rem;
    }
    
    /* Genel metin */
    p, div, span {
        color: #e0e0e0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Kartlar */
    .main-container {
        background: rgba(42, 42, 62, 0.8);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Churn Band Renkleri - Daha Parlak */
    .churn-mor {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
        margin: 1rem 0;
    }
    
    .churn-kirmizi {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5);
        margin: 1rem 0;
    }
    
    .churn-sari {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.5);
        margin: 1rem 0;
    }
    
    .churn-yesil {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        margin: 1rem 0;
    }
    
    /* Info/Alert kutuları */
    .stAlert {
        background-color: #2a2a3e;
        border-left: 4px solid #667eea;
        color: #e0e0e0;
        font-size: 1.1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Sidebar gizle */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Tabs - Dark Mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(42, 42, 62, 0.5);
        border-radius: 10px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(42, 42, 62, 0.5);
        color: #b8b8d1;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL YÜKLEME (CACHE)
# =========================================================
@st.cache_resource
def load_models():
    """Model ve tokenizer'ı yükle"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "bert_based_classification_models")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()
    
    emb_model = AutoModel.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()
    
    return tokenizer, clf_model, emb_model, device

@st.cache_data
def load_data():
    """Veri setini yükle"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base_path, "df_weigthed_final.pkl")
    
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    
    return df

@st.cache_resource
def load_summary_models():
    """Özet ve duygu analizi modellerini yükle - Çalışan dosyadaki basit yaklaşım"""
    from transformers import pipeline
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Kontrol edilecek yollar (öncelik sırasına göre)
    possible_paths = [
        os.path.join(base_path, "yerel_modeller"),  # Proje klasöründe
        os.path.join(os.path.expanduser("~"), "Desktop", "yerel_modeller"),  # Masaüstünde
        os.path.join(os.path.expanduser("~"), "PycharmProjects", "PythonProject", "duygu_ozet", "yerel_modeller"),  # PycharmProjects'te
        r"C:\Users\smt20\Desktop\yerel_modeller",  # Doğrudan yol
    ]
    
    yerel_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ozet_path = os.path.join(path, "ozetleme_modeli")
            duygu_path = os.path.join(path, "duygu_modeli")
            if os.path.exists(ozet_path) and os.path.exists(duygu_path):
                yerel_path = path
                break
    
    if yerel_path is None:
        return None, None
    
    ozet_path = os.path.join(yerel_path, "ozetleme_modeli")
    duygu_path = os.path.join(yerel_path, "duygu_modeli")
    
    device_id = 0 if torch.cuda.is_available() else -1
    
    try:
        # Çalışan dosyadaki basit yaklaşım: Direkt pipeline kullan, tokenizer'ı manuel yükleme
        # Pipeline otomatik olarak tokenizer'ı yükleyecek
        summarizer = pipeline(
            "summarization",
            model=ozet_path,
            tokenizer=ozet_path,
            device=device_id
        )
        
        classifier = pipeline(
            "zero-shot-classification",
            model=duygu_path,
            tokenizer=duygu_path,
            device=device_id
        )
        
        return summarizer, classifier
    except Exception as e:
        # Hata mesajını exception olarak fırlat (show_summary_analysis'te yakalanacak)
        import traceback
        error_detail = traceback.format_exc()
        raise Exception(f"Model yükleme hatası:\n{str(e)}\n\nDetay:\n{error_detail}")

# =========================================================
# SABİTLER
# =========================================================
LABEL_NAMES = [
    "fiyat farkı talebi",
    "garanti sorunu",
    "iade reddi",
    "iade süreci tamamlanmamış",
    "kargo teslimat",
    "satıcı sipariş iptali",
    "teslim edilmeyen paket",
    "uygulama",
    "yanlış veya eksik ürün gönderimi",
    "ürün ile ilgili sorunlar"
]

# CHURN SİNYAL RİSKLERİ (CATEGORY_WEIGHTS)
CATEGORY_WEIGHTS = {
    "1. Kesin Kopuş": 1.00,
    "7. Yasal Tehdit": 0.95,
    "3. Çözümsüzlük & Güven Kaybı": 0.85,
    "2. Duygusal Kopuş": 0.75,
    "5. Sabır Tükenişi": 0.70,
    "6. Tekrarlayan Problem": 0.65,
    "4. Mağduriyet": 0.60,
    "8. İlk Kez Sorun": 0.30
}

# Eski isimle uyumluluk için
CHURN_SIGNAL_RISK = CATEGORY_WEIGHTS

KEYWORDS = [
    # 1️⃣ KESİN KOPUŞ
    ("1. Kesin Kopuş", "bir daha asla"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "bir daha alışveriş"),
    ("1. Kesin Kopuş", "alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "alışveriş yapmayı düşünmüyorum"),
    ("1. Kesin Kopuş", "güvenerek alışveriş yaptım"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "bir daha asla alışveriş"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayı"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayı düşünmüyorum"),
    ("1. Kesin Kopuş", "bir daha asla alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "güvenerek alışveriş yaptım"),
    ("1. Kesin Kopuş", "alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "bir daha"),
    ("1. Kesin Kopuş", "bir daha alışveriş"),
    ("1. Kesin Kopuş", "bir daha hepsiburada"),
    
    # 2️⃣ DUYGUSAL KOPUŞ
    ("2. Duygusal Kopuş", "hayal"),
    ("2. Duygusal Kopuş", "hayal kırıklığı"),
    ("2. Duygusal Kopuş", "hayal kırıklığına"),
    ("2. Duygusal Kopuş", "bir hayal kırıklığı"),
    ("2. Duygusal Kopuş", "pişman oldum"),
    ("2. Duygusal Kopuş", "güvenerek alışveriş yaptım"),
    ("2. Duygusal Kopuş", "büyük bir hayal"),
    ("2. Duygusal Kopuş", "hayal kırıklığı hepsiburada"),
    ("2. Duygusal Kopuş", "hayal kırıklığı yarattı"),
    ("2. Duygusal Kopuş", "beni hayal kırıklığına"),
    ("2. Duygusal Kopuş", "beni hayal kırıklığına uğrattı"),
    ("2. Duygusal Kopuş", "büyük bir hayal kırıklığı yaşadım"),
    ("2. Duygusal Kopuş", "dalga geçer gibi"),
    
    # 3️⃣ ÇÖZÜMSÜZLÜK & GÜVEN KAYBI
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş"),
    ("3. Çözümsüzlük & Güven Kaybı", "bir çözüm"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "herhangi bir çözüm"),
    ("3. Çözümsüzlük & Güven Kaybı", "bir çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş yapılmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm bekliyorum"),
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş yapılmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "ulaşamıyorum"),
    ("3. Çözümsüzlük & Güven Kaybı", "bilgi verilmedi"),
    ("3. Çözümsüzlük & Güven Kaybı", "herhangi bir çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "henüz bir çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "sonuç alamadım"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm sunulmuyor"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm yok"),
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş alamadım"),
    
    # 4️⃣ MAĞDURİYET
    ("4. Mağduriyet", "mağdur"),
    ("4. Mağduriyet", "mağduriyet"),
    ("4. Mağduriyet", "mağduriyetim"),
    ("4. Mağduriyet", "mağduriyetimin"),
    ("4. Mağduriyet", "mağdur oldum"),
    ("4. Mağduriyet", "mağduriyet yaşıyorum"),
    ("4. Mağduriyet", "yaşadığım mağduriyet"),
    ("4. Mağduriyet", "mağduriyetimin giderilmesini"),
    ("4. Mağduriyet", "mağduriyetimin giderilmesini"),
    ("4. Mağduriyet", "yaşadığım mağduriyet"),
    ("4. Mağduriyet", "mağduriyetim devam ediyor"),
    ("4. Mağduriyet", "ve mağduriyetimin giderilmesini talep ediyorum"),
    ("4. Mağduriyet", "mağdur edildim"),
    ("4. Mağduriyet", "mağduriyet yaşıyorum"),
    
    # 5️⃣ SABIR TÜKENİŞİ
    ("5. Sabır Tükenişi", "defalarca"),
    ("5. Sabır Tükenişi", "pişman"),
    ("5. Sabır Tükenişi", "defalarca aramama rağmen"),
    ("5. Sabır Tükenişi", "sürekli"),
    ("5. Sabır Tükenişi", "en kısa sürede"),
    ("5. Sabır Tükenişi", "hala"),
    ("5. Sabır Tükenişi", "halen"),
    ("5. Sabır Tükenişi", "aynı sorun"),
    ("5. Sabır Tükenişi", "sorun devam ediyor"),
    ("5. Sabır Tükenişi", "artık"),
    ("5. Sabır Tükenişi", "acilen"),
    ("5. Sabır Tükenişi", "en kısa sürede giderilmesini bekliyorum"),
    
    # 6️⃣ TEKRARLAYAN PROBLEM
    ("6. Tekrarlayan Problem", "benzer sorunlar"),
    ("6. Tekrarlayan Problem", "benzer bir sorun"),
    ("6. Tekrarlayan Problem", "benzer sorunlar yaşadım"),
    ("6. Tekrarlayan Problem", "benzer bir sorun yaşadım"),
    ("6. Tekrarlayan Problem", "benzer bir sorun yaşamıştım"),
    ("6. Tekrarlayan Problem", "benzer sorunların tekrar"),
    ("6. Tekrarlayan Problem", "benzer sorunların tekrar yaşanmaması"),
    ("6. Tekrarlayan Problem", "benzer durumların tekrar yaşanmaması"),
    ("6. Tekrarlayan Problem", "daha önce de benzer"),
    ("6. Tekrarlayan Problem", "önce de benzer bir sorun"),
    
    # 7️⃣ YASAL TEHDİT
    ("7. Yasal Tehdit", "tüketici hakem"),
    ("7. Yasal Tehdit", "tüketici hakem heyeti"),
    ("7. Yasal Tehdit", "hakem heyeti"),
    ("7. Yasal Tehdit", "hukuki"),
    ("7. Yasal Tehdit", "cimer"),
    ("7. Yasal Tehdit", "yasal haklarımı"),
    ("7. Yasal Tehdit", "tüketici hakları"),
    
    # 8️⃣ İLK KEZ SORUN
    ("8. İlk Kez Sorun", "ilk kez böyle bir sorun"),
    ("8. İlk Kez Sorun", "ilk kez başıma geliyor"),
    ("8. İlk Kez Sorun", "ilk kez böyle bir durum"),
    ("8. İlk Kez Sorun", "ilk kez böyle bir sorunla"),
    ("8. İlk Kez Sorun", "ilk kez böyle bir durumla"),
    ("8. İlk Kez Sorun", "ilk kez"),
]

ALT_KATEGORI_WEIGHTS = {
    "teslim edilmeyen paket": 1.00,
    "yanlış veya eksik ürün gönderimi": 0.90,
    "kargo teslimat": 0.80,
    "satıcı sipariş iptali": 0.75,
    "iade süreci tamamlanmamış": 0.65,
    "ürün ile ilgili sorunlar": 0.60,
    "uygulama": 0.55,
    "iade reddi": 0.40,
    "garanti sorunu": 0.40,
    "fiyat farkı talebi": 0.20
}

# Eski isimle uyumluluk için
ALT_KATEGORI_RISK = ALT_KATEGORI_WEIGHTS

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def clean_reviews_tr(text):
    """Türkçe metin temizleme"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.replace("İ", "i").replace("I", "ı").lower()
    
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(?:https?|www)\S+\b", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-zçğıöşü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def length_score(tokens):
    """Uzunluk skoru hesapla"""
    if tokens < 20:
        return 0
    elif tokens < 50:
        return 8
    elif tokens < 100:
        return 15
    elif tokens < 200:
        return 20
    else:
        return 25

def churn_signal_score_and_report(temiz_metin):
    """Churn sinyal skoru ve raporu hesapla"""
    active = {}
    
    # Hangi churn sinyalleri var?
    for cat, phrase in KEYWORDS:
        if re.search(rf"\b{re.escape(phrase)}\b", temiz_metin):
            active[cat] = CATEGORY_WEIGHTS[cat]
    
    if not active:
        return 0, []
    
    # En güçlü 2 sinyal (skor için)
    sorted_active = sorted(active.items(), key=lambda x: x[1], reverse=True)
    
    churn_signal_score = sorted_active[0][1] * 30
    if len(sorted_active) > 1:
        churn_signal_score += sorted_active[1][1] * 15
    
    # Tüm aktif sinyallerin listesi - CATEGORY_WEIGHTS'a göre sıralı (yüksekten düşüğe)
    all_signals = [cat for cat, _ in sorted(active.items(), key=lambda x: CATEGORY_WEIGHTS.get(x[0], 0), reverse=True)]
    
    return churn_signal_score, all_signals

def alt_kategori_score(alt_kategori):
    """Alt kategori skoru hesapla"""
    if not alt_kategori:
        return 0
    return ALT_KATEGORI_WEIGHTS.get(alt_kategori, 0) * 20

def churn_band(score):
    """Churn band belirle"""
    if score >= 70:
        return "MOR"
    elif score >= 50:
        return "KIRMIZI"
    elif score >= 35:
        return "SARI"
    else:
        return "YEŞİL"

def get_churn_color(band):
    """Churn band rengi - CANLI TONLAR"""
    colors = {
        "MOR": "#9333ea",  # Kritik Riskli (Mor)
        "KIRMIZI": "#dc2626",  # Yüksek Riskli (Kırmızı)
        "SARI": "#facc15",  # Orta Riskli (Sarı)
        "YEŞİL": "#22c55e"  # Düşük Riskli (Yeşil)
    }
    return colors.get(band, "#667eea")

def get_churn_label(band):
    """Churn band görsel label"""
    labels = {
        "MOR": "Kritik Riskli (MOR)",
        "KIRMIZI": "Yüksek Riskli (KIRMIZI)",
        "SARI": "Orta Riskli (SARI)",
        "YEŞİL": "Düşük Riskli (YEŞİL)"
    }
    return labels.get(band, band)

def remove_category_number(category):
    """Kategori isminden sayıyı kaldır (örn: '5. Sabır Tükenişi' -> 'Sabır Tükenişi')"""
    # Sayı ve nokta ile başlayan kısmı kaldır
    import re
    return re.sub(r'^\d+\.\s*', '', category).strip()

def get_category_icon(category_name):
    """Kategori için uygun ikon döndür"""
    icons = {
        "Kesin Kopuş": "🚫",
        "Duygusal Kopuş": "💔",
        "Çözümsüzlük & Güven Kaybı": "❓",
        "Mağduriyet": "😔",
        "Sabır Tükenişi": "😤",
        "Tekrarlayan Problem": "🔄",
        "Yasal Tehdit": "⚖️",
        "İlk Kez Sorun": "🆕"
    }
    return icons.get(category_name, "📌")

def get_responsible_unit(alt_kategori):
    """Alt kategori için sorumlu birim döndür"""
    unit_mapping = {
        "ürün ile ilgili sorunlar": "Ürün & Kalite Sorunları",
        "iade süreci tamamlanmamış": "Finans & İade İşlemleri",
        "iade reddi": "Finans & İade İşlemleri",
        "kargo teslimat": "Lojistik & Teslimat",
        "teslim edilmeyen paket": "Lojistik & Teslimat",
        "fiyat farkı talebi": "Finans & İade İşlemleri",
        "yanlış veya eksik ürün gönderimi": "Ürün & Kalite Sorunları",
        "satıcı sipariş iptali": "Sistem & Sipariş Yönetimi",
        "uygulama": "Sistem & Sipariş Yönetimi",
        "garanti sorunu": "Ürün & Kalite Sorunları"
    }
    return unit_mapping.get(alt_kategori, "Genel")

# =========================================================
# ANA TAHMİN FONKSİYONU
# =========================================================
def predict_complaint(baslik, sikayet_metni, df, tokenizer, clf_model, emb_model, device, top_k_similar=5):
    """Şikayet analizi yap"""
    # Başlık boşsa sadece metin kullan
    if baslik and baslik.strip():
        full_text = f"{baslik} {sikayet_metni}"
    else:
        full_text = sikayet_metni
    
    # 1. ALT KATEGORİ (BERT)
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        logits = clf_model(**inputs).logits
    
    probs = F.softmax(logits, dim=1)[0]
    top_idx = torch.argmax(probs).item()
    
    alt_kategori = LABEL_NAMES[top_idx]
    olasilik = round(probs[top_idx].item() * 100, 2)
    
    # Tüm kategorilerin olasılıklarını al
    all_probs = {LABEL_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(len(LABEL_NAMES))}
    
    # 2. CHURN SCORE
    temiz_metin = clean_reviews_tr(full_text)
    token_len = len(temiz_metin.split())
    
    # Alt kategori skoru
    alt_score = ALT_KATEGORI_WEIGHTS.get(alt_kategori, 0) * 20
    
    # Churn sinyal skoru
    active = {}
    for cat, phrase in KEYWORDS:
        if re.search(rf"\b{re.escape(phrase)}\b", temiz_metin):
            active[cat] = CATEGORY_WEIGHTS[cat]
    
    if active:
        sorted_active = sorted(active.items(), key=lambda x: x[1], reverse=True)
        churn_signal_score = sorted_active[0][1] * 30
        if len(sorted_active) > 1:
            churn_signal_score += sorted_active[1][1] * 15
    else:
        churn_signal_score = 0
    
    # Length skoru
    length_score_value = length_score(token_len)
    
    # Toplam churn score
    churn_score = churn_signal_score + alt_score + length_score_value
    
    # Aktif sinyaller listesi - CATEGORY_WEIGHTS'a göre sıralı (yüksekten düşüğe)
    if active:
        triggered = [cat for cat, _ in sorted(active.items(), key=lambda x: CATEGORY_WEIGHTS.get(x[0], 0), reverse=True)]
    else:
        triggered = []
    
    churn_band_value = churn_band(churn_score)
    
    # 3. EN BENZER 5 ŞİKAYET
    def get_embedding(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            return emb_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    
    query_emb = get_embedding(full_text)
    corpus_emb = np.vstack(df["embedding"].values)
    
    sims = cosine_similarity(query_emb, corpus_emb)[0]
    top_idx = np.argsort(sims)[::-1][:top_k_similar]
    
    similarity_df = df.iloc[top_idx][
        ["tarih_saat", "kullanici", "baslik", "sikayet_metni"]
    ].copy()
    similarity_df["benzerlik_skoru"] = [round(sims[i], 4) for i in top_idx]
    similarity_df = similarity_df.reset_index(drop=True)
    
    return {
        "alt_kategori": alt_kategori,
        "olasilik": olasilik,
        "all_probs": all_probs,
        "churn_score": round(churn_score, 2),
        "churn_band": churn_band_value,
        "churn_signal_score": round(churn_signal_score, 2),
        "length_score": length_score_value,
        "alt_kategori_score": round(alt_score, 2),
        "triggered_categories": triggered,
        "similar_complaints": similarity_df,
        "token_len": token_len
    }

# =========================================================
# DASHBOARD FONKSİYONU
# =========================================================
def show_dashboard(df):
    """Dashboard - KPI Kartları, Kategori Dağılımları, Grafikler"""
    
    # Dark mode CSS - Filtreler dahil
    st.markdown("""
    <style>
    .stSelectbox label { color: #fff !important; font-weight: 600 !important; font-size: 1rem !important; }
    div[data-baseweb="select"] > div { background-color: #2a2a3e !important; color: #fff !important; border: 2px solid #667eea !important; }
    div[data-baseweb="select"] span { color: #fff !important; font-weight: 500 !important; }
    div[data-baseweb="select"] svg { fill: #fff !important; }
    [data-baseweb="popover"] { background-color: #2a2a3e !important; }
    [data-baseweb="popover"] li { color: #fff !important; background-color: #2a2a3e !important; }
    [data-baseweb="popover"] li:hover { background-color: #3a3a5e !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📊 Dashboard")
    
    # Filtreler - Sıra: Ana Kategori, Alt Kategori, Churn Band
    st.markdown("### 🎛️ Filtreler")
    f1, f2, f3 = st.columns(3)
    
    df_copy = df.copy()
    
    # 1. Ana Kategori
    with f1:
        ana_kats = ["Tümü"] + sorted(df_copy['Ana_Kategori'].dropna().unique().tolist()) if 'Ana_Kategori' in df_copy.columns else ["Tümü"]
        sel_ana = st.selectbox("Ana Kategori", ana_kats)
    
    # 2. Alt Kategori - Ana kategori seçilirse sadece ona ait alt kategoriler
    with f2:
        if sel_ana != "Tümü" and 'Ana_Kategori' in df_copy.columns and 'Alt_Kategori' in df_copy.columns:
            # Seçili ana kategoriye ait alt kategoriler
            filtered_df = df_copy[df_copy['Ana_Kategori'] == sel_ana]
            alt_kats = ["Tümü"] + sorted(filtered_df['Alt_Kategori'].dropna().unique().tolist())
        else:
            # Tüm alt kategoriler
            alt_kats = ["Tümü"] + sorted(df_copy['Alt_Kategori'].dropna().unique().tolist()) if 'Alt_Kategori' in df_copy.columns else ["Tümü"]
        sel_alt = st.selectbox("Alt Kategori", alt_kats)
    
    # 3. Churn Band
    with f3:
        bands = ["Tümü"] + df_copy['churn_band'].dropna().unique().tolist() if 'churn_band' in df_copy.columns else ["Tümü"]
        sel_band = st.selectbox("Churn Band", bands)
    
    # Filtreleme
    fdf = df_copy.copy()
    if sel_ana != "Tümü":
        fdf = fdf[fdf['Ana_Kategori'] == sel_ana]
    if sel_alt != "Tümü":
        fdf = fdf[fdf['Alt_Kategori'] == sel_alt]
    if sel_band != "Tümü":
        fdf = fdf[fdf['churn_band'] == sel_band]
    
    n = len(fdf)
    
    # CANLI RENKLER
    colors = {'MOR': '#9333ea', 'KIRMIZI': '#dc2626', 'SARI': '#facc15', 'YEŞİL': '#22c55e'}
    
    # Churn Band hesapla
    mor = (fdf['churn_band'] == 'MOR').sum() if 'churn_band' in fdf.columns else 0
    kirmizi = (fdf['churn_band'] == 'KIRMIZI').sum() if 'churn_band' in fdf.columns else 0
    sari = (fdf['churn_band'] == 'SARI').sum() if 'churn_band' in fdf.columns else 0
    yesil = (fdf['churn_band'] == 'YEŞİL').sum() if 'churn_band' in fdf.columns else 0
    avg_score = fdf['churn_score'].mean() if 'churn_score' in fdf.columns and n > 0 else 0
    
    # ═══════════════════════════════════════════════════════════════
    # KPI KARTLARI - FİLTRELERİN ALTINDA (ÇERÇEVE İLE, EŞİT BOYUT)
    # ═══════════════════════════════════════════════════════════════
    high_risk = mor + kirmizi
    high_pct = (high_risk/n*100) if n > 0 else 0
    
    kpi_style = """
    <div style="background: #1a1a2e; border: 2px solid {border}; border-radius: 12px; padding: 1rem; text-align: center; min-height: 120px; display: flex; flex-direction: column; justify-content: center; overflow: hidden; word-wrap: break-word;">
        <p style="color: #888; margin: 0; font-size: 0.85rem; white-space: nowrap;">{icon} {label}</p>
        <h2 style="color: {color}; margin: 0.2rem 0; font-size: 1.8rem; font-weight: 700; line-height: 1.2; overflow: hidden; text-overflow: ellipsis;">{value}</h2>
        <p style="color: #666; margin: 0; font-size: 0.8rem; white-space: nowrap;">{sub}</p>
    </div>
    """
    
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(kpi_style.format(border='#667eea', icon='📊', label='Toplam', color='#fff', value=f'{n:,}', sub='Şikayet'), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_style.format(border='#f59e0b', icon='📉', label='Ort. Skor', color='#f59e0b', value=f'{avg_score:.1f}', sub='Churn'), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_style.format(border='#ef4444', icon='🚨', label='Yüksek Risk', color='#ef4444', value=f'{high_risk:,}', sub=f'%{high_pct:.1f}'), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_style.format(border=colors['MOR'], icon='🟣', label='Kritik Riskli (MOR)', color=colors['MOR'], value=f'{mor:,}', sub=f'%{(mor/n*100) if n > 0 else 0:.1f}'), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_style.format(border=colors['KIRMIZI'], icon='🔴', label='Yüksek Riskli (KIRMIZI)', color=colors['KIRMIZI'], value=f'{kirmizi:,}', sub=f'%{(kirmizi/n*100) if n > 0 else 0:.1f}'), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # CHURN BAND + BİRİM TEK KPI KARTI
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🎨 Churn Band & Birim Dağılımı")
    
    mor_pct = (mor/n*100) if n > 0 else 0
    kirmizi_pct = (kirmizi/n*100) if n > 0 else 0
    sari_pct = (sari/n*100) if n > 0 else 0
    yesil_pct = (yesil/n*100) if n > 0 else 0
    
    col_band, col_birim = st.columns(2)
    
    # Churn Band Kartı
    with col_band:
        st.markdown("#### 🎯 Churn Band")
        fig_band = go.Figure(data=[go.Pie(
            labels=['Kritik Riskli (MOR)', 'Yüksek Riskli (KIRMIZI)', 'Orta Riskli (SARI)', 'Düşük Riskli (YEŞİL)'],
            values=[mor, kirmizi, sari, yesil],
            hole=0.6,
            marker=dict(colors=[colors['MOR'], colors['KIRMIZI'], colors['SARI'], colors['YEŞİL']]),
            textinfo='label+value+percent',
            textfont=dict(size=13, color='#fff', family='Arial Black'),
            textposition='outside',
            pull=[0.05, 0.02, 0, 0]
        )])
        fig_band.add_annotation(
            text=f"<b>{n:,}</b><br>Toplam<br>Ort: {avg_score:.1f}",
            x=0.5, y=0.5, font=dict(size=16, color='#fff', family='Arial Black'), showarrow=False
        )
        fig_band.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            height=350,
            margin=dict(l=80, r=80, t=50, b=80),  # Margin'leri artırdık ki etiketler taşmasın
            showlegend=False
        )
        st.plotly_chart(fig_band, use_container_width=True)
    
    # Birim Kartı (Sadece Ana Kategoriler) - Yeşil, Sarı, Kırmızı, Mor
    with col_birim:
        st.markdown("#### 📁 Birim Dağılımı")
        if 'Ana_Kategori' in fdf.columns and n > 0:
            birim_counts = fdf['Ana_Kategori'].value_counts().sort_values(ascending=True)
            
            # Mavi tonları (açıktan koyuya)
            mavi_tonlar = ['#3b82f6', '#2563eb', '#1d4ed8', '#1e40af', '#1e3a8a', '#1e3a8a']
            bar_colors = mavi_tonlar[:len(birim_counts)] if len(birim_counts) <= len(mavi_tonlar) else (mavi_tonlar * 2)[:len(birim_counts)]
            
            fig_birim = go.Figure()
            fig_birim.add_trace(go.Bar(
                x=birim_counts.values,
                y=birim_counts.index,
                orientation='h',
                marker=dict(color=bar_colors),
                text=[f"{v:,}" for v in birim_counts.values],
                textposition='inside',
                textfont=dict(color='#fff', size=16, family='Arial Black'),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.9)', font_size=14, font_family='Arial', font_color='#fff')
            ))
            fig_birim.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                height=350,
                margin=dict(l=180, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#fff', family='Arial'))
            )
            st.plotly_chart(fig_birim, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # ALT KATEGORİ DAĞILIMI - STACKED BAR (MOR, KIRMIZI, SARI, YEŞİL)
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 📊 Alt Kategori Şikayet Dağılımı (Churn Band Renkli)")
    
    if 'Alt_Kategori' in fdf.columns and 'churn_band' in fdf.columns and n > 0:
        # Her alt kategori için churn band sayıları
        alt_cats = fdf['Alt_Kategori'].value_counts().head(10).index.tolist()
        
        # Her band için sayıları hesapla ve birimleri ekle
        mor_vals = []
        kirmizi_vals = []
        sari_vals = []
        yesil_vals = []
        alt_cats_with_unit = []
        
        for alt_cat in alt_cats:
            cat_df = fdf[fdf['Alt_Kategori'] == alt_cat]
            mor_vals.append((cat_df['churn_band'] == 'MOR').sum())
            kirmizi_vals.append((cat_df['churn_band'] == 'KIRMIZI').sum())
            sari_vals.append((cat_df['churn_band'] == 'SARI').sum())
            yesil_vals.append((cat_df['churn_band'] == 'YEŞİL').sum())
            # Birim ekle (kısa format)
            birim = get_responsible_unit(alt_cat)
            # Birim ismini kısalt
            birim_short = birim.replace("Ürün & Kalite Sorunları", "Ürün").replace("Finans & İade İşlemleri", "Finans").replace("Lojistik & Teslimat", "Lojistik").replace("Sistem & Sipariş Yönetimi", "Sistem")
            alt_cats_with_unit.append(f"{birim_short} | {alt_cat}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Düşük Riskli (YEŞİL)', y=alt_cats_with_unit, x=yesil_vals, orientation='h', marker_color=colors['YEŞİL']))
        fig.add_trace(go.Bar(name='Orta Riskli (SARI)', y=alt_cats_with_unit, x=sari_vals, orientation='h', marker_color=colors['SARI']))
        fig.add_trace(go.Bar(name='Yüksek Riskli (KIRMIZI)', y=alt_cats_with_unit, x=kirmizi_vals, orientation='h', marker_color=colors['KIRMIZI']))
        fig.add_trace(go.Bar(name='Kritik Riskli (MOR)', y=alt_cats_with_unit, x=mor_vals, orientation='h', marker_color=colors['MOR']))
        
        fig.update_layout(
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            height=450,
            margin=dict(l=200, r=30, t=20, b=10),
            xaxis=dict(showgrid=False, zeroline=False, title=dict(text='Şikayet Sayısı', font=dict(size=14, color='#fff', family='Arial Black')), tickfont=dict(size=12, color='#fff')),
            yaxis=dict(
                showgrid=False, 
                tickfont=dict(size=11, color='#fff', family='Arial'),
                autorange='reversed',
                tickmode='linear',
                tickangle=0
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=12, color='#ffffff', family='Arial Black'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # CHURN SİNYAL ANALİZİ - İKİ GRAFİK
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🔥 Churn Sinyal Analizi")
    
    sig_col1, sig_col2 = st.columns(2)
    
    with sig_col1:
        st.markdown("#### 📊 Churn Sinyal Analizi")
        if 'top_churn_signal_1' in fdf.columns and n > 0:
            signal1_counts = fdf['top_churn_signal_1'].value_counts()
            signal2_counts = fdf['top_churn_signal_2'].value_counts() if 'top_churn_signal_2' in fdf.columns else pd.Series()
            
            all_signals = signal1_counts.add(signal2_counts, fill_value=0)
            # CATEGORY_WEIGHTS'a göre sırala (yüksekten düşüğe)
            all_signals_dict = all_signals.to_dict()
            # Ağırlığa göre sırala, sonra sayıya göre
            all_signals_sorted_list = sorted(
                all_signals_dict.items(),
                key=lambda x: (CATEGORY_WEIGHTS.get(x[0], 0), x[1]),
                reverse=True
            )[:8]
            all_signals_sorted = pd.Series(dict(all_signals_sorted_list))
            
            signal_names = [s.split('. ')[1] if pd.notna(s) and '. ' in str(s) else str(s) for s in all_signals_sorted.index]
            
            signal_scores = []
            for sig in all_signals_sorted.index:
                mask1 = fdf['top_churn_signal_1'] == sig
                mask2 = fdf['top_churn_signal_2'] == sig if 'top_churn_signal_2' in fdf.columns else pd.Series([False]*len(fdf))
                combined = fdf[mask1 | mask2]
                avg = combined['churn_score'].mean() if len(combined) > 0 else 0
                signal_scores.append(avg)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=all_signals_sorted.values,
                y=signal_names,
                orientation='h',
                marker=dict(color=signal_scores, colorscale=[[0, '#3b82f6'], [0.4, '#2563eb'], [0.7, '#1d4ed8'], [1, '#1e40af']]),
                text=[f"{int(v):,}" for v in all_signals_sorted.values],
                textposition='inside',
                textfont=dict(color='#fff', size=15, family='Arial Black'),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.9)', font_size=14, font_family='Arial', font_color='#fff')
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                height=400,
                margin=dict(l=150, r=60, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(
                    showgrid=False, 
                    tickfont=dict(size=11, color='#fff', family='Arial'), 
                    autorange='reversed',
                    tickmode='linear'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with sig_col2:
        st.markdown("#### 📊 Birim × Ortalama Churn Skoru")
        if 'Ana_Kategori' in fdf.columns and 'churn_score' in fdf.columns and n > 0:
            # Birim bazlı ortalama churn skoru
            birim_churn = fdf.groupby('Ana_Kategori').agg(
                avg_churn=('churn_score', 'mean'),
                count=('churn_score', 'count')
            ).reset_index().sort_values('avg_churn', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=birim_churn['avg_churn'],
                y=birim_churn['Ana_Kategori'],
                orientation='h',
                marker=dict(
                    color=birim_churn['avg_churn'],
                    colorscale=[[0, '#3b82f6'], [0.4, '#2563eb'], [0.7, '#1d4ed8'], [1, '#1e40af']]
                ),
                text=[f"{v:.1f}<br>({c:,})" for v, c in zip(birim_churn['avg_churn'], birim_churn['count'])],
                textposition='inside',
                textfont=dict(color='#fff', size=12, family='Arial Black'),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.9)', font_size=14, font_family='Arial', font_color='#fff')
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                height=400,
                margin=dict(l=180, r=20, t=10, b=10),
                xaxis=dict(title=dict(text='Ort. Churn Skoru', font=dict(size=14, color='#fff', family='Arial Black')), showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=12, color='#fff')),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#fff', family='Arial'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════
    # CHURN SKORU DAĞILIMI
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 📈 Churn Skoru Dağılımı")
    
    if 'churn_score' in fdf.columns and n > 0:
        fig = px.histogram(
            fdf, x='churn_score', nbins=30,
            color='churn_band',
            color_discrete_map=colors
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            height=350,
            margin=dict(l=40, r=40, t=30, b=40),
            xaxis=dict(title=dict(text='Churn Skoru', font=dict(size=14, color='#fff', family='Arial Black')), tickfont=dict(size=12, color='#fff')),
            yaxis=dict(title=dict(text='Şikayet Sayısı', font=dict(size=14, color='#fff', family='Arial Black')), tickfont=dict(size=12, color='#fff')),
            legend=dict(font=dict(size=12, color='#ffffff', family='Arial Black'))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        median_score = fdf['churn_score'].median()
        high_count = len(fdf[fdf['churn_score'] >= 60])
        st.markdown(f"""
        <p style="color: #888; font-size: 0.9rem; text-align: center;">
        📈 Ortalama: <b>{avg_score:.1f}</b> | Medyan: <b>{median_score:.1f}</b> | Yüksek Risk (60+): <b>{high_count}</b> şikayet
        </p>
        """, unsafe_allow_html=True)

# =========================================================
# ŞİKAYET ANALİZİ FONKSİYONU (MEVCUT EKRAN)
# =========================================================
def show_complaint_analysis(tokenizer, clf_model, emb_model, device, df):
    """Şikayet Analizi sekmesi - Mevcut ekran"""
    # Başlık
    st.title("📊 Şikayet Analiz Sistemi")
    st.markdown("---")
    
    # Örnek metinler
    ornek_baslik = "Sipariş Görüntüleme Sorunu"
    ornek_metin = """Hepsiburada'dan sipariş verdim ancak siparişim 'Siparişlerim' kısmında görünmüyor. Sipariş veremez olduk. Artık lütfen yardımcı olur musunuz?"""
    
    # Session state ile ilk yükleme kontrolü
    if 'initial_analysis_done' not in st.session_state:
        st.session_state.initial_analysis_done = False
    
    # ANA LAYOUT - SOL: INPUTLAR, SAĞ: SONUÇLAR
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📝 Şikayet Başlığı (Opsiyonel)")
        baslik = st.text_input(
            "Şikayet Başlığı",
            value=ornek_baslik,
            placeholder="Şikayet başlığı (opsiyonel)",
            label_visibility="collapsed"
        )
        
        st.markdown("### 📄 Şikayet Metni")
        sikayet_metni = st.text_area(
            "Şikayet Metni",
            value=ornek_metin,
            height=400,
            placeholder="Şikayet metni",
            label_visibility="collapsed"
        )
        
        # ANALİZ BUTONU
        analiz_butonu = st.button(
            "🔍 Analiz et",
            type="primary",
            use_container_width=True
        )
        
        # Benzer şikayet sayısı sabit 10
        top_k = 10
    
    with col_right:
        st.markdown("### 📊 Tahmin Sonuçları")
        
        # İlk yüklemede veya butona basıldığında analiz yap
        should_analyze = analiz_butonu or (not st.session_state.initial_analysis_done and sikayet_metni and sikayet_metni.strip())
        
        # SONUÇLAR SAĞ KOLONDA - SADECE METRİKLER VE CHURN
        if should_analyze:
            if not sikayet_metni or not sikayet_metni.strip():
                st.warning("⚠️ Lütfen şikayet metnini doldurun!")
            else:
                with st.spinner("🔄 Analiz yapılıyor..."):
                    try:
                        results = predict_complaint(
                            baslik, sikayet_metni, df, 
                            tokenizer, clf_model, emb_model, device, top_k
                        )
                        
                        # Sonuçları session state'e kaydet
                        st.session_state.analysis_results = results
                        st.session_state.last_metin = sikayet_metni
                        
                        st.success("✅ Analiz tamamlandı!")
                        st.markdown("---")
                        
                        # İlk analiz tamamlandı olarak işaretle
                        st.session_state.initial_analysis_done = True
                        
                        # SORUMLU BİRİM VE ALT KATEGORİ - EN ÜSTTE
                        responsible_unit = get_responsible_unit(results["alt_kategori"])
                        alt_kategori_title = results["alt_kategori"].title()
                        
                        st.markdown(f'<p style="font-size: 1.6rem; font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">📋 Sorumlu Birim: <strong>{responsible_unit}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 1.5rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.5rem;">Alt Kategori: <strong>{alt_kategori_title}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 1.4rem; margin-top: 0.5rem; margin-bottom: 1rem;">Güven Oranı: <strong>%{results["olasilik"]}</strong></p>', unsafe_allow_html=True)
                        
                        # CHURN ANALİZİ - KOMPAKT VE YANYANA
                        churn_score = results["churn_score"]
                        churn_band_value = results["churn_band"]
                        churn_band_label = get_churn_label(churn_band_value)
                        color = get_churn_color(churn_band_value)
                        
                        # Churn Skoru ve Band yanyana
                        st.markdown(f"""
                        <div style="background: rgba(42, 42, 62, 0.8); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border: 2px solid {color};">
                            <p style="font-size: 1.4rem; font-weight: 700; margin: 0; color: #fff;">
                                Churn Skoru: <span style="color: {color};">{churn_score}</span> 
                                <span style="background: {color}; color: #fff; padding: 0.2rem 0.8rem; border-radius: 8px; margin-left: 0.5rem; font-size: 1.2rem;">{churn_band_label}</span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Gauge grafik küçültüldü
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=churn_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            number={'font': {'size': 28, 'color': '#ffffff'}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickcolor': '#ffffff', 'tickfont': {'size': 12}},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 35], 'color': "rgba(34, 197, 94, 0.3)"},  # Düşük Riskli (YEŞİL): #22c55e
                                    {'range': [35, 50], 'color': "rgba(250, 204, 21, 0.3)"},  # Orta Riskli (SARI): #facc15
                                    {'range': [50, 70], 'color': "rgba(220, 38, 38, 0.3)"},  # Yüksek Riskli (KIRMIZI): #dc2626
                                    {'range': [70, 100], 'color': "rgba(147, 51, 234, 0.3)"}  # Kritik Riskli (MOR): #9333ea
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 2},
                                    'thickness': 0.75,
                                    'value': churn_score
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#ffffff",
                            height=180,
                            margin=dict(t=30, b=30, l=15, r=15)  # Üst ve alt margin artırıldı
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Hata oluştu: {str(e)}")
                        st.exception(e)
        else:
            st.info("👈 Sol taraftaki formu doldurup 'Analiz et' butonuna tıklayın.")
    
    # TETİKLENEN KATEGORİLER VE BENZER ŞİKAYETLER - ALTA TAM GENİŞLİKTE
    if should_analyze and sikayet_metni and sikayet_metni.strip():
        try:
            # Analiz sonuçlarını session state'ten al (zaten yapılmışsa)
            if 'analysis_results' in st.session_state and st.session_state.get('last_metin') == sikayet_metni:
                results = st.session_state.analysis_results
            else:
                # Eğer session state'te yoksa, yukarıdaki analiz sonuçlarını kullan
                # Bu durumda results zaten mevcut olmalı
                if 'analysis_results' in st.session_state:
                    results = st.session_state.analysis_results
                else:
                    # Son çare olarak tekrar analiz yap
                    with st.spinner("🔄 Analiz yapılıyor..."):
                        results = predict_complaint(
                            baslik, sikayet_metni, df, 
                            tokenizer, clf_model, emb_model, device, top_k
                        )
                        st.session_state.analysis_results = results
                        st.session_state.last_metin = sikayet_metni
            
            # CHURN SİNYALLERİ - TETİKLENEN KATEGORİLER
            st.markdown('<div style="background: rgba(42, 42, 62, 0.8); border-radius: 15px; padding: 2rem; border: 2px solid rgba(102, 126, 234, 0.3); margin: 1rem 0;"><p style="font-size: 1.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem;">Churn Sinyalleri</p>', unsafe_allow_html=True)
            
            if results["triggered_categories"]:
                # CATEGORY_WEIGHTS'a göre sıralı
                sorted_categories = sorted(
                    results["triggered_categories"],
                    key=lambda x: CATEGORY_WEIGHTS.get(x, 0),
                    reverse=True
                )
                category_names = [remove_category_number(cat) for cat in sorted_categories]
                
                num_categories = len(category_names)
                cols = st.columns(4)
                
                if num_categories <= 4:
                    start_col = (4 - num_categories) // 2
                    
                    for idx, cat_name in enumerate(category_names):
                        col_idx = start_col + idx
                        icon = get_category_icon(cat_name)
                        
                        with cols[col_idx]:
                            category_html = f"""
                            <div style="
                                background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
                                border: 2px solid rgba(102, 126, 234, 0.5);
                                border-radius: 15px;
                                padding: 1rem;
                                margin: 0.5rem 0;
                                text-align: center;
                                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                            ">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                                <div style="font-size: 1rem; font-weight: 600; color: #ffffff; word-wrap: break-word;">{cat_name}</div>
                            </div>
                            """
                            st.markdown(category_html, unsafe_allow_html=True)
                else:
                    for row_start in range(0, num_categories, 4):
                        row_categories = category_names[row_start:row_start+4]
                        cols = st.columns(4)
                        
                        for idx, cat_name in enumerate(row_categories):
                            icon = get_category_icon(cat_name)
                            
                            with cols[idx]:
                                category_html = f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
                                    border: 2px solid rgba(102, 126, 234, 0.5);
                                    border-radius: 15px;
                                    padding: 1rem;
                                    margin: 0.5rem 0;
                                    text-align: center;
                                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                                ">
                                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                                    <div style="font-size: 1rem; font-weight: 600; color: #ffffff; word-wrap: break-word;">{cat_name}</div>
                                </div>
                                """
                                st.markdown(category_html, unsafe_allow_html=True)
            else:
                st.markdown('<p style="text-align: center; color: #888; font-size: 1rem;">Tetiklenen churn sinyali bulunamadı.</p>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # BENZER ŞİKAYETLER
            st.markdown('<div style="background: rgba(42, 42, 62, 0.8); border-radius: 15px; padding: 2rem; border: 2px solid rgba(102, 126, 234, 0.3); margin: 1rem 0;"><p style="font-size: 1.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem;">Benzer Şikayetler</p>', unsafe_allow_html=True)
            
            if not results["similar_complaints"].empty:
                for idx, row in results["similar_complaints"].iterrows():
                    similarity_pct = row['benzerlik_skoru'] * 100
                    
                    # Ana container - 2 kolon: içerik ve benzerlik skoru
                    col_main, col_score = st.columns([4, 1])
                    
                    with col_main:
                        # Üstte: Kullanıcı ve Tarih yan yana
                        col_user, col_date = st.columns(2)
                        with col_user:
                            if 'kullanici' in row and pd.notna(row['kullanici']):
                                st.markdown(f'<p style="color: #888; font-size: 0.9rem; margin: 0;"><strong>👤 Kullanıcı:</strong> {row["kullanici"]}</p>', unsafe_allow_html=True)
                        with col_date:
                            if 'tarih_saat' in row and pd.notna(row['tarih_saat']):
                                st.markdown(f'<p style="color: #888; font-size: 0.9rem; margin: 0;"><strong>📅 Tarih:</strong> {row["tarih_saat"]}</p>', unsafe_allow_html=True)
                        
                        # Ortada: Şikayet Başlığı
                        st.markdown(f'<p style="font-size: 1.2rem; margin-top: 0.5rem; margin-bottom: 0.5rem;"><strong>Şikayet Başlığı:</strong> {row["baslik"]}</p>', unsafe_allow_html=True)
                        
                        # Altta: Şikayet Metni
                        with st.expander("📄 Şikayet Metni"):
                            st.markdown(f'<p style="color: #b8b8d1;">{row["sikayet_metni"]}</p>', unsafe_allow_html=True)
                    
                    with col_score:
                        st.markdown(f'<div style="text-align: right;"><p style="font-size: 1.5rem; font-weight: 700; color: #667eea; margin: 0;">%{similarity_pct:.2f}</p></div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
            else:
                st.markdown('<p style="text-align: center; color: #888; font-size: 1rem;">Benzer şikayet bulunamadı.</p>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Hata oluştu: {str(e)}")
            st.exception(e)
    
    # BENZER ŞİKAYETLER - TAM GENİŞLİKTE EN ALTA (GÖRSELDEKİ GİBİ) - ESKİ KOD (BUTON İÇİN)
    if analiz_butonu and baslik and sikayet_metni and not should_analyze:
        try:
            results = predict_complaint(
                baslik, sikayet_metni, df, 
                tokenizer, clf_model, emb_model, device, top_k
            )
            
            st.markdown("---")
            st.markdown("---")
            
            # CHURN SİNYALLERİ - TAM GENİŞLİKTE VE ORTALANMIŞ (KUTU İÇİNDE)
            st.markdown('<div style="background: rgba(42, 42, 62, 0.8); border-radius: 15px; padding: 2rem; border: 2px solid rgba(102, 126, 234, 0.3); margin: 1rem 0;"><p style="font-size: 1.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem;">Churn Sinyalleri</p>', unsafe_allow_html=True)
            
            if results["triggered_categories"]:
                # CATEGORY_WEIGHTS'a göre sıralı (zaten sıralı geliyor ama emin olmak için tekrar sırala)
                sorted_categories = sorted(
                    results["triggered_categories"],
                    key=lambda x: CATEGORY_WEIGHTS.get(x, 0),
                    reverse=True
                )
                category_names = [remove_category_number(cat) for cat in sorted_categories]
                
                num_categories = len(category_names)
                
                # Her zaman 4 kolon kullan, tam genişlikte
                cols = st.columns(4)
                
                # Kategorileri ortalamak için başlangıç pozisyonunu hesapla
                if num_categories <= 4:
                    start_col = (4 - num_categories) // 2
                    
                    for idx, cat_name in enumerate(category_names):
                        col_idx = start_col + idx
                        icon = get_category_icon(cat_name)
                        
                        with cols[col_idx]:
                            category_html = f"""
                            <div style="
                                background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
                                border: 2px solid rgba(102, 126, 234, 0.5);
                                border-radius: 15px;
                                padding: 1rem;
                                margin: 0.5rem 0;
                                text-align: center;
                                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                            ">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                                <div style="font-size: 1rem; font-weight: 600; color: #ffffff; word-wrap: break-word;">{cat_name}</div>
                            </div>
                            """
                            st.markdown(category_html, unsafe_allow_html=True)
                else:
                    # 4'ten fazla kategori varsa satır satır göster
                    for row_start in range(0, num_categories, 4):
                        row_categories = category_names[row_start:row_start+4]
                        cols = st.columns(4)
                        
                        for idx, cat_name in enumerate(row_categories):
                            icon = get_category_icon(cat_name)
                            
                            with cols[idx]:
                                category_html = f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
                                    border: 2px solid rgba(102, 126, 234, 0.5);
                                    border-radius: 15px;
                                    padding: 1rem;
                                    margin: 0.5rem 0;
                                    text-align: center;
                                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                                ">
                                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                                    <div style="font-size: 1rem; font-weight: 600; color: #ffffff; word-wrap: break-word;">{cat_name}</div>
                                </div>
                                """
                                st.markdown(category_html, unsafe_allow_html=True)
            else:
                st.info("⚠️ Hiçbir kategori tetiklenmedi.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # BENZER ŞİKAYETLER - BÜYÜK KUTUDA TAM GENİŞLİKTE (SABİT 10)
            st.markdown(f'<div style="background: rgba(42, 42, 62, 0.8); border-radius: 15px; padding: 2rem; border: 2px solid rgba(102, 126, 234, 0.3); margin: 1rem 0;"><p style="font-size: 1.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem;">Benzer Top 10 Şikayet</p>', unsafe_allow_html=True)
            
            for idx, row in results["similar_complaints"].iterrows():
                similarity_pct = row['benzerlik_skoru'] * 100
                
                # Benzerlik skoru sağda, başlık solda
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Benzerlik skoru ve başlık
                    st.markdown(f'<p style="font-size: 1.2rem;"><strong>Şikayet Başlığı:</strong> {row["baslik"]}</p>', unsafe_allow_html=True)
                    
                    # Ad ve tarih yan yana
                    col1a, col1b = st.columns(2)
                    with col1a:
                        st.markdown(f'<p style="font-size: 1.1rem;"><strong>Kullanıcı:</strong> {row["kullanici"]}</p>', unsafe_allow_html=True)
                    with col1b:
                        st.markdown(f'<p style="font-size: 1.1rem;"><strong>Tarih:</strong> {row["tarih_saat"]}</p>', unsafe_allow_html=True)
                
                with col2:
                    # Benzerlik skoru sağda
                    st.markdown(f'<div style="text-align: right;"><p style="font-size: 1.5rem; font-weight: 700; color: #667eea; margin: 0;">%{similarity_pct:.2f}</p></div>', unsafe_allow_html=True)
                
                with st.expander("📄 Şikayet Metni"):
                    st.write(row['sikayet_metni'])
                
                st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass  # Hata durumunda sessizce geç


# =========================================================
# ZAMAN SERİSİ ANALİZİ SAYFASI
# =========================================================
def dataset_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Kategorileri 10 kategoriye düşür"""
    # Gerekli kolonları kontrol et
    if "sorun" not in df.columns:
        # Alt_Kategori veya başka bir kolon olabilir
        if "Alt_Kategori" in df.columns:
            df = df.rename(columns={"Alt_Kategori": "sorun"})
        else:
            st.error("Dosyada 'sorun' veya 'Alt_Kategori' kolonu bulunamadı.")
            return pd.DataFrame()
    
    # Sadece gerekli kolonları al (text varsa al, yoksa sadece tarih_saat ve sorun)
    cols_to_keep = ["tarih_saat", "sorun"]
    if "text" in df.columns:
        cols_to_keep.append("text")
    
    df = df[cols_to_keep].copy()

    etiket_eslestirme = {
        'ürün ile ilgili sorunlar': 'ürün ile ilgili sorunlar',
        'teslim edilmeyen paket': 'müşteriye teslim edilmeyen paket',
        'kargoya teslim edilmeyen paket': 'müşteriye teslim edilmeyen paket',
        'kargoya geç teslim': 'müşteriye teslim edilmeyen paket',
        'geç teslimat': 'kargo teslimat sorunu',
        'hasarlı paket': 'kargo teslimat sorunu',
        'iade süreci tamamlanmamış': 'iade süreci',
        'eksik ücret iadesi': 'iade süreci',
        'iade reddi': 'iade reddi',
        'uygulama sorunu': 'uygulama',
        'kupon sorunu': 'uygulama',
        'ödeme sorunu': 'uygulama',
        'siparişi iptal edememe': 'uygulama',
        'satıcı sipariş iptali': 'satıcı sipariş iptali',
        'yanlış veya eksik ürün gönderimi': 'yanlış veya eksik ürün gönderimi',
        'kullanılmış ürün gönderimi': 'yanlış veya eksik ürün gönderimi',
        'garanti sorunu': 'garanti sorunu',
        'fiyat farkı talebi': 'fiyat farkı talebi',
    }

    df["sorun"] = df["sorun"].astype(str).str.strip()
    df["kategoriler"] = df["sorun"].map(etiket_eslestirme)
    df = df[df["kategoriler"].notna()].copy()
    return df

def find_strong_active_start(ts: pd.DataFrame, window: int = 7, min_avg: float = 5.0):
    """
    Strong active start:
    - y'nin window günlük rolling ortalaması min_avg ve üstüne ilk çıktığı gün
    - min_avg artırıldı: 2.0 -> 5.0 (daha az veri olan günleri filtrelemek için)
    """
    if ts.empty:
        return None
    roll = ts["y"].rolling(window=window, min_periods=window).mean()
    valid = roll[roll >= min_avg]
    if valid.empty:
        return None
    return valid.index[0]

def slice_to_strong_active(ts: pd.DataFrame, window: int = 7, min_avg: float = 5.0):
    """
    ts'yi strong active start'tan itibaren kırpar.
    - min_avg artırıldı: 2.0 -> 5.0 (daha az veri olan günleri filtrelemek için)
    """
    start = find_strong_active_start(ts, window=window, min_avg=min_avg)
    if start is None:
        return ts.copy(), None
    return ts.loc[start:].copy(), start

def show_time_series_analysis():
    """Zaman Serisi Tahmin ve Anomali Tespiti sekmesi"""
    st.title("📈 Zaman Serisi Tahmin ve Anomali Tespiti")
    st.markdown("---")
    st.markdown(
        """
        1) Excel yükleyin  
        2) Kategori seçin (opsiyonel)  
        3) Bölüm butonları ile tahmin ve anomali analizlerini çalıştırın.
        """
    )

    # Cache'li yardımcılar
    @st.cache_data
    def read_excel_file(file):
        return pd.read_excel(file)

    @st.cache_data
    def prepare_df(df: pd.DataFrame):
        # dataset_preprocessing kullan
        d = dataset_preprocessing(df)
        d["tarih_saat"] = pd.to_datetime(d["tarih_saat"], errors="coerce")
        d = d.dropna(subset=["tarih_saat"])
        return d

    @st.cache_data
    def resample_counts(df: pd.DataFrame, freq: str, min_count: int = 1):
        """
        Veriyi yeniden örnekle ve çok az veri olan günleri/ayları filtrele
        min_count: Minimum veri sayısı (bu değerin altındaki günler/aylar filtrelenir)
        """
        ts = (
            df.set_index("tarih_saat")
            .resample(freq)
            .size()
            .reset_index(name="y")
        )
        ts.columns = ["ds", "y"]
        
        # Çok az veri olan günleri/ayları filtrele
        if min_count > 0:
            ts = ts[ts["y"] >= min_count].copy()
        
        return ts

    @st.cache_data
    def run_prophet(ts: pd.DataFrame, periods: int, freq: str):
        from prophet import Prophet

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=(freq == "D"),
            daily_seasonality=False,
        )
        model.fit(ts)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return forecast

    @st.cache_data
    def detect_anomalies(ts: pd.DataFrame, forecast: pd.DataFrame, sigma: float):
        merged = ts.merge(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            on="ds",
            how="left",
        )
        merged["anomaly"] = 0
        merged.loc[merged["y"] > merged["yhat_upper"] + sigma * (merged["yhat_upper"] - merged["yhat"]), "anomaly"] = 1
        merged.loc[merged["y"] < merged["yhat_lower"] - sigma * (merged["yhat"] - merged["yhat_lower"]), "anomaly"] = -1
        return merged

    def plot_forecast(actual: pd.DataFrame, forecast: pd.DataFrame, title: str, color="#ef4444"):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=actual["ds"],
                y=actual["y"],
                mode="lines",
                name="Gerçek",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode="lines",
                name="Tahmin",
                line=dict(color=color, dash="dash", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_upper"],
                mode="lines",
                name="Üst",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_lower"],
                mode="lines",
                name="Alt",
                fill="tonexty",
                fillcolor="rgba(239, 68, 68, 0.2)",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#000',
            title=dict(text=title, font=dict(size=20, color='#000', family='Arial Black')),
            xaxis=dict(
                title=dict(text="Tarih", font=dict(size=18, color='#000', family='Arial Black')),
                tickfont=dict(size=16, color='#000', family='Arial'),
                gridcolor='#e0e0e0',
                linecolor='#000',
            ),
            yaxis=dict(
                title=dict(text="Şikayet Sayısı", font=dict(size=18, color='#000', family='Arial Black')),
                tickfont=dict(size=16, color='#000', family='Arial'),
                gridcolor='#e0e0e0',
                linecolor='#000',
            ),
            hovermode="x unified",
            height=520,
            legend=dict(
                font=dict(size=16, color='#000', family='Arial Black'),
                bgcolor='rgba(255,255,255,0.8)',
            ),
        )
        return fig

    def plot_anomaly(df_anom: pd.DataFrame, title: str):
        fig = go.Figure()
        normal = df_anom[df_anom["anomaly"] == 0]
        pos = df_anom[df_anom["anomaly"] == 1]
        neg = df_anom[df_anom["anomaly"] == -1]

        fig.add_trace(
            go.Scatter(
                x=normal["ds"],
                y=normal["y"],
                mode="lines+markers",
                name="Normal",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=4, color="#1f77b4"),
            )
        )
        if not pos.empty:
            fig.add_trace(
                go.Scatter(
                    x=pos["ds"],
                    y=pos["y"],
                    mode="markers",
                    name="Pozitif Anomali",
                    marker=dict(color="#d62728", size=12, symbol="triangle-up", line=dict(width=2, color="#000")),
                )
            )
        if not neg.empty:
            fig.add_trace(
                go.Scatter(
                    x=neg["ds"],
                    y=neg["y"],
                    mode="markers",
                    name="Negatif Anomali",
                    marker=dict(color="#ff7f0e", size=12, symbol="triangle-down", line=dict(width=2, color="#000")),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df_anom["ds"],
                y=df_anom["yhat"],
                mode="lines",
                name="Beklenen",
                line=dict(color="#2ca02c", dash="dash", width=3),
            )
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#000',
            title=dict(text=title, font=dict(size=20, color='#000', family='Arial Black')),
            xaxis=dict(
                title=dict(text="Tarih", font=dict(size=18, color='#000', family='Arial Black')),
                tickfont=dict(size=16, color='#000', family='Arial'),
                gridcolor='#e0e0e0',
                linecolor='#000',
            ),
            yaxis=dict(
                title=dict(text="Şikayet Sayısı", font=dict(size=18, color='#000', family='Arial Black')),
                tickfont=dict(size=16, color='#000', family='Arial'),
                gridcolor='#e0e0e0',
                linecolor='#000',
            ),
            hovermode="x unified",
            height=520,
            legend=dict(
                font=dict(size=16, color='#000', family='Arial Black'),
                bgcolor='rgba(255,255,255,0.8)',
            ),
        )
        return fig

    # Veri yükleme - Varsayılan dosya veya opsiyonel Excel
    base_path = os.path.dirname(os.path.abspath(__file__))
    default_pkl_path = os.path.join(base_path, "df_weigthed_final.pkl")
    
    df_raw = None
    
    # Varsayılan dosyayı yükle
    if os.path.exists(default_pkl_path):
        try:
            with open(default_pkl_path, "rb") as f:
                df_raw = pickle.load(f)
        except Exception as exc:
            st.warning(f"⚠️ Varsayılan dosya yüklenemedi: {exc}")
            df_raw = None
    
    # Opsiyonel Excel yükleme
    uploaded = st.file_uploader("📁 Opsiyonel: Farklı Excel dosyası yükleyin", type=["xlsx", "xls"], help="Varsayılan veri yerine farklı bir Excel dosyası kullanmak isterseniz yükleyin.")
    if uploaded:
        try:
            df_raw = read_excel_file(uploaded)
            
            # Gerekli kolonları kontrol et
            missing_cols = []
            if "tarih_saat" not in df_raw.columns:
                missing_cols.append("tarih_saat")
            if "sorun" not in df_raw.columns and "Alt_Kategori" not in df_raw.columns:
                missing_cols.append("sorun veya Alt_Kategori")
            
            if missing_cols:
                st.warning(f"⚠️ Eksik kolonlar: {', '.join(missing_cols)}")
                with st.expander("📊 Mevcut Kolonlar"):
                    st.write(", ".join(df_raw.columns.astype(str)))
            else:
                with st.expander("📊 Önizleme"):
                    st.dataframe(df_raw.head(10))
                    st.caption(", ".join(df_raw.columns.astype(str)))
        except Exception as exc:
            st.error(f"Dosya okunamadı: {exc}")
            if df_raw is None:
                st.info("Varsayılan veri kullanılacak.")

    if df_raw is None:
        st.error("❌ Veri yüklenemedi. Lütfen varsayılan dosyanın mevcut olduğundan veya bir Excel dosyası yüklediğinizden emin olun.")
        return

    # Gerekli kolonları kontrol et (tarih_saat zorunlu, sorun veya Alt_Kategori zorunlu)
    if "tarih_saat" not in df_raw.columns:
        st.error("⚠️ Dosyada 'tarih_saat' kolonu bulunamadı.")
        return
    if "sorun" not in df_raw.columns and "Alt_Kategori" not in df_raw.columns:
        st.error("⚠️ Dosyada 'sorun' veya 'Alt_Kategori' kolonu bulunamadı.")
        return

    st.markdown("---")
    st.subheader("⚙️ Kategori Seçimi")
    
    # Otomatik preprocessing
    df_clean = prepare_df(df_raw)
    if df_clean.empty:
        st.error("Geçerli veri bulunamadı. Tarih verileri parse edilemedi.")
        return

    # Kategori seçimi
    cats = ["Tümü"] + sorted(df_clean["kategoriler"].dropna().astype(str).unique().tolist())
    category_filter = st.selectbox("Kategori/Segment", options=cats)

    def filter_df(df: pd.DataFrame):
        d = df.copy()
        if category_filter and category_filter != "Tümü":
            d = d[d["kategoriler"] == category_filter]
        return d

    st.info("Veri hazır. Aşağıdaki bölümlerden istediğinizi çalıştırın.")

    # Bölüm 1: Günlük Tahmin
    with st.expander("📅 Günlük Tahmin", expanded=True):
        horizon_d = st.slider("Tahmin Ufku (Gün)", 7, 90, 30, step=7)
        use_strong_active = st.checkbox("Strong Active Start Kullan (Kategori bazında)", value=True)
        if st.button("🚀 Günlük Tahmin Çalıştır"):
            with st.spinner("Günlük tahmin hesaplanıyor..."):
                try:
                    df_use = filter_df(df_clean)
                    
                    # Kategori bazında işlem yap
                    if category_filter != "Tümü" and use_strong_active:
                        # Kategori seçilmişse, kategori bazında strong active start bul
                        # Minimum 5 veri olan günleri filtrele
                        ts = resample_counts(df_use, freq="D", min_count=5)
                        if len(ts) < 14:
                            st.warning("En az 14 günlük veri gerekli.")
                        else:
                            # ts'yi ds'yi index yaparak hazırla (find_strong_active_start için)
                            ts_indexed = ts.set_index("ds")
                            
                            # Strong active start'ı bul ve kırp (min_avg=5.0 ile daha agresif filtreleme)
                            ts_sliced, start_date = slice_to_strong_active(ts_indexed, window=7, min_avg=5.0)
                            
                            # ts_sliced'i tekrar ds kolonlu DataFrame'e çevir
                            ts_sliced = ts_sliced.reset_index()
                            
                            if start_date is not None:
                                st.info(f"📌 Strong Active Start: {start_date.strftime('%Y-%m-%d')} tarihinden itibaren gösteriliyor.")
                            
                            if len(ts_sliced) < 14:
                                st.warning("Strong active start sonrası yeterli veri yok. Tüm veri kullanılıyor.")
                                ts_sliced = ts
                                start_date = None
                            
                            # Prophet tahmini
                            fc = run_prophet(ts_sliced, periods=horizon_d, freq="D")
                            
                            # Grafik için sliced kısmı göster
                            fig = plot_forecast(ts_sliced, fc, f"Günlük Tahmin - {category_filter} ({horizon_d} gün)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Tablo
                            table = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_d)
                            table.columns = ["Tarih", "Tahmin", "Alt", "Üst"]
                            st.dataframe(table, use_container_width=True)
                            st.download_button(
                                "📥 Günlük Tahmin (CSV)",
                                table.to_csv(index=False).encode("utf-8"),
                                file_name=f"gunluk_tahmin_{horizon_d}.csv",
                                mime="text/csv",
                            )
                    else:
                        # Tümü seçilmişse veya strong active kullanılmıyorsa normal işlem
                        # Minimum 5 veri olan günleri filtrele
                        ts = resample_counts(df_use, freq="D", min_count=5)
                        if len(ts) < 14:
                            st.warning("En az 14 günlük veri gerekli.")
                        else:
                            fc = run_prophet(ts, periods=horizon_d, freq="D")
                            title = f"Günlük Tahmin ({horizon_d} gün)"
                            if category_filter != "Tümü":
                                title = f"Günlük Tahmin - {category_filter} ({horizon_d} gün)"
                            fig = plot_forecast(ts, fc, title)
                            st.plotly_chart(fig, use_container_width=True)
                            table = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_d)
                            table.columns = ["Tarih", "Tahmin", "Alt", "Üst"]
                            st.dataframe(table, use_container_width=True)
                            st.download_button(
                                "📥 Günlük Tahmin (CSV)",
                                table.to_csv(index=False).encode("utf-8"),
                                file_name=f"gunluk_tahmin_{horizon_d}.csv",
                                mime="text/csv",
                            )
                except Exception as exc:
                    st.error(f"Günlük tahmin hatası: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

    # Bölüm 2: Haftalık Tahmin
    with st.expander("📆 Haftalık Tahmin", expanded=True):
        horizon_w = st.slider("Tahmin Ufku (Hafta)", 4, 24, 12, step=4)
        if st.button("🚀 Haftalık Tahmin Çalıştır"):
            with st.spinner("Haftalık tahmin hesaplanıyor..."):
                try:
                    df_use = filter_df(df_clean)
                    # Minimum 15 veri olan haftaları filtrele
                    ts_w = resample_counts(df_use, freq="W", min_count=15)
                    if len(ts_w) < 8:
                        st.warning("En az 8 haftalık veri gerekli.")
                    else:
                        fc_w = run_prophet(ts_w, periods=horizon_w, freq="W")
                        fig_w = plot_forecast(ts_w, fc_w, f"Haftalık Tahmin ({horizon_w} hafta)", color="#10b981")
                        st.plotly_chart(fig_w, use_container_width=True)
                        table_w = fc_w[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_w)
                        table_w.columns = ["Tarih", "Tahmin", "Alt", "Üst"]
                        st.dataframe(table_w, use_container_width=True)
                        st.download_button(
                            "📥 Haftalık Tahmin (CSV)",
                            table_w.to_csv(index=False).encode("utf-8"),
                            file_name=f"haftalik_tahmin_{horizon_w}.csv",
                            mime="text/csv",
                        )
                except Exception as exc:
                    st.error(f"Haftalık tahmin hatası: {exc}")

    # Bölüm 3: Anomali Tespiti
    with st.expander("🔍 Anomali Tespiti", expanded=True):
        freq_label = st.selectbox("Frekans", ["Günlük (D)", "Haftalık (W)"])
        sigma = st.slider("Eşik (σ)", 1.0, 5.0, 2.0, step=0.5)
        if st.button("🚀 Anomali Analizi Çalıştır"):
            with st.spinner("Anomali analizi yapılıyor..."):
                try:
                    df_use = filter_df(df_clean)
                    freq_code = "D" if "Günlük" in freq_label else "W"
                    # Günlük için minimum 5, haftalık için minimum 15 veri filtrele
                    min_count_anom = 5 if freq_code == "D" else 15
                    ts_a = resample_counts(df_use, freq=freq_code, min_count=min_count_anom)
                    if len(ts_a) < 10:
                        st.warning("En az 10 veri noktası gerekli.")
                    else:
                        fc_a = run_prophet(ts_a, periods=0, freq=freq_code)
                        anom = detect_anomalies(ts_a, fc_a, sigma=sigma)
                        fig_a = plot_anomaly(anom, f"Anomali Tespiti (σ={sigma})")
                        st.plotly_chart(fig_a, use_container_width=True)
                        anom_list = anom[anom["anomaly"] != 0].copy()
                        if anom_list.empty:
                            st.success("Anomali tespit edilmedi.")
                        else:
                            anom_list["anomali_tipi"] = anom_list["anomaly"].map({1: "Pozitif", -1: "Negatif"})
                            st.dataframe(anom_list[["ds", "y", "yhat", "anomali_tipi"]], use_container_width=True)
                            st.download_button(
                                "📥 Anomali Listesi (CSV)",
                                anom_list.to_csv(index=False).encode("utf-8"),
                                file_name=f"anomali_listesi_{freq_code}.csv",
                                mime="text/csv",
                            )
                except Exception as exc:
                    st.error(f"Anomali analizi hatası: {exc}")

# =========================================================
# ÖZET ANALİZİ FONKSİYONU
# =========================================================
def show_summary_analysis():
    """Özet ve Duygu Analizi sekmesi"""
    st.title("📝 Şikayet Özet ve Duygu Analizi")
    st.markdown("---")
    
    # Modelleri yükle
    try:
        summarizer, classifier = load_summary_models()
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        st.exception(e)
        return
    
    if summarizer is None or classifier is None:
        st.error("⚠️ Modeller yüklenemedi. Lütfen model dosyalarının doğru konumda olduğundan emin olun.")
        
        # Kontrol edilen yolları göster
        base_path = os.path.dirname(os.path.abspath(__file__))
        checked_paths = [
            os.path.join(base_path, "yerel_modeller"),
            os.path.join(os.path.expanduser("~"), "Desktop", "yerel_modeller"),
            os.path.join(os.path.expanduser("~"), "PycharmProjects", "PythonProject", "duygu_ozet", "yerel_modeller"),
            r"C:\Users\smt20\Desktop\yerel_modeller",
        ]
        
        st.info("💡 Kontrol edilen yollar:")
        for path in checked_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            st.text(f"{exists} {path}")
        
        st.info("💡 Model yolu: `yerel_modeller/ozetleme_modeli` ve `yerel_modeller/duygu_modeli` klasörleri gerekli.")
        return
    
    # İKİ KOLONLU LAYOUT - SOL: INPUT, SAĞ: SONUÇLAR
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("### 📝 Şikayet Başlığı (Opsiyonel)")
        baslik = st.text_input(
            "Şikayet Başlığı",
            placeholder="Şikayet başlığı (opsiyonel)",
            label_visibility="collapsed"
        )
        
        st.markdown("### 📄 Şikayet Metni")
        sikayet_metni = st.text_area(
            "Şikayet Metni",
            height=400,
            placeholder="Şikayet metnini buraya yapıştırın...",
            label_visibility="collapsed"
        )
        
        # Analiz butonu
        analiz_butonu = st.button(
            "🔍 Analiz Et",
            type="primary",
            use_container_width=True
        )
    
    with col_right:
        st.markdown("### 📊 Analiz Sonuçları")
        
        # Analiz yap
        if analiz_butonu or (sikayet_metni and sikayet_metni.strip()):
            if not sikayet_metni or not sikayet_metni.strip():
                st.warning("⚠️ Lütfen şikayet metnini girin.")
            else:
                # Tam metin oluştur
                if baslik and baslik.strip():
                    full_text = f"{baslik} {sikayet_metni}"
                else:
                    full_text = sikayet_metni
                
                with st.spinner("🔄 Analiz yapılıyor, lütfen bekleyin..."):
                    try:
                        # Özet
                        ozet_sonuc = summarizer(
                            full_text,
                            num_beams=5,
                            max_length=300,
                            min_length=20,
                            length_penalty=2.0,
                            early_stopping=True,
                            no_repeat_ngram_size=4
                        )
                        ozet_metni = ozet_sonuc[0]["summary_text"]
                        
                        # Duygu analizi
                        ETIKETLER = ["Öfke", "Hayal Kırıklığı", "Bıkkınlık", "Endişe", "Şaşkınlık", "Tehdit"]
                        duygu_sonuc = classifier(
                            full_text,
                            ETIKETLER,
                            multi_label=False
                        )
                        
                        # =========================================================
                        # ÖZET BÖLÜMÜ - RENKLİ KART
                        # =========================================================
                        st.markdown("---")
                        st.markdown("#### 📝 Şikayet Özeti")
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                            border-left: 4px solid #667eea;
                            border-radius: 12px;
                            padding: 1.5rem;
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
                        ">
                            <p style="
                                color: #e0e0e0;
                                font-size: 1.1rem;
                                line-height: 1.8;
                                margin: 0;
                                text-align: justify;
                            ">{ozet_metni}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # =========================================================
                        # DUYGU SKORLARI GRAFİĞİ
                        # =========================================================
                        st.markdown("---")
                        st.markdown("#### 📊 Duygu Skorları")
                        
                        # Duygu skorlarını hazırla
                        duygu_labels = duygu_sonuc["labels"]
                        duygu_scores = [s * 100 for s in duygu_sonuc["scores"]]
                        
                        # Bar chart oluştur
                        fig_duygu = go.Figure()
                        
                        # Duygu renkleri
                        duygu_renkler = {
                            "Öfke": "#ef4444",
                            "Hayal Kırıklığı": "#f59e0b",
                            "Bıkkınlık": "#8b5cf6",
                            "Endişe": "#3b82f6",
                            "Şaşkınlık": "#10b981",
                            "Tehdit": "#dc2626"
                        }
                        
                        # Her duygu için renkli bar
                        bar_colors = [duygu_renkler.get(label, "#667eea") for label in duygu_labels]
                        
                        fig_duygu.add_trace(go.Bar(
                            x=duygu_labels,
                            y=duygu_scores,
                            marker=dict(
                                color=bar_colors,
                                line=dict(color='#1a1a2e', width=2)
                            ),
                            text=[f"%{s:.1f}" for s in duygu_scores],
                            textposition='outside',
                            textfont=dict(size=14, color='#ffffff', family='Arial Black'),
                            hovertemplate='<b>%{x}</b><br>Skor: %{y:.2f}%<extra></extra>'
                        ))
                        
                        fig_duygu.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#fff',
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=60),
                            xaxis=dict(
                                title=dict(text="Duygular", font=dict(size=14, color='#fff', family='Arial Black')),
                                tickfont=dict(size=12, color='#fff', family='Arial'),
                                showgrid=False
                            ),
                            yaxis=dict(
                                title=dict(text="Skor (%)", font=dict(size=14, color='#fff', family='Arial Black')),
                                tickfont=dict(size=12, color='#fff'),
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)',
                                range=[0, 100]
                            ),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_duygu, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Analiz hatası: {str(e)}")
                        st.exception(e)
        else:
            # Başlangıç mesajı
            st.markdown("""
            <div style="
                background: rgba(42, 42, 62, 0.5);
                border: 2px dashed rgba(102, 126, 234, 0.3);
                border-radius: 12px;
                padding: 3rem;
                text-align: center;
                margin-top: 2rem;
            ">
                <p style="color: #b8b8d1; font-size: 1.2rem; margin: 0;">
                    👈 Sol taraftaki formu doldurup<br>
                    <strong style="color: #667eea;">"Analiz Et"</strong> butonuna tıklayın
                </p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# ANA FONKSİYON - MENÜ İLE
# =========================================================
def main():
    # Model ve veri yükleme (her iki sekme için)
    with st.spinner("Model ve veriler yükleniyor..."):
        tokenizer, clf_model, emb_model, device = load_models()
        df = load_data()
    
    # MENÜ - TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Şikayet Analizi", "📊 Dashboard", "📈 Zaman Serisi", "📝 Özet"])
    
    with tab1:
        show_complaint_analysis(tokenizer, clf_model, emb_model, device, df)
    
    with tab2:
        show_dashboard(df)
    
    with tab3:
        show_time_series_analysis()
    
    with tab4:
        show_summary_analysis()

if __name__ == "__main__":
    main()

