# 📊 Şikayet Analiz Sistemi

Modern, kapsamlı bir müşteri şikayet analiz ve churn risk yönetim platformu.

## 🚀 Özellikler

### 🔍 Şikayet Analizi
- **Otomatik Kategori Tahmini**: BERT modeli ile şikayetleri 10 farklı alt kategoriye sınıflandırır
- **Churn Skoru Hesaplama**: Dinamik algoritma ile müşteri kaybı riskini ölçer
- **Churn Band Sınıflandırması**: 
  - 🟣 **Kritik (MOR)**: ≥70 skor
  - 🔴 **Yüksek (KIRMIZI)**: ≥50 skor
  - 🟡 **Orta (SARI)**: ≥35 skor
  - 🟢 **Düşük (YEŞİL)**: <35 skor
- **Benzer Şikayetler**: Cosine similarity ile en benzer 10 şikayeti bulur
- **Churn Sinyal Analizi**: 8 farklı churn sinyalini tespit eder

### 📊 Dashboard
- **KPI Kartları**: Toplam şikayet, ortalama churn skoru, yüksek riskli şikayetler
- **Churn Band Dağılımı**: Görsel pie chart ile risk dağılımı
- **Birim Bazlı Analiz**: Ana kategorilere göre şikayet dağılımı
- **Alt Kategori Analizi**: Churn band renkli stacked bar chart
- **Churn Sinyal Analizi**: En çok tetiklenen sinyaller
- **Birim × Churn Skoru**: Birim bazlı ortalama churn skorları
- **Churn Skoru Dağılımı**: Histogram ile skor dağılımı
- **Gelişmiş Filtreleme**: Ana kategori, alt kategori, churn band ve tarih aralığı

### 📈 Zaman Serisi Analizi
- **Günlük Tahmin**: Prophet modeli ile günlük şikayet tahmini
- **Haftalık Tahmin**: Haftalık trend analizi
- **Anomali Tespiti**: İstatistiksel yöntemlerle anomali tespiti
- **Kategori Bazlı Analiz**: Kategori seçimine göre özelleştirilmiş analiz
- **Strong Active Start**: Veri kalitesi için otomatik filtreleme

## 📦 Kurulum

### Gereksinimler
- Python 3.8+
- CUDA destekli GPU (opsiyonel, CPU'da da çalışır)

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd customer_complaint
```

2. **Sanal ortam oluşturun (önerilir):**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# veya
source venv/bin/activate  # Linux/Mac
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Model dosyalarını kontrol edin:**
   - `bert_based_classification_models/` klasörü mevcut olmalı
   - `df_weigthed_final.pkl` veri dosyası mevcut olmalı

5. **Uygulamayı çalıştırın:**
```bash
streamlit run streamlit_app_v3.py
```

## 🌐 Streamlit Cloud'a Deploy Etme

1. **GitHub'a yükleyin:**
   - Repository'yi GitHub'a push edin
   - Büyük dosyalar (.pkl, model dosyaları) için Git LFS kullanın veya harici depolama kullanın

2. **Streamlit Cloud'a bağlayın:**
   - [streamlit.io](https://streamlit.io/cloud) adresine gidin
   - GitHub hesabınızla giriş yapın
   - Repository'yi seçin
   - Main file: `streamlit_app_v3.py`
   - Deploy edin!

## 📁 Proje Yapısı

```
customer_complaint/
├── streamlit_app_v3.py          # Ana uygulama dosyası
├── requirements.txt               # Python bağımlılıkları
├── README.md                      # Bu dosya
├── bert_based_classification_models/      # BERT model dosyaları
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── df_weigthed_final.pkl         # Veri dosyası
└── .gitignore                    # Git ignore dosyası
```

## 🔧 Kullanım

### Şikayet Analizi
1. "🔍 Şikayet Analizi" sekmesine gidin
2. Şikayet başlığı ve metnini girin (başlık opsiyonel)
3. "Analiz et" butonuna tıklayın veya otomatik analiz bekleyin
4. Sonuçları inceleyin:
   - Sorumlu birim
   - Alt kategori
   - Churn skoru ve band
   - Tetiklenen kategoriler
   - Benzer şikayetler

### Dashboard
1. "📊 Dashboard" sekmesine gidin
2. Filtreleri kullanın:
   - Ana Kategori
   - Alt Kategori (ana kategori seçildiğinde otomatik filtrelenir)
   - Churn Band
3. Grafikleri ve KPI'ları inceleyin

### Zaman Serisi Analizi
1. "📈 Zaman Serisi" sekmesine gidin
2. Opsiyonel: Excel dosyası yükleyin (varsayılan veri kullanılır)
3. Kategori/Segment seçin
4. Tahmin veya anomali analizi butonlarına tıklayın

## 🎨 Özellikler

- **Dark Mode**: Modern, göz yormayan karanlık tema
- **Responsive Design**: Tüm ekran boyutlarına uyumlu
- **Interactive Charts**: Plotly ile interaktif grafikler
- **Real-time Analysis**: Anlık analiz ve tahmin
- **Advanced Filtering**: Gelişmiş filtreleme seçenekleri

## 📝 Notlar

- Model dosyaları büyük olduğu için Git LFS kullanılması önerilir
- İlk yükleme sırasında modeller indirileceği için biraz zaman alabilir
- GPU kullanımı performansı önemli ölçüde artırır

## 📄 Lisans

Bu proje özel kullanım içindir.

## 👤 Geliştirici

Oğuzhan EREZ

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen pull request göndermeden önce değişikliklerinizi test edin.

