# ⚡ Hızlı Başlangıç - Streamlit Cloud

## 🎯 3 Adımda Deploy

### 1️⃣ GitHub'a Yükle

```bash
cd şikayet_heybesi_V2
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/KULLANICI_ADINIZ/REPO_ADI.git
git branch -M main
git push -u origin main
```

### 2️⃣ Streamlit Cloud'a Bağla

1. [share.streamlit.io](https://share.streamlit.io) → Sign in (GitHub)
2. New app → Repository seç → Deploy!

### 3️⃣ Paylaş!

Uygulamanız hazır! Linki herkesle paylaşın 🎉

---

## 📋 Gereksinimler

- ✅ GitHub hesabı
- ✅ Streamlit Cloud hesabı (GitHub ile ücretsiz)
- ✅ Repository Public olmalı

## ⚠️ Model Dosyaları

Model dosyaları büyük olabilir. İki seçenek:

**A) Git LFS kullan (önerilen):**
```bash
git lfs install
git lfs track "*.pkl" "*.safetensors" "bert_oversampling_model/**"
git add .gitattributes
git add .
git commit -m "Add large files"
git push
```

**B) Model dosyalarını harici depolamaya al**
- Google Drive/Dropbox kullan
- İlk çalıştırmada indir

## 🆘 Sorun mu var?

- Logları kontrol et: Streamlit Cloud → App → Logs
- `requirements.txt` doğru mu kontrol et
- Model dosyaları repository'de mi kontrol et

