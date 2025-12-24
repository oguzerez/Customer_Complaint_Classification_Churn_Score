# 🚀 Streamlit Cloud'a Deploy Etme Rehberi

## Adım 1: GitHub Repository Oluşturma

1. [GitHub](https://github.com) adresine gidin ve giriş yapın
2. Sağ üstteki **"+"** butonuna tıklayın → **"New repository"**
3. Repository adı: `sikayet-analiz-sistemi` (veya istediğiniz isim)
4. **Public** seçin (Streamlit Cloud için gerekli)
5. **"Create repository"** butonuna tıklayın

## Adım 2: Dosyaları GitHub'a Yükleme

### Yöntem 1: Git Komutları (Önerilen)

Terminal/PowerShell'de şu komutları çalıştırın:

```bash
cd şikayet_heybesi_V2

# Git repository başlat
git init

# Tüm dosyaları ekle
git add .

# İlk commit
git commit -m "Initial commit: Şikayet Analiz Sistemi"

# GitHub repository'nizi ekleyin (URL'i kendi repository'nizle değiştirin)
git remote add origin https://github.com/oguzerez/sikayet-analiz-sistemi.git

# Dosyaları yükle
git branch -M main
git push -u origin main
```

### Yöntem 2: GitHub Desktop (Kolay)

1. [GitHub Desktop](https://desktop.github.com/) indirin ve kurun
2. GitHub Desktop'u açın
3. **File → Add Local Repository**
4. `şikayet_heybesi_V2` klasörünü seçin
5. **Publish repository** butonuna tıklayın

### Yöntem 3: GitHub Web Arayüzü

1. GitHub repository sayfanızda **"uploading an existing file"** linkine tıklayın
2. Tüm dosyaları sürükleyip bırakın
3. **"Commit changes"** butonuna tıklayın

## Adım 3: Streamlit Cloud'a Bağlama

1. [share.streamlit.io](https://share.streamlit.io) adresine gidin
2. **"Sign in"** butonuna tıklayın
3. GitHub hesabınızla giriş yapın
4. **"New app"** butonuna tıklayın
5. Formu doldurun:
   - **Repository**: `oguzerez/sikayet-analiz-sistemi`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app_v3.py`
6. **"Deploy!"** butonuna tıklayın

## ⚠️ Önemli Notlar

### Model Dosyaları Büyük Olabilir

Model dosyaları (`bert_oversampling_model/` ve `df_weigthed_final.pkl`) büyük olabilir. İki seçenek:

**Seçenek 1: Git LFS Kullanın (Önerilen)**
```bash
# Git LFS kurulumu
git lfs install

# Büyük dosyaları LFS'e ekle
git lfs track "*.pkl"
git lfs track "*.safetensors"
git lfs track "bert_oversampling_model/**"

git add .gitattributes
git add .
git commit -m "Add large files with Git LFS"
git push
```

**Seçenek 2: Model Dosyalarını Harici Depolamaya Alın**
- Google Drive, Dropbox veya başka bir depolama servisi kullanın
- Uygulama ilk çalıştırmada modelleri indirsin

### Streamlit Cloud Limitleri

- **RAM**: 1 GB (ücretsiz)
- **CPU**: Paylaşımlı
- **Disk**: 1 GB
- **Dosya boyutu**: 200 MB (tek dosya)

Eğer model dosyaları çok büyükse, Streamlit Cloud'un ücretli planına geçmeniz gerekebilir.

## 🔧 Sorun Giderme

### Model Yükleme Hatası
- Model dosyalarının repository'de olduğundan emin olun
- Git LFS kullanıyorsanız, Streamlit Cloud'un Git LFS'i desteklediğinden emin olun

### Bağımlılık Hataları
- `requirements.txt` dosyasının doğru olduğundan emin olun
- Streamlit Cloud loglarını kontrol edin

### Veri Dosyası Bulunamadı
- `df_weigthed_final.pkl` dosyasının repository'de olduğundan emin olun
- Dosya yolu doğru mu kontrol edin

## 📞 Destek

Sorun yaşarsanız:
1. Streamlit Cloud loglarını kontrol edin
2. GitHub Issues'da sorun bildirin
3. Streamlit Community Forum'da yardım isteyin

## ✅ Başarılı Deploy Sonrası

Deploy başarılı olduktan sonra:
- Uygulamanız `https://oguzerez-sikayet-analiz-sistemi.streamlit.app` adresinde olacak
- Bu linki herkesle paylaşabilirsiniz!
- Otomatik güncellemeler: GitHub'a push ettiğinizde otomatik olarak güncellenir

