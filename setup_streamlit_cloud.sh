#!/bin/bash
# Streamlit Cloud için GitHub'a yükleme scripti

echo "🚀 Streamlit Cloud Deploy Setup"
echo "================================"
echo ""

# Git LFS kontrolü
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS bulunamadı. Büyük dosyalar için Git LFS kurmanız önerilir."
    echo "   Kurulum: https://git-lfs.github.com/"
    read -p "Devam etmek istiyor musunuz? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Git repository kontrolü
if [ -d ".git" ]; then
    echo "✅ Git repository zaten başlatılmış"
else
    echo "📦 Git repository başlatılıyor..."
    git init
fi

# Git LFS kurulumu (varsa)
if command -v git-lfs &> /dev/null; then
    echo "📦 Git LFS kurulumu..."
    git lfs install
    git lfs track "*.pkl"
    git lfs track "*.safetensors"
    git lfs track "bert_oversampling_model/**"
    git add .gitattributes
    echo "✅ Git LFS yapılandırıldı"
fi

# Dosyaları ekle
echo "📁 Dosyalar ekleniyor..."
git add .

# Commit
echo "💾 Commit yapılıyor..."
git commit -m "Initial commit: Şikayet Analiz Sistemi - Streamlit Cloud ready"

echo ""
echo "✅ Hazır!"
echo ""
echo "📝 Sonraki adımlar:"
echo "1. GitHub'da yeni repository oluşturun"
echo "2. Şu komutu çalıştırın (URL'i kendi repository'nizle değiştirin):"
echo "   git remote add origin https://github.com/KULLANICI_ADINIZ/sikayet-analiz-sistemi.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. share.streamlit.io adresine gidin ve deploy edin!"
echo ""

