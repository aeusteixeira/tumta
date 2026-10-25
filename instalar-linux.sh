#!/usr/bin/env bash
# Instala o Tumtá no Linux: ambiente Python, modelo do Demucs e atalho no menu.
# O atalho roda direto desta pasta, entao um "git pull" ja atualiza o programa.
#
#   ./instalar-linux.sh            instala (ou repara/atualiza as dependencias)
#   ./instalar-linux.sh --remover  tira o atalho do menu (a pasta e os videos ficam)
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONES="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
ATALHO="$APPS/tumta.desktop"
PACOTES=(demucs librosa opencv-python soundfile pillow "pywebview[qt]" "yt-dlp[default]" deno
         huggingface_hub pyyaml)

passo() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
falha() { printf '\n\033[31mErro:\033[0m %s\n' "$1" >&2; exit 1; }

if [[ "${1:-}" == "--remover" ]]; then
    rm -f "$ATALHO" "$ICONES"/{192x192,512x512}/apps/tumta.png
    update-desktop-database "$APPS" 2>/dev/null || true
    echo "Atalho removido. Pra apagar tudo, apague a pasta $RAIZ"
    exit 0
fi

passo "Conferindo o que o sistema precisa"
command -v python3 >/dev/null || falha "falta o Python 3 (sudo dnf install python3)"
command -v ffmpeg >/dev/null || falha "falta o ffmpeg (no Fedora: sudo dnf install ffmpeg, com o RPM Fusion ativo)"
command -v wl-paste >/dev/null || command -v xclip >/dev/null ||
    echo "Aviso: sem wl-paste nem xclip o botao 'Colar link' nao funciona (Ctrl+V funciona)."

passo "Ambiente Python (.venv)"
[[ -x "$RAIZ/.venv/bin/python" ]] || python3 -m venv "$RAIZ/.venv"
PY="$RAIZ/.venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip

passo "Dependências (a primeira vez baixa uns 3 GB por causa do PyTorch)"
"$PY" -m pip install --upgrade "${PACOTES[@]}"

passo "Modelo do Demucs"
(cd "$RAIZ" && "$PY" -c "import empacotar; empacotar.garantir_modelo()")

passo "Atalho no menu"
for tam in 192 512; do
    install -Dm644 "$RAIZ/docs/img/icone-$tam.png" "$ICONES/${tam}x${tam}/apps/tumta.png"
done
mkdir -p "$APPS"
cat > "$ATALHO" <<EOF
[Desktop Entry]
Type=Application
Name=Tumtá
GenericName=Partitura de atabaque e djembe
Comment=Partitura de atabaque e djembe em vídeo, a partir de qualquer música
Exec="$PY" "$RAIZ/app.py"
Path=$RAIZ
Icon=tumta
Terminal=false
Categories=AudioVideo;Audio;Music;
Keywords=atabaque;djembe;tambor;partitura;percussão;
EOF
update-desktop-database "$APPS" 2>/dev/null || true
gtk-update-icon-cache -q "$ICONES" 2>/dev/null || true

if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    PLACA="placa de vídeo (CUDA)"
else
    PLACA="processador (sem placa NVIDIA disponível, fica mais lento)"
fi

passo "Pronto!"
echo "Abra o Tumtá pelo menu de aplicativos."
echo "Separação do tambor vai rodar no: $PLACA"
echo "Os vídeos ficam em: $(xdg-user-dir DOCUMENTS 2>/dev/null || echo "$HOME/Documents")/Tumta"
