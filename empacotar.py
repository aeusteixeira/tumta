"""Monta o programa para distribuir: dist/Partitura Atabaque/ e um .zip pronto pra mandar.

    .venv-build\\Scripts\\python.exe empacotar.py

Usa o ambiente .venv-build (PyTorch so de processador, roda em qualquer PC).
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAIZ = Path(__file__).parent
NOME = "Tumta"
BUILD = RAIZ / "build"
DIST = RAIZ / "dist"


def desenhar_icone(destino, tamanho=256):
    """Icone da marca Tumta: quadrado urucum com o aro do tambor e os pontos das maos."""
    k = 8  # desenha grande e reduz, pra borda ficar lisa
    t = 256 * k
    img = Image.new("RGBA", (t, t), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    urucum, madeira, anil = (240, 138, 44, 255), (23, 17, 13, 255), (74, 160, 230, 255)
    d.rounded_rectangle([0, 0, t - 1, t - 1], radius=56 * k, fill=urucum)
    c, r, aro = t / 2, 86 * k, 22 * k
    d.ellipse([c - r, c - r, c + r, c + r], fill=madeira)
    d.ellipse([c - r + aro, c - r + aro, c + r - aro, c + r - aro], fill=(31, 22, 16, 255))
    p = 22 * k  # pontos das maos: esquerda anil (mais baixo), direita urucum (mais alto)
    for (x, y), cor in (((c - 34 * k, c + 6 * k), anil), ((c + 34 * k, c - 18 * k), urucum)):
        d.ellipse([x - p, y - p, x + p, y + p], fill=cor)
    img = img.resize((tamanho, tamanho), Image.LANCZOS)
    img.save(destino, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.save(Path(destino).with_suffix(".png"))


def garantir_modelo():
    """O modelo do Demucs nao vai pro GitHub (84 MB); baixa do Hugging Face se faltar."""
    import yaml
    from huggingface_hub import hf_hub_download
    pasta = RAIZ / "modelo"
    bag = yaml.safe_load((pasta / "htdemucs.yaml").read_text())
    for sig in bag["models"]:
        destino = pasta / f"{sig}.safetensors"
        if not destino.exists():
            print(f"Baixando o modelo {destino.name}...")
            shutil.copy(hf_hub_download("adefossez/HTDemucs", destino.name), destino)


def achar_ffmpeg():
    for nome in ("ffmpeg", "ffprobe"):
        caminho = shutil.which(nome)
        if not caminho:
            sys.exit(f"Nao achei o {nome} no PATH")
        yield Path(caminho)


def main():
    BUILD.mkdir(exist_ok=True)
    garantir_modelo()
    icone = BUILD / "icone.ico"
    desenhar_icone(icone)

    sep = ";"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
           "--name", NOME, "--icon", str(icone),
           "--workpath", str(BUILD), "--distpath", str(DIST), "--specpath", str(BUILD),
           "--add-data", f"{RAIZ / 'ui'}{sep}ui",
           "--add-data", f"{RAIZ / 'modelo'}{sep}modelo",
           "--hidden-import", "separacao",
           "--collect-submodules", "demucs",
           "--collect-data", "librosa",
           "--collect-submodules", "yt_dlp", "--collect-all", "yt_dlp_ejs",
           "--exclude-module", "matplotlib", "--exclude-module", "IPython",
           "--exclude-module", "torchaudio", "--exclude-module", "tensorboard",
           "--exclude-module", "pytest",
           str(RAIZ / "app.py")]
    for ff in achar_ffmpeg():
        cmd += ["--add-binary", f"{ff}{sep}ffmpeg"]
    deno = Path(sys.prefix) / "Scripts" / "deno.exe"  # motor de JavaScript do yt-dlp (YouTube)
    if not deno.exists():
        sys.exit(r"Falta o deno: .venv-build\Scripts\python.exe -m pip install deno")
    cmd += ["--add-binary", f"{deno}{sep}deno"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=RAIZ)

    pasta = DIST / NOME
    (pasta / "LEIA-ME.txt").write_text(LEIAME, encoding="utf-8")
    zipar(pasta, DIST / f"{NOME}.zip")


def zipar(pasta, destino):
    print(f"Compactando {destino.name}...")
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arq in sorted(pasta.rglob("*")):
            if arq.is_file():
                z.write(arq, Path(pasta.name) / arq.relative_to(pasta))
    print(f"  {destino}  ({destino.stat().st_size / 1e6:.0f} MB)")


LEIAME = """Tumta - Partitura de Atabaque e Djembe
===================================

Como usar
1. Extraia esta pasta inteira (botao direito no .zip > Extrair tudo).
2. Abra "Tumta.exe".
3. Arraste a musica pra janela (ou cole o link do YouTube na aba "Link do YouTube"),
   escolha Atabaque ou Djembe e clique em "Gerar videos".

Os videos ficam em Documentos\\Tumta\\<nome da musica>:
  - video completo: a musica inteira com as maos mostrando onde bater
  - aulas: cada levada passo a passo (so TUM, so TA, junto devagar, no andamento, com a musica)
  - trechos: os pedacos da musica em que cada levada aparece

Se o Windows mostrar "O Windows protegeu o computador", clique em
"Mais informacoes" > "Executar assim mesmo" (aparece porque o programa nao e assinado).

Uma musica de 5 minutos leva uns 15 a 25 minutos pra gerar tudo, dependendo do computador.
So precisa de internet pra baixar musica por link do YouTube.
"""

if __name__ == "__main__":
    main()
