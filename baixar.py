"""Download do audio de videos (YouTube e outros sites que o yt-dlp aceita)."""
import re
import shutil
import sys
from pathlib import Path


def eh_link(texto):
    return re.match(r"https?://", str(texto).strip(), re.I) is not None


def achar_deno():
    """Motor de JavaScript que o yt-dlp usa pro YouTube (vai junto no programa instalado)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    for caminho in (base / "deno" / "deno.exe", Path(sys.prefix) / "Scripts" / "deno.exe",
                    Path(sys.prefix) / "bin" / "deno"):
        if caminho.exists():
            return str(caminho)
    return shutil.which("deno")


def _opcoes(**extra):
    opcoes = {"noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
              "socket_timeout": 20}
    deno = achar_deno()
    if deno:
        opcoes["js_runtimes"] = {"deno": {"path": deno}}
    opcoes.update(extra)
    return opcoes


def tipo_do_erro(erro):
    """'internet', 'invalido' ou 'indisponivel', pela mensagem do yt-dlp."""
    msg = str(erro).lower()
    if any(p in msg for p in ("getaddrinfo", "failed to resolve", "name resolution", "timed out",
                              "unreachable", "urlopen error", "connection", "network")):
        return "internet"
    if "unsupported url" in msg or "is not a valid url" in msg:
        return "invalido"
    return "indisponivel"


def informacoes(url):
    """Titulo e duracao do video, sem baixar. Levanta yt_dlp.utils.DownloadError se falhar."""
    import yt_dlp
    with yt_dlp.YoutubeDL(_opcoes(skip_download=True)) as ydl:
        info = ydl.extract_info(url.strip(), download=False)
    return {"titulo": info.get("title") or "", "duracao": info.get("duration") or 0,
            "canal": info.get("uploader") or info.get("channel") or "",
            "site": info.get("extractor_key") or ""}


def baixar(url, pasta):
    """Baixa so o audio. Devolve (caminho do arquivo, titulo do video)."""
    import yt_dlp

    print("Baixando o audio do video...", flush=True)

    def gancho(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                print(f"  download: {100 * d['downloaded_bytes'] / total:5.1f}% de "
                      f"{total / 1e6:.1f} MB", end=chr(13), flush=True)

    pasta.mkdir(parents=True, exist_ok=True)
    opcoes = _opcoes(format="bestaudio/best", outtmpl=str(pasta / "%(title)s [%(id)s].%(ext)s"),
                     windowsfilenames=True, progress_hooks=[gancho])
    try:
        with yt_dlp.YoutubeDL(opcoes) as ydl:
            info = ydl.extract_info(url.strip(), download=True)
            caminho = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as erro:
        sys.exit(f"Nao consegui baixar o video ({tipo_do_erro(erro)}): {erro}")
    print(f"Video: {info.get('title') or caminho.stem}", flush=True)
    return caminho, info.get("title") or caminho.stem
