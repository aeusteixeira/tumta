"""Tumtá: programa com janela. Escolhe a musica (arquivo ou link), o instrumento e as opcoes,
e gera os videos.

    python app.py              abre a janela
    python app.py --cli ...    roda direto o partitura.py (e assim que a janela gera os videos)
"""
import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

CONGELADO = getattr(sys, "frozen", False)
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
PASTA_DADOS = Path(os.environ.get("APPDATA", Path.home())) / "Tumta"
CONFIG = PASTA_DADOS / "config.json"
SEM_JANELA = 0x08000000 if os.name == "nt" else 0
TIPOS_ARQUIVO = ("Música ou vídeo (*.mp3;*.wav;*.m4a;*.aac;*.flac;*.ogg;*.opus;*.wma;*.mp4;"
                 "*.mkv;*.mov;*.webm;*.avi)", "Todos os arquivos (*.*)")
MESES = "jan fev mar abr mai jun jul ago set out nov dez".split()
NOMES_TIPO = {0: "TUM", 1: "TA", 2: "TOM"}


def preparar_ambiente():
    """O ffmpeg vai junto no programa instalado; poe ele no PATH."""
    pasta = BASE / "ffmpeg"
    if pasta.exists():
        os.environ["PATH"] = str(pasta) + os.pathsep + os.environ.get("PATH", "")


def rodar_cli(argv):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = sys.stdout
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    import partitura
    partitura.main(argv)


# ----------------------------------------------------------------- janela

def ler_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def salvar_config(cfg):
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


def pasta_padrao():
    if not CONGELADO:
        return str(Path(__file__).parent / "saida")
    docs = Path.home() / "Documents"
    return str((docs if docs.exists() else Path.home()) / "Tumta")


def ler_area_transferencia():
    """Texto copiado (Ctrl+C) no Windows."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    if not user32.OpenClipboard(None):
        return ""
    try:
        h = user32.GetClipboardData(13)  # CF_UNICODETEXT
        if not h:
            return ""
        p = kernel32.GlobalLock(h)
        try:
            return ctypes.wstring_at(p) if p else ""
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


def quando(iso):
    try:
        d = datetime.fromisoformat(iso).date()
    except (TypeError, ValueError):
        return ""
    dias = (datetime.now().date() - d).days
    return "hoje" if dias == 0 else "ontem" if dias == 1 else f"{d.day} {MESES[d.month - 1]}"


def celulas_da_levada(lv):
    """Quadradinhos da levada: 8 colcheias, ou 16 semicolcheias se ela usa o 'e' e o 'a'."""
    base, maos = lv["base"], lv.get("maos", {})
    passo = 1 if any(base[p] >= 0 for p in range(1, 16, 2)) else 2
    return [{"tipo": NOMES_TIPO.get(base[p]), "mao": "DE"[maos[str(p)]] if str(p) in maos else ""}
            for p in range(0, 16, passo)]


class Api:
    def __init__(self):
        self._janela = None
        self._proc = None
        self._cancelado = False
        self._atual = None

    def _evento(self, **dados):
        if self._janela:
            self._janela.evaluate_js(f"window.evento({json.dumps(dados, ensure_ascii=False)})")

    # --- chamadas vindas da pagina

    def inicio(self):
        cfg = ler_config()
        return {"pasta_saida": cfg.get("pasta_saida") or pasta_padrao(),
                "instrumento": cfg.get("instrumento", "atabaque")}

    def escolher_musica(self):
        import webview
        r = self._janela.create_file_dialog(webview.FileDialog.OPEN, file_types=TIPOS_ARQUIVO)
        return self._info_musica(r[0]) if r else None

    def escolher_pasta(self):
        import webview
        r = self._janela.create_file_dialog(webview.FileDialog.FOLDER)
        if not r:
            return None
        cfg = ler_config()
        cfg["pasta_saida"] = r[0]
        salvar_config(cfg)
        return r[0]

    def colar(self):
        try:
            return ler_area_transferencia().strip()
        except Exception:
            return ""

    def verificar_link(self, url, pedido):
        """Procura o video sem baixar; a resposta chega como evento 'link'."""
        threading.Thread(target=self._verificar, args=(url, pedido), daemon=True).start()

    def recentes(self):
        itens = []
        for r in ler_config().get("recentes", []):
            if r["tipo"] == "arquivo" and not Path(r["fonte"]).exists():
                continue
            instrumento = "Atabaque" if r.get("instrumento") == "atabaque" else "Djembe"
            itens.append({**r, "meta": f"{instrumento} · {quando(r.get('quando'))}"})
        return itens[:6]

    def abrir(self, caminho):
        if caminho and Path(caminho).exists():
            os.startfile(caminho)

    def cancelar(self):
        self._cancelado = True
        if self._proc and self._proc.poll() is None:
            # derruba tambem os processos filhos (desenho em paralelo, ffmpeg)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self._proc.pid)],
                           capture_output=True, creationflags=SEM_JANELA)

    def gerar(self, op):
        if self._proc and self._proc.poll() is None:
            return False
        cfg = ler_config()
        cfg["instrumento"] = op["instrumento"]
        salvar_config(cfg)
        args = [op["musica"], "--saida", op["pasta_saida"], "--instrumento", op["instrumento"],
                "--velocidade", str(op["velocidade"])]
        if op.get("aulas"):
            args.append("--trechos")
        if op.get("clique"):
            args.append("--clique")
        if op.get("tres_sons"):
            args.append("--tres-sons")
        cmd = [sys.executable] if CONGELADO else [sys.executable, str(Path(__file__).resolve())]
        cmd += ["--cli"] + args
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self._cancelado = False
        self._atual = op
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, env=env, creationflags=SEM_JANELA)
        threading.Thread(target=self._acompanhar, args=(self._proc,), daemon=True).start()
        return True

    def resultado(self, pasta):
        """O que foi gerado, do jeito que a tela de 'Pronto!' mostra."""
        pasta = Path(pasta)
        r = {"pasta": str(pasta), "completo": None, "aulas": [], "trechos": [], "textos": []}
        if not pasta.exists():
            return r
        completos = sorted(pasta.glob("* - video completo.mp4"))
        r["completo"] = str(completos[0]) if completos else None
        try:
            dados = json.loads((pasta / "levadas.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            dados = {"levadas": []}
        por_numero = {lv["numero"]: lv for lv in dados.get("levadas", [])}
        for v in sorted((pasta / "aulas").glob("*.mp4"), key=lambda p: _ordem(p.name)):
            m = re.search(r"levada (\d+)", v.name)
            lv = por_numero.get(int(m.group(1))) if m else (dados["levadas"] or [None])[0]
            r["aulas"].append({"nome": v.stem.replace(" - aula", "").capitalize(), "caminho": str(v),
                               "celulas": celulas_da_levada(lv) if lv else []})
        for v in sorted((pasta / "trechos").glob("*.mp4"), key=lambda p: _ordem(p.name)):
            m = re.match(r"trecho (\d+) - (.+) \(levada (\d+)\)", v.stem)
            r["trechos"].append({"nome": f"Trecho {m.group(1)}" if m else v.stem, "caminho": str(v),
                                 "detalhe": f"{m.group(2)} · levada {m.group(3)}" if m else ""})
        for t in ("resumo.txt", "partitura.txt"):
            if (pasta / t).exists():
                r["textos"].append({"nome": t, "caminho": str(pasta / t)})
        return r

    def _info_musica(self, caminho):
        p = Path(caminho)
        tamanho = p.stat().st_size / 1e6 if p.exists() else 0
        return {"caminho": str(p), "nome": p.name, "mb": round(tamanho, 1)}

    def soltou_arquivo(self, caminho):
        self._evento(tipo="musica", **self._info_musica(caminho))

    # --- trabalho em segundo plano

    def _verificar(self, url, pedido):
        try:
            import baixar
            import yt_dlp
            try:
                info = baixar.informacoes(url)
                self._evento(tipo="link", pedido=pedido, estado="ok", **info)
            except yt_dlp.utils.DownloadError as erro:
                self._evento(tipo="link", pedido=pedido, estado=baixar.tipo_do_erro(erro))
        except Exception as erro:  # qualquer outra falha: trata como indisponivel
            self._evento(tipo="link", pedido=pedido, estado="indisponivel", detalhe=str(erro))

    def _guardar_recente(self, pasta):
        op = self._atual or {}
        if not op.get("musica"):
            return
        cfg = ler_config()
        item = {"nome": op.get("nome") or Path(op["musica"]).name, "fonte": op["musica"],
                "tipo": "youtube" if op.get("por_link") else "arquivo",
                "instrumento": op.get("instrumento"), "quando": datetime.now().isoformat(),
                "pasta": pasta}
        antigos = [r for r in cfg.get("recentes", []) if r.get("fonte") != item["fonte"]]
        cfg["recentes"] = [item] + antigos[:9]
        salvar_config(cfg)

    def _acompanhar(self, proc):
        etapa, pasta, cauda = None, None, []
        inicio = time.time()

        def mudar(nova, **extra):
            nonlocal etapa
            etapa = nova
            self._evento(tipo="etapa", etapa=nova, **extra)

        buf = b""
        while True:
            pedaco = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
            if not pedaco:
                break
            buf += pedaco
            partes = re.split(rb"[\r\n]", buf)
            buf = partes.pop()
            for bruto in partes:
                linha = bruto.decode("utf-8", "replace").strip()
                if not linha:
                    continue
                cauda = (cauda + [linha])[-12:]
                m = re.search(r"(\d+(?:\.\d+)?)%", linha)
                if linha.startswith("Baixando o audio"):
                    mudar("baixar")
                elif linha.startswith("Video: "):
                    self._evento(tipo="titulo", titulo=linha[7:])
                elif linha.startswith("Nao consegui baixar"):
                    erro = re.search(r"\((\w+)\)", linha)
                    self._evento(tipo="falha_link", estado=erro.group(1) if erro else "indisponivel")
                elif linha.startswith("Lendo "):
                    mudar("ler")
                elif linha.startswith("Separando"):
                    mudar("separar")
                elif linha.startswith("Usando a percussao"):
                    mudar("separar", pct=100)
                elif linha.startswith("Detectando"):
                    mudar("detectar")
                elif "batidas:" in linha:
                    self._evento(tipo="info", texto=linha)
                elif linha.startswith("Gerando o video..."):
                    mudar("video")
                elif linha.startswith("=== Levada"):
                    k = re.findall(r"\d+", linha)
                    mudar("aulas", k=int(k[0]), n=int(k[1]))
                elif linha.startswith("Cortando os trechos"):
                    mudar("trechos")
                elif linha.startswith("Tudo em:"):
                    pasta = linha.split(":", 1)[1].strip()
                elif m and etapa in ("baixar", "separar", "video", "aulas") and (
                        "video:" in linha or "download:" in linha or "|" in linha):
                    mb = re.search(r"de ([\d.]+) MB", linha)
                    self._evento(tipo="pct", etapa=etapa, pct=float(m.group(1)),
                                 mb=float(mb.group(1)) if mb else None)
        proc.wait()
        if self._cancelado:
            self._evento(tipo="cancelado")
        elif proc.returncode == 0 and pasta:
            self._guardar_recente(pasta)
            self._evento(tipo="fim", segundos=round(time.time() - inicio), **self.resultado(pasta))
        else:
            self._evento(tipo="erro", log="\n".join(cauda))


def _ordem(nome):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", nome)]


def abrir_janela():
    import webview
    from webview.dom import DOMEventHandler

    api = Api()
    janela = webview.create_window("Tumtá", str(BASE / "ui" / "index.html"), js_api=api,
                                   width=1060, height=800, min_size=(960, 720),
                                   background_color="#17110D")
    api._janela = janela

    def ao_soltar(e):
        arquivos = e.get("dataTransfer", {}).get("files", [])
        if arquivos and arquivos[0].get("pywebviewFullPath"):
            api.soltou_arquivo(arquivos[0]["pywebviewFullPath"])

    def ao_carregar():
        janela.dom.document.events.drop += DOMEventHandler(ao_soltar, True, True)

    janela.events.loaded += ao_carregar
    janela.events.closing += lambda: api.cancelar()
    webview.start()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    preparar_ambiente()
    if "--cli" in sys.argv:
        rodar_cli(sys.argv[sys.argv.index("--cli") + 1:])
    else:
        abrir_janela()
