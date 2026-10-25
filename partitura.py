"""Gera uma partitura animada de atabaque/djembe (TUM e TA, ou TUM/TOM/TA com --tres-sons)
a partir de uma musica ou video.

Uso:
    python partitura.py musica.mp3
    python partitura.py video.mp4 --clique --velocidade 0.75

Saidas (em ./saida/<nome da musica>/):
    <nome> - video completo.mp4   video com o som e as batidas animadas
    partitura.txt                 partitura em texto, compasso por compasso
    batidas.json                  todas as batidas com tempo, tipo e mao
    aulas/, trechos/, resumo.txt  com --aula ou --trechos
"""
import argparse
import importlib.util
import json
import re
import multiprocessing
import os
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import cv2
import librosa
import numpy as np
import soundfile as sf

import aula
from baixar import baixar, eh_link
from cena_tumta import CenaTumta
from tambor import CenaTambor

SR = 44100
TUM, TA, ABERTO = 0, 1, 2
D, E = 0, 1
NOMES = {TUM: "TUM", TA: "TA", ABERTO: "TOM"}
SIMBOLOS = {TUM: "O", TA: "X", ABERTO: "o"}
CORES = {TUM: (235, 150, 40), TA: (30, 140, 255), ABERTO: (90, 190, 70)}  # BGR: azul, laranja, verde

W, H, FPS = 1280, 720, 30
# no programa com janela, nao abre uma janela preta pra cada ffmpeg
SEM_JANELA = {"creationflags": 0x08000000} if os.name == "nt" else {}
HIT_X = 250
SEGUNDOS_FUTURO = 3.0
SEGUNDOS_PASSADO = HIT_X / ((W - HIT_X - 40) / SEGUNDOS_FUTURO)
PX_POR_SEG = (W - HIT_X - 40) / SEGUNDOS_FUTURO


# ---------------------------------------------------------------- audio

def extrair_audio(entrada, destino, inicio, duracao, velocidade):
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if inicio:
        cmd += ["-ss", str(inicio)]
    cmd += ["-i", str(entrada)]
    if duracao:
        cmd += ["-t", str(duracao)]
    if velocidade != 1:
        cmd += ["-af", f"atempo={velocidade}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(SR), str(destino)]
    if subprocess.run(cmd, **SEM_JANELA).returncode != 0:
        sys.exit(f"Nao consegui ler o audio de {entrada}")


def separar_percussao(wav, cache, usar_demucs):
    """Isola o tambor. Com Demucs o resultado e guardado em cache pra proxima vez."""
    if usar_demucs and cache.exists():
        print("Usando a percussao ja separada antes.")
        y, _ = librosa.load(cache, sr=SR, mono=True)
        return y
    if usar_demucs and importlib.util.find_spec("demucs"):
        print("Separando o tambor do resto da musica (Demucs)...", flush=True)
        try:
            import separacao
            audio, _ = sf.read(wav, dtype="float32", always_2d=True)
            y = separacao.separar_tambor(audio, SR)
            cache.parent.mkdir(parents=True, exist_ok=True)
            sf.write(cache, y, SR)
            return y
        except Exception as erro:  # sem memoria, modelo faltando...
            print(f"Demucs falhou ({erro}); usando separacao simples.")
    elif usar_demucs:
        print("Demucs nao instalado; usando separacao simples (veja o LEIAME).")
    y, _ = librosa.load(wav, sr=SR, mono=True)
    print("Separando a percussao (HPSS)...")
    return librosa.effects.percussive(y, margin=3.0)


# ------------------------------------------------------------- analise

def detectar_batidas(y_perc, sensibilidade):
    hop = 256
    env = librosa.onset.onset_strength(y=y_perc, sr=SR, hop_length=hop)
    # sensibilidade alta = limiar adaptativo mais baixo = pega batidas mais fracas
    delta = 0.02 + (1.0 - sensibilidade) * 0.3
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop, delta=delta,
                                        wait=int(0.06 * SR / hop), units="frames")
    return librosa.frames_to_time(frames, sr=SR, hop_length=hop)


def medir_golpes(y_perc, tempos):
    """Volume de cada golpe e o quanto ele e 'medio' em relacao ao grave (em dB)."""
    n_fft = 4096
    n = int(0.07 * SR)
    janela = np.hanning(n)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    grave = (freqs >= 40) & (freqs < 150)
    medio = (freqs >= 150) & (freqs < 4000)
    total = (freqs >= 40) & (freqs < 12000)
    volume, brilho = [], []
    for t in tempos:
        s = max(0, int((t - 0.01) * SR))
        seg = y_perc[s:s + n]
        seg = np.pad(seg, (0, n - len(seg))) * janela
        spec = np.abs(np.fft.rfft(seg, n_fft)) ** 2
        volume.append(10 * np.log10(spec[total].sum() + 1e-12))
        brilho.append(10 * np.log10((spec[medio].sum() + 1e-12) / (spec[grave].sum() + 1e-12)))
    return np.array(volume), np.array(brilho)


def medir_agudo(y_perc, tempos):
    """No comeco do golpe, quanto de agudo (estalo do tapa) ha em relacao ao corpo do som."""
    n_fft = 4096
    n = int(0.03 * SR)
    janela = np.hanning(n)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    corpo = (freqs >= 200) & (freqs < 1200)
    estalo = (freqs >= 2500) & (freqs < 9000)
    agudo = []
    for t in tempos:
        s = max(0, int((t - 0.005) * SR))
        seg = y_perc[s:s + n]
        seg = np.pad(seg, (0, n - len(seg))) * janela
        spec = np.abs(np.fft.rfft(seg, n_fft)) ** 2
        agudo.append(10 * np.log10((spec[estalo].sum() + 1e-12) / (spec[corpo].sum() + 1e-12)))
    return np.array(agudo)


def corte_dois_grupos(x):
    """Ponto que divide os valores em 2 grupos (k-means em 1 dimensao)."""
    c0, c1 = np.percentile(x, 25), np.percentile(x, 75)
    for _ in range(50):
        alto = np.abs(x - c1) < np.abs(x - c0)
        if alto.all() or not alto.any():
            break
        novo0, novo1 = x[~alto].mean(), x[alto].mean()
        if np.isclose(novo0, c0) and np.isclose(novo1, c1):
            break
        c0, c1 = novo0, novo1
    return (c0 + c1) / 2


def classificar(brilho, vies):
    """Separa em TUM (grave, centro) e TA (mais medio/agudo, tapa) com 2 grupos."""
    if len(brilho) < 2:
        return np.full(len(brilho), TUM, dtype=int)
    corte = corte_dois_grupos(brilho) + vies
    return np.where(brilho > corte, TA, TUM).astype(int)


def separar_aberto(tipos, agudo, vies):
    """Divide os golpes de borda em TA (tapa, estalado) e TOM (aberto, som cheio)."""
    borda = tipos == TA
    if borda.sum() < 2:
        return tipos
    corte = corte_dois_grupos(agudo[borda]) + vies
    tipos = tipos.copy()
    tipos[borda & (agudo <= corte)] = ABERTO
    return tipos


def acompanhar_pulso(y_perc, duracao):
    hop = 512
    env = librosa.onset.onset_strength(y=y_perc, sr=SR, hop_length=hop)
    tempo, frames = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=hop,
                                            start_bpm=100, tightness=1600)
    bpm = float(np.atleast_1d(tempo)[0])
    # pulso dobrado (comum com muita semicolcheia): fica com metade, na fase mais forte
    if bpm > 150 and len(frames) >= 4:
        par, impar = frames[::2], frames[1::2]
        frames = par if env[par].sum() >= env[impar].sum() else impar
        bpm /= 2
    pulsos = librosa.frames_to_time(frames, sr=SR, hop_length=hop)
    # pulso pela metade (padrao longo que se repete): coloca um pulso no meio de cada par
    if bpm < 65 and len(pulsos) >= 2:
        meios = (pulsos[:-1] + pulsos[1:]) / 2
        pulsos = np.sort(np.concatenate([pulsos, meios]))
        bpm *= 2
    if len(pulsos) < 2:
        bpm = bpm or 100.0
        pulsos = np.arange(0, duracao, 60.0 / bpm)
    return bpm, np.asarray(pulsos)


def grade_semicolcheias(pulsos, duracao):
    """4 subdivisoes por pulso; estende a grade ate o fim da musica."""
    intervalo = np.median(np.diff(pulsos))
    antes = np.arange(pulsos[0] - intervalo, -intervalo / 2, -intervalo)[::-1]
    depois = np.arange(pulsos[-1] + intervalo, duracao + intervalo, intervalo)
    p = np.concatenate([antes, pulsos, depois])
    slots = [np.linspace(a, b, 4, endpoint=False) for a, b in zip(p[:-1], p[1:])]
    return np.concatenate(slots), p


def atribuir_maos(tempos, slots):
    """Cada posicao da contagem tem sempre a mesma mao (aula.mao_da_posicao)."""
    passo = np.median(np.diff(slots))
    idx = np.clip(np.searchsorted(slots, tempos - passo / 2), 0, len(slots) - 1)
    em_semicolcheia = aula.usa_semicolcheias(idx)
    maos = np.zeros(len(tempos), dtype=int)
    anterior = E
    for k, i in enumerate(idx):
        maos[k] = aula.mao_da_posicao(int(i), em_semicolcheia, anterior)
        anterior = maos[k]
    return maos


def quantizar(tempos, tipos, maos, slots):
    grade, grade_maos = {}, {}
    if len(slots) < 2:
        return grade, grade_maos
    passo = np.median(np.diff(slots))
    for t, tipo, mao in zip(tempos, tipos, maos):
        i = int(np.argmin(np.abs(slots - t)))
        if abs(slots[i] - t) <= passo / 2 and grade.get(i) != TUM:
            grade[i], grade_maos[i] = tipo, mao
    return grade, grade_maos


# -------------------------------------------------------------- saidas

def achar_deslocamento(grade):
    """Escolhe qual pulso e o '1' do compasso: onde mais cai TUM."""
    melhor, pontos_max = 0, -1
    for d in range(0, 16, 4):
        pontos = sum(1 for i, tipo in grade.items() if tipo == TUM and (i + d) % 16 == 0)
        if pontos > pontos_max:
            melhor, pontos_max = d, pontos
    return melhor


def compasso_de(i, deslocamento):
    return (i + deslocamento) // 16, (i + deslocamento) % 16


def legenda(tres_sons):
    if tres_sons:
        return ("O = TUM (grave, centro)   o = TOM (aberto, borda)   X = TA (tapa, borda)"
                "   . = pausa")
    return "O = TUM (grave, centro do couro)   X = TA (tapa, borda)   . = pausa"


def escrever_txt(caminho, grade, grade_maos, deslocamento, bpm, titulo, tres_sons=False):
    linhas = [f"Partitura de atabaque/djembe - {titulo}",
              f"Andamento: ~{bpm:.0f} BPM  |  compasso 4/4, 4 subdivisoes por tempo",
              "Legenda:  " + legenda(tres_sons),
              "          D = mao direita   E = mao esquerda (linha de baixo de cada compasso)",
              "",
              "comp  |  1 e & a | 2 e & a | 3 e & a | 4 e & a |"]
    if grade:
        compassos = sorted({compasso_de(i, deslocamento)[0] for i in grade})
        for m in range(compassos[0], compassos[-1] + 1):
            celulas, maos = [], []
            for pos in range(16):
                i = m * 16 + pos - deslocamento
                tipo = grade.get(i)
                celulas.append(SIMBOLOS[tipo] if tipo is not None else ".")
                maos.append("DE"[grade_maos[i]] if tipo is not None else " ")
            grupos = [" ".join(celulas[k:k + 4]) for k in range(0, 16, 4)]
            grupos_maos = [" ".join(maos[k:k + 4]) for k in range(0, 16, 4)]
            linhas.append(f"{m + 1:>4}  |  " + " |  ".join(grupos) + " |")
            linhas.append("      |  " + " |  ".join(grupos_maos) + " |")
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


def gerar_cliques(tempos, tipos, n_amostras):
    out = np.zeros(n_amostras, dtype=np.float32)
    rng = np.random.default_rng(0)
    for t, tipo in zip(tempos, tipos):
        s = int(t * SR)
        if tipo == TUM:
            tt = np.arange(int(0.2 * SR)) / SR
            som = np.sin(2 * np.pi * 85 * tt) * np.exp(-tt * 22)
        elif tipo == ABERTO:
            tt = np.arange(int(0.15 * SR)) / SR
            som = np.sin(2 * np.pi * 420 * tt) * np.exp(-tt * 30) * 0.7
        else:
            tt = np.arange(int(0.06 * SR)) / SR
            ruido = np.diff(rng.standard_normal(len(tt) + 1))
            som = (0.5 * ruido + np.sin(2 * np.pi * 1800 * tt)) * np.exp(-tt * 70) * 0.6
        fim = min(n_amostras, s + len(som))
        out[s:fim] += som[:fim - s]
    return out


def ascii(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


# --------------------------------------------------------------- video

PAINEL_Y = 380  # a faixa de toques ocupa a parte de baixo da tela
LANE_Y = {TA: 432, TUM: 512}  # o TOM (aberto) anda na faixa da borda, junto com o TA
LANE_ALT = 70
GRADE_Y = 575


def tem_video(entrada):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream_disposition=attached_pic", "-of", "json", str(entrada)],
                         capture_output=True, text=True, **SEM_JANELA).stdout
    streams = json.loads(out or "{}").get("streams", [])
    return any(not s.get("disposition", {}).get("attached_pic") for s in streams)


def ler_frames(entrada, inicio, duracao, velocidade):
    """Gera os frames do video original ja no tamanho e velocidade do video final."""
    cmd = ["ffmpeg", "-loglevel", "error"]
    if inicio:
        cmd += ["-ss", str(inicio)]
    cmd += ["-i", str(entrada)]
    if duracao:
        cmd += ["-t", str(duracao)]
    filtro = (f"setpts=PTS/{velocidade},fps={FPS},"
              f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2")
    cmd += ["-an", "-vf", filtro, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **SEM_JANELA)
    tamanho = W * H * 3
    ultimo = np.zeros((H, W, 3), np.uint8)
    while True:
        dados = proc.stdout.read(tamanho)
        if len(dados) == tamanho:
            ultimo = np.frombuffer(dados, np.uint8).reshape(H, W, 3)
        yield ultimo


def texto(img, s, x, y, escala, cor, esp=1, centro=False, fonte=cv2.FONT_HERSHEY_DUPLEX):
    if centro:
        (lw, lh), _ = cv2.getTextSize(s, fonte, escala, esp)
        x, y = int(x - lw / 2), int(y + lh / 2)
    cv2.putText(img, s, (int(x), int(y)), fonte, escala, (0, 0, 0), esp + 3, cv2.LINE_AA)
    cv2.putText(img, s, (int(x), int(y)), fonte, escala, cor, esp, cv2.LINE_AA)


def escurecer(cor, f):
    return tuple(int(c * f) for c in cor)


def painel_estatico():
    img = np.full((H - PAINEL_Y, W, 3), (24, 20, 18), np.uint8)
    for tipo, y in LANE_Y.items():
        y -= PAINEL_Y
        cv2.rectangle(img, (0, y - LANE_ALT // 2), (W, y + LANE_ALT // 2), (40, 35, 31), -1)
        texto(img, NOMES[tipo], 25, y + 10, 0.9, CORES[tipo], 2)
    cv2.line(img, (HIT_X, LANE_Y[TA] - PAINEL_Y - LANE_ALT // 2 - 6),
             (HIT_X, LANE_Y[TUM] - PAINEL_Y + LANE_ALT // 2 + 6), (240, 240, 240), 3)
    return img


def faixa(tipo):
    return LANE_Y[TUM if tipo == TUM else TA]


def desenhar_frame(fundo, painel, t, dados):
    img = fundo.copy()
    tempos, tipos, pulsos_ext, slots, grade, desl, bpm, total, titulo, velocidade = dados
    img[PAINEL_Y:] = cv2.addWeighted(img[PAINEL_Y:], 0.08, painel, 0.92, 0)

    # cabecalho: titulo, tempo, BPM e barra de progresso
    cv2.rectangle(img, (0, 0), (W, 62), (0, 0, 0), -1)
    texto(img, ascii(titulo)[:55], 20, 30, 0.7, (235, 235, 235))
    vel = f"  vel {velocidade:g}x" if velocidade != 1 else ""
    texto(img, f"{int(t // 60)}:{int(t % 60):02d} / {int(total // 60)}:{int(total % 60):02d}"
               f"  ~{bpm:.0f} BPM{vel}", W - 400, 30, 0.6, (200, 200, 200))
    cv2.rectangle(img, (20, 46), (W - 20, 52), (60, 55, 50), -1)
    cv2.rectangle(img, (20, 46), (20 + int((W - 40) * min(1, t / total)), 52), (120, 200, 120), -1)

    t0, t1 = t - SEGUNDOS_PASSADO, t + SEGUNDOS_FUTURO
    topo, base_y = LANE_Y[TA] - LANE_ALT // 2, LANE_Y[TUM] + LANE_ALT // 2

    # linhas de pulso e de compasso
    a, b = np.searchsorted(pulsos_ext, [t0, t1])
    for k in range(a, b):
        x = int(HIT_X + (pulsos_ext[k] - t) * PX_POR_SEG)
        m, pos = compasso_de(k * 4, desl)
        if pos == 0:
            cv2.line(img, (x, topo), (x, base_y), (120, 115, 110), 2)
            texto(img, str(m + 1), x + 4, topo - 6, 0.5, (170, 170, 170), 1,
                  fonte=cv2.FONT_HERSHEY_SIMPLEX)
        else:
            cv2.line(img, (x, topo), (x, base_y), (65, 60, 56), 1)

    # notas andando ate a linha de batida
    a, b = np.searchsorted(tempos, [t0, t1])
    ultima = None
    for k in range(a, b):
        nt, tipo = tempos[k], tipos[k]
        x = int(HIT_X + (nt - t) * PX_POR_SEG)
        cor = CORES[tipo] if nt >= t else escurecer(CORES[tipo], 0.35)
        cv2.circle(img, (x, faixa(tipo)), 27, cor, -1, cv2.LINE_AA)
        texto(img, NOMES[tipo], x, faixa(tipo), 0.5, (255, 255, 255), 1, centro=True)
        if nt <= t:
            ultima = k

    # nomes das faixas por cima das notas que ja passaram
    for tipo, y in LANE_Y.items():
        cv2.rectangle(img, (0, y - LANE_ALT // 2), (105, y + LANE_ALT // 2), (40, 35, 31), -1)
        texto(img, NOMES[tipo], 25, y + 10, 0.9, CORES[tipo], 2)

    # brilho na hora de bater + palavra grande em cima do video
    if ultima is not None and t - tempos[ultima] < 0.25:
        f = 1 - (t - tempos[ultima]) / 0.25
        tipo = tipos[ultima]
        cv2.circle(img, (HIT_X, faixa(tipo)), int(32 + 16 * f), CORES[tipo], 4, cv2.LINE_AA)
        texto(img, NOMES[tipo] + "!", W // 2, 220, 2.6 + 0.8 * f,
              escurecer(CORES[tipo], 0.75 + 0.25 * f), 6, centro=True)

    # partitura do compasso atual, contando 1 e & a 2 e & a ...
    i_atual = max(0, int(np.searchsorted(slots, t, "right")) - 1)
    m, _ = compasso_de(i_atual, desl)
    inicio = m * 16 - desl
    x0, lado = 70, (W - 140) // 16
    texto(img, f"Compasso {m + 1}", 20, GRADE_Y - 12, 0.55, (210, 210, 210), 1)
    for pos in range(16):
        i = inicio + pos
        x = x0 + pos * lado
        tipo = grade.get(i)
        cor_caixa = CORES[tipo] if tipo is not None else (55, 50, 46)
        if i < i_atual and tipo is not None:
            cor_caixa = escurecer(cor_caixa, 0.55)
        cv2.rectangle(img, (x + 3, GRADE_Y), (x + lado - 3, GRADE_Y + 75), cor_caixa, -1)
        if tipo is not None:
            texto(img, NOMES[tipo], x + lado / 2, GRADE_Y + 37, 0.65, (255, 255, 255), 2,
                  centro=True)
        contagem = str(pos // 4 + 1) if pos % 4 == 0 else ("&" if pos % 4 == 2 else "")
        if contagem:
            texto(img, contagem, x + lado / 2, GRADE_Y + 100, 0.75 if pos % 4 == 0 else 0.5,
                  (230, 230, 230) if pos % 4 == 0 else (140, 140, 140), 1, centro=True)
        if pos and pos % 4 == 0:
            cv2.line(img, (x, GRADE_Y - 6), (x, GRADE_Y + 81), (160, 160, 160), 2)
        if i == i_atual:
            cv2.rectangle(img, (x + 1, GRADE_Y - 3), (x + lado - 1, GRADE_Y + 78),
                          (255, 255, 255), 3)
    return img


_CENA = None


def _iniciar_processo(cena):
    global _CENA
    _CENA = cena


def _quadro(f):
    return _CENA.frame(f / FPS).tobytes()


def renderizar(caminho, audio, total, gerar_frame=None, cena=None):
    """Gera o video. Com `cena`, desenha os quadros em paralelo em todos os nucleos."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(audio),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(caminho)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, **SEM_JANELA)
    n = int(total * FPS) + 1

    def escrever(quadros):
        for f, dados in enumerate(quadros):
            proc.stdin.write(dados)
            if f % (FPS * 5) == 0:
                print(f"  video: {100 * f / n:5.1f}%", end=chr(13), flush=True)

    if cena is not None:
        nucleos = max(1, (os.cpu_count() or 2) - 1)
        lote = nucleos * 6  # em lotes, pra memoria nao encher se o ffmpeg ficar pra tras

        def quadros(pool):
            for ini in range(0, n, lote):
                yield from pool.map(_quadro, range(ini, min(n, ini + lote)), chunksize=2)

        with multiprocessing.Pool(nucleos, _iniciar_processo, (cena,)) as pool:
            escrever(quadros(pool))
    else:
        escrever(gerar_frame(f / FPS).tobytes() for f in range(n))
    proc.stdin.close()
    proc.wait()
    print("  video: 100.0%")
    if proc.returncode != 0:
        sys.exit("ffmpeg falhou ao gerar o video")


def cena_faixas(entrada, args, dados):
    frames = None
    if not args.sem_fundo and tem_video(entrada):
        frames = ler_frames(entrada, args.inicio, args.duracao, args.velocidade)
    painel = painel_estatico()
    fundo_liso = np.full((H, W, 3), (30, 26, 24), np.uint8)
    return lambda t: desenhar_frame(next(frames) if frames else fundo_liso, painel, t, dados)


def fmt_tempo(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def nova_cena(estilo, *dados, etapas=None, tres_sons=False, instrumento="djembe", levada=None,
              trechos=None, titulo_tumta=None):
    """Cena do video no estilo escolhido (tumta = layout novo; tambor = desenho anterior)."""
    if estilo == "tambor":
        return CenaTambor(*dados, etapas, tres_sons, instrumento)
    dados = list(dados)
    if titulo_tumta:
        dados[10] = titulo_tumta
    return CenaTumta(*dados, etapas=etapas, tres_sons=tres_sons, instrumento=instrumento,
                     levada=levada, trechos=trechos)


def gerar_aula(info, titulo, nome, tmp, bpm, audio, slots, desl, tres_sons=False,
               instrumento="djembe", estilo="tumta", musica=None, levada=None):
    """Video-aula passo a passo de uma levada. Devolve o texto da levada."""
    tempos, tipos, maos, slots_aula, etapas, som, maos_base = aula.montar_aula(
        info, bpm, audio, slots, desl, SR)
    resumo = (f"{titulo}\n"
              f"Aparece em {info['apoio']} de {info['total']} compassos com tambor.\n"
              + ("O = TUM   o = TOM (aberto)   X = TA (tapa)" if tres_sons else "O = TUM   X = TA")
              + "   D/E = mao direita/esquerda\n\n"
              + aula.texto_base(info["base"], maos_base) + "\n\nPassos do video:\n"
              + "\n".join(f"  {e['numero']}. {e['titulo']} ({e['bpm']:.0f} BPM)" for e in etapas))
    Path(f"{nome}.txt").write_text(resumo + "\n", encoding="utf-8")
    print("\n" + resumo + "\n")

    wav_aula = Path(tmp) / "aula.wav"
    sf.write(wav_aula, som, SR)
    total = len(som) / SR
    grade_aula = dict(zip(np.searchsorted(slots_aula, tempos - 1e-6).tolist(), tipos.tolist()))
    cena = nova_cena(estilo, tempos, tipos, maos, slots_aula, grade_aula, {}, 0, compasso_de, bpm,
                     total, titulo, 1.0, slots_aula[::4], etapas=etapas, tres_sons=tres_sons,
                     instrumento=instrumento, levada=levada, titulo_tumta=musica)
    print("Gerando o video-aula...")
    renderizar(f"{nome}.mp4", wav_aula, total, cena=cena)
    print(f"  {nome}.mp4")
    return resumo


def nome_da_musica(stem):
    """Nome limpo pra pasta: sem '(320k)', '- Topic' e afins do download."""
    nome = re.sub(r"\s*\(\d+\s*k(bps)?\)\s*$", "", stem, flags=re.I)
    nome = re.sub(r"\s*\d+\s*kbps\s*$", "", nome, flags=re.I)
    nome = re.sub(r"^\S+\.(com|net|org|io)\S*\s+", "", nome, flags=re.I)
    nome = re.sub(r"\s*-\s*Topic\s*$", "", nome, flags=re.I)
    return nome.strip() or stem


def salvar_levadas(pasta, levadas, trechos=()):
    """Padrao e mao de cada levada (e os trechos), pra janela do programa desenhar."""
    dados = {
        "levadas": [{"numero": k, "base": [int(v) for v in info["base"]],
                     "maos": {str(p): int(m) for p, m in aula.maos_da_base(info["base"]).items()}}
                    for k, info in enumerate(levadas, 1)],
        "trechos": [{"numero": n, "inicio": round(t["ini"], 2), "fim": round(t["fim_s"], 2),
                     "levada": t["levada"] + 1} for n, t in enumerate(trechos, 1)],
    }
    (pasta / "levadas.json").write_text(json.dumps(dados, indent=1), encoding="utf-8")


def trechos_no_tempo(grade, desl, slots):
    """[(inicio_s, fim_s, levada)] pra o video completo marcar os trechos e a levada atual."""
    levadas = aula.achar_levadas(grade, desl, compasso_de)
    if not levadas:
        return []
    def tempo(m):
        return float(slots[min(len(slots) - 1, max(0, m * 16 - desl))])
    return [(tempo(t["inicio"]), tempo(t["fim"] + 1), t["levada"] + 1)
            for t in aula.achar_trechos(levadas, grade, desl, compasso_de)]


def gerar_aulas(args, nome, pasta, tmp, grade, desl, bpm, audio, slots, video_principal):
    pasta_aulas = pasta / "aulas"
    pasta_aulas.mkdir(exist_ok=True)
    if not args.trechos:
        info = aula.achar_base(grade, desl, compasso_de)
        if info is None:
            print("Nao achei uma levada que se repete pra montar a aula.")
            return
        salvar_levadas(pasta, [info])
        gerar_aula(info, f"Aula: {nome}", pasta_aulas / "levada base - aula", tmp, bpm,
                   audio, slots, desl, args.tres_sons, args.instrumento, args.estilo, nome)
        return

    levadas = aula.achar_levadas(grade, desl, compasso_de)
    if not levadas:
        print("Nao achei levadas que se repetem pra montar as aulas.")
        return
    trechos = aula.achar_trechos(levadas, grade, desl, compasso_de)

    def tempo_compasso(m):
        i = min(len(slots) - 1, max(0, m * 16 - desl))
        return float(slots[i])

    linhas = [f"Trechos e levadas - {nome}", ""]
    for n, t in enumerate(trechos, 1):
        ini, fim = tempo_compasso(t["inicio"]), tempo_compasso(t["fim"] + 1)
        t["ini"], t["fim_s"] = ini, fim
        linhas.append(f"  Trecho {n}: {fmt_tempo(ini)} a {fmt_tempo(fim)}  "
                      f"(compassos {t['inicio'] + 1}-{t['fim'] + 1})  ->  levada {t['levada'] + 1}")
    linhas.append("")

    for k, info in enumerate(levadas, 1):
        titulo = f"Levada {k}: {nome}"
        print(f"\n=== Levada {k} de {len(levadas)} ===")
        resumo = gerar_aula(info, titulo, pasta_aulas / f"levada {k} - aula", tmp, bpm, audio,
                            slots, desl, args.tres_sons, args.instrumento, args.estilo, nome, k)
        linhas += [resumo, "", "-" * 60, ""]

    if video_principal and Path(video_principal).exists():
        print("\nCortando os trechos do video principal...")
        pasta_trechos = pasta / "trechos"
        pasta_trechos.mkdir(exist_ok=True)
        for n, t in enumerate(trechos, 1):
            ini = max(0.0, t["ini"] - 2.0)
            arquivo = (f"trecho {n} - {fmt_tempo(t['ini']).replace(':', 'm')}s a "
                       f"{fmt_tempo(t['fim_s']).replace(':', 'm')}s (levada {t['levada'] + 1}).mp4")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ini:.2f}",
                   "-to", f"{t['fim_s'] + 0.5:.2f}", "-i", str(video_principal),
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac",
                   "-b:a", "192k", str(pasta_trechos / arquivo)]
            subprocess.run(cmd, **SEM_JANELA)
            print(f"  {arquivo}")

    Path(pasta / "resumo.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    salvar_levadas(pasta, levadas, trechos)
    print(f"\nAulas e trechos em: {pasta}")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Partitura animada de atabaque/djembe (TUM e TA)")
    ap.add_argument("entrada", help="arquivo de musica ou video (mp3, wav, mp4, ...) ou link do YouTube")
    ap.add_argument("--saida", default=str(Path(__file__).parent / "saida"), help="pasta de saida")
    ap.add_argument("--clique", action="store_true",
                    help="mistura um som de TUM/TA no audio do video, pra conferir as batidas")
    ap.add_argument("--rapido", action="store_true",
                    help="nao usa o Demucs (mais rapido, mas erra mais em musica cheia)")
    ap.add_argument("--sensibilidade", type=float, default=0.75,
                    help="0 a 1; maior = pega mais batidas fracas (padrao 0.75)")
    ap.add_argument("--vies", type=float, default=0.0,
                    help="ajusta a divisao TUM/TA: positivo = mais TUM, negativo = mais TA")
    ap.add_argument("--instrumento", choices=["djembe", "atabaque"], default="djembe",
                    help="desenho do tambor no video")
    ap.add_argument("--tres-sons", action="store_true",
                    help="separa os toques de borda em TOM (aberto) e TA (tapa)")
    ap.add_argument("--vies-tapa", type=float, default=0.0,
                    help="com --tres-sons: positivo = mais TOM, negativo = mais TA")
    ap.add_argument("--deslocar-compasso", type=int, default=0,
                    help="se o '1' do compasso ficar no lugar errado, mude por 1, 2 ou 3 tempos")
    ap.add_argument("--inicio", type=float, default=0, help="comeca a partir deste segundo")
    ap.add_argument("--duracao", type=float, default=0, help="processa so estes segundos")
    ap.add_argument("--velocidade", type=float, default=1.0,
                    help="deixa a musica mais lenta pra treinar, ex: 0.75 (0.5 a 2)")
    ap.add_argument("--estilo", choices=["tumta", "tambor", "faixas"], default="tumta",
                    help="tumta = layout com fotos das maos (padrao); tambor = desenho anterior; "
                         "faixas = notas andando em linhas")
    ap.add_argument("--sem-fundo", action="store_true",
                    help="no estilo faixas, nao mostra o video original no fundo")
    ap.add_argument("--aula", action="store_true",
                    help="gera tambem um video-aula passo a passo da levada base")
    ap.add_argument("--trechos", action="store_true",
                    help="divide a musica em trechos por levada: uma aula por levada + cortes")
    ap.add_argument("--sem-video", action="store_true", help="gera so a partitura em texto/json")
    args = ap.parse_args(argv)

    if not 0.5 <= args.velocidade <= 2:
        sys.exit("--velocidade precisa estar entre 0.5 e 2")
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    if eh_link(args.entrada):
        entrada, titulo = baixar(args.entrada, saida / ".downloads")
        nome = nome_da_musica(re.sub(r'[<>:"/\|?*]', "", titulo).strip())
    else:
        entrada = Path(args.entrada)
        if not entrada.exists():
            sys.exit(f"Arquivo nao encontrado: {entrada}")
        nome = nome_da_musica(entrada.stem)
    pasta = saida / nome
    if args.velocidade != 1:
        pasta = saida / f"{nome} ({args.velocidade:g}x)"
    pasta.mkdir(parents=True, exist_ok=True)
    video_completo = pasta / f"{nome} - video completo.mp4"
    txt_partitura, json_batidas = pasta / "partitura.txt", pasta / "batidas.json"

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print(f"Lendo {entrada.name}...")
        extrair_audio(entrada, wav, args.inicio, args.duracao, args.velocidade)
        audio, _ = sf.read(wav, dtype="float32", always_2d=True)
        total = len(audio) / SR

        chave = (f"{entrada.stem}_{entrada.stat().st_size}_{args.inicio:g}_{args.duracao:g}"
                 f"_{args.velocidade:g}.wav")
        y_perc = separar_percussao(wav, saida / ".cache" / chave, not args.rapido)
        print("Detectando as batidas...")
        tempos = detectar_batidas(y_perc, args.sensibilidade)
        volume, brilho = medir_golpes(y_perc, tempos)
        # ignora golpes muito mais fracos que os fortes (chocalho, vazamento de outros sons)
        if len(tempos):
            manter = volume >= np.percentile(volume, 90) - (8 + 14 * args.sensibilidade)
            tempos, volume, brilho = tempos[manter], volume[manter], brilho[manter]
        if len(tempos) == 0:
            sys.exit("Nenhuma batida encontrada. Tente --sensibilidade 0.9")
        tipos = classificar(brilho, args.vies)
        if args.tres_sons:
            tipos = separar_aberto(tipos, medir_agudo(y_perc, tempos), args.vies_tapa)
        forca = volume - volume.max()
        bpm, pulsos = acompanhar_pulso(y_perc, total)
        slots, pulsos_ext = grade_semicolcheias(pulsos, total)
        maos = atribuir_maos(tempos, slots)
        grade, grade_maos = quantizar(tempos, tipos, maos, slots)
        desl = achar_deslocamento(grade)
        if args.deslocar_compasso:
            desl = (desl - args.deslocar_compasso * 4) % 16

        contagem = ", ".join(f"{int((tipos == k).sum())} {NOMES[k]}"
                             for k in ((TUM, ABERTO, TA) if args.tres_sons else (TUM, TA)))
        print(f"  {len(tempos)} batidas: {contagem}  |  ~{bpm:.0f} BPM")

        escrever_txt(txt_partitura, grade, grade_maos, desl, bpm, nome, args.tres_sons)
        json_batidas.write_text(json.dumps({
            "arquivo": entrada.name, "bpm": round(bpm, 1),
            "pulsos": [round(float(p), 3) for p in pulsos],
            "batidas": [{"tempo": round(float(t), 3), "tipo": NOMES[int(k)],
                         "mao": "DE"[int(mao)], "volume_db": round(float(f), 1)}
                        for t, k, mao, f in zip(tempos, tipos, maos, forca)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")

        if not args.sem_video:
            audio_video = wav
            if args.clique:
                cliques = gerar_cliques(tempos, tipos, len(audio))
                mix = audio * 0.75 + cliques[:, None] * 0.5
                pico = np.abs(mix).max()
                if pico > 0.99:
                    mix /= pico / 0.99
                audio_video = Path(tmp) / "mix.wav"
                sf.write(audio_video, mix, SR)
            print("Gerando o video...")
            if args.estilo == "faixas":
                dados = (tempos, tipos, pulsos_ext, slots, grade, desl, bpm, total, nome,
                         args.velocidade)
                renderizar(video_completo, audio_video, total,
                           gerar_frame=cena_faixas(entrada, args, dados))
            else:
                cena = nova_cena(args.estilo, tempos, tipos, maos, slots, grade, grade_maos, desl,
                                 compasso_de, bpm, total, nome, args.velocidade, pulsos_ext,
                                 tres_sons=args.tres_sons, instrumento=args.instrumento,
                                 trechos=trechos_no_tempo(grade, desl, slots) if args.trechos else None)
                renderizar(video_completo, audio_video, total, cena=cena)

        if args.aula or args.trechos:
            video = None if args.sem_video else video_completo
            gerar_aulas(args, nome, pasta, tmp, grade, desl, bpm, audio, slots, video)

    print("\nPronto!")
    print(f"  Tudo em: {pasta}")


if __name__ == "__main__":
    main()
