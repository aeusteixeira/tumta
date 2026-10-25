"""Cena do video: tambor visto de cima com duas maos batendo TUM (centro) e TA (borda)."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
TUM, TA, ABERTO = 0, 1, 2
D, E = 0, 1
NOMES = {TUM: "TUM", TA: "TA", ABERTO: "TOM"}
NOMES_MAO = {D: "mão direita", E: "mão esquerda"}
LETRA_MAO = {D: "D", E: "E"}
CORES = {TUM: (235, 150, 40), TA: (30, 140, 255), ABERTO: (90, 190, 70)}  # BGR: azul, laranja, verde
FUNDO = (28, 24, 22)

CENTRO = np.array([400.0, 350.0])
RAIO = 250
PAINEL_X = 800

# linha do tempo vertical: as batidas sobem ate LINHA_Y
COLUNAS = {E: (840, 1030), D: (1060, 1250)}
LINHA_Y = 270
LINHA_FIM = 670
PX_SEG = 165

FONTE = "C:/Windows/Fonts/segoeui.ttf"
FONTE_NEGRITO = "C:/Windows/Fonts/segoeuib.ttf"

# pontos de referencia dentro do desenho da mao (pixels do sprite)
PALMA = np.array([120.0, 250.0])
DEDOS = np.array([120.0, 100.0])
ESCALA = 0.85


# ---------------------------------------------------------------- texto

@lru_cache(maxsize=512)
def _mascara_texto(s, tamanho, negrito, ancora):
    fonte = ImageFont.truetype(FONTE_NEGRITO if negrito else FONTE, tamanho)
    x0, y0, x1, y1 = fonte.getbbox(s, anchor=ancora)
    img = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
    ImageDraw.Draw(img).text((-x0, -y0), s, font=fonte, fill=255, anchor=ancora)
    return np.asarray(img, np.float32) / 255.0, x0, y0


def texto(img, s, x, y, tamanho, cor, negrito=False, alpha=1.0, ancora="la"):
    mascara, dx, dy = _mascara_texto(s, tamanho, negrito, ancora)
    colar_mascara(img, mascara, int(x + dx), int(y + dy), cor, alpha)


def largura_texto(s, tamanho, negrito=False):
    return _mascara_texto(s, tamanho, negrito, "la")[0].shape[1]


def colar_mascara(img, mascara, x, y, cor, alpha=1.0):
    h, w = mascara.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = mascara[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + np.array(cor, np.float32) * a).astype(np.uint8)


# ----------------------------------------------------------------- maos

def desenhar_mao(direita):
    """Costas da mao direita vista de cima, dedos para cima. A esquerda e o espelho."""
    k = 3
    sw, sh = 240, 700
    p = lambda x, y: (int(x * k), int(y * k))
    pele = np.array([118, 160, 212], np.float32)

    antebraco = np.array([p(84, 300), p(158, 300), p(172, 700), p(68, 700)], np.int32)
    palma = np.array([p(68, 200), p(174, 196), p(168, 262), p(154, 308), p(88, 310),
                      p(72, 270)], np.int32)
    dedos = [(86, 204, 80, 50, 13), (111, 196, 110, 30, 14), (136, 198, 139, 42, 13),
             (160, 206, 167, 86, 11)]
    polegar = (74, 284, 30, 214, 15)

    # silhueta inteira de uma vez, contorno so por fora
    silhueta = np.zeros((sh * k, sw * k), np.uint8)
    cv2.fillPoly(silhueta, [antebraco, palma], 255, cv2.LINE_AA)
    cv2.polylines(silhueta, [palma], True, 255, 16 * k, cv2.LINE_AA)
    for x0, y0, x1, y1, r in dedos + [polegar]:
        cv2.line(silhueta, p(x0, y0), p(x1, y1), 255, int(2 * r * k), cv2.LINE_AA)
    contorno = silhueta.astype(np.float32) / 255
    miolo = cv2.erode(silhueta, np.ones((7, 7), np.uint8)).astype(np.float32) / 255
    miolo = cv2.GaussianBlur(miolo, (0, 0), 1.5)

    cor = np.zeros((sh * k, sw * k, 3), np.float32)
    cor[:] = pele
    # leve volume: mais escuro nas laterais de cada dedo e no antebraco
    volume = np.zeros(silhueta.shape, np.float32)
    for x0, y0, x1, y1, r in dedos + [polegar]:
        cv2.line(volume, p(x0, y0), p(x1, y1), 1.0, int(r * 0.9 * k), cv2.LINE_AA)
    cv2.fillPoly(volume, [palma], 1.0)
    cv2.fillPoly(volume, [np.array([p(96, 300), p(146, 300), p(156, 700), p(84, 700)])], 0.8)
    volume = cv2.GaussianBlur(volume, (0, 0), 6 * k)[..., None]
    cor = cor * (0.82 + 0.22 * volume)

    # separacao entre os dedos, unhas e dobras
    for (xa, ya, xa1, ya1, _), (xb, yb, xb1, yb1, _) in zip(dedos, dedos[1:]):
        topo = ((xa1 + xb1) / 2, max(ya1, yb1) + 18)
        cv2.line(cor, p(*topo), p((xa + xb) / 2, (ya + yb) / 2 - 6), (78, 112, 160), 2 * k,
                 cv2.LINE_AA)
    for x0, y0, x1, y1, r in dedos + [polegar]:
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) + 90
        cv2.ellipse(cor, p(x1 + (x0 - x1) * 0.10, y1 + (y0 - y1) * 0.10),
                    (int(r * 0.6 * k), int(r * 0.85 * k)), ang, 0, 360, (166, 186, 230), -1,
                    cv2.LINE_AA)
        for frac in (0.48, 0.72):
            cx, cy = x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
            cv2.ellipse(cor, p(cx, cy), (int(r * 0.5 * k), int(2.5 * k)), ang - 90, 200, 340,
                        (92, 128, 178), k, cv2.LINE_AA)

    borda = np.array([58, 88, 132], np.float32)
    cor = cor * miolo[..., None] + borda * (1 - miolo[..., None])
    sprite = np.dstack([np.clip(cor, 0, 255), contorno * 255]).astype(np.uint8)
    sprite = cv2.resize(sprite, (sw, sh), interpolation=cv2.INTER_AREA)
    if not direita:
        sprite = cv2.flip(sprite, 1)
    sombra = cv2.GaussianBlur(sprite[..., 3], (0, 0), 9)
    return sprite, sombra


def colar_sprite(img, sprite, M, alpha=1.0, sombra=False):
    h, w = sprite.shape[:2]
    cantos = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    x0, y0 = max(0, int(cantos[:, 0].min())), max(0, int(cantos[:, 1].min()))
    x1, y1 = min(W, int(cantos[:, 0].max()) + 1), min(H, int(cantos[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy()
    M2[:, 2] -= (x0, y0)
    torto = cv2.warpAffine(sprite, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    roi = img[y0:y1, x0:x1]
    if sombra:
        a = torto.astype(np.float32)[..., None] / 255 * alpha
        roi[:] = (roi * (1 - a)).astype(np.uint8)
    else:
        a = torto[..., 3:4].astype(np.float32) / 255 * alpha
        roi[:] = (roi * (1 - a) + torto[..., :3] * a).astype(np.uint8)


def matriz(palma, angulo, escala):
    M = cv2.getRotationMatrix2D(tuple(PALMA), angulo, escala)
    M[:, 2] += palma - PALMA
    return M


def palma_para_dedos(dedos, angulo, escala):
    """Posicao da palma para que o meio dos dedos caia no ponto dado."""
    a = math.radians(angulo)
    v = DEDOS - PALMA
    rot = np.array([v[0] * math.cos(a) + v[1] * math.sin(a),
                    -v[0] * math.sin(a) + v[1] * math.cos(a)]) * escala
    return dedos - rot


def montar_alvos(tres_sons=False):
    """Onde cada mao toca: (palma, angulo, ponto de contato) para TUM, TA e TOM.
    Com 3 sons, o TOM fica na borda e o TA (tapa) com os dedos um pouco mais para dentro."""
    alvos, repouso = {}, {}
    for mao, lado in ((D, 1), (E, -1)):
        palma_tum = CENTRO + np.array([lado * 0.37 * RAIO, 0.10 * RAIO])
        alvos[mao, TUM] = (palma_tum, lado * 4, palma_tum)
        ang = math.radians(90 - lado * 35)
        dedos_ta = CENTRO + 0.70 * RAIO * np.array([math.cos(ang), math.sin(ang)])
        dedos_ta[0] = CENTRO[0] + lado * abs(dedos_ta[0] - CENTRO[0])
        palma_ta = palma_para_dedos(dedos_ta, lado * 35, ESCALA)
        alvos[mao, TA] = alvos[mao, ABERTO] = (palma_ta, lado * 35, dedos_ta)
        repouso[mao] = (palma_ta + np.array([lado * 30, 55]), lado * 30)
        if tres_sons:
            ang = math.radians(90 - lado * 30)
            dedos_tapa = CENTRO + 0.56 * RAIO * np.array([math.cos(ang), math.sin(ang)])
            dedos_tapa[0] = CENTRO[0] + lado * abs(dedos_tapa[0] - CENTRO[0])
            alvos[mao, TA] = (palma_para_dedos(dedos_tapa, lado * 30, ESCALA), lado * 30,
                              dedos_tapa)
    return alvos, repouso


def suave(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------- tambor

def borda_djembe(img, cx, cy):
    """Aro com a corda trancada em volta."""
    cv2.circle(img, (cx + 8, cy + 12), RAIO + 34, (14, 12, 11), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), RAIO + 26, (38, 64, 102), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), RAIO + 26, (24, 40, 66), 3, cv2.LINE_AA)
    for g in range(0, 360, 8):
        a, b = math.radians(g), math.radians(g + 4)
        p1 = (int(cx + (RAIO + 6) * math.cos(a)), int(cy + (RAIO + 6) * math.sin(a)))
        p2 = (int(cx + (RAIO + 22) * math.cos(b)), int(cy + (RAIO + 22) * math.sin(b)))
        cv2.line(img, p1, p2, (150, 172, 190), 3, cv2.LINE_AA)


def borda_atabaque(img, cx, cy):
    """Corpo de madeira, aro, corda em volta do corpo e as cravelhas para fora."""
    cv2.circle(img, (cx + 10, cy + 14), RAIO + 60, (14, 12, 11), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), RAIO + 34, (30, 48, 78), -1, cv2.LINE_AA)
    for g in range(0, 360, 3):  # veios da madeira
        a = math.radians(g + (g * 7) % 5)
        r0, r1 = RAIO + 14, RAIO + 33
        cv2.line(img, (int(cx + r0 * math.cos(a)), int(cy + r0 * math.sin(a))),
                 (int(cx + r1 * math.cos(a)), int(cy + r1 * math.sin(a))), (26, 41, 66), 1,
                 cv2.LINE_AA)
    cv2.circle(img, (cx, cy), RAIO + 30, (110, 140, 165), 5, cv2.LINE_AA)  # corda
    n = 14
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        c = np.array([cx, cy], float)
        pts = [c + u * (RAIO + 20) + v * 11, c + u * (RAIO + 62) + v * 7,
               c + u * (RAIO + 62) - v * 7, c + u * (RAIO + 20) - v * 11]
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], (52, 92, 138), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (24, 44, 72), 2, cv2.LINE_AA)
        cv2.line(img, tuple((c + u * (RAIO + 26)).astype(int)),
                 tuple((c + u * (RAIO + 56)).astype(int)), (78, 124, 170), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), RAIO + 12, (46, 70, 104), 14, cv2.LINE_AA)  # aro
    cv2.circle(img, (cx, cy), RAIO + 19, (22, 34, 56), 2, cv2.LINE_AA)


def fundo_tambor(tres_sons=False, instrumento="djembe"):
    img = np.full((H, W, 3), FUNDO, np.uint8)
    cx, cy = int(CENTRO[0]), int(CENTRO[1])
    if instrumento == "atabaque":
        borda_atabaque(img, cx, cy)
    else:
        borda_djembe(img, cx, cy)

    # couro: degrade do centro para a borda + textura
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / RAIO
    rng = np.random.default_rng(3)
    ruido = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 7) * 60
    ruido += cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 1.2) * 6
    centro_cor, borda_cor = np.array([148, 196, 226.0]), np.array([84, 132, 178.0])
    t = np.clip(d, 0, 1)[..., None] ** 1.6
    couro = centro_cor * (1 - t) + borda_cor * t + ruido[..., None]
    mascara = cv2.GaussianBlur((d <= 1).astype(np.float32), (0, 0), 1.0)[..., None]
    img = (img * (1 - mascara) + np.clip(couro, 0, 255) * mascara).astype(np.uint8)

    # zonas: centro (TUM) e borda (TA)
    zona = img.copy()
    cv2.circle(zona, (cx, cy), int(0.45 * RAIO), CORES[TUM], -1, cv2.LINE_AA)
    img = cv2.addWeighted(zona, 0.16, img, 0.84, 0)
    anel = np.zeros((H, W), np.uint8)
    cv2.circle(anel, (cx, cy), RAIO, 255, -1, cv2.LINE_AA)
    cv2.circle(anel, (cx, cy), int(0.68 * RAIO), 0, -1, cv2.LINE_AA)
    a = anel.astype(np.float32)[..., None] / 255 * 0.16
    img = (img * (1 - a) + np.array(CORES[TA], np.float32) * a).astype(np.uint8)
    for r in (int(0.45 * RAIO), int(0.68 * RAIO)):
        for g in range(0, 360, 12):
            cv2.ellipse(img, (cx, cy), (r, r), 0, g, g + 6, (70, 90, 110), 1, cv2.LINE_AA)
    texto(img, "TUM", cx, cy - int(0.22 * RAIO), 30, CORES[TUM], True, 0.55, "mm")
    texto(img, "centro", cx, cy - int(0.22 * RAIO) + 26, 16, (60, 70, 80), False, 0.8, "mm")
    if tres_sons:
        texto(img, "TOM", cx - 52, cy - int(0.84 * RAIO), 30, CORES[ABERTO], True, 0.7, "mm")
        texto(img, "·", cx, cy - int(0.84 * RAIO), 30, (70, 90, 110), True, 0.7, "mm")
        texto(img, "TA", cx + 36, cy - int(0.84 * RAIO), 30, CORES[TA], True, 0.6, "mm")
    else:
        texto(img, "TA", cx, cy - int(0.84 * RAIO), 30, CORES[TA], True, 0.6, "mm")
    texto(img, "borda", cx, cy - int(0.84 * RAIO) + 26, 16, (50, 60, 70), False, 0.8, "mm")

    # colunas da linha do tempo (parte fixa)
    for mao in (E, D):
        x0, x1 = COLUNAS[mao]
        cv2.rectangle(img, (x0, LINHA_Y - 30), (x1, LINHA_FIM), (38, 34, 31), -1)
        texto(img, LETRA_MAO[mao], (x0 + x1) / 2, 186, 34, (235, 235, 235), True, 1, "mm")
        texto(img, NOMES_MAO[mao], (x0 + x1) / 2, 216, 16, (160, 160, 160), False, 1, "mm")
    if tres_sons:
        texto(img, "TUM = palma no centro  ·  TOM = dedos juntos na borda, som cheio", PAINEL_X,
              LINHA_FIM + 8, 15, (150, 150, 150))
        texto(img, "TA = tapa, mão solta, som estalado", PAINEL_X, LINHA_FIM + 28, 15,
              (150, 150, 150))
    else:
        texto(img, "TUM = palma no centro  ·  TA = dedos na borda", PAINEL_X, LINHA_FIM + 12, 16,
              (150, 150, 150))
    return img


def anel(img, centro, raio, cor, esp, alpha):
    r = int(raio) + esp + 2
    x, y = int(centro[0]), int(centro[1])
    x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + r), min(H, y + r)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    camada = roi.copy()
    cv2.circle(camada, (x - x0, y - y0), int(raio), cor, esp, cv2.LINE_AA)
    roi[:] = cv2.addWeighted(camada, alpha, roi, 1 - alpha, 0)


# --------------------------------------------------------------- cena

class CenaTambor:
    def __init__(self, tempos, tipos, maos, slots, grade, grade_maos, desl, compasso_de,
                 bpm, total, titulo, velocidade, pulsos_ext, etapas=None, tres_sons=False,
                 instrumento="djembe"):
        self.tempos, self.tipos, self.maos = tempos, tipos, maos
        self.pulsos_ext = pulsos_ext
        self.slots, self.grade, self.grade_maos = slots, grade, grade_maos
        self.desl, self.compasso_de = desl, compasso_de
        self.bpm, self.total, self.titulo, self.velocidade = bpm, total, titulo, velocidade
        self.fundo = fundo_tambor(tres_sons, instrumento)
        self.sprites = {D: desenhar_mao(True), E: desenhar_mao(False)}
        self.alvos, self.repouso = montar_alvos(tres_sons)
        self.por_mao = {m: (tempos[maos == m], tipos[maos == m]) for m in (D, E)}
        self.titulo_curto = self._encurtar(titulo, 440, 24)
        self.etapas = etapas or []

    @staticmethod
    def _encurtar(s, largura, tamanho):
        if largura_texto(s, tamanho, True) <= largura:
            return s
        while s and largura_texto(s + "...", tamanho, True) > largura:
            s = s[:-1]
        return s.rstrip() + "..."

    # posicao de uma mao no instante t: (palma, angulo, altura 0=no couro 1=levantada)
    def pose(self, mao, t):
        ts, tps = self.por_mao[mao]
        k = int(np.searchsorted(ts, t, "right")) - 1
        rep_palma, rep_ang = self.repouso[mao]
        if k >= 0:
            a_palma, a_ang, _ = self.alvos[mao, tps[k]]
            ta = ts[k]
        else:
            a_palma, a_ang, ta = rep_palma, rep_ang, -1e9
        if k + 1 < len(ts):
            b_palma, b_ang, _ = self.alvos[mao, tps[k + 1]]
            tb = ts[k + 1]
        else:
            b_palma, b_ang, tb = rep_palma, rep_ang, 1e9
        intervalo = tb - ta
        subida = min(0.16, intervalo * 0.4)
        descida = min(0.10, intervalo * 0.35)

        if k >= 0 and t - ta < subida:
            return a_palma, a_ang, suave((t - ta) / subida)
        if k + 1 < len(ts) and tb - t < descida:
            return b_palma, b_ang, suave((tb - t) / descida)

        ini, fim = ta + subida, tb - descida
        if fim - ini > 0.9:  # pausa longa: volta pro descanso no meio
            if t - ini < 0.35:
                f = suave((t - ini) / 0.35)
                return a_palma + (rep_palma - a_palma) * f, a_ang + (rep_ang - a_ang) * f, 1.0
            if fim - t < 0.35:
                f = suave((fim - t) / 0.35)
                return b_palma + (rep_palma - b_palma) * f, b_ang + (rep_ang - b_ang) * f, 1.0
            return rep_palma, rep_ang, 1.0
        f = suave((t - ini) / max(1e-3, fim - ini))
        return a_palma + (b_palma - a_palma) * f, a_ang + (b_ang - a_ang) * f, 1.0

    def frame(self, t):
        img = self.fundo.copy()
        tempos, tipos, maos = self.tempos, self.tipos, self.maos

        # proxima batida de cada mao: alvo que vai fechando
        for mao in (D, E):
            ts, tps = self.por_mao[mao]
            k = int(np.searchsorted(ts, t, "right"))
            if k < len(ts) and ts[k] - t < 0.7:
                p = 1 - (ts[k] - t) / 0.7
                _, _, ponto = self.alvos[mao, tps[k]]
                anel(img, ponto, 78 - 44 * p, CORES[tps[k]], 4, 0.25 + 0.65 * p)

        # maos: sombra, depois a mao (a mais levantada por cima)
        poses = {m: self.pose(m, t) for m in (D, E)}
        ordem = sorted((D, E), key=lambda m: poses[m][2])
        for mao in ordem:
            palma, ang, alt = poses[mao]
            esc = ESCALA * (1 + 0.12 * alt)
            M = matriz(palma + np.array([6 + 16 * alt, 8 + 26 * alt]), ang, esc)
            colar_sprite(img, self.sprites[mao][1], M, 0.55 - 0.25 * alt, sombra=True)
        for mao in ordem:
            palma, ang, alt = poses[mao]
            M = matriz(palma, ang, ESCALA * (1 + 0.12 * alt))
            colar_sprite(img, self.sprites[mao][0], M)

        # ondas no lugar da batida
        a, b = np.searchsorted(tempos, [t - 0.35, t + 1e-6])
        for k in range(a, b):
            f = (t - tempos[k]) / 0.35
            _, _, ponto = self.alvos[maos[k], tipos[k]]
            anel(img, ponto, 30 + 80 * f, CORES[tipos[k]], 6, 0.9 * (1 - f))

        self._painel(img, t)
        self._aviso_etapa(img, t)
        return img

    def _etapa(self, t):
        for etapa in self.etapas:
            if etapa["inicio"] <= t < etapa["fim"]:
                return etapa
        return self.etapas[-1] if self.etapas else None

    def _aviso_etapa(self, img, t):
        """Faixa grande sobre o tambor no comeco de cada passo."""
        etapa = self._etapa(t)
        if not etapa or t - etapa["inicio"] > 3.0:
            return
        dt = t - etapa["inicio"]
        alpha = min(1.0, dt / 0.25, (3.0 - dt) / 0.5)
        y0, y1 = 18, 128
        roi = img[y0:y1, 30:770]
        roi[:] = (roi * (1 - 0.85 * alpha)).astype(np.uint8)
        texto(img, f"Passo {etapa['numero']} de {etapa['de']}", 400, 50, 22, (200, 200, 200),
              False, alpha, "mm")
        texto(img, etapa["titulo"], 400, 84, 36, (255, 255, 255), True, alpha, "mm")
        texto(img, etapa["descricao"], 400, 114, 18, (190, 190, 190), False, alpha, "mm")

    def _painel(self, img, t):
        x0 = PAINEL_X
        tempos, tipos, maos = self.tempos, self.tipos, self.maos
        texto(img, self.titulo_curto, x0, 22, 24, (240, 240, 240), True)
        vel = f"   ·   velocidade {self.velocidade:g}x" if self.velocidade != 1 else ""
        total = self.total
        etapa = self._etapa(t)
        if etapa:
            texto(img, f"Passo {etapa['numero']} de {etapa['de']}  ·  {etapa['titulo']}  ·  "
                       f"{etapa['bpm']:.0f} BPM", x0, 60, 18, (200, 200, 200))
        else:
            texto(img, f"{int(t // 60)}:{int(t % 60):02d} / {int(total // 60)}:{int(total % 60):02d}"
                       f"   ·   ~{self.bpm:.0f} BPM{vel}", x0, 60, 18, (170, 170, 170))
        cv2.rectangle(img, (x0, 92), (W - 30, 97), (60, 55, 50), -1)
        cv2.rectangle(img, (x0, 92), (x0 + int((W - 30 - x0) * min(1, t / total)), 97),
                      (120, 200, 120), -1)

        # contagem 1 2 3 4 do compasso
        i_atual = max(0, int(np.searchsorted(self.slots, t, "right")) - 1)
        m, pos_atual = self.compasso_de(i_atual, self.desl)
        for b in range(4):
            c = (x0 + 22 + b * 56, 133)
            if b == pos_atual // 4:
                cv2.circle(img, c, 21, (240, 240, 240), -1, cv2.LINE_AA)
                texto(img, str(b + 1), c[0], c[1], 22, (30, 30, 30), True, 1, "mm")
            else:
                cv2.circle(img, c, 21, (110, 110, 110), 2, cv2.LINE_AA)
                texto(img, str(b + 1), c[0], c[1], 20, (150, 150, 150), True, 1, "mm")
        if etapa:
            rel = m - etapa["slot0"] // 16
            rotulo = "contagem... prepare" if rel == 0 else f"compasso {rel}"
            texto(img, rotulo, x0 + 240, 122, 18, (240, 200, 120) if rel == 0 else (170, 170, 170))
        else:
            texto(img, f"compasso {m + 1}", x0 + 240, 122, 18, (170, 170, 170))

        # linha do tempo: as batidas sobem ate a linha branca, uma coluna por mao
        topo = LINHA_Y - 30
        fim = t + (LINHA_FIM - LINHA_Y) / PX_SEG
        a, b = np.searchsorted(self.pulsos_ext, [t - 30 / PX_SEG, fim])
        for k in range(a, b):
            y = int(LINHA_Y + (self.pulsos_ext[k] - t) * PX_SEG)
            mk, pos = self.compasso_de(k * 4, self.desl)
            for mao in (E, D):
                c0, c1 = COLUNAS[mao]
                cv2.line(img, (c0, y), (c1, y), (95, 90, 85) if pos == 0 else (55, 51, 47),
                         2 if pos == 0 else 1)
            texto(img, str(pos // 4 + 1), COLUNAS[E][0] - 14, y, 14,
                  (200, 200, 200) if pos == 0 else (110, 110, 110), pos == 0, 1, "mm")

        a, b = np.searchsorted(tempos, [t - 0.25, fim])
        for k in range(a, b):
            c0, c1 = COLUNAS[maos[k]]
            y = LINHA_Y + (tempos[k] - t) * PX_SEG
            cor = CORES[tipos[k]]
            if tempos[k] < t:  # ja tocou: some subindo
                f = (t - tempos[k]) / 0.25
                cor = tuple(int(v * (1 - 0.7 * f)) for v in cor)
            y0, y1 = int(max(topo, y - 11)), int(min(LINHA_FIM, y + 11))
            if y1 <= y0:
                continue
            cv2.rectangle(img, (c0 + 6, y0), (c1 - 6, y1), cor, -1, cv2.LINE_AA)
            if y1 - y0 > 14:
                texto(img, NOMES[tipos[k]], (c0 + c1) / 2, y, 16, (255, 255, 255), True, 1, "mm")

        # linha de batida; acende na coluna da mao que acabou de tocar
        for mao in (E, D):
            c0, c1 = COLUNAS[mao]
            ts, tps = self.por_mao[mao]
            k = int(np.searchsorted(ts, t, "right")) - 1
            if k >= 0 and t - ts[k] < 0.2:
                f = 1 - (t - ts[k]) / 0.2
                cv2.rectangle(img, (c0, LINHA_Y - 14), (c1, LINHA_Y + 14), CORES[tps[k]], 3)
                cv2.line(img, (c0, LINHA_Y), (c1, LINHA_Y), CORES[tps[k]], int(3 + 5 * f))
            else:
                cv2.line(img, (c0, LINHA_Y), (c1, LINHA_Y), (240, 240, 240), 3)
        # cobre o que passou da linha
        img[topo - 4:topo, PAINEL_X - 30:W] = FUNDO
