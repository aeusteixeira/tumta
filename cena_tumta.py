"""Cena do video no layout Tumtá (design "Tumta Video"): tambor com fotos das maos a esquerda,
pista das duas maos a direita e tela de contagem antes de cada levada.

Usa as fotos de assets/ quando existem (mao-direita, mao-esquerda, djembe-topo, fundo-terreiro);
sem elas, desenha o tambor e usa um fundo escuro, como o modo "usarFotos = false" do design.
"""
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFilter, ImageFont

W, H = 1280, 720
TUM, TA, ABERTO = 0, 1, 2
D, E = 0, 1
NOMES = {TUM: "TUM", TA: "TA", ABERTO: "TOM"}

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
ASSETS = BASE / "assets"
FONTES = BASE / "fontes"


def hexa(c):
    """'#F08A2C' -> (B, G, R) do OpenCV."""
    c = c.lstrip("#")
    return (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))


COR_SOM = {TUM: "#4AA0E6", TA: "#F08A2C", ABERTO: "#5CC46E"}
TEXTO_SOM = {TUM: "#08131F", TA: "#17110D", ABERTO: "#0E1F11"}
FUNDO = "#120D0A"
CLARO = "#F4EBDD"

# palco: tambor de 560 px em (70, 90) -> centro (350, 370)
TAMBOR_X, TAMBOR_Y, TAMBOR_LADO = 70, 90, 560
CENTRO = (350, 370)
RAIO_COURO = 248
PIVO = np.array([120.0, 264.0])   # palma dentro da caixa de 240 x 480 da foto da mao
DEDOS = np.array([120.0, 120.0])  # meio dos dedos (onde bate o TA)

# painel da direita
PX, PW = 720, 520
PASSO_COLCHEIA = 46               # pixels por colcheia na pista
TOPO_AGORA = 26                   # linha do "agora" dentro da pista
ALTURA_PISTA = 370
COL_E = (PX + 50, PX + 50 + 228)
COL_D = (PX + 50 + 228 + 14, PX + PW)


# ------------------------------------------------------------------ texto

@lru_cache(maxsize=64)
def fonte(tamanho, peso="Regular", serif=False):
    if serif:
        return ImageFont.truetype(str(FONTES / "YoungSerif-Regular.ttf"), tamanho)
    f = ImageFont.truetype(str(FONTES / "Figtree.ttf"), tamanho)
    f.set_variation_by_name(peso)
    return f


@lru_cache(maxsize=4096)
def mascara_texto(s, tamanho, peso, serif, ancora, espaco):
    f = fonte(tamanho, peso, serif)
    if espaco:  # espacamento entre letras (letter-spacing)
        larg = sum(f.getlength(ch) + espaco for ch in s) - espaco
        asc, desc = f.getmetrics()
        img = Image.new("L", (int(larg) + 4, asc + desc + 4), 0)
        d, x = ImageDraw.Draw(img), 2
        for ch in s:
            d.text((x, 2), ch, font=f, fill=255)
            x += f.getlength(ch) + espaco
        m = np.asarray(img, np.float32) / 255
        dx = -m.shape[1] / 2 if ancora[0] == "m" else (-m.shape[1] if ancora[0] == "r" else 0)
        dy = -m.shape[0] / 2 if ancora[1] == "m" else 0
        return m, int(dx), int(dy)
    x0, y0, x1, y1 = f.getbbox(s, anchor=ancora)
    img = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
    ImageDraw.Draw(img).text((-x0, -y0), s, font=f, fill=255, anchor=ancora)
    return np.asarray(img, np.float32) / 255, x0, y0


def largura(s, tamanho, peso="Regular", serif=False):
    return fonte(tamanho, peso, serif).getlength(s)


def texto(img, s, x, y, tamanho, cor, peso="Regular", serif=False, ancora="la", alpha=1.0,
          espaco=0, clip=None):
    m, dx, dy = mascara_texto(s, tamanho, peso, serif, ancora, espaco)
    pintar(img, m * alpha, int(x + dx), int(y + dy), hexa(cor) if isinstance(cor, str) else cor, clip)


def pintar(img, mascara, x, y, cor, clip=None):
    """Pinta `cor` usando a mascara (0..1) na posicao (x, y)."""
    h, w = mascara.shape
    cx0, cy0, cx1, cy1 = clip or (0, 0, img.shape[1], img.shape[0])
    x0, y0, x1, y1 = max(cx0, x), max(cy0, y), min(cx1, x + w), min(cy1, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = mascara[y0 - y:y1 - y, x0 - x:x1 - x, None]
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + np.array(cor, np.float32) * a).astype(np.uint8)


# --------------------------------------------------------------- sprites

class Sprite:
    """Imagem com alfa ja pre-multiplicado, pronta pra colar rapido."""

    def __init__(self, rgba_pil):
        a = np.asarray(rgba_pil.convert("RGBA"), np.float32) / 255
        self.alfa = a[..., 3:4]
        self.cor = a[..., 2::-1] * 255 * self.alfa  # RGB -> BGR, pre-multiplicado
        self.h, self.w = a.shape[:2]

    def colar(self, img, x, y, opacidade=1.0, clip=None):
        cx0, cy0, cx1, cy1 = clip or (0, 0, img.shape[1], img.shape[0])
        x, y = int(round(x)), int(round(y))
        x0, y0, x1, y1 = max(cx0, x), max(cy0, y), min(cx1, x + self.w), min(cy1, y + self.h)
        if x1 <= x0 or y1 <= y0 or opacidade <= 0:
            return
        a = self.alfa[y0 - y:y1 - y, x0 - x:x1 - x] * opacidade
        c = self.cor[y0 - y:y1 - y, x0 - x:x1 - x] * opacidade
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - a) + c).astype(np.uint8)


def pil_rgba(c, a=1.0):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(round(255 * a)))


def caixa_arredondada(w, h, raio, fundo=None, borda=None, esp=0, brilho=None, brilho_raio=0,
                      brilho_a=0.6, escala=3):
    """Retangulo arredondado com borda e brilho em volta (box-shadow), suavizado."""
    m = brilho_raio * 2
    W2, H2 = (w + 2 * m) * escala, (h + 2 * m) * escala
    img = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    if brilho:
        sombra = Image.new("L", (W2, H2), 0)
        ImageDraw.Draw(sombra).rounded_rectangle(
            [m * escala, m * escala, (m + w) * escala - 1, (m + h) * escala - 1], raio * escala, fill=255)
        forma = sombra.copy()
        sombra = sombra.filter(ImageFilter.GaussianBlur(brilho_raio * escala / 2))
        if not fundo:  # como o box-shadow do CSS: o brilho fica so do lado de fora
            sombra = Image.fromarray((np.asarray(sombra, np.float32)
                                      * (1 - np.asarray(forma, np.float32) / 255)).astype(np.uint8))
        camada = Image.new("RGBA", (W2, H2), pil_rgba(brilho))
        camada.putalpha(sombra.point(lambda v: int(v * brilho_a)))
        img = Image.alpha_composite(img, camada)
    d = ImageDraw.Draw(img)
    caixa = [m * escala, m * escala, (m + w) * escala - 1, (m + h) * escala - 1]
    if fundo:
        d.rounded_rectangle(caixa, raio * escala, fill=pil_rgba(fundo))
    if borda and esp:
        d.rounded_rectangle(caixa, raio * escala, outline=pil_rgba(borda), width=esp * escala)
    return img.resize((W2 // escala, H2 // escala), Image.LANCZOS), m


def pilula(s, fundo, cor, tamanho=15, peso="ExtraBold", px=12, py=5):
    f = fonte(tamanho, peso)
    x0, y0, x1, y1 = f.getbbox(s)
    w, h = int(x1 - x0) + 2 * px, tamanho + 2 * py + 3
    img, _ = caixa_arredondada(w, h, h // 2, fundo=fundo)
    ImageDraw.Draw(img).text((w / 2, h / 2), s, font=f, fill=pil_rgba(cor), anchor="mm")
    return img


def carregar_imagem(nome):
    """Abre a imagem de assets/; devolve None se faltar ou estiver incompleta."""
    caminho = ASSETS / nome
    if not caminho.exists():
        return None
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        img = Image.open(caminho)
        img.load()
        return img
    except (OSError, SyntaxError):
        return None


# ------------------------------------------------------------- desenhos

def mao_sprite(mao):
    """Foto da mao na caixa de 240 x 480, com o antebraco sumindo embaixo (mask do design)."""
    foto = carregar_imagem("mao-direita.webp" if mao == D else "mao-esquerda.webp")
    if foto is None:
        return None
    foto = foto.convert("RGBA").resize((240, 480), Image.LANCZOS)
    a = np.asarray(foto, np.float32).copy()
    y = np.arange(480, dtype=np.float32)[:, None]
    some = np.clip((0.96 * 480 - y) / ((0.96 - 0.72) * 480), 0, 1)
    a[..., 3] *= some
    rgba = a / 255
    alfa = rgba[..., 3]
    cor = rgba[..., 2::-1] * alfa[..., None] * 255
    return np.dstack([cor, alfa * 255]).astype(np.float32)  # BGR pre-multiplicado + alfa


def fundo_tela():
    foto = carregar_imagem("fundo-terreiro.jpg")
    if foto is not None:
        foto = foto.convert("RGB")
        esc = max(W / foto.width, H / foto.height)
        foto = foto.resize((math.ceil(foto.width * esc), math.ceil(foto.height * esc)), Image.LANCZOS)
        x0, y0 = (foto.width - W) // 2, (foto.height - H) // 2
        img = np.asarray(foto.crop((x0, y0, x0 + W, y0 + H)), np.float32)[..., ::-1]
    else:
        # sem a foto: madeira escura com uma luz suave em cima do tambor
        rng = np.random.default_rng(7)
        img = np.zeros((H, W, 3), np.float32) + np.array(hexa("#1C140F"), np.float32)
        veios = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=2)
        img += veios[..., None] * 18
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        luz = np.exp(-(((xx - CENTRO[0]) / 520) ** 2 + ((yy - CENTRO[1]) / 420) ** 2))
        img *= (0.55 + 0.6 * luz)[..., None]
    # escurece a direita, onde fica o painel (gradiente do design)
    x = np.arange(W, dtype=np.float32) / W
    a = np.interp(x, [0, 0.5, 0.56, 1], [0.25, 0.3, 0.86, 0.92])[None, :, None]
    img = img * (1 - a) + np.array(hexa(FUNDO), np.float32) * a
    return np.clip(img, 0, 255).astype(np.uint8)


def sombra_redonda(img, cx, cy, r, dy, sigma, forca):
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (int(cx), int(cy + dy)), int(r), 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), sigma)[..., None] * forca
    img[:] = (img * (1 - m)).astype(np.uint8)


def desenhar_tambor(img, instrumento):
    cx, cy, R = CENTRO[0], CENTRO[1], TAMBOR_LADO / 2
    sombra_redonda(img, cx, cy, R, 40, 40, 0.8)
    sombra_redonda(img, cx, cy, R, 10, 12, 0.6)
    foto = carregar_imagem("djembe-topo.webp" if instrumento == "djembe" else "atabaque-topo.png")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.hypot(xx - cx, yy - cy)
    dentro = lambda r: np.clip(r - dist + 0.5, 0, 1)[..., None]  # borda suavizada
    if foto is not None:
        foto = foto.convert("RGBA").resize((TAMBOR_LADO, TAMBOR_LADO), Image.LANCZOS)
        camada = np.zeros((H, W, 4), np.float32)
        camada[TAMBOR_Y:TAMBOR_Y + TAMBOR_LADO, TAMBOR_X:TAMBOR_X + TAMBOR_LADO] = np.asarray(foto, np.float32)
        a = camada[..., 3:4] / 255 * dentro(R)
        img[:] = (img * (1 - a) + camada[..., 2::-1] * a).astype(np.uint8)
    else:
        ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 90) % 360
        if instrumento == "djembe":
            # corda trancada: repeating-conic-gradient(#E9D5B0 0 3deg, #5A3820 3deg 9deg)
            aro = np.where((ang % 9) < 3, 1.0, 0.0)[..., None]
            cor_aro = aro * np.array(hexa("#E9D5B0"), np.float32) + (1 - aro) * np.array(hexa("#5A3820"), np.float32)
        else:
            # atabaque: madeira com as cravelhas claras
            cravelha = np.where((ang % 30) < 7, 1.0, 0.0)[..., None]
            cor_aro = cravelha * np.array(hexa("#B98A55"), np.float32) + (1 - cravelha) * np.array(hexa("#4A2E1A"), np.float32)
        img[:] = (img * (1 - dentro(R)) + cor_aro * dentro(R)).astype(np.uint8)
        a = dentro(R - 20)
        img[:] = (img * (1 - a) + np.array(hexa("#4A2E1A"), np.float32) * a).astype(np.uint8)
        # couro: radial-gradient(circle at 45% 40%, #E3BC84, #C08A50)
        fx, fy = TAMBOR_X + 0.45 * TAMBOR_LADO, TAMBOR_Y + 0.40 * TAMBOR_LADO
        t = np.clip(np.hypot(xx - fx, yy - fy) / (R - 32) / 1.15, 0, 1)[..., None]
        couro = (1 - t) * np.array(hexa("#E3BC84"), np.float32) + t * np.array(hexa("#C08A50"), np.float32)
        a = dentro(R - 32)
        img[:] = (img * (1 - a) + couro * a).astype(np.uint8)
        # miolo mais claro (centro, TUM)
        t = np.clip(np.hypot(xx - fx, yy - fy) / 150, 0, 1)[..., None]
        miolo = (1 - t) * np.array(hexa("#EEE2C8"), np.float32) + t * np.array(hexa("#D9C49E"), np.float32)
        a = dentro(150)
        img[:] = (img * (1 - a) + miolo * a).astype(np.uint8)
    # circulo tracejado do centro: rgba(23,17,13,.35)
    tracos = img.copy()
    for g in range(0, 360, 8):
        cv2.ellipse(tracos, (cx, cy), (150, 150), 0, g, g + 4, (13, 17, 23), 2, cv2.LINE_AA)
    img[:] = cv2.addWeighted(tracos, 0.38, img, 0.62, 0)


def logo(img, x, y):
    cv2.circle(img, (x + 14, y + 14), 12, hexa("#F08A2C"), 4, cv2.LINE_AA)
    cv2.circle(img, (x + 10, y + 16), 3, hexa("#4AA0E6"), -1, cv2.LINE_AA)
    cv2.circle(img, (x + 19, y + 9), 3, hexa("#F08A2C"), -1, cv2.LINE_AA)
    texto(img, "tumtá", x + 38, y + 14, 18, "#D8CABA", serif=True, ancora="lm")


def palma_para_dedos(dedos, angulo):
    """Posicao da palma para que o meio dos dedos caia no ponto dado (angulo anti-horario)."""
    a = math.radians(angulo)
    v = DEDOS - PIVO
    rot = np.array([v[0] * math.cos(a) + v[1] * math.sin(a), -v[0] * math.sin(a) + v[1] * math.cos(a)])
    return dedos - rot


def montar_alvos(tres_sons=False):
    """Onde cada mao toca: (palma, angulo, ponto de contato). TUM: palma no centro, cada mao do
    seu lado; TA: dedos na borda com a mao inclinada. Mesma regra do video anterior."""
    c, R = np.array(CENTRO, float), RAIO_COURO
    alvos, repouso = {}, {}
    for mao, lado in ((D, 1), (E, -1)):
        palma_tum = c + np.array([lado * 0.48 * R, 0.12 * R])
        alvos[mao, TUM] = (palma_tum, lado * 4, palma_tum)
        def borda(raio, giro):
            ang = math.radians(90 - lado * giro)
            dedos = c + raio * R * np.array([math.cos(ang), math.sin(ang)])
            dedos[0] = c[0] + lado * abs(dedos[0] - c[0])
            return palma_para_dedos(dedos, lado * giro), lado * giro, dedos
        alvos[mao, TA] = alvos[mao, ABERTO] = borda(0.72, 35)
        if tres_sons:  # o tapa entra um pouco mais
            alvos[mao, TA] = borda(0.58, 30)
        palma_ta = alvos[mao, ABERTO][0]
        repouso[mao] = (palma_ta + np.array([lado * 30, 55]), lado * 30)
    return alvos, repouso


def titulo_curto(s):
    """Tira o '(feat. ...)' do nome pra caber no video."""
    curto = re.sub(r"\s*[\(\[](feat|ft)\.?[^\)\]]*[\)\]]", "", s, flags=re.I).strip()
    return curto or s


def caber(s, larg_max, tam, tam_min, serif=False, peso="Regular"):
    """Diminui a letra ate caber; se nem assim couber, corta com reticencias."""
    while tam > tam_min and largura(s, tam, peso, serif) > larg_max:
        tam -= 1
    if largura(s, tam, peso, serif) > larg_max:
        while s and largura(s + "…", tam, peso, serif) > larg_max:
            s = s[:-1]
        s = s.rstrip() + "…"
    return s, tam


def suave(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ cena

class CenaTumta:
    def __init__(self, tempos, tipos, maos, slots, grade, grade_maos, desl, compasso_de, bpm, total,
                 titulo, velocidade, pulsos_ext, etapas=None, tres_sons=False, instrumento="djembe",
                 levada=None, trechos=None):
        self.tempos, self.tipos, self.maos = np.asarray(tempos), np.asarray(tipos), np.asarray(maos)
        self.slots = np.asarray(slots)
        self.desl, self.compasso_de = desl, compasso_de
        self.bpm, self.total, self.titulo, self.velocidade = bpm, total, titulo, velocidade
        self.etapas = etapas or []
        self.tres_sons, self.instrumento = tres_sons, instrumento
        self.levada = levada
        self.trechos = trechos or []  # [(inicio_s, fim_s, levada)]
        # cada batida na grade (em semicolcheias, com fracao) pra posicionar na pista
        self.pos_batidas = np.interp(self.tempos, self.slots, np.arange(len(self.slots)))
        self.por_mao = {m: np.nonzero(self.maos == m)[0] for m in (D, E)}
        self.alvos, self.repouso = montar_alvos(tres_sons)
        self.titulo_painel = caber(titulo_curto(titulo), PW, 34, 24, serif=True)
        self._montar()

    # --- partes fixas

    def _chips(self, levada=None):
        nomes = []
        if levada:
            nomes.append((f"Levada {levada}", "#F08A2C"))
        nomes.append(("Atabaque" if self.instrumento == "atabaque" else "Djembe", "#D8CABA"))
        if self.velocidade != 1:
            nomes.append((f"{self.velocidade:g}×".replace(".", ","), "#D8CABA"))
        return [Sprite(pilula(n, "#2E2118", c, 13, "Bold", 10, 3)) for n, c in nomes]

    def _montar(self):
        base = fundo_tela()
        desenhar_tambor(base, self.instrumento)
        logo(base, 40, 28)
        self.base = base
        self.maos_foto = {m: mao_sprite(m) for m in (D, E)}
        self.pilulas = [
            (Sprite(pilula("TOM / TA · borda" if self.tres_sons else "TA · borda", "#F08A2C", "#17110D")), 350, 152),
            (Sprite(pilula("TUM · centro", "#2E7FC8", "#FFFFFF")), 350, 236),
        ]
        self.chips_sem_levada = self._chips(self.levada)
        self.chips_por_levada = {}
        # cabecalho das colunas
        self.y_cab = 214
        self.y_pista = 282
        self.y_barra = 138
        self.y_batidas = 160
        topo = self.y_pista
        for (x0, x1) in (COL_E, COL_D):
            box, _ = caixa_arredondada(x1 - x0, ALTURA_PISTA, 10, fundo="#1C1510")
            Sprite(box).colar(base, x0, topo)
        for (x0, x1), letra, nome in ((COL_E, "E", "mão esquerda"), (COL_D, "D", "mão direita")):
            cxm = (x0 + x1) / 2
            texto(base, letra, cxm, self.y_cab, 28, CLARO, "ExtraBold", ancora="mt")
            texto(base, nome, cxm, self.y_cab + 32, 13, "#A89684", ancora="mt")
        # notas (normal e acesa), batidas e caixa do "agora"
        self.notas = {}
        for tipo in (TUM, TA, ABERTO):
            for aceso in (False, True):
                box, m = caixa_arredondada(216, 28, 7, fundo=COR_SOM[tipo],
                                           borda="#FFFFFF" if aceso else None, esp=2 if aceso else 0,
                                           brilho=COR_SOM[tipo] if aceso else None,
                                           brilho_raio=11 if aceso else 0, brilho_a=0.85)
                ImageDraw.Draw(box).text((m + 108, m + 14), NOMES[tipo], font=fonte(15, "ExtraBold"),
                                         fill=pil_rgba(TEXTO_SOM[tipo]), anchor="mm")
                self.notas[tipo, aceso] = (Sprite(box), m)
        agora, m = caixa_arredondada(PW + 6 - 44, 40, 12, borda=CLARO, esp=2, brilho=CLARO,
                                     brilho_raio=12, brilho_a=0.25)
        self.agora = (Sprite(agora), m)
        self.batidas = [Sprite(self._linha_batidas(k)) for k in range(4)]
        self.contagens = {}

    def _linha_batidas(self, ativa):
        f, fe = fonte(17, "ExtraBold"), fonte(13)
        img = Image.new("RGBA", (300, 40), (0, 0, 0, 0))
        x = 0
        for k in range(4):
            if k == ativa:
                c, _ = caixa_arredondada(40, 40, 20, fundo=CLARO)
                img.alpha_composite(c, (x, 0))
                ImageDraw.Draw(img).text((x + 20, 20), str(k + 1), font=f, fill=pil_rgba("#17110D"), anchor="mm")
            else:
                c, _ = caixa_arredondada(40, 40, 20, borda="#4A3828", esp=2)
                img.alpha_composite(c, (x, 0))
                ImageDraw.Draw(img).text((x + 20, 20), str(k + 1), font=fonte(17, "Bold"),
                                         fill=pil_rgba("#A89684"), anchor="mm")
            x += 46
            ImageDraw.Draw(img).text((x, 20), "e", font=fe, fill=pil_rgba("#7D6B5B"), anchor="lm")
            x += 8 + 6
        return img

    # --- tempo

    def _pos(self, t):
        """Posicao fracionaria na grade de semicolcheias no instante t."""
        return float(np.interp(t, self.slots, np.arange(len(self.slots))))

    def _etapa(self, t):
        for e in self.etapas:
            if e["inicio"] <= t < e["fim"]:
                return e
        return None

    def _levada_em(self, t):
        for ini, fim, lv in self.trechos:
            if ini <= t < fim:
                return lv
        return None

    def _contagem(self, t):
        """Durante o compasso de contagem: (etapa ou None, tempo 0..3). Senao None."""
        if self.etapas:
            e = self._etapa(t)
            if e is None:
                return None
            s0 = e["slot0"]
            if s0 + 16 < len(self.slots) and self.slots[s0] <= t < self.slots[s0 + 16]:
                return e, int((self._pos(t) - s0) // 4)
            return None
        # video completo: o compasso antes da primeira batida
        if not len(self.tempos):
            return None
        i0 = int(round(self.pos_batidas[0]))
        m0, _ = self.compasso_de(i0, self.desl)
        ini = (m0 - 1) * 16 - self.desl
        fim = m0 * 16 - self.desl
        if ini >= 0 and fim < len(self.slots) and self.slots[ini] <= t < self.slots[fim]:
            return None, int((self._pos(t) - ini) // 4)
        return None

    # --- maos: mesmo movimento do video anterior (tambor.py), so trocando o desenho pela foto

    def _pose(self, mao, t):
        """(palma, angulo anti-horario em graus, altura 0 = no couro .. 1 = levantada)."""
        ts = self.tempos[self.por_mao[mao]]
        tps = self.tipos[self.por_mao[mao]]
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

    def _desenhar_mao(self, img, mao, pos, angulo, lift):
        foto = self.maos_foto[mao]
        if foto is None:
            return
        s = 1 + 0.07 * lift
        th = math.radians(-angulo)
        Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * s
        t = np.array(pos) + Rm @ (np.array([0.0, -8 * lift]) - PIVO)
        M = np.hstack([Rm, t[:, None]])
        cantos = np.array([[0, 0], [240, 0], [0, 480], [240, 480]], np.float32) @ Rm.T + t
        sig = (8 + 26 * lift) / 2
        oy = 8 + 34 * lift
        mg = int(3 * sig + oy) + 2
        x0 = max(0, int(cantos[:, 0].min()) - mg)
        y0 = max(0, int(cantos[:, 1].min()) - mg)
        x1 = min(W, int(cantos[:, 0].max()) + mg)
        y1 = min(H, int(cantos[:, 1].max()) + mg)
        if x1 <= x0 or y1 <= y0:
            return
        M2 = M.copy()
        M2[:, 2] -= (x0, y0)
        tw = cv2.warpAffine(foto, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        roi = img[y0:y1, x0:x1].astype(np.float32)
        # sombra (drop-shadow)
        a = tw[..., 3] / 255
        sombra = cv2.GaussianBlur(a, (0, 0), max(0.5, sig))
        sombra = np.roll(sombra, int(oy), axis=0)
        sombra[:int(oy)] = 0
        roi *= (1 - sombra * (0.6 - 0.25 * lift))[..., None]
        roi = roi * (1 - a[..., None]) + tw[..., :3]
        img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)

    def _ondas(self, img, t):
        a, b = np.searchsorted(self.tempos, [t - 0.38, t + 1e-6])
        for k in range(a, b):
            q = (t - self.tempos[k]) / 0.38
            mao, tipo = self.maos[k], self.tipos[k]
            x, y = (int(v) for v in self.alvos[mao, tipo][2])
            r = (50 + 110 * q) / 2
            cor = hexa(COR_SOM[tipo])
            mg = int(r + 30)
            x0, y0, x1, y1 = max(0, x - mg), max(0, y - mg), min(W, x + mg), min(H, y + mg)
            camada = np.zeros((y1 - y0, x1 - x0), np.float32)
            cv2.circle(camada, (x - x0, y - y0), int(r), 1.0, 4, cv2.LINE_AA)
            brilho = cv2.GaussianBlur(camada, (0, 0), 10) * 2.2
            m = np.clip(camada + brilho * 0.6, 0, 1) * (1 - q)
            roi = img[y0:y1, x0:x1]
            roi[:] = (roi * (1 - m[..., None]) + np.array(cor, np.float32) * m[..., None]).astype(np.uint8)
            if k == b - 1:
                yy = y - 80 - 20 * q
                texto(img, NOMES[tipo], x, yy + 2, 22, (0, 0, 0), "ExtraBold", ancora="mm", alpha=0.6 * (1 - q))
                texto(img, NOMES[tipo], x, yy, 22, "#FFFFFF", "ExtraBold", ancora="mm", alpha=1 - q)

    # --- painel

    def _painel(self, img, t):
        etapa = self._etapa(t)
        lv = etapa and self.levada or self._levada_em(t) or self.levada
        if lv not in self.chips_por_levada:
            self.chips_por_levada[lv] = self._chips(lv)
        x = PX
        for chip in self.chips_por_levada[lv]:
            chip.colar(img, x, 28)
            x += chip.w + 8
        titulo, tam = self.titulo_painel
        texto(img, titulo, PX, 62 + (34 - tam) * 0.6, tam, CLARO, serif=True)
        if etapa:
            sub = f"Passo {etapa['numero']} de {etapa['de']} · {etapa['titulo']} · {etapa['bpm']:.0f} BPM"
        else:
            i = max(0, int(np.searchsorted(self.slots, t, "right")) - 1)
            m, _ = self.compasso_de(i, self.desl)
            sub = (f"{int(t // 60)}:{int(t % 60):02d} / {int(self.total // 60)}:{int(self.total % 60):02d}"
                   f" · ~{self.bpm:.0f} BPM · compasso {max(1, m + 1)}")
        texto(img, sub, PX, 106, 16, "#BFAE9C")
        # barra de progresso com as marcas dos trechos
        y = self.y_barra
        cv2.rectangle(img, (PX, y), (PX + PW, y + 8), hexa("#2A1F17"), -1, cv2.LINE_AA)
        if etapa:
            frac = (t - etapa["inicio"]) / max(1e-6, etapa["fim"] - etapa["inicio"])
        else:
            frac = t / max(1e-6, self.total)
        cv2.rectangle(img, (PX, y), (PX + int(PW * min(1, frac)), y + 8), hexa("#F08A2C"), -1, cv2.LINE_AA)
        if not etapa:
            for ini, _, _ in self.trechos:
                xm = PX + int(PW * ini / max(1e-6, self.total))
                cv2.rectangle(img, (xm, y - 3), (xm + 1, y + 11), hexa(CLARO), -1)
        # tempo atual do compasso
        pos = self._pos(t)
        _, p16 = self.compasso_de(int(pos), self.desl)
        self.batidas[p16 // 4].colar(img, PX, self.y_batidas)
        self._pista(img, pos)

    def _pista(self, img, pos):
        topo = self.y_pista
        clip = (PX - 8, topo, PX + PW + 8, topo + ALTURA_PISTA)
        agora_n = pos / 2  # em colcheias
        base_n = math.floor(agora_n)
        y_de = lambda n: topo + TOPO_AGORA + (n - agora_n) * PASSO_COLCHEIA
        for n in range(base_n - 1, base_n + 10):
            y = y_de(n)
            if y < topo - 30 or y > topo + ALTURA_PISTA + 30:
                continue
            p16 = (n * 2 + self.desl) % 16
            colcheia = p16 // 2
            tempo = colcheia % 2 == 0
            um = colcheia == 0
            rotulo = str(colcheia // 2 + 1) if tempo else "e"
            cor_linha = "#6B5A4B" if um else ("#33261C" if tempo else "#241B14")
            yi = int(round(y))
            if topo <= yi < topo + ALTURA_PISTA:
                cv2.line(img, (PX + 50, yi), (PX + PW, yi), hexa(cor_linha), 1)
            texto(img, rotulo, PX + 36, y, 15 if tempo else 12,
                  CLARO if um else ("#A89684" if tempo else "#6B5A4B"),
                  "ExtraBold" if um else "SemiBold", ancora="rm", clip=clip)
        # notas que estao na janela da pista
        lo = agora_n * 2 - 4
        hi = agora_n * 2 + 2 * 10
        a, b = np.searchsorted(self.pos_batidas, [lo, hi])
        for k in range(a, b):
            n = self.pos_batidas[k] / 2
            y = y_de(n)
            aceso = abs(n - agora_n) < 0.3
            op = max(0.0, 1 + (y - (topo + TOPO_AGORA)) / 30) if y < topo + TOPO_AGORA - 4 else 1.0
            spr, m = self.notas[self.tipos[k], aceso]
            x = (COL_E[0] + 6) if self.maos[k] == E else (COL_D[0] + 6)
            spr.colar(img, x - m, y - 14 - m, op, clip)
        spr, m = self.agora
        spr.colar(img, PX + 44 - m, topo + 6 - m, 1.0, (PX - 20, topo - 20, W, topo + ALTURA_PISTA))
        texto(img, "AGORA", PX + PW + 6, topo - 18, 11, CLARO, "ExtraBold", ancora="ra", espaco=1.3)

    # --- contagem (antes de cada levada)

    def _tela_contagem(self, etapa, tempo):
        chave = (etapa["numero"] if etapa else 0, tempo)
        if chave in self.contagens:
            return self.contagens[chave].copy()
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = hexa(FUNDO)
        nomes = []
        if self.levada:
            nomes.append((f"Levada {self.levada}", "#F08A2C"))
        nomes.append(("Atabaque" if self.instrumento == "atabaque" else "Djembe", "#D8CABA"))
        bpm = etapa["bpm"] if etapa else self.bpm
        vel = f"{self.velocidade:g}×".replace(".", ",") + " · " if self.velocidade != 1 else ""
        nomes.append((f"{vel}~{bpm:.0f} BPM", "#D8CABA"))
        chips = [Sprite(pilula(n, "#2E2118", c, 14, "Bold", 12, 4)) for n, c in nomes]
        total_w = sum(c.w for c in chips) + 8 * (len(chips) - 1)
        x = (W - total_w) / 2
        y0 = 150 if etapa else 175
        for c in chips:
            c.colar(img, x, y0)
            x += c.w + 8
        titulo, tam = caber(titulo_curto(self.titulo), W - 120, 40, 26, serif=True)
        texto(img, titulo, W / 2, y0 + 70, tam, CLARO, serif=True, ancora="mm")
        y = y0 + 120
        if etapa:
            texto(img, f"Passo {etapa['numero']} de {etapa['de']} · {etapa['titulo']}", W / 2, y, 22,
                  "#F08A2C", "Bold", ancora="mm")
            y += 30
        # os 4 tempos
        cy = y + 95
        tamanhos = [150 if k == tempo else 110 for k in range(4)]
        larg = sum(tamanhos) + 24 * 3
        x = (W - larg) / 2
        for k, tam in enumerate(tamanhos):
            cx = x + tam / 2
            if k == tempo:
                cv2.circle(img, (int(cx), int(cy)), tam // 2 + 14, (40, 43, 46), -1, cv2.LINE_AA)
                cv2.circle(img, (int(cx), int(cy)), tam // 2, hexa(CLARO), -1, cv2.LINE_AA)
                texto(img, str(k + 1), cx, cy, 72, "#17110D", "ExtraBold", ancora="mm")
            elif k < tempo:
                cv2.circle(img, (int(cx), int(cy)), tam // 2, hexa("#3A2B1F"), -1, cv2.LINE_AA)
                texto(img, str(k + 1), cx, cy, 44, "#7D6B5B", "ExtraBold", ancora="mm")
            else:
                cv2.circle(img, (int(cx), int(cy)), tam // 2 - 2, hexa("#3A2B1F"), 3, cv2.LINE_AA)
                texto(img, str(k + 1), cx, cy, 44, "#4A3828", "ExtraBold", ancora="mm")
            x += tam + 24
        texto(img, "Prepare as mãos · começa no próximo 1", W / 2, cy + 120, 20, "#BFAE9C", ancora="mm")
        # legenda
        itens = [(TUM, "centro"), (TA, "borda")] if not self.tres_sons else [(TUM, "centro"), (ABERTO, "aberto"), (TA, "tapa")]
        partes = []
        for tipo, desc in itens:
            p = Sprite(pilula(NOMES[tipo], COR_SOM[tipo], TEXTO_SOM[tipo], 15, "ExtraBold", 12, 5))
            partes.append((p, desc, largura(desc, 15)))
        larg = sum(p.w + 8 + w for p, _, w in partes) + 24 * (len(partes) - 1)
        x = (W - larg) / 2
        yl = cy + 165
        for p, desc, w in partes:
            p.colar(img, x, yl - p.h / 2)
            texto(img, desc, x + p.w + 8, yl, 15, "#A89684", ancora="lm")
            x += p.w + 8 + w + 24
        self.contagens[chave] = img
        return img.copy()

    # --- quadro

    def frame(self, t):
        cont = self._contagem(t)
        if cont is not None:
            return self._tela_contagem(*cont)
        img = self.base.copy()
        poses = {m: self._pose(m, t) for m in (D, E)}
        for mao in sorted((D, E), key=lambda m: poses[m][2]):  # a mais levantada por cima
            palma, ang, alt = poses[mao]
            self._desenhar_mao(img, mao, palma, ang, alt)
        self._ondas(img, t)
        for spr, cx, top in self.pilulas:
            spr.colar(img, cx - spr.w / 2, top)
        self._painel(img, t)
        return img
