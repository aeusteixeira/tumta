"""Aula passo a passo da levada base: acha o compasso que mais se repete e monta os passos."""
import numpy as np

TUM, TA, ABERTO = 0, 1, 2
D, E = 0, 1
SIMBOLOS = {TUM: "O", TA: "X", ABERTO: "o", -1: "."}
VAZIO = -1


def _grosso(P):
    """Pra comparar compassos, TOM e TA contam como o mesmo toque de borda (a separacao entre
    os dois oscila de um compasso pro outro); o consenso depois decide qual vale em cada posicao."""
    P = np.asarray(P)
    return np.where(P == ABERTO, TA, P)


def achar_base(grade, desl, compasso_de, tolerancia=3):
    """Levada que mais se repete: compasso com mais 'parecidos' e o consenso entre eles."""
    compassos = {}
    for i, tipo in grade.items():
        m, pos = compasso_de(i, desl)
        compassos.setdefault(m, [VAZIO] * 16)[pos] = tipo
    numeros = sorted(m for m, p in compassos.items() if sum(v != VAZIO for v in p) >= 4)
    if not numeros:
        return None
    P = np.array([compassos[m] for m in numeros])
    G = _grosso(P)
    dist = (G[:, None, :] != G[None, :, :]).sum(2)
    apoio = (dist <= tolerancia).sum(1)
    centro = int(np.argmax(apoio))
    membros = dist[centro] <= tolerancia

    base = []
    for pos in range(16):
        valores, contagem = np.unique(P[membros, pos], return_counts=True)
        base.append(int(valores[np.argmax(contagem)]))

    # trecho mais longo em que a base aparece seguida (pra tocar junto com a musica)
    parecidos = {m for m, ok in zip(numeros, (G != _grosso(base)).sum(1) <= tolerancia) if ok}
    melhor_inicio, melhor_len = numeros[centro], 1
    for m in sorted(parecidos):
        if m - 1 in parecidos:
            continue
        n = 1
        while m + n in parecidos:
            n += 1
        if n > melhor_len:
            melhor_inicio, melhor_len = m, n
    return {"base": base, "apoio": int(membros.sum()), "total": len(numeros),
            "trecho_inicio": melhor_inicio, "trecho_len": melhor_len}


def padroes_por_compasso(grade, desl, compasso_de):
    compassos = {}
    for i, tipo in grade.items():
        m, pos = compasso_de(i, desl)
        compassos.setdefault(m, [VAZIO] * 16)[pos] = tipo
    return {m: p for m, p in compassos.items() if sum(v != VAZIO for v in p) >= 4}


def _consenso(P):
    base = []
    for pos in range(16):
        valores, contagem = np.unique(P[:, pos], return_counts=True)
        base.append(int(valores[np.argmax(contagem)]))
    return base


def _trecho_mais_longo(parecidos):
    melhor_inicio, melhor_len = None, 0
    for m in sorted(parecidos):
        if m - 1 in parecidos:
            continue
        n = 1
        while m + n in parecidos:
            n += 1
        if n > melhor_len:
            melhor_inicio, melhor_len = m, n
    return melhor_inicio, melhor_len


def achar_levadas(grade, desl, compasso_de, tolerancia=3, minimo=6, maximo=5):
    """Todas as levadas que se repetem (a mais comum primeiro), cada uma com seu consenso."""
    compassos = padroes_por_compasso(grade, desl, compasso_de)
    restantes = sorted(compassos)
    total = len(restantes)
    levadas = []
    while restantes and len(levadas) < maximo:
        P = np.array([compassos[m] for m in restantes])
        G = _grosso(P)
        dist = (G[:, None, :] != G[None, :, :]).sum(2)
        apoio = (dist <= tolerancia).sum(1)
        centro = int(np.argmax(apoio))
        if apoio[centro] < minimo:
            break
        base = _consenso(P[dist[centro] <= tolerancia])
        perto = (G != _grosso(base)).sum(1) <= tolerancia
        membros = [m for m, ok in zip(restantes, perto) if ok] or \
                  [m for m, ok in zip(restantes, dist[centro] <= tolerancia) if ok]
        if any((_grosso(base) != _grosso(lv["base"])).sum() <= tolerancia for lv in levadas):
            restantes = [m for m in restantes if m not in membros]
            continue
        inicio, n = _trecho_mais_longo(set(membros))
        levadas.append({"base": base, "apoio": len(membros), "total": total,
                        "compassos": membros, "trecho_inicio": inicio, "trecho_len": n})
        restantes = [m for m in restantes if m not in membros]
    return levadas


def achar_trechos(levadas, grade, desl, compasso_de, tolerancia=3, minimo=4):
    """Partes da musica em que uma mesma levada se mantem (aceita quebras de ate 4 compassos)."""
    compassos = padroes_por_compasso(grade, desl, compasso_de)
    rotulo = {}
    for m, p in compassos.items():
        dists = [int((_grosso(p) != _grosso(lv["base"])).sum()) for lv in levadas]
        if dists and min(dists) <= tolerancia + 1:
            rotulo[m] = int(np.argmin(dists))
    def juntar(trechos):
        juntos = []
        for t in trechos:
            if juntos and juntos[-1]["levada"] == t["levada"] and \
                    t["inicio"] - juntos[-1]["fim"] <= 5:
                juntos[-1]["fim"] = t["fim"]
            else:
                juntos.append(dict(t))
        return juntos

    trechos = juntar({"levada": rotulo[m], "inicio": m, "fim": m} for m in sorted(rotulo))
    # some com os pedacinhos e junta de novo o que ficou separado so por eles
    trechos = [t for t in trechos if t["fim"] - t["inicio"] + 1 >= minimo]
    return juntar(trechos)


def usa_semicolcheias(posicoes):
    """A levada usa o 'e' e o 'a' de cada tempo (semicolcheias) com frequencia?"""
    posicoes = np.asarray(posicoes)
    return len(posicoes) > 0 and np.mean(posicoes % 2 == 1) > 0.15


def mao_da_posicao(pos, em_semicolcheia, anterior):
    """Mao fixa por posicao, como se ensina percussao: com semicolcheias, '1' e '&' com a
    direita e 'e' e 'a' com a esquerda; so com colcheias, alterna a cada colcheia."""
    if em_semicolcheia:
        return D if pos % 2 == 0 else E
    if pos % 2 == 1:  # semicolcheia solta numa levada de colcheias
        return 1 - anterior
    return D if (pos // 2) % 2 == 0 else E


def maos_da_base(base):
    posicoes = [p for p in range(16) if base[p] != VAZIO]
    semi = usa_semicolcheias(posicoes)
    maos, anterior = {}, E
    for p in posicoes:
        maos[p] = anterior = mao_da_posicao(p, semi, anterior)
    return maos


def texto_base(base, maos):
    linha = " |  ".join(" ".join(SIMBOLOS[base[k]] for k in range(g, g + 4)) for g in (0, 4, 8, 12))
    linha_maos = " |  ".join(" ".join("DE"[maos[k]] if k in maos else " " for k in range(g, g + 4))
                             for g in (0, 4, 8, 12))
    return f"  1 e & a | 2 e & a | 3 e & a | 4 e & a\n  {linha} |\n  {linha_maos} |"


# ---------------------------------------------------------------- sons

def som_tum(sr):
    t = np.arange(int(0.35 * sr)) / sr
    freq = 70 + 60 * np.exp(-t * 25)
    som = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 9)
    som[:int(0.004 * sr)] += np.random.default_rng(1).standard_normal(int(0.004 * sr)) * 0.3
    return som * 0.9


def som_ta(sr):
    t = np.arange(int(0.12 * sr)) / sr
    ruido = np.random.default_rng(2).standard_normal(len(t))
    ruido = np.convolve(ruido, [1, -0.6], "same")
    tom = np.sin(2 * np.pi * 380 * t) * 0.6 + np.sin(2 * np.pi * 760 * t) * 0.25
    return (ruido * 0.45 + tom) * np.exp(-t * 38) * 0.7


def som_tom(sr):
    t = np.arange(int(0.25 * sr)) / sr
    tom = np.sin(2 * np.pi * 330 * t) + np.sin(2 * np.pi * 660 * t) * 0.3
    som = tom * np.exp(-t * 16)
    som[:int(0.003 * sr)] += np.random.default_rng(3).standard_normal(int(0.003 * sr)) * 0.2
    return som * 0.6


def som_metronomo(sr, forte):
    t = np.arange(int(0.03 * sr)) / sr
    return np.sin(2 * np.pi * (2000 if forte else 1400) * t) * np.exp(-t * 150) * 0.35


def somar(destino, som, inicio):
    s = int(inicio)
    if s >= len(destino):
        return
    fim = min(len(destino), s + len(som))
    destino[s:fim] += som[:fim - s]


# ---------------------------------------------------------------- aula

def montar_aula(info, bpm, audio_musica, slots_musica, desl, sr):
    """Monta a linha do tempo da aula. Devolve tempos, tipos, maos, slots, etapas e o audio."""
    base = info["base"]
    maos_base = maos_da_base(base)
    lento = max(50.0, round(bpm * 0.6))
    presentes = [k for k in (TUM, ABERTO, TA) if k in base]
    tres = ABERTO in base
    so_um = {TUM: ("Só os TUM", "palma no centro, devagar"),
             ABERTO: ("Só os TOM", "aberto: dedos juntos na borda, som cheio"),
             TA: ("Só os TA", "tapa: mão solta, som estalado" if tres else "dedos na borda, devagar")}
    nomes = {TUM: "TUM", ABERTO: "TOM", TA: "TA"}
    todos = set(presentes)
    passos = []
    if len(presentes) > 1:
        passos += [(*so_um[k], lento, 4, {k}) for k in presentes]
    juntos = " e ".join(nomes[k] for k in presentes)
    passos += [("Tudo junto, devagar", f"{juntos} na ordem da levada", lento, 6, todos),
               ("No andamento da música", f"{bpm:.0f} BPM, sem a música", bpm, 6, todos)]
    sons = {TUM: som_tum(sr), TA: som_ta(sr), ABERTO: som_tom(sr)}

    tempos, tipos, maos, slots, etapas, pedacos = [], [], [], [], [], []
    agora = 0.0
    tem_trecho = info["trecho_len"] >= 2 and info["trecho_inicio"] >= 1
    total_passos = len(passos) + (1 if tem_trecho else 0)

    for n, (titulo, desc, andamento, compassos, quais) in enumerate(passos, 1):
        passo = 60.0 / andamento / 4
        n_slots = (compassos + 1) * 16  # +1 compasso de contagem
        audio = np.zeros(int(n_slots * passo * sr) + sr // 2)
        for j in range(n_slots):
            t = j * passo
            slots.append(agora + t)
            if j % 4 == 0:
                somar(audio, som_metronomo(sr, j % 16 == 0) * (1.6 if j < 16 else 0.6), t * sr)
            tipo = base[j % 16]
            if j >= 16 and tipo in quais:
                tempos.append(agora + t)
                tipos.append(tipo)
                maos.append(maos_base[j % 16])
                somar(audio, sons[tipo], t * sr)
        duracao = n_slots * passo
        etapas.append({"inicio": agora, "fim": agora + duracao, "titulo": titulo,
                       "descricao": desc, "bpm": andamento, "numero": n, "de": total_passos,
                       "slot0": len(slots) - n_slots})
        pedacos.append(audio[:int(duracao * sr)])
        agora += duracao

    if tem_trecho:
        # trecho real da musica: 1 compasso de contagem + ate 8 compassos com a base
        m0 = info["trecho_inicio"] - 1
        n_comp = min(8, info["trecho_len"]) + 1
        i0 = m0 * 16 - desl
        idx = np.arange(i0, i0 + n_comp * 16 + 1)
        if idx[0] >= 0 and idx[-1] < len(slots_musica):
            ts = slots_musica[idx]
            ini, fim = ts[0], ts[-1]
            for j, t in enumerate(ts[:-1]):
                slots.append(agora + t - ini)
                tipo = base[j % 16]
                if j >= 16 and tipo != VAZIO:
                    tempos.append(agora + t - ini)
                    tipos.append(tipo)
                    maos.append(maos_base[j % 16])
            trecho = audio_musica[int(ini * sr):int(fim * sr)].mean(axis=1).copy()
            borda = int(0.05 * sr)
            trecho[:borda] *= np.linspace(0, 1, borda)
            trecho[-borda:] *= np.linspace(1, 0, borda)
            etapas.append({"inicio": agora, "fim": agora + fim - ini, "titulo": "Com a música",
                           "descricao": f"trecho em {int(ini // 60)}:{int(ini % 60):02d} da música",
                           "bpm": bpm, "numero": total_passos, "de": total_passos,
                           "slot0": len(slots) - n_comp * 16})
            pedacos.append(trecho)
            agora += fim - ini

    audio = np.concatenate(pedacos)
    pico = np.abs(audio).max()
    if pico > 0:
        audio = audio / pico * 0.9
    return (np.array(tempos), np.array(tipos, dtype=int), np.array(maos, dtype=int),
            np.array(slots), etapas, audio, maos_base)
