"""Separa o tambor do resto da musica com o Demucs (htdemucs), no proprio processo.

Usa o modelo da pasta `modelo/` (vai junto no programa instalado, sem precisar de internet);
se ela nao existir, baixa o modelo do Demucs na primeira vez.
"""
import sys
from pathlib import Path

import numpy as np


def pasta_modelo():
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / "modelo"


def carregar_modelo():
    pasta = pasta_modelo()
    if (pasta / "htdemucs.yaml").exists():
        import yaml
        from demucs.apply import BagOfModels
        from demucs.hf import load_safetensors_model
        bag = yaml.safe_load((pasta / "htdemucs.yaml").read_text())
        modelos = [load_safetensors_model(pasta / f"{sig}.safetensors") for sig in bag["models"]]
        return BagOfModels(modelos, bag.get("weights"), bag.get("segment"))
    from demucs.pretrained import get_model
    return get_model("htdemucs")


def separar_tambor(audio, sr):
    """`audio` em (amostras, canais) no sr do modelo (44100). Devolve o tambor em mono."""
    import torch
    from demucs.apply import apply_model

    modelo = carregar_modelo()
    modelo.eval()
    if sr != modelo.samplerate:
        raise ValueError(f"o Demucs espera {modelo.samplerate} Hz")
    x = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    # mesma normalizacao do `demucs` de linha de comando
    ref = x.mean(0)
    media, desvio = ref.mean(), ref.std() + 1e-8
    x = (x - media) / desvio
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        fontes = apply_model(modelo, x[None], device=dispositivo, split=True, overlap=0.25,
                             progress=True)[0]
    tambor = fontes[modelo.sources.index("drums")] * desvio + media
    return tambor.mean(0).cpu().numpy()
