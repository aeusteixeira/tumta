# Partitura de atabaque / djembe

Analisa uma música ou vídeo, encontra as batidas do atabaque/djembe, separa em
**TUM** (grave, centro do couro) e **TA** (tapa, borda) e gera um vídeo pra treinar.

## Programa com janela (Tumtá)

`python app.py` (ou o `.exe` montado, veja abaixo) abre a janela do **Tumtá**: cole o link
do YouTube (ou arraste/escolha um arquivo), escolha **Atabaque** ou **Djembe** (muda o
desenho do tambor), as opções, e clique em **Gerar vídeos**. A janela confere o link antes
(vídeo encontrado, link inválido, indisponível, sem internet), mostra o andamento e, no fim,
os vídeos com o desenho de cada levada. "Recentes" lista as músicas já geradas.

O visual segue o design *Tumta v4*: `ui/index.html`, as fontes em `ui/fontes/` (Young Serif
e Figtree, licença OFL) e os ramos de folhas do fundo em `ui/folhas.js`.

### Montar o .exe pra distribuir

```
.venv-build\Scripts\python.exe empacotar.py
```

Gera `dist/Tumta/` e `dist/Tumta.zip`. O `.venv-build` tem o PyTorch só de processador,
então roda em qualquer PC com Windows 10/11; o modelo do Demucs (`modelo/`), o ffmpeg e o
deno (que o yt-dlp usa pro YouTube) vão junto. Só o download por link precisa de internet.
Quem usa o programa só extrai o zip e abre `Tumta.exe`; os vídeos vão pra `Documentos\Tumta`.

## Jeito fácil (sem janela)

Arraste a música ou o vídeo em cima de **`Gerar partitura.bat`**.
O resultado fica numa pasta com o nome da música, dentro de `saida/`:

```
saida/
  Nome da Música/
    Nome da Música - video completo.mp4   tambor visto de cima + mãos + colunas E/D subindo
    partitura.txt                         partitura em texto (O = TUM, X = TA, embaixo D/E)
    batidas.json                          todas as batidas com tempo, tipo e mão
    resumo.txt                            levadas encontradas e trechos de cada uma
    aulas/levada 1 - aula.mp4             aula passo a passo de cada levada (+ .txt)
    trechos/trecho 1 - 0m56s a 1m22s (levada 2).mp4   cortes do vídeo completo
```

`aulas/`, `trechos/` e `resumo.txt` aparecem quando você responde `s` na pergunta das aulas
(ou usa `--trechos`). Com velocidade diferente de 1, a pasta ganha o sufixo, ex.: `Nome (0.75x)`.

Cada posição da contagem tem sempre a mesma mão, como se ensina percussão: se a levada usa
semicolcheias, o "1" e o "&" vão com a direita e o "e" e o "a" com a esquerda (a mão que não
toca fica parada esperando). Se usa só colcheias, as mãos alternam a cada colcheia.

## Três sons (opcional)

Por padrão o programa usa só **TUM** (centro) e **TA** (borda), que é o que quem está começando
toca. Pra quem já separa o aberto do tapa, responda `s` na última pergunta do
`Gerar partitura.bat` (ou use `--tres-sons`). O aberto e o tapa soam parecidos, então essa
separação é uma estimativa: se não bater com o que você ouve, ajuste com `--vies-tapa`.

## Pela linha de comando

```
.venv\Scripts\python.exe partitura.py musica.mp3 [opções]
```

| Opção | Pra que serve |
|---|---|
| `--aula` | gera também `aulas/levada base - aula.mp4`: a levada base passo a passo (só TUM, só TA, junto devagar, no andamento, com a música) |
| `--trechos` | acha todas as levadas da música e os trechos de cada uma: uma aula por levada em `aulas/`, os cortes do vídeo em `trechos/` e um `resumo.txt` |
| `--instrumento atabaque` | desenha um atabaque (aro e cravelhas) em vez do djembe (corda trançada) |
| `--tres-sons` | separa os toques de borda em **TOM** (aberto: dedos juntos, som cheio, verde) e **TA** (tapa: mão solta, som estalado, laranja). Na partitura: `O` = TUM, `o` = TOM, `X` = TA |
| `--velocidade 0.75` | deixa mais lento pra treinar (0.5 a 2) |
| `--clique` | toca um som de TUM/TA junto, pra conferir se as batidas estão certas |
| `--inicio 30 --duracao 40` | usa só um trecho (do segundo 30 até o 70) |
| `--sensibilidade 0.9` | pega mais batidas fracas (padrão 0.75; menor = só as fortes) |
| `--vies 3` | se estiver marcando TA demais, puxa pra TUM (negativo faz o contrário) |
| `--vies-tapa 3` | com `--tres-sons`: se estiver marcando TA demais, puxa pra TOM (negativo faz o contrário) |
| `--deslocar-compasso 2` | se o "1" do compasso estiver no lugar errado, muda 1, 2 ou 3 tempos |
| `--rapido` | não usa o Demucs (mais rápido, mas erra mais em música com muitos instrumentos) |
| `--estilo tambor` | usa o desenho anterior do tambor (o padrão é o layout Tumtá, com fotos do tambor e das mãos; as fotos ficam em `assets/`) |
| `--estilo faixas` | visual mais antigo, com as notas andando em linhas |
| `--sem-fundo` | no estilo faixas, não mostra o vídeo original atrás |

### Demucs (já instalado)

O Demucs separa o tambor do resto da música antes da análise. Ele já está instalado e é usado
automaticamente; a separação de cada música fica guardada em `saida/.cache`. Pra reinstalar:

```
.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install demucs
```
