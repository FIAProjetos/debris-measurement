# Medidor de Detritos — Visão Computacional

Aplicação Python para **medição dimensional de detritos** em tempo real via webcam. Utiliza um marcador **ArUco** para calibração de escala e segmentação por contorno (OpenCV).

## Integrantes

- Khadija do Rocio Vieira de Lima (RM558971)
- Ricardo Fernandes de Aquino (RM554597)

## Bibliotecas utilizadas

| Biblioteca | Uso |
|---|---|
| **OpenCV** (`opencv-contrib-python`) | Captura de vídeo, detecção ArUco, segmentação por contorno, overlay |
| **NumPy** | Operações numéricas sobre frames e contornos |

## Pré-requisitos

- Python 3.10+
- Webcam
- Marcador ArUco impresso (gerado pelo script incluído)

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Gerar o marcador ArUco

```bash
python assets/generate_aruco.py
```

Isso cria `assets/aruco_marker_id0.png`. Imprima em **escala 100% (tamanho real)**, sem redimensionar. O quadrado preto deve medir **5 cm × 5 cm**.

## Executar

```bash
python main.py
```

## Trocar de câmera (webcam externa)

No OpenCV, cada câmera conectada recebe um índice numérico:

| Índice | Dispositivo típico |
|--------|--------------------|
| `0` | Câmera integrada do notebook |
| `1` | Primeira webcam USB externa |
| `2` | Segunda câmera (se existir) |

### Usar a webcam externa

Edite `src/config.py` e altere o valor:

```python
CAMERA_INDEX = 1
```

Salve o arquivo e execute `python main.py` novamente.

### Descobrir qual índice usar

Se não souber qual índice corresponde a cada câmera, liste as disponíveis:

```bash
python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Índice {i}: câmera disponível')
        cap.release()
    else:
        print(f'Índice {i}: indisponível')
"
```

Para visualizar uma câmera específica, troque o valor de `i` e pressione `Q` para fechar:

```bash
python -c "
import cv2
i = 1  # troque 0, 1, 2...
cap = cv2.VideoCapture(i)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow('teste', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
"
```

Quando a imagem correta aparecer, use esse número em `CAMERA_INDEX`.

**Dica:** conecte a webcam USB **antes** de abrir o programa — em alguns sistemas o índice muda se ela for plugada depois.

### Janela lenta ou pequena

A janela agora abre **redimensionável** — arraste a borda ou maximize para ampliar.

Se ainda estiver lenta com a webcam externa, ajuste em `src/config.py`:

```python
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

Resoluções menores reduzem o processamento sem afetar a precisão da medição (desde que o ArUco e o detrito fiquem visíveis).

**Importante:** no Linux, a mesma webcam USB pode aparecer em **dois índices** — um em baixa resolução (ex.: índice 1 → 640×480) e outro em alta (ex.: índice 2 → 1280×720 ou 1920×1080). Se a imagem estiver pixelada e o contorno impreciso, rode o script de listagem acima e troque para o índice com maior resolução.

### Contorno impreciso

- Use fundo contrastante (pedra escura em papel branco, ou vice-versa).
- Pressione `T` para alternar modos: **auto** (padrão), **claro**, **escuro**, **adaptativo**.
- Pressione `D` para ver a máscara branca/preta — se a pedra não aparecer sólida, mude o modo ou a iluminação.
- Garanta resolução ≥ 960px de largura (`CAMERA_INDEX` correto + `FRAME_WIDTH = 1280`).

## Como usar

1. Imprima e recorte o marcador ArUco.
2. Coloque o marcador **ao lado do detrito**, no **mesmo plano** (mesma superfície).
3. Posicione o detrito sobre um fundo com **bom contraste** (ex.: detrito escuro em superfície clara).
4. Aponte a webcam — as medidas aparecem no topo da tela.
5. Pressione **S** para salvar.

### Controles

| Tecla | Ação |
|---|---|
| `S` | Salvar imagem anotada + JSON em `output/` |
| `T` | Alternar modo de detecção (auto / claro / escuro / adaptativo) |
| `D` | Mostrar máscara de segmentação (debug do contorno) |
| `R` | Resetar suavização das medidas |
| `Q` / `Esc` | Sair |

## Arquivos salvos

Cada captura gera dois arquivos em `output/`:

- `detrito_YYYYMMDD_HHMMSS.png` — imagem com anotações (contorno, dimensões)
- `detrito_YYYYMMDD_HHMMSS.json` — metadados (largura, altura, área em cm, timestamp)

## Estrutura do projeto

```
debris-measurement/
├── main.py
├── requirements.txt
├── assets/
│   └── generate_aruco.py
├── src/
│   ├── camera.py
│   ├── aruco_calibrator.py
│   ├── debris_detector.py
│   ├── measurer.py
│   ├── renderer.py
│   └── snapshot.py
└── output/          # imagens salvas (gerado em runtime)
```

## Dicas para melhor precisão

- Use boa iluminação uniforme.
- Mantenha câmera e objeto estáveis durante a medição.
- O marcador ArUco deve estar sempre visível no frame.
- Se o detrito não for detectado, pressione `T` para alternar o modo (tente **auto** ou **adaptativo**).
