# PARTE B — Redes convolucionales, arquitecturas, secuencias, transformers y regularización

## 7) Convoluciones 2D — fórmula general y parámetros

**Idea clave:** conv2d aplica kernels sobre imagenes; output depende de kernel, padding, stride, dilation.

Fórmula (altura):

```
out_h = floor((in_h + 2*padding - dilation*(kernel_h - 1) - 1) / stride) + 1
```

Parámetros:

- `kernel_size` (k), `stride` (s), `padding` (p), `dilation` (d), `groups`.
    

Consejos:

- Para conservar tamaño con `kernel=3` usar `padding=1` y `stride=1`.
    
- `dilation` aumenta receptive field: efectiva kernel = `d*(k-1)+1`.
    
- `groups=in_channels` → depthwise conv.
    

Pool:

- `MaxPool2d(kernel=2, stride=2)` reduce dimensiones a la mitad (p.ej. 32→16).
    

Pitfalls:

- `stride=0` inválido.
    
- Cuidado con `ceil_mode=True` en pooling (redondeo hacia arriba).
    

Snippet:

```python
conv = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
x = torch.randn(8,3,64,64); out = conv(x)  # shape (8,16,64,64)
```

---

## 8) DenseNet vs ResNet — mecanismos y consecuencias

**ResNet (skip-add):** `y = F(x) + x`. Facilita paso de gradientes por sumas identitarias. Requiere que `F(x)` tenga misma shape que `x` (hacer 1x1 conv si cambia canales).

**DenseNet (concatenación):** cada capa `l` recibe `[x0, x1, ..., x_{l-1}]` concatenados.

- `growth_rate` (k) indica cuántos feature maps añade cada capa.
    
- `Transition blocks` (1x1 conv + pool) reducen canales/espacio para controlar crecimiento.
    

Consecuencias:

- DenseNet reutiliza features, suele usar menos parámetros para rendimiento comparable.
    
- Sin transition blocks, canales crecen linealmente con las capas → explosion de memoria.
    

Pitfalls:

- Confundir suma (ResNet) con concatenación (DenseNet) en preguntas.
    

---

## 9) RNN / LSTM / GRU — shapes, ecuaciones y uso práctico

**Shapes (batch_first=True):**

- Input: `(B, T, input_size)`.
    
- RNN output: `output, h_n`
    
    - `output`: `(B, T, num_directions * hidden_size)` (hidden por paso)
        
    - `h_n`: `(num_layers * num_directions, B, hidden_size)` (último hidden por capa)
        

**LSTM:** devuelve `(output, (h_n, c_n))` donde `c_n` es estado de celda.

Ecuaciones (resumen):

- RNN simple: `h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`.
    
- LSTM: gates `i,f,o` y celda `c_t = f*c_{t-1} + i * g`; `h_t = o * tanh(c_t)`.
    
- GRU: puertas `z` y `r`, menos parámetros que LSTM.
    

Consejos:

- Usar `pack_padded_sequence` para sequences con padding y acelerar / evitar que padding afecte al RNN.
    
- Inicializar `h0` opcionalmente, PyTorch pone zeros si no.
    

Pitfalls:

- Confundir `h_n` con salida por tiempo.
    
- No usar `batch_first` y equivocarse en shapes.
    

---

## 10) Transformers — encoder/decoder, teacher forcing, inferencia

**Arquitectura alta:** encoder procesa input y produce representaciones; decoder genera tokens condicionados en encoder outputs y tokens previos.

**Teacher forcing:** en entrenamiento el decoder recibe el token real `y_{t-1}`; acelera aprendizaje pero provoca _exposure bias_ en inferencia.

**Inferencia:** token-por-token (autoregresivo) porque futuros tokens no son conocidos; se usan técnicas como greedy, beam search, sampling. Para eficiencia en transformers se cachean `K,V` de pasos previos.

Pitfalls:

- Confundir entrenamiento (paralelizable con teacher forcing) con inferencia (secuencial).
    
- No explicar que en transformers se puede paralelizar el entrenamiento gracias a teacher forcing y masks.
    

---

## 11) Atención (Scaled dot-product) — matemática y máscaras

**Fórmula esencial:**

```
scores = Q @ K^T / sqrt(d_k)    # (B, T_q, T_k)
attn = softmax(scores, dim=-1)
context = attn @ V              # (B, T_q, d_v)
```

**Multi-head:** proyectar Q,K,V en H cabezas (d_k = d_model / H), calcular attention por cabeza, concatenar y aplicar linear final.

**Máscaras:**

- `padding_mask`: evita atención a posiciones de padding (aplica sobre scores antes de softmax con `-inf`).
    
- `causal_mask` (look-ahead): triangular superior para evitar ver futuros en decoder.
    

Por qué dividir por `sqrt(d_k)`:

- Si d_k grande, scores tienen varianza alta que causa softmax muy peaky; división estabiliza.
    

Pitfalls:

- No aplicar masks correctamente → leaking positions o atenciones a padding.
    

---

## 12) Positional encoding — sinusoidal vs learned

**Sinusoidal (paper):**

```
PE(pos,2i)   = sin(pos / (10000^(2i/d_model)))
PE(pos,2i+1) = cos(pos / (10000^(2i/d_model)))
```

Ventajas:

- No parametrizado (no learnable), permite extrapolar a secuencias más largas; codifica relaciones relativas en combinaciones lineales.
    

**Learned positional embeddings:** se aprenden y a menudo funcionan mejor en práctica si longitud fija conocida, pero no extrapolan.

Pitfalls:

- No confundir positional encodings con embeddings de tokens.
    

---

## 13) NLP — vocabulario, tokens, padding, truncamiento

**Conceptos:**

- **Token**: unidad (palabra/subword/char).
    
- **Vocab**: mapa token → índice. Incluye `<pad>`, `<unk>`, `<sos>`, `<eos>`.
    
- **Padding**: completar secuencias a misma longitud; usar `padding_idx` en embedding.
    
- **Truncamiento**: recortar secuencias largas.
    
- **Embeddings**: mapear tokens a vectores densos; tamaño `(vocab_size, emb_dim)`.
    

Consejos:

- Limitar vocab con `max_vocab_size` para memoria; descartar rare words o mapear a `<unk>`.
    
- Normalizar texto (lowercase, strip punctuation) antes de tokenizar si conviene.
    

Pitfalls:

- No alinear batches si no haces padding; olvidar `padding_idx` y que padding aprenda parámetros inútiles.
    

---

## 14) Regularización y Data Augmentation

**Regularización técnica:**

- **Dropout**, **L2 weight decay**, **early stopping**, **data augmentation**, **batchnorm** (regularizador implícito).
    
- Seleccionar técnica según tipo de overfitting y datos.
    

**Data augmentation para imágenes:**

- Comunes: `RandomHorizontalFlip`, `RandomRotation`, `RandomResizedCrop`, `ColorJitter`, `Normalize`.
    
- Evitar transformaciones que cambien la etiqueta semántica (p.ej. flip para texto/OCR).
    

Consejos prácticos:

- Aplicar `Normalize(mean,std)` tras augmentations.
    
- Mantener aleatoriedad reproducible con seeds en DataLoader workers.
    

Pitfalls:

- Augment demasiado agresivo que daña señal (ej. cortar objetos pequeños).
    
- Aplicar augmentation en validación (no se debe).
    

---

## 15) Resumen de “qué llevar al parcial” (lista de chequeo)

- Saber definiciones: dtype, device, ndim, shape.
    
- Saber aplicar broadcasting y explicar reglas (ejemplo).
    
- Implementar early stopping y explicar por qué.
    
- Saber formulas de conv2d (incluyendo dilation) y pool.
    
- Diferenciar ResNet (suma) y DenseNet (concat), explicar transition blocks.
    
- Shapes de RNN/LSTM/GRU y uso de `pack_padded_sequence`.
    
- Explicar attention (Q,K,V), fórmula, por qué dividir por `sqrt(d_k)`.
    
- Explicar masks (padding y causal) y por qué se usan.
    
- Saber qué es positional encoding y su fórmula.
    
- Saber cómo funciona embedding y `padding_idx`.
    
- Saber listar augmentations útiles y cuándo son perjudiciales.
    

---

Si querés, ahora:

1. Genero **respuestas modelo** (listas para copiar) para cada pregunta textual de los PDFs; o
    
2. Te dejo **flashcards** (pregunta+respuesta breve) por cada subtema; o
    
3. Hacemos un **simulacro** de 30–45 minutos con preguntas aleatorias y yo te corrijo.
    

¿Qué querés que haga ahora?