
---

# 1) Broadcasting — explicación exhaustiva y ejemplos

**Idea**: broadcasting permite operar tensores de distintas formas sin crear copias explícitas, “estirando” dimensiones de tamaño 1 según reglas precisas.

## Reglas formales (PyTorch / NumPy compatible)

1. Alinea las shapes por la **derecha** (última dimensión con última, etc.).
    
2. Para cada dimensión (de derecha a izquierda):
    
    - Si las dimensiones son iguales → OK.
        
    - Si una dimensión es 1 en uno de los tensores → el tensor con 1 se **repite** (virtually) a lo largo de esa dimensión.
        
    - Si una dimensión no existe en uno de los tensores (tensor tiene menos dims) → se comporta como si tuviera dimensión 1 en esa posición.
        
    - Si ninguna de las condiciones anteriores se cumple, se lanza un error (shapes incompatibles).
        

## Consecuencia: la operación devuelve una shape que es el máximo por dimensión después de alinear por la derecha.

## Ejemplos con explicación paso a paso

### Ejemplo 1 — `(4,3) + (3,)`

- Align: `(4,3)` y `(1,3)` (porque `(3,)` se ve como `(1,3)` al alinear por la derecha).
    
- Comparación por dimensión (izquierda→derecha):
    
    - Dim 0: 4 vs 1 → 1 se replica 4 veces.
        
    - Dim 1: 3 vs 3 → iguales.
        
- Resultado shape: `(4,3)`.
    
- Código:
    
    ```python
    x = torch.arange(12.).reshape(4,3)
    y = torch.tensor([10.,20,30])
    z = x + y  # broadcasting: y se repite 4 veces
    ```
    

### Ejemplo 2 — `(4,1,3) + (3,)`

- Align: `(4,1,3)` with `(1,1,3)`.
    
- Dim by dim: (4 vs 1) → replicate; (1 vs 1) → replicate; (3 vs 3) → OK.
    
- Resultado `(4,1,3)` → si hicieras `squeeze` podrías obtener `(4,3)`.
    

### Ejemplo 3 — `(5,4)` + `(3,)` → **ERROR**

- Align: `(5,4)` vs `(1,3)`: last dims 4 vs 3 incompatible → excepción.
    

## Efecto en memoria y compute

- Broadcasting **no siempre copia** datos físicamente; PyTorch crea views/strides que indexan el mismo buffer repetidas veces hasta que una operación requiere una copia (p.ej. `.clone()` o asignación a distinto shape). Sin embargo, algunas operaciones pueden materializar copias si el backend lo necesita.
    

## Casos prácticos y uso común

- Añadir bias a salidas conv / linear: `out + bias` donde `bias` shape `(out_channels,)`. Broadcasting automático.
    
- Normalización por canal: `x - mean` donde `mean` shape `(C,)` y `x` shape `(N,C,H,W)` → mean se comporta como `(1,C,1,1)`.
    

## Pitfalls / errores comunes

- Pensar que `(4,) + (4,1)` es OK: al alinear por la derecha se comparan `(4,)` ↔ `(4,1)` → se interpreta `(1,4)` vs `(4,1)` y falla si no hay compatibilidad. Siempre alinear mentalmente por la derecha.
    
- Operaciones in-place peligrosas con broadcasting: `a += b` cuando b se broadcastea puede intentar escribir en memoria compartida y causar error si shapes no son compatibles para asignación in-place. Mejor usar no-inplace o `.expand`/`.repeat` explícito.
    

## Expand vs Repeat vs View

- `expand`: crea view que repite virtualmente un tensor sin copiar datos, sólo cambia strides; sólo funciona si la dimensión a expandir tiene tamaño 1.
    
    ```python
    t = torch.tensor([1.,2.])       # shape (2,)
    t_exp = t.unsqueeze(0).expand(4,2)  # view: shape (4,2) sin copia
    ```
    
- `repeat`: replica datos físicamente (copia real).
    
    ```python
    t_rep = t.unsqueeze(0).repeat(4,1)  # copia
    ```
    
- `expand` falla si la dimensión no es 1. `repeat` siempre funciona pero consume memoria.
    

---

# 2) Reshape / View / Contiguity / Squeeze / Unsqueeze — detalle técnico

- **`tensor.reshape(new_shape)`**:
    
    - Intenta devolver una view con nueva shape sin copiar. Si la memoria no está contigua en el orden requerido, puede devolver una copia (materializa).
        
    - Acepta `-1` para inferir una dimensión.
        
- **`tensor.view(new_shape)`**:
    
    - Requiere que el tensor sea **contiguous** en memoria según su stride. Si no lo es, falla. Para forzar, hacer `tensor.contiguous().view(...)` (que hace copia si hace falta).
        
- **Contiguidad**:
    
    - Operaciones como `transpose` o `permute` devuelven un tensor con strides cambiados — no contiguo. `.contiguous()` devuelve copia contigua.
        
- **`squeeze(dim=None)`**:
    
    - Si `dim` es None elimina todas las dims de tamaño 1.
        
    - Si `dim` es int elimina solo esa dimensión si su tamaño es 1; si no lo es, error.
        
- **`unsqueeze(dim)`**:
    
    - Inserta una dimensión de tamaño 1 en `dim`.
        
- **Ejemplo**:
    
    ```python
    x = torch.randn(2,1,3)
    y = x.squeeze(1)        # shape (2,3)
    z = y.unsqueeze(1)      # back to (2,1,3)
    ```
    
- **Importante**: usar `view` en tensores no contiguos -> RuntimeError. Use `.reshape` o `.contiguous().view()`.
    

---

# 3) Cat vs Stack — diferencias y ejemplos

- **`torch.cat(tensors, dim)`**:
    
    - Concatena a lo largo de una dimensión existente. Todos los tensores deben tener la misma shape excepto en `dim`.
        
    - Ejemplo: `torch.cat([A,B], dim=0)` con `A.shape=(2,3), B.shape=(5,3)` → result `(7,3)`.
        
- **`torch.stack(tensors, dim)`**:
    
    - Crea una nueva dimensión y apila los tensores en esa dimensión. Todos los tensores deben tener exactamente la misma shape.
        
    - Ejemplo: `stack([A,B], dim=0)` con `A.shape=(2,3)` -> result `(2,2,3)` (si apilas 2 tensores).
        
- **Uso típico**:
    
    - Cat: combinar batches.
        
    - Stack: agregar eje de "experimentos" o "canales".
        

---

# 4) Conv2d / stride / padding / dilation / groups — fórmula y ejemplos

**Fórmula general (altura)**:

```
out_h = floor((in_h + 2*padding - dilation*(kernel_h - 1) - 1) / stride) + 1
```

- `kernel_h` tamaño kernel vertical.
    
- `dilation` expande el receptive field (espacio entre kernel elements).
    
- `groups` divide convolución en sub-convoluciones independientes (depthwise conv: groups = in_channels).
    
- `padding` puede ser `'same'` (en frameworks) o explícito int. En PyTorch, `padding=1` agrega 1 píxel por lado.
    
- **Ejemplo**:
    
    - `in: 32`, `kernel=3`, `pad=1`, `stride=1`, `dilation=1` → `out = floor((32 + 2 - 2)/1) + 1 = 32`.
        
- **Stride > 1** reduce tamaño; stride = 2 => reducción a la mitad (aprox).
    
- **Dilation**: si `dilation=2` y `kernel=3` el kernel "actúa" como si tuviera tamaño 5 en spacing.
    

## Padding types

- `padding=0` (default): no padding.
    
- `padding=1`: añade 1 pixel por cada lado.
    
- For asymmetric padding, usar `F.pad` antes de conv.
    

## Groups

- `groups = in_channels` con `out_channels = K*in_channels` da depthwise conv.
    
- `groups > 1` particiona filtros y limita conectividad.
    

## Ejemplo numérico

```python
conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
x = torch.randn(8,3,64,64)
out = conv(x)  # out.shape -> (8,16,32,32)
```

## Pooling

- MaxPool2d kernel=2 stride=2 en `(B,C,32,32)` → `(B,C,16,16)`.
    

## Pitfalls

- Stride 0 no permitido (debe ser >=1).
    
- Cuidado con `ceil_mode=True` en pool que cambia la fórmula (redondea hacia arriba).
    

---

# 5) Dropout — comportamiento matemático y train vs eval

- **Definición**: durante entrenamiento, cada unidad de la capa es “apagada” con probabilidad `p` (Bernoulli(p)). Las activaciones sobrevivientes se escalan para mantener expectativa.
    
- **Two conventions**:
    
    - PyTorch: en train, aplica máscara `m ~ Bernoulli(1-p)`, y hace `output = input * m / (1-p)`. En eval no aplica mask; salida = input (no escalado).
        
    - Otra convención (en papers): no escalan en train y escalan en eval (menos común hoy).
        
- **Impacto**:
    
    - Regulariza reduciendo co-adaptaciones.
        
    - `p` alto (ej. 0.5) -> fuerte regularización; p pequeño (0.1) -> ligero.
        
- **Ejemplo en código**:
    
    ```python
    dropout = nn.Dropout(p=0.5)
    model.train(); out = dropout(x)   # aplica máscara y scaling
    model.eval(); out = dropout(x)    # identidad
    ```
    
- **Pitfall**: olvidar `model.eval()` en eval → salidas estocásticas y metricas inconsistentes.
    

---

# 6) Early stopping — implementación robusta

- **Idea**: monitorizar `val_loss` (o métrica) y parar cuando no mejora después de `patience` epochs. Guardar mejor checkpoint.
    
- **Consideraciones**:
    
    - `delta` mínimo (mejora pequeña) evita flapping.
        
    - `mode`: minimize loss vs maximize accuracy.
        
    - Restablecer scheduler si usás LR scheduler (reduce on plateau).
        
- **Snippet robusto**:
    
    ```python
    best = float('inf')
    patience = 5
    counter = 0
    delta = 1e-4
    for epoch in range(epochs):
        train_one_epoch(...)
        val_loss = validate(...)
        if val_loss < best - delta:
            best = val_loss
            counter = 0
            torch.save(model.state_dict(), "best.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    ```
    

---

# 7) RNN / LSTM / GRU — shapes, internals y diferencias

## Shapes (batch_first=True)

- Input: `(batch, seq_len, input_size)`
    
- RNN output tuple: `output, h_n`
    
    - `output`: `(batch, seq_len, num_directions * hidden_size)` — salida por cada paso de tiempo (hidden states en cada t).
        
    - `h_n`: `(num_layers * num_directions, batch, hidden_size)` — último hidden state de cada capa (y dirección).
        
- LSTM returns `(output, (h_n, c_n))` donde `c_n` es la celda final con misma shape que `h_n`.
    

## Equations compactas

- **RNN (tanh)**:
    
    ```
    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
    ```
    
- **LSTM** (gates):
    
    ```
    i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)   # input gate
    f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)   # forget gate
    g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)      # candidate
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)   # output gate
    
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    ```
    
- **GRU** (gates simplificados):
    
    ```
    z_t = sigmoid(W_z x_t + U_z h_{t-1})
    r_t = sigmoid(W_r x_t + U_r h_{t-1})
    h_tilde = tanh(W_h x_t + U_h (r_t * h_{t-1}))
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    ```
    

## Consideraciones prácticas

- **Packed sequences**: usar `pack_padded_sequence` / `pad_packed_sequence` para secuencias de diferentes longitudes y acelerar cálculo.
    
- **Initial hidden states**: si no pasás `h0`, PyTorch inicializa ceros.
    
- **Bidireccional**: `num_directions = 2`.
    

---

# 8) DenseNet vs ResNet — por qué y cómo funcionan

- **ResNet**: block residual: `y = F(x) + x`. La suma preserva la dimensionalidad por lo que `F(x)` debe producir la misma shape; se usa `1x1 conv` si cambia channels. Mitiga degradación y permite entrenar redes muy profundas.
    
- **DenseNet**: cada capa concatena todas las salidas previas: `x_l = H_l([x_0, x_1, ..., x_{l-1}])`.
    
    - **growth_rate** `k`: número de mapas de características que añade cada capa (corta el crecimiento).
        
    - **Transition block**: reduce `channels` con `1x1 conv` y reduce spatial con `avgpool` — controla tamaño y compute.
        
- **Comparación**:
    
    - ResNet suma (fácil backprop de gradiente mediante caminos de identidad).
        
    - DenseNet concatena (fomenta reutilización de features, menos parámetros para mismo performance).
        
- **Si eliminás transition blocks** → la concatenación se acumula y el número de canales crece mucho (O(L * k)), memoria explotará.
    

---

# 9) Attention / Multi-head attention — math y shapes

## Scaled dot-product attention (matricial)

- Input: `Q (B, T_q, d_k)`, `K (B, T_k, d_k)`, `V (B, T_k, d_v)`
    
- Compute:
    
    ```
    scores = Q @ K.transpose(-2,-1) / sqrt(d_k)   # (B, T_q, T_k)
    attn = softmax(scores, dim=-1)                # (B, T_q, T_k)
    context = attn @ V                             # (B, T_q, d_v)
    ```
    
- **Masking**: aplicar `scores.masked_fill(mask == 0, -inf)` antes del softmax.
    

## Multi-head

- Proyectar Q,K,V H veces con matrices lineales en paralelo:
    
    - Q_proj: `(B, T, H, d_k)` → reshape a `(B*H, T, d_k)`, calcular attention por cabeza y luego concatenar.
        
    - Output concat y linear final.
        
- **Shapes**:
    
    - d_model = H * d_k
        
    - After concat: `(B, T_q, H * d_v)` → final linear → `(B, T_q, d_model)`.
        

## Máscaras

- **Padding mask**: shape `(B, 1, 1, T_k)` o `(B, 1, T_q, T_k)` para broadcast. Bloquea posiciones de K/V que son padding.
    
- **Causal (look-ahead) mask**: triangular superior mask `(T, T)` que evita ver futuros: `mask[i,j] = 0 if j>i`.
    

## Numérico: por qué divide por `sqrt(d_k)`

- Si `d_k` grande, los productos escalares tienden a grandes valores medios, lo que hace softmax saturado (gradientes pequeños). Division estabiliza distribuciones y gradientes.
    

---

# 10) Positional encoding — por qué y propiedades

- **Why**: atención es permutation-invariant; PE encodes position.
    
- **Sinusoidal formula**:
    
    ```
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    ```
    
- **Propiedades**:
    
    - Permite extrapolación a secuencias más largas (funciones periódicas).
        
    - Las combinaciones lineales de PE permiten que modelos aprendan relaciones relativas: por ejemplo PE(pos+k) se expresa como función de PE(pos).
        
- **Alternativa**: learned positional embeddings (parametrizadas), a veces mejores en práctica pero no extrapolan.
    

---

# 11) Teacher forcing y exposure bias — detalle

- **Teacher forcing**: durante entrenamiento del decoder, en t paso alimentas token real `y_{t-1}` como input en vez de la predicción `\hat{y}_{t-1}`.
    
- **Ventajas**: acelera convergencia, reduce errores tempranos.
    
- **Problema**: **exposure bias** — en inferencia el modelo no ve tokens perfectos, por lo que se puede acumular error y degradar performance.
    
- **Mitigaciones**:
    
    - Scheduled sampling: con prob p usar la predicción en vez del target (annealing p durante training).
        
    - Data augmentation de tokens y robustificación del decoder.
        

---

# 12) Inference en generación: por qué autoregresivo y estrategias

- **Por qué**: cada token depende de tokens previos; en inferencia aún no conoces futuros → debes predecir secuencialmente.
    
- **Estrategias de decodificación**:
    
    - **Greedy**: pick argmax at each step (rápido, miopía).
        
    - **Beam search**: mantener top-k hypotheses; trade-off calidad/compute.
        
    - **Sampling**: muestreo de softmax (temperature, top-k, top-p (nucleus)).
        
- **Complejidad**: autoregresivo requiere recomputar parte de modelo; optimizaciones: cache de keys/values (transformers) para no recomputar encoder/previos K/V.
    

---

# 13) Embeddings — shape, inicializar pre-trained

- **`nn.Embedding(num_embeddings, embedding_dim)`**: mapping from integer indices `[0..num_embeddings-1]` to dense vectors `(embedding_dim)`.
    
- **Input**: shape `(seq_len,)` o `(batch, seq_len)`.
    
- **Output**: if input `(batch, seq)` → `(batch, seq, embedding_dim)`.
    
- **Cargar pesos pre-trained**:
    
    ```python
    embedding = nn.Embedding(num_embeddings, dim)
    embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
    ```
    
    O usa `padding_idx` para que padding tenga embedding = 0 y no aprenda.
    

---

# 14) DataLoader `num_workers` — detalle y errores comunes

- `num_workers=0` → carga en proceso principal (determinístico, más lento).
    
- `num_workers>0` → cada worker carga batch; datos pasan por cola multiproceso.
    
- **Windows**: `spawn` start method provoca que todo el módulo sea importable (si DataLoader crea objetos no picklables puede fallar). Evitar lambdas en datasets o usar `if __name__ == "__main__":`.
    
- **Efectos**:
    
    - Aumenta throughput IO y preprocessing.
        
    - Demasiados workers en CPUs limitadas → overhead y contención.
        
- **Bug común**: reproducibilidad y random seeds: cada worker hereda seed distinto; usar `worker_init_fn` y `torch.manual_seed`.
    

---

# 15) Convocatoria práctica: errores típicos en ejercicios del parcial

- Olvidar `model.eval()` en validación (Dropout activo, BatchNorm en modo train).
    
- Hacer inplace ops con broadcasting que fallan (`x += y` cuando shapes no compatibles para write).
    
- Usar `view` en tensor no contiguo.
    
- No usar `pack_padded_sequence` cuando secuencias tienen longitudes variadas y se asume que RNN maneja padding sin máscara.
    
- Confundir `cat` y `stack` en respuestas — es muy típico en parciales.
    

---

# 16) Preguntas rápidas y respuestas cortas (listas para memorizar)

Voy a darte las **respuestas “listas para poner en el parcial”** para 15 preguntas críticas — muy concisas.

1. **dtype/device/ndim/shape**: `dtype` tipo numérico; `device` CPU/CUDA; `ndim` número de ejes; `shape` tupla con tamaño por eje.
    
2. **Mover a GPU**:
    
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    ```
    
3. **Broadcasting (ejemplo)**: `(4,3) + (3,)` → `(4,3)` replicando `(3,)` por la dimensión 0.
    
4. **Reshape vs view**: `reshape` puede copiar si no contiguo; `view` exige contiguidad.
    
5. **Squeeze/unsqueeze**: `squeeze` elimina dims tamaño 1; `unsqueeze` inserta dim 1.
    
6. **Cat vs Stack**: `cat` concatena sobre dim existente; `stack` crea nueva dim y apila.
    
7. **model.train() / model.eval()**: activa/desactiva comportamientos específicos (dropout/batchnorm).
    
8. **torch.no_grad()**: desactiva grads para inferencia (ahorra memoria/cálculo).
    
9. **num_workers**: procesos para carga paralela; mejora throughput; cuidado con Windows/pickling.
    
10. **Early stopping**: guardar best val_loss, aumentar contador si no mejora, parar si counter >= patience.
    
11. **Conv2d out formula**: see formula with dilation.
    
12. **MaxPool2d**: reduce resolución, ejemplo `(8,3,32,32)` -> `(8,3,16,16)` con kernel=2,stride=2.
    
13. **LSTM vs RNN**: LSTM tiene gates y state c_t que permiten dependencias largas; RNN simple sufre vanishing gradients.
    
14. **Teacher forcing**: alimentar target real como input del decoder durante train; crea exposure bias.
    
15. **Scaled dot-product**: `scores = Q @ K^T / sqrt(d_k)` → `attn = softmax(scores)` → `context = attn @ V`.
    

---
