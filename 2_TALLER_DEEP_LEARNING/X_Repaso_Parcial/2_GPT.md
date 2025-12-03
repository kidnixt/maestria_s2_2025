
# 1 — PyTorch y tensores (breve + útil)

- **Tensor**: estructura N-D que contiene datos y gradientes.
- **dtype**: tipo numérico (float32, int64, etc.). Importante para precisión y memoria.
- **device**: dónde está el tensor (CPU o GPU: `cpu`, `cuda:0`). Mover con `.to(device)` o `.cuda()`.
- **ndim**: número de dimensiones (rank).
- **shape**: tupla con tamaños por dimensión (p. ej. `(3,4)`).
- **reshape / view**: cambian la forma sin copiar datos en general (`view` exige continuidad).
- **squeeze(x)**: elimina dimensiones de tamaño 1.
- **unsqueeze(x, dim)**: añade dimensión de tamaño 1 en `dim`.
- **torch.cat(list, dim)**: concatena tensores a lo largo de `dim` (requiere mismas otras dims).
- **torch.stack(list, dim)**: apila creando una nueva dimensión (todos tienen misma shape; el resultado tiene +1 dim).
- **broadcasting**: reglas que permiten operar tensores de shapes distintas cuando dimensiones son compatibles (uno de ellos debe ser 1 o igual). Ejemplo: `(4,3) + (3,) -> (4,3)` se replica la dim `(3,)` sobre la primera dimensión.  


# 2 — GPU, device y operaciones entre dispositivos

- **`torch.cuda.is_available()`**: comprueba GPU.
- **Mover tensor**: `x = x.to(device)` o `x = x.cuda()` y volver con `.cpu()`.
- **Error mixto**: operar tensor CPU con tensor CUDA lanza excepción; solución: mover ambos al mismo `device`.
    

# 3 — Dataset / DataLoader (práctico)

- **Dataset personalizado**: implementar `__len__` y `__getitem__`.
- **DataLoader params**:
    - `batch_size`: tamaño por batch (impacta memoria, estabilidad, velocidad).
    - `shuffle`: mezcla datos (útil en train).
    - `num_workers`: procesos para cargar datos (aumenta throughput; cuidado con Windows y colisiones).
- **Efectos**: mayor `batch_size` → menos ruido, mayor memoria; `num_workers`>0 acelera IO.
    

# 4 — Train / Eval (modo de modelo)

- **`model.train()`**: activa comportamientos de entrenamiento (Dropout, BatchNorm en modo train).
- **`model.eval()`**: desactiva dropout, BatchNorm usa estadísticas acumuladas; para evaluación.
- **`torch.no_grad()`**: desactiva cálculo de gradientes (memoria y tiempo). Siempre usar en validación/inferencia.
    

# 5 — Pérdidas: train_loss / val_loss y early stopping

- **train_loss**: promedio de loss en batches de entrenamiento (mide ajuste al train set).
- **val_loss**: loss en conjunto de validación (mide generalización).
- **Early stopping**: parar si `val_loss` no mejora N épocas (paciencia). Implementación típica: guardar `best_val` y contador de no-mejora. Ejemplo breve (pseudo):
    
    ```python
    best = inf; patience=5; counter=0
    for epoch:
        train()
        val = validate()
        if val < best - delta:
            best = val; counter = 0; save_model()
        else:
            counter += 1
            if counter >= patience: break
    ```
    

# 6 — Capas básicas

- **nn.Linear(in, out, bias=True)**: `y = x @ W^T + b`. Si input `x` tiene shape `(B, in)`, output `(B, out)`.
- **nn.Dropout(p)**: durante train, con prob `p` apaga unidades (multiplica por Bernoulli(1-p)) y escala el resto; en eval no hace nada. `p` alto → más regularización.
- **nn.Embedding(num_embeddings, embedding_dim)**: mapea índices enteros a vectores densos `(seq_len, embedding_dim)`; útil para NLP. Para usar pre-trained: cargar pesos en `embedding.weight`.

# 7 — Convoluciones y Pooling (shapes)

- **Conv2d(in_ch, out_ch, kernel_size, stride, padding)**: salida calculada por:  ```
	out_h = floor((in_h + 2*pad - kernel_h)/stride) + 1```
	igual para ancho. `padding` corrige el borde.
- **MaxPool2d(kernel_size, stride)**: reduce resolución tomando máximos; output shape se calcula similar al conv.
- **Stride = 0** no tiene sentido — stride mínimo es 1. (Si se pone 0 falla).

# 8 — RNN / LSTM / GRU: formas y diferencias

- **`nn.RNN(input_size, hidden_size, num_layers, batch_first=True)`**:
    - Input shape: `(B, T, input_size)` si `batch_first`.
    - Output: `output, h_n` donde `output` es `(B, T, hidden_size)` (o `hidden_size * num_directions` si bidireccional) y `h_n` es `(num_layers * num_directions, B, hidden_size)`.
- **Diferencias clave**:
    
    - RNN: simple recurrent, sufre gradientes desvanecientes/explosivos.
    - LSTM: usa gate mechanisms (forget, input, output) y celda `c_t` → maneja mejor dependencias largas.
    - GRU: simplifica LSTM (menos parámetros) manteniendo buen rendimiento.
        

# 9 — DenseNet vs ResNet (conciso)

- **ResNet (residual block)**: suma $x + F(x)$ (skip-add). Evita degradación en redes profundas.
- **DenseNet (conexiones densas)**: cada capa concatena todas las salidas previas como entrada $x_l = H_l([x_0, x_1, ..., x_{l-1}]$.
    - **Ventaja DenseNet**: mejor reutilización de características, parámetros eficientes por crecimiento (`growth_rate`).
    - **Transition blocks**: reducen dimensiones (1x1 conv + pooling) para controlar tamaño y coste computacional. Si los eliminas, modelo crece mucho en parámetros y memoria y se vuelve ineficiente.


# 10 — Regularización y Data Augmentation

- **Regularización**: técnicas para reducir overfitting: Dropout, L2 weight decay, early stopping, data augmentation, batchnorm (parcial).
- **Data augmentation imágenes**: ejemplos comunes:
    - RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop, ColorJitter, GaussianNoise, Normalize, Cutout, RandomResizedCrop.
- **Cuando es contraproducente**: rotaciones 180° en dígitos con orientación importante (ej. texto), o crop que elimina la clase objetivo.
    

# 11 — Seq2Seq y Transformers (fundamentos, breve)

- **Seq2Seq (encoder-decoder)**: encoder procesa input (p. ej. frase source) y produce representaciones; decoder genera output condicionado en la representación y tokens previos. En traducción: encoder → contexto, decoder → produce palabras una a una.
- **Teacher Forcing**: durante entrenamiento, alimentas al decoder con el token real $y_{t-1}$ en vez del token predicho. Acelera convergencia pero crea discrepancia en inferencia.
- **Token** : indica inicio de secuencia para que decoder sepa cuándo empezar.
- **Fin de secuencia (inferencia)**: se genera token a token hasta que se produce `<EOS>` o se alcanza longitud máxima. Razón: en generación la siguiente predicción depende del token ya generado.
    

# 12 — Atención (Scaled dot-product) y máscaras

- **Scaled dot-product attention**:
   ```
    scores = Q @ K^T / sqrt(d_k)
    attention = softmax(scores, dim=-1)
    context = attention @ V
    ```
    Q,K,V dimensiones: `(B, heads, T_q, d_k)` etc. El factor `1/sqrt(d_k)` previene que los logits sean muy grandes.
    
- **Máscaras**:
    - **Padding mask**: evita atención a posiciones de padding.
    - **Causal mask / look-ahead mask**: en decoder evita que cada posición vea futuros tokens (triangular superior masked).
    - **Masking en entrenamiento**: permite parallelizar el cálculo (pero manteniendo la restricción causal).


# 13 — Positional Encoding (formula y propósito)

- **Propósito**: introducir información posicional en modelos sin recurrencia.
- **Sinusoidal formula (paper "Attention is all you need")**:
    ```
    PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
    ```
    Esto permite que la red infiera relaciones posicionales mediante combinaciones lineales.

# 14 — Inference en generación de texto (por qué token a token)

- **Razón**: en inferencia no conocés tokens futuros; la probabilidad de cada token depende de tokens previos generados. En entrenamiento con teacher forcing se conoce la secuencia completa y se puede paralelizar, pero en inferencia debes decodificar autoregresivamente (greedy, beam search, sampling).
    


# 16 — Weights & Biases (WandB)

- **WandB**: plataforma para tracking de experimentos, métricas, modelos, artefactos.
- **Run**: ejecución individual que registra métricas.
- **Sweep**: experimento para búsqueda de hiperparámetros (define espacio y estrategia; lanza muchos runs).

---

# 17 — Fragmentos de código útiles (micro-snippets listos)
- **Mover tensor a GPU y volver**:
    
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    x = x.cpu()
    ```
    
- **Squeeze / Unsqueeze**:
    
    ```python
    x = x.squeeze()        # elimina dims 1
    x = x.unsqueeze(0)     # añade dim 0
    ```
    
- **Cat vs Stack**:
    
    ```python
    a = torch.tensor([1,2])
    b = torch.tensor([3,4])
    torch.cat([a,b], dim=0)   # -> [1,2,3,4]
    torch.stack([a,b], dim=0) # -> [[1,2],[3,4]]
    ```
    
- **Early stopping (compacto)**: ver snippet en sección 5.

---

# 18 — Preguntas de repaso (para practicar)

Te dejo **40 preguntas rápidas** tipo parcial; respóndelas en voz alta o por escrito para probarte:

1. ¿Qué contiene `tensor.shape`, `tensor.dtype`, `tensor.device` y `tensor.ndim`?
2. ¿Cómo pasar un tensor a GPU? Muestra el código.
3. Explica broadcasting con un ejemplo `(4,3)` + `(3,)`.
4. Diferencia entre `reshape` y `view`.
5. ¿Qué hacen `squeeze` y `unsqueeze`?
6. `torch.cat` vs `torch.stack` — ¿cuándo usar cada uno?
7. ¿Qué hace `model.train()` y `model.eval()`? ¿Qué pasa con Dropout en cada caso?
8. ¿Por qué usar `with torch.no_grad()` en validación?
9. ¿Qué es `num_workers` en DataLoader y qué efecto tiene?
10. Define `batch_size` y su impacto en entrenamiento.
11. ¿Cómo implementás early stopping con `val_loss` y paciencia=5?
12. ¿Qué hace `nn.Linear(4,2)` a un input `(3,4)`? ¿Cuál será la shape?
13. ¿Qué hace `nn.Embedding(6,3)` con input `[0,2,4]`?
14. Formula salida de Conv2d para `out_h`.
15. ¿Qué hace MaxPool2d(kernel=2, stride=2) a un tensor `(8,3,32,32)`?
16. Diferencia conceptual entre RNN y LSTM.
17. Forma de salida `output, h_n` de un RNN con `batch_first=True`.
18. ¿Qué significa `growth_rate` en DenseNet?
19. ¿Qué hacen los transition blocks en DenseNet?
20. Diferencia entre skip-connection (ResNet) y concatenación (DenseNet).
21. Explica teacher forcing y por qué puede crear discrepancia.    
22. ¿Por qué en inferencia no se puede paralelizar la generación completa?
23. Escribe la fórmula de attention (dot-product).
24. ¿Qué hace el factor `1/sqrt(d_k)` en attention?
25. ¿Qué tipo de máscara evita que el decoder vea futuros tokens?
26. Escribe la fórmula de positional encoding sinusoidal.
27. ¿Qué es mode collapse en GANs y una forma para mitigarlo?
28. ¿Qué hace `nn.Dropout(p=0.5)` durante entrenamiento? ¿Y en eval?
29. ¿Cómo inicializas un embedding con pesos pre-entrenados? (breve)
30. ¿Qué es un sweep en WandB? ¿Y un run?
31. ¿Qué transformaciones incluirías en un pipeline de augmentations para imágenes? Nombra 4.
32. Da un ejemplo de augmentation que sería contraproducente para OCR.
33. ¿Qué es padding mask y cuándo se usa?
34. ¿Cómo evitar operar tensors en CPU y CUDA a la vez?
35. ¿Qué pasa si olvidas `model.eval()` durante validación?
36. ¿Qué implica `bias=True` en `nn.Linear`?
37. ¿Cómo calcular `out_w` en una conv 2D? (fórmula)
38. ¿Qué hace `torch.unsqueeze(x, dim=0)`?
39. Explica brevemente qué representan `Q`, `K`, `V` en attention.
40. Menciona 3 técnicas de regularización distintas a Dropout.


---

# 19 — Cómo uso esto contigo ahora

Puedo:

- Generarte **respuestas modelo** para cualquiera de las preguntas reales de los parciales (listas para copiar).
    
- Hacer un **simulacro de 30 minutos** con preguntas tipo parcial (te corrijo).
    
- Prepararte **tu defensa obligatoria** (texto listo para decir: transformaciones, cambios del modelo, loss).
    

---

¿Querés que ahora **genere respuestas modelo** (muy concisas) para cada una de las preguntas que aparecen textualmente en tus PDFs, listas para que las aprendas y las copies en el parcial? Si es así empiezo por la **Sección PyTorch y Tensores** y te doy respuestas modelo para cada subpregunta (listas para usar).