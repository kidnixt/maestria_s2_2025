
# 📘 Apuntes – Deep Learning: **Transformers (Parte 2)**

_(Basado en Teórico MDL 21 – Arquitectura Encoder/Decoder del Transformer)_

---

# 1. 🎯 Visión general del Transformer (Vaswani et al., 2017)

El Transformer introduce una arquitectura completamente basada en **atención**, compuesta de dos bloques principales:
- **Encoder** (izquierda)
- **Decoder** (derecha)
Ambos construidos repitiendo **N bloques** idénticos.

## 1.1 Encoder

Cada bloque aplica, en este orden:
1. **Multi-Head Attention**
2. **Add & LayerNorm**
3. **Feed-Forward Network (FFN)**
4. **Add & LayerNorm**

## 1.2 Decoder

Cada bloque aplica:
1. **Masked Multi-Head Self-Attention**
2. **Multi-Head Attention sobre las salidas del encoder**
3. **Feed-Forward Network**
4. **Add & LayerNorm** después de cada operación

Al final del decoder:

- Proyección lineal
- Softmax sobre el vocabulario

---

# 2. 🔡 Embeddings + Positional Encoding

Dada una secuencia de entrada:
$$ 
x = (x_1, \dots, x_{T_x}),\quad x_t \in V.  
$$

Cada token se convierte en un embedding:

$$
e_t = E(x_t) \in \mathbb{R}^{d_{\text{model}}}.  
$$
Se suma el **Positional Encoding** correspondiente:
$$
z_t^{(0)} = e_t + PE_t.  
$$

En forma matricial:
$$
\mathbf{z}^{(0)} = \begin{bmatrix}
(\mathbf{z}_1^{(0)})^{\text{T}} \\
\vdots \\
(\mathbf{z}_{T_x}^{(0)})^{\text{T}}
\end{bmatrix} \sim (T_x, d_{\text{model}}).
$$

Esta matriz entra al primer encoder.

---

# 3. 🧱 Bloque del Encoder

Para cada capa $l = 1,\dots,N$:

Entrada:  
$$ 
Z^{(l-1)} \sim (T_x, d_{\text{model}}).  
$$
## 3.1 Multi-Head Self-Attention

$$
H^{(l)} = \mathrm{MHA}(Z^{(l-1)}, Z^{(l-1)}, Z^{(l-1)}).  
$$
## 3.2 Residual + LayerNorm
$$ 
\tilde{Z}^{(l)} = \mathrm{LayerNorm}(Z^{(l-1)} + H^{(l)}).  
$$
## 3.3 Feed-Forward Network
$$ 
F^{(l)} = \mathrm{FFN}(\tilde{Z}^{(l)}).  
$$
## 3.4 Residual + LayerNorm final

$$ 
Z^{(l)} = \mathrm{LayerNorm}(\tilde{Z}^{(l)} + F^{(l)}).  
$$

---

# 4. 🧠 Multi-Head Attention (MHA)

Dadas matrices:

- $Q \sim (m, d_{\text{model}})$
- $K \sim (n, d_{\text{model}})$
- $V \sim (n, d_{\text{model}})$

Cada head $i$ proyecta:
$$
Q_i = Q W_i^Q,\quad K_i = K W_i^K,\quad V_i = V W_i^V,  
$$

con:

- $W_i^Q, W_i^K \sim (d_{\text{model}}, d_k)$
- $W_i^V \sim (d_{\text{model}}, d_v)$

## 4.1 Atención por head

$$
\text{head}_i = \mathrm{Softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i  
\quad \sim (m \times d_v).  
$$
## 4.2 Salida final del MHA

$$
\mathrm{MHA}(Q,K,V) = \mathrm{Concat}(\text{head}_1,\dots,\text{head}_h), W^O  
$$

donde $W^O \sim (h d_v, d_{\text{model}})$.

---

# 5. 🔁 Self-Attention en el Encoder

En el encoder:

$$ 
Q = K = V = Z^{(l-1)} \sim (T_x, d_{\text{model}})  
$$
Cada head:
$$
\text{head}_i^{(l)} =  
\mathrm{Softmax}\left(  
\frac{Z^{(l-1)} W_i^Q (Z^{(l-1)} W_i^K)^\top}{\sqrt{d_k}}  
\right) Z^{(l-1)} W_i^V.  
$$

Cada fila de la salida es una **combinación lineal** de todas las posiciones de entrada. La concatenación de todas las heads y la proyección final dan $H^{l}$

---

# 6. 🧮 Layer Normalization

Dado $u = (u_1,\dots,u_d)$:
- **LayerNorm** normaliza el vector a lo largo de sus coordenadas (no en batch)
- Se calculan media y varianza

$$
\mu = \frac{1}{d} \sum u_i,\qquad  
\sigma^2 = \frac{1}{d}\sum (u_i - \mu)^2  
$$

- Luego se normaliza cada coordenada
$$
\hat{u}_i = \frac{u_i - \mu}{\sqrt{\sigma^2 + \epsilon}}  
$$

- Luego se aplica un **re-escalado y desplazamiento** con parámetros entrenables:
$$
\mathrm{LayerNorm}(u) = \gamma \odot \hat{u} + \beta,  
\quad \gamma,\beta \sim d.  
$$

Para una matriz $U \sim (T,d)$ se aplica **fila a fila**.

$$
\text{LayerNorm}(\mathbf{U}) = \begin{bmatrix}
\text{LayerNorm}(\mathbf{u}_1) \\
\vdots \\
\text{LayerNorm}(\mathbf{u}_T)
\end{bmatrix}
$$
---

# 7. ⚙️ Feed-Forward Network (FFN)

Se usa una FNN aplicada **independientemente** a cada posición de la secuencia (misma red para todos los tokens)

Para cada vector $z \sim (d_{model})$
$$
\mathrm{FFN}(z) = \sigma(z W_1 + b_1), W_2 + b_2  
$$

con:

- $W_1 \sim (d_{\text{model}}, d_{\text{ff}})$
- $W_2 \sim (d_{\text{ff}}, d_{\text{model}})$
- $\sigma$: ReLU en Vaswani, GELU moderno

En forma matricial:

$$
\text{FFN}(\mathbf{Z}) = \begin{bmatrix}
\text{FFN}(\mathbf{Z}_{1,:}) \\
\vdots \\
\text{FFN}(\mathbf{Z}_{T,:})
\end{bmatrix}
$$
Es decir:
- La misma MLP se aplica fila a fila.
- No hay interacción entre posiciones en la FNN: toda la mezcla entre tokens viene de la atención.

---

# 8. 🔤 Tokens especiales del Decoder

- $BOS$: inicio
- $EOS$: final

Durante entrenamiento:
- Entrada al decoder:  
    $(\text{BOS}, y_1, \dots, y_{T_y-1})$
- Salida esperada:  
    $(y_1, \dots, y_{T_y}, \text{EOS})$

Durante inferencia:
- Comienza con $BOS$
- Genera tokens hasta producir $EOS$

---

# 9. 🟥 Decoder: Máscara Causal

Se obtienen embeddings + positional encodings:
$$
Z^{(0)}_{\text{dec}} \sim (T_y, d_{\text{model}}).  
$$

En la primera subcapá se aplica Masket Multi-Head Self-Attention:

$$
Q = K = V = Z^{(l-1)}_{\text{dec}}.  
$$

La máscara causal impone:
$$
(\mathbf{Q}\mathbf{K}^{\text{T}})_{tj} \to 
\begin{cases}
-\infty & j > t, \\
(\mathbf{Q}\mathbf{K}^{\text{T}})_{tj} & j \le t.
\end{cases}
$$

→ La softmax ignora posiciones futuras.

---

# 10. 🟩 Decoder: Atención Encoder–Decoder (Cross-Attention)

Queries: son las salidas de la subcapa anterior del decoder
$$
Q = \tilde{Z}^{(l)}_{\text{dec}} \sim (T_y, d_{model})  
$$

Keys/Values: son las salidas finales del encoder

$$
K = V = Z^{(N)}_{\text{enc}}  
$$

Atención:

[  
H^{(l)}_{\text{enc-dec}} = \mathrm{MHA}(Q,K,V).  
]

Luego:

[  
\hat{Z}^{(l)}_{\text{dec}}  
= \mathrm{LayerNorm}!\bigl( \tilde{Z}^{(l)}_{\text{dec}} + H^{(l)}_{\text{enc-dec}}\bigr)  
]

Después FFN + Add & Norm igual al encoder.

---

# 11. 🧾 Salida final del Decoder

La última capa del decoder produce:

[  
Z^{(N)}_{\text{dec}} \in (T_y, d_{\text{model}}).  
]

Para cada posición:

[  
o_t = Z^{(N)}_{\text{dec},t}, W_{\text{out}} + b_{\text{out}}  
\quad \in \mathbb{R}^{|V|}  
]

Probabilidad del token:

[  
p(y_t \mid y_{<t}, x) = \mathrm{Softmax}(o_t).  
]

---

# 12. 📏 Observaciones sobre la secuencia de entrada

- El encoder procesa **cualquier longitud (T_x)**.
    
- No hay recurrencia temporal.
    
- El positional encoding permite manejar posiciones arbitrarias.
    
- Para batches con longitudes distintas → se usa **padding + máscara** para que los no sean atendidos.
    

---

# 13. 📐 Observaciones sobre la secuencia de salida

- El decoder genera:
    

[  
p(y_t \mid y_{<t}, x)  
]

- La máscara causal asegura que no se mire el futuro.
    
- El largo de salida (T_y) es variable.
    
- El cross-attention permite que (T_x \neq T_y).
    

---

# 14. 🌀 Positional Encoding: Motivación

El self-attention es **invariante a permutaciones**.  
Depende solo de productos escalares entre filas de Q, K y V.

Por eso se suma:

[  
z_t = e_t + PE_t.  
]

El vector (PE_t) debe tener dimensión (d_{\text{model}}).

---

# 15. 🔊 Positional Encoding Sinusoidal

Para posición (pos) y coordenada (i):

[  
PE(pos, 2i) = \sin!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),  
]

[  
PE(pos, 2i+1) = \cos!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).  
]

Propiedades:

- Frecuencias decrecen geométricamente
    
- No requiere parámetros
    
- Permiten extrapolación a secuencias más largas
    

---

# 16. 📚 Bibliografía

- Vaswani et al., 2017 – _Attention Is All You Need_
    
- Alammar, 2018 – _The Illustrated Transformer_
    

---

# ✅ Conclusiones

- El Transformer encoder–decoder combina **self-attention**, **cross-attention**, **FFNs**, **LayerNorm** y **residuals** en una arquitectura flexible.
    
- El encoder procesa secuencias completas con **self-attention bidireccional**.
    
- El decoder usa:
    
    - **máscara causal** (autoregresión),
        
    - **cross-attention** (consulta al encoder).
        
- La salida final se obtiene mediante una capa lineal seguida de softmax.
    
- El positional encoding permite introducir la noción de **orden**.
    
- La arquitectura es totalmente paralelizable, sin recurrencia, y maneja secuencias de longitudes variables.
    

---

Si querés, seguimos inmediatamente con el próximo PDF 📘🔥