
# 📘 Apuntes – Deep Learning: Atención en Seq2Seq (Bahdanau Attention)

---

## 1. 🎯 Motivación: ¿Por qué surge la Atención?

Los modelos encoder–decoder clásicos (pre-2014) comprimían **toda la secuencia de entrada** en un único vector:

$$
c = h_{T_x}  
$$

donde $h_{T_x}$ es el **último hidden** del encoder.
### ❗ Problemas

- Cuanto más larga la oración → peor el rendimiento.
- Toda la información debe entrar en un solo vector → **cuello de botella**.
- El decoder debe traducir sin poder “mirar” nuevamente la secuencia fuente.
- Dificultad para capturar dependencias largas.

Bahdanau et al. (2014) identifican esta limitación y proponen:

> **Aprender a alinear y traducir simultáneamente**, permitiendo que el decoder mire dinámicamente distintas partes del input.

---

## 2. 🧱 Encoder–Decoder clásico (sin atención)

### Encoder

Procesa la secuencia de entrada:

$$  
x=(x_1,\ldots,x_{T_x})  
$$
y genera hiddens:

$$
h_t = f(x_t, h_{t-1})  
$$
El vector de contexto:

$$ 
c = q(h_1, \ldots, h_{T_x}) = h_{T_x}  
$$

### Decoder

Predice:
$$ 
p(y)=\prod_{t=1}^{T_y} p(y_t \mid y_{1:t-1}, c)  
$$

Con una RNN:
$$  
p(y_t \mid y_{1:t-1},c)=g(y_{t-1}, s_t, c)  
$$

donde $s_t$ es el hidden del decoder.

👉 Todo depende del **mismo** vector $c$.  
De allí el cuello de botella.

### Traducción antes de Bahdanau
![[Pasted image 20251126145138.png]]

---

## 3. 🧠 Idea clave de Bahdanau: Atención + Alineamiento suave

En lugar de usar un único vector de contexto: $c$ se propone usar **un vector distinto para cada paso del decoder**: $c_i$ donde $i$ es el índice de la palabra target.

El modelo aprende una función de **alineamiento**:
$$  
e_{ij} = a(s_{i-1}, h_j)  
$$

que indica qué tan relevante es el hidden del encoder $h_j$ para generar la palabra $y_i$.

Luego se normaliza con softmax:

$$  
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}.  
$$

El context vector:

$$  
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j  
$$

⭐ Interpretable:  
$\alpha_{ij}$ ≈ probabilidad de alinear $y_i$ con $x_j$.


---

## 4. 📡 Queries, Keys y Values (paralelo moderno)

- **Keys** → hiddens del encoder $h_1,\ldots,h_{T_x}$
- **Values** → los mismos hiddens (o combinación)
- **Queries** → hiddens del decoder ${s_i}$

**Atención =**  
**“Qué value mirar” según “qué query tengo”.**

Esto anticipa la terminología usada en Transformers.

---

## 5. 🧬 Arquitectura completa del modelo con Atención

### 5.1 Encoder

Produce:
$$  
h_1,\ldots,h_{T_x}  
$$
### 5.2 Atención

Para cada paso i del decoder:

1. Calcular scores:      $$ 
    e_{ij} = a(s_{i-1},h_j)  
    $$
2. Softmax:  
$$  
    \alpha_{ij} = \text{Softmax}(e_i)_j  
    $$
3. Context vector:  $$  
    c_i = \sum_j \alpha_{ij} h_j  
    $$
### 5.3 Decoder

Actualización del estado:

$$  
s_i = f(s_{i-1}, y_{i-1}, c_i)  
$$
Predicción:

$$  
p(y_i \mid y_{1:i-1}, x) = g(y_{i-1}, s_i, c_i)  
$$
---

## 6. 🔍 Cálculo de e₍ᵢⱼ₎: Mecanismo de Alineación

Bahdanau propone parametrizar (a(\cdot)) como un **MLP**.

### Paso 1: Proyecciones

$$  
W_s s_{i-1}, \quad W_h h_j  
$$
### Paso 2: Suma + no linealidad

$$  
\tilde{v}_{ij} = \tanh(W_s s_{i-1} + W_h h_j)  
$$
### Paso 3: Score escalar

$$  
e_{ij} = v^\top \tilde{v}_{ij}  
$$
donde (v) es un vector aprendible.

---

## 7. 🎛️ Interpretación probabilística

- $\alpha_{ij}$ = “probabilidad suave” de que $y_i$ se traduzca desde $x_j$
- $c_i$ = “esperanza” de las anotaciones $h_j$ bajo esa distribución

👉 Atención = **alineamiento suave**  
ya que no selecciona una única posición sino una distribución.

---

## 8. 🎨 Visualización de matrices de atención

El PDF presenta varios ejemplos (frases francesas → atención sobre palabras correspondientes).

Estos heatmaps muestran:

- qué palabras del input miró el decoder para generar cada palabra output
- correspondencias casi diagonales para secuencias similares
- desviaciones cuando el idioma requiere reordenar palabras

👉 Interpretabilidad clara: la atención “mira” las regiones relevantes.

---

## 9. 🧩 Integración en un Decoder GRU

El decoder combina tres insumos:

- embedding de $y_{i-1}: \tilde{e}_{i-1}$
- hidden previo: $s_{i-1}$
- context vector: $c_i$

Actualiza:
$$  
s_i = \text{GRU}([\tilde{e}_{i-1}, c_i], s_{i-1})  
$$

Salida:
$$  
o_i = W_o s_i + b_o  
$$

Distribución de probabilidad:
$$  
p(y_i) = \text{Softmax}(o_i)  
$$

---

## 10. 🧱 Resumen del paso i del Decoder

1. **Cálculo de scores**  
$$ e_{ij} = a(s_{i-1}, h_j) $$

2. **Atención (softmax)**  
$$\alpha_{ij} = \mathrm{Softmax}(e_i)_j $$

3. **Context vector**  
$$c_i = \sum_j \alpha_{ij} h_j $$

4. **Actualización del estado**  
$$s_i = \text{GRU}([e_{i-1}, c_i], s_{i-1})$$
5. **Predicción**  
$$p(y_i) = \text{Softmax}(W_o s_i + b_o)$$


---

# 📘 Tabla comparativa: Encoder–Decoder clásico vs. Bahdanau Attention

| Característica                   | Encoder–Decoder clásico | Bahdanau Attention |
| -------------------------------- | ----------------------- | ------------------ |
| Vector de contexto               | Uno solo (c)            | Uno por paso (cᵢ)  |
| Alineamiento                     | No existe               | Sí, suave (αᵢⱼ)    |
| Dependencias largas              | Malas                   | Mucho mejores      |
| Interpretabilidad                | Nula                    | Alta (heatmaps)    |
| Reordenamiento de palabras       | Difícil                 | Natural            |
| Capacidad para secuencias largas | Limitada                | Escala mejor       |
| Complejidad                      | Simple                  | Moderada           |

---

# ✅ Conclusiones

- El encoder–decoder original tiene un **cuello de botella**: toda la info del input debe comprimirse en un único vector.
- Bahdanau introduce el **mecanismo de atención**, permitiendo que el decoder “mire” dinámicamente distintas partes del input.
- Esto se logra mediante una distribución (\alpha_{ij}) sobre los hiddens del encoder.
- Cada palabra target tiene su **propio context vector** (c_i).
- Atención = mecanismo de alineamiento suave, interpretable, flexible y efectivo.
- El modelo resultante **mejora significativamente** tareas como la traducción automática.
- Este mecanismo sentó las bases para la atención moderna usada en los **Transformers**.
