
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


----


# 📘 Apuntes – Deep Learning: Atención en Seq2Seq (Bahdanau et al., 2014)


---

## 1. 🎯 Extracto / motivación (abstract)

- Problema: los encoder–decoder clásicos codifican la oración fuente en **un único vector de longitud fija**, lo cual es un **cuello de botella** para traducción y tareas similares.
- Idea de Bahdanau et al. (2014): **permitir que el modelo "soft-search"** automáticamente qué partes de la oración fuente son relevantes para predecir cada palabra objetivo, en lugar de comprimirlo todo en un único vector.
- Resultado: **aprender a alinear y traducir conjuntamente**.

---

## 2. 🔁 Encoder–Decoder clásico (antes de Bahdanau)

- Entrada: $x=(x_1,\dots,x_{T_x})$
- Encoder (ej.: LSTM) produce hiddens $h_t = f(x_t,h_{t-1})$.
- Context vector clásico: $c = q({h_1,\dots,h_{T_x}})$ (habitualmente $c=h_{T_x})$.
- Decoder modela la traducción como:

$$
p(y) = \prod_{t=1}^{T_y} p(y_t \mid y_{1:t-1}, c),  
$$

con un RNN-decoder: 
$$p(y_t \mid \cdot) = g(y_{t-1}, s_t, c))$$donde $s_t$ es el hidden del decoder.

![[Pasted image 20251127124046.png]]

---

## 3. 🧭 Limitación identificada

- El uso de un **vector fijo** $c$ reúne toda la información; esto falla en oraciones largas o con estructura compleja.
- Bahdanau propone que **cada palabra target tenga su propio context vector** $c_i$ calculado como una combinación ponderada de los hiddens del encoder.

---

## 4. 🧩 Modelo con atención (formulación central)

- Nuevo modelado para cada $i$:

$$
p(y_i \mid y_{1:i-1}, x) = g(y_{i-1}, s_i, c_i)  
$$

- La actualización del hidden del decoder incorpora $c_i$:

$$
s_i = f(s_{i-1}, y_{i-1}, c_i).  
$$

- Importante: **$c_i$ depende de $i$** (un vector distinto por cada paso del decoder).

---

## 5. 🏗️ Arquitectura (esquema)

- Encoder produce la secuencia de anotaciones (hiddens) $h_1,\dots,h_{T_x}$.
- Para cada paso $i$ del decoder:
    - Se calcula $c_i$ a partir de los $h_j$.
    - El decoder (con su hidden previo $s_{i-1}$ y la palabra previa $y_{i-1})$ genera $s_i$ y predice $y_i$.
- En entrenamiento suele usarse _teacher forcing_ (dar $y_{i-1}$ real como entrada).
    
(La PPT muestra un diagrama con encoder → atención → decoder, y flujo train/inference.)
![[Pasted image 20251127124146.png]]


---

## 6. 🔢 Cálculo del context vector $c_i$ (fórmulas)

Definición:
$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} , h_j.  
$$
Los pesos $\alpha_{ij}$ son una distribución (softmax) sobre las posiciones de la fuente:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}  
= \text{Softmax}_j([e_{i1},\dots,e_{iT_x}]).  
$$

Donde el **score de alineación** $e_{ij}$ se define como:

$$
e_{ij} = a(s_{i-1}, h_j),  
$$

es decir, una función (aprendida) que mide la compatibilidad entre el estado previo del decoder $s_{i-1}$ y el hidden del encoder $h_j$.

---

## 7. 🔁 Terminología moderna (keys, values, queries)

- **Keys:** ${h_1,\dots,h_{T_x}}$ (hiddens del encoder)
- **Values:** usualmente las mismas anotaciones ${h_1,\dots,h_{T_x}}$. Son combinaciones del hidden del encoder
- **Queries:** hiddens del decoder ${s_1,\dots,s_{T_y}}$

Interpretación: para cada query (decoder step) se calcula una combinación ponderada de values según compatibilidades con las keys.

---

## 8. 🧾 Intuición probabilística

- $\alpha_{ij}$ puede verse como la **probabilidad** de que la palabra target $y_i$ esté alineada con la fuente $x_j$.
- $c_i$ es la **esperanza** (weighted average) de las anotaciones $h_j$ bajo esa distribución: el "contexto" relevante para generar $y_i$.

---

## 9. 🖼️ Visualizaciones (qué muestran los heatmaps)

- La PPT incluye ejemplos de heatmaps de atención sobre frases en francés: muestran qué tokens fuente contribuyen a cada token objetivo.
- Observación práctica: la atención revela correspondencias (a menudo casi diagonales) y reordenamientos cuando el idioma lo requiere — proporciona **interpretabilidad**.

---

## 10. Atención de Bahdanau

En cada paso del decoder, el modelo **decide a qué partes de la secuencia fuente prestar atención** para generar la siguiente palabra.

- El encoder produce hiddens
- El decoder tiene un hidden previo $s_{i-1}$ y la palabra anterior $y_{i-1}$
- La atención calcula pesos $\alpha_{ij}$ sobre los hiddens del encoder y un context vector $c_i$
- El decoder usa $c_i$ para actualizar su estado y predecir $y_1$

### Entradas del decoder en el paso $i$

En el paso $i$ del decoder:
- **Input**: La palabra target anterior.
- **prev_hidden**: hidden state previo del decoder
- **encoder_outputs**: hiddens del encoder.

La palabra anterior se encaja:

$$\text{embedded} = E(y_{i-1})
\quad\quad \tilde{e}_{i-1} = \text{Dropout}(E(y_{i-1}))$$
 
Embedding y context vector se usarán luego como entrada al GRU del decoder.
## 11. ⚙️ Parametrización del alignment (cómo se calcula $e_{ij}$)

Bahdanau parametriza $a(\cdot)$ como una **red feedforward** entrenable. En la PPT se resume el cálculo así:

1. Proyectar el hidden del decoder y cada hidden del encoder:
$$
W_s s_{i-1}, \quad W_h h_j.  
$$

2. Sumar y aplicar no linealidad:
$$
\text{sum}_{ij} = W_s s_{i-1} + W_h h_j,  
$$
$$
\tilde{v}_{ij} = \tanh(\text{sum}_{ij}).  
$$
3. Reducir a score escalar con vector (v):
$$
e_{ij} = v^\top \tilde{v}_{ij}.  
$$

- $v$ es un parámetro (vector) independiente de $i,j$.
- Las $\tilde{v}_{ij}$ son vectores intermedios que dependen de la posición $j$ y del paso $i$.
	- Son vector de **alineación**

---

## 11. 🔢 Cálculo de atención: scores → softmax

A partir  de los vectores $\tilde{v}_{ij}$, se obtienen **scores escalares** de alineación

- A partir de $e_{ij}$ se obtiene $\alpha_{ij}$ vía softmax (ya mostrado).
	- Lo que da un score para cada posición $j$ del input.
- $\tilde{v}_{j} \rightarrow$ vector intermedio dependiente de $j$
- $v$ (no depende de $i$ y $j$) $\rightarrow$ parámetros del mecanismo de atención
- $\alpha_{ij}$ es la probabilidad de alinear $y_i$ con la posición $j$ del input.
- El vector $\alpha_i = (\alpha_{i1},\dots,\alpha_{iT_x})$ son los pesos de atención para el paso $i$.

---

## 12. 🧩 Construcción de $c_i$ (repetición breve)

Pesos de atención  $\alpha_{ij}$ y hidden del encoder $h_j$ forman el **context vector**

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j.  
$$

Interpretación: el modelo decide “dónde mirar” y forma (c_i) como la suma ponderada de las anotaciones.

---

## 13. 🔁 Decoder GRU con atención (entradas y actualización)

Entradas al decoder en el paso (i) (según PPT):

- **Input**: la palabra target anterior (y_{i-1}).
    
- **Prev hidden**: (s_{i-1}).
    
- **Encoder outputs**: ({h_1,\dots,h_{T_x}}).
    

La palabra previa se convierte en embedding y se aplica dropout:

[  
\text{embedded} = E(y_{i-1}),\qquad \tilde{e}_{i-1} = \text{Dropout}(E(y_{i-1})).  
]

Luego el GRU del decoder actualiza el estado combinando el embedding y el context vector:

[  
s_i = \mathrm{GRU}([\tilde{e}_{i-1}, c_i],; s_{i-1}).  
]

Así (s_i) incorpora tanto la historia del decoder como la información de la fuente relevante (vía (c_i)).

---

## 14. 🔚 Salida y distribución sobre la próxima palabra

A partir del nuevo hidden (s_i) se calcula el logit y la probabilidad:

[  
o_i = W_o s_i + b_o,  
]

[  
p(y_i \mid y_{1:i-1}, x) = \mathrm{Softmax}(o_i).  
]

---

## 15. ✅ Resumen paso a paso (operacional) — paso (i)

1. **Calcular scores** (e_{ij}) usando (a(s_{i-1}, h_j)) (proyección + tanh + (v^\top)).
    
2. **Obtener pesos** (\alpha_{ij} = \text{Softmax}_j(e_{i1},\dots,e_{iT_x})).
    
3. **Formar context vector** (c_i = \sum_j \alpha_{ij} h_j).
    
4. **Actualizar hidden** del decoder: (s_i = \mathrm{GRU}([\tilde{e}_{i-1}, c_i], s_{i-1})).
    
5. **Predecir**: (o_i = W_o s_i + b_o) → (p(y_i)=\text{Softmax}(o_i)).
    

(Esta lista sigue estrictamente el flujo que aparece en la PPT.)

---

## 16. 📚 Parámetros aprendidos principales (recap)

- Pesos del encoder (p. ej. LSTM).
    
- Pesos del decoder (GRU) y embeddings (E(\cdot)).
    
- Parámetros del alignment network: (W_s, W_h, v).
    
- Proyección final (W_o, b_o).
    

---

## 17. 📝 Conclusión (fiel al paper / PPT)

- Bahdanau et al. introducen un **mecanismo de atención suave** que evita el cuello de botella del vector único (c).
    
- La atención **aprende a alinear** dinámicamente la entrada y el output en cada paso, mejorando traducción y tareas seq2seq.
    
- La arquitectura mantiene interpretabilidad (matrices de (\alpha)) y puede entrenarse end-to-end junto con encoder y decoder.
    

---

¿Querés que te lo deje ahora en **Markdown descargable (.md)** tal cual —o lo exporto también a **.pdf** con el mismo formato— para que lo tengas como apuntes listos para imprimir?