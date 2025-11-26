# 📘 Apuntes – Deep Learning: Atención (Parte 2 – Vaswani et al., 2017)

---

## 1. 🎯 Motivación: ¿Qué es la Atención según “Attention is All You Need”?

El paper de Vaswani et al. define atención como:

> Un mecanismo que toma una **query** y un conjunto de **pares key–value**, y devuelve una **combinación ponderada de los values**, donde el peso asignado a cada value depende de una función de compatibilidad entre la query y la key correspondiente.

Formalmente:

$$\text{Attention}: Q \times {K:V} \rightarrow V. $$

Dado un conjunto de keys ${k_i}$, values ${v_i}$ y una query $q$:
$$  
\text{Attention}(q, {k_i:v_i}) = \sum_{i=1}^{n} \alpha_i(q) , v_i  
$$

donde cada $\alpha_i$ indica **qué tanto mirar** a cada value.

---

## 2. 🧱 Estructuras y espacios vectoriales

Se asume:

- Query: $q \in \mathbb{R}^p$
- Key: $k_i \in \mathbb{R}^p$
- Value: $v_i \in \mathbb{R}^d$

La atención produce un vector en $\mathbb{R}^d$:
$$  
z = \sum_{i=1}^{n} \alpha_i v_i  
$$

Las $\alpha_i$ son compatibilidades normalizadas (softmax).

---

## 3. ⚖️ Casos extremos de compatibilidad

### 🟥 Caso 1: _Winner-takes-all_

La atención escoge solo un índice:

$$ 
\alpha_i = 1_{{i = \arg\max_j (q \cdot k_j)}}  
$$
→ Interpretación: el modelo “fija la vista” en un único elemento.

### 🟦 Caso 2: _Indecisión total_

Distribución uniforme:

$$ 
\alpha_i = \frac{1}{n}  
$$

→ La query mira a todos los values por igual.

### 🔥 Caso intermedio: Softmax

$$ 
\alpha_i = \operatorname{Softmax}_i(q \cdot k_1, \ldots, q \cdot k_n)  
$$

Controlado por **temperatura** $\tau$:

- $\tau \to 0$: aproximación al winner-takes-all
- $\tau \to \infty$: distribución uniforme

---

## 4. 🔁 Atención para múltiples queries

Si tenemos queries ${q_j}_{j=1}^m$:

$$ \text{Attention}({q_j},{k_i:v_i})
\left{  
\sum_{i=1}^n \alpha_i(q_j) v_i  
\right}_{j=1}^m  
$$

Cada query obtiene su propio vector de salida.

---

## 5. 🧮 Notación matricial: el corazón de los Transformers

Vaswani et al. definieron el mecanismo de atención en forma **completamente vectorizada**:

$$ 
\text{Attention}(Q,K,V) = 
\operatorname{Softmax}(QK^{\top}) V  
$$
Donde:

- $Q \in \mathbb{R}^{m \times p}$ → consultas
- $K \in \mathbb{R}^{n \times p}$ → claves
- $V \in \mathbb{R}^{n \times d}$ → valores

### 5.1 Producto QKᵀ

Da matriz de similitudes $m\times n$:
$$  
QK^\top_{j,i} = q_j \cdot k_i  
$$

Cada fila → una query comparada con todas las keys.

### 5.2 Softmax fila a fila
$$  
A = \operatorname{Softmax}(QK^\top) $$

Cada fila de A es una **distribución sobre keys**.

### 5.3 Multiplicación por V
 
$$Z = AV$$

Cada fila de Z es la combinación ponderada de values para una query.

---

## 6. 🔥 Self-Attention

La atención anterior es **genérica**.  
Self-attention es un caso particular:

> Aplicar atención _dentro_ de la propia secuencia.

### 6.1 Construcción de Q, K, V

Para cada posición $x_i$ se aprenden proyecciones:
$$
q_i = W_Q x_i,\quad k_i = W_K x_i,\quad v_i = W_V x_i $$

El mecanismo compara cada posición consigo misma y con todas las demás.

### 6.2 Resultado final

Una secuencia de salida:
$$
Y = (y_1, \ldots, y_n)  $$
donde cada $y_i$ incorpora:

- el significado de $x_i$
- relaciones con toda la secuencia completa

👉 Es la base del Transformer Encoder.

---

## 7. 📋 Fórmula final de Self-Attention escalar

Dada una query $q$ y un conjunto de keys y values:

$$
z = \sum_{j=1}^n \alpha_j v_j  $$
donde:
$$
\alpha_j = \frac{\exp(f(k_j, q))}{\sum_{i=1}^n \exp(f(k_i, q))}  $$
y $f$ es la función de compatibilidad.

---

## 8. 🚀 Scaled Dot-Product Attention

Transformers definen $f$ como:
$$
f(k,q) = \frac{k q^\top}{\sqrt{d_k}}  $$

donde $d_k$ es la dimensión de las keys.

### ¿Por qué escalar?

Cuando $d_k$ crece:

- dot-products tienen varianza mayor
- softmax se “satura” más fácilmente
- entrenamiento pierde estabilidad

Escalar por $\sqrt{d_k}$ estabiliza los valores.

---

## 9. 🧮 Self-Attention (versión matricial final)

$$  
\text{Attention}(Q,K,V)
=
\operatorname{Softmax}  
\left(  
\frac{QK^\top}{\sqrt{d_k}}  
\right)V  
$$
Esta es exactamente la fórmula usada en los Transformers.

---

## 10. 🌀 Invarianza bajo permutaciones

### Propiedad

La self-attention **no tiene noción del orden** de la secuencia.

Es completamente invariante a permutar los tokens, porque $QKᵀ$ depende solo de similitudes entre tokens, no de posiciones.

### Consecuencia

Transformers NECESITAN agregar **información posicional**.

### Métodos:

- 🔹 Positional encodings sinusoidales (fijos)
- 🔹 Positional encodings aprendidos
- 🔹 Representaciones relativas (Shaw et al.)

Estas permitirán que el modelo entienda orden, distancia y dirección.

---

# 🧷 Tabla comparativa: Bahdanau vs. Scaled Dot-Product Attention

|Aspecto|Bahdanau|Dot-Product (Transformers)|
|---|---|---|
|Compatibilidad|MLP no lineal|Producto escalar|
|Escalamiento|No|Sí, 1/√dk|
|Costo computacional|Alto|Mucho menor|
|Paralelizable|Poco|Totalmente|
|Se usa en|Seq2Seq clásico|Transformers|

---

# 🧠 Diagrama conceptual

```
QKᵀ → Softmax → combinación con V → salida
```

Self-attention aplica esta operación por **toda la secuencia en paralelo**.

---

# ✅ Conclusiones

- La atención de Vaswani et al. formaliza el mecanismo como una transformación **query → weighted sum de values**, con compatibilidades basadas en keys.
- La versión matricial permite aplicar atención **a múltiples queries simultáneamente**, clave para el paralelismo masivo de Transformers.
- El mecanismo base del Transformer es la **Scaled Dot-Product Attention**, diseñada para estabilidad numérica en altas dimensiones.
- La **Self-Attention** permite que cada token “vea” a todos los demás tokens de la secuencia.
- El mecanismo es **invariante al orden**, por lo que se agregan **positional encodings**.
- Esta formulación simplificada, paralelizable y efectiva es la base de **Attention is All You Need**, que reemplaza completamente a las RNNs en las tareas de NLP.

