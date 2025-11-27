
# 📘 Apuntes – Deep Learning: Definición de Attention



---

## 1. ✳️ Frase inicial (definición textual)

> “An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.  
> The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.”

---

## 2. 🧭 Espacios y notación (Q, K, V)

- Se definen los espacios:
    - Espacio de **queries**: $Q$
    - Espacio de **keys**: $K$
    - Espacio de **values**: $V$
    - Espacio de diccionarios (pares key–value): $\{K:V\}$
- La función de atención se formaliza como:  
    $$
    \text{Attention}: Q \times \{K:V\} \to V.  
    $$
- En la PPT se asume la simplificación (Q = K).
    

---

## 3. 🧮 Forma escalar (entrada vectorial)

-  $Q=\mathbb{R}^p \quad\wedge\quad q = [q_1,\dots, q_p]$ 
- $K=\mathbb{R}^p \quad\wedge\quad k = [k_1,\dots, k_p]$
- $V=\mathbb{R}^d \quad\wedge\quad v = [v_1,\dots, q_d]$

Entonces, la función Attention es de la forma

$$
\text{Attention}\big(q,\{k_i:v_i\}_{i=1}^n\big) = \sum_{i=1}^n \alpha_i\big(q,\{k_i\}_{i=1}^n\big) v_i,  
$$

donde las $\alpha_i$ calculan la **compatibilidad** entre $q$ y las keys.

---
## 4. 🔺 Casos extremos de compatibilidad

- **Winner-takes-all (extremo):** 
- Interpretración: el modelo "fija la vista" en un único elemento.

$$
    \alpha_i(q,\{k_i\}_{i=1}^n) = \mathbf{1}{i = \arg\max_j (q\cdot k_j)}.  
    $$
- **Indecisión (extremo uniforme):** 
- Interpretación: La query mira a todos los valores por igual.
    $$
    \alpha_i(q,\{k_i\}_{i=1}^n) = \frac{1}{n}.  
    $$

Estas son dos formas límite de la función de compatibilidad.

---

## 5. 🔁 Caso intermedio: softmax sobre productos escalares

El caso práctico usual, compatibilidad mediante softmax de productos escalares:

$$
\alpha_i(q,\{k_i\}_{i=1}^n) = \operatorname{Softmax}_i\big[q\cdot k_1, \ldots, q\cdot k_n\big]  
$$

y menciona el parámetro de **temperatura** (\tau) para controlar extremos:

- $\tau \to 0$ → winner-takes-all.
- $\tau \to \infty$ → distribución uniforme.

---

## 6. 🔗 Diagrama conceptual (proceso)

- Input: $q, \{k_i\}_{i=1}^n, \{v_i\}_{i=1}^n$
- Paso 1: calcular compatibilidades $e_i$ entre $q$ y cada $k_i$
- Paso 2: normalizar con softmax → $\alpha_i$
- Paso 3: suma ponderada → salida $z=\sum_i \alpha_i v_i$.


![[Pasted image 20251127151550.png]]

---

## 7. 🧩 Atención para múltiples queries

Si tenemos varias queries $\{q_j\}_{j=1}^m$, la función devuelve los $m$ outputs:
$$
\text{Attention}\big(\{q_j\}_{j=1}^m,\{k_i:v_i\}_{i=1}^n\big) = 
\left\{
\sum_{i=1}^n \alpha_i(q_j,\{k_i\}_{i=1}^n) v_i
\right\}_{j=1}^m
$$
Cada query obtiene su propia distribución $\alpha$.

---

## 8. 🧾 Notación matricial compacta

Agrupando queries, keys y values en matrices:

- $Q \in \mathbb{R}^{m\times p}$ (filas: queries)
    
- (K \in \mathbb{R}^{n\times p}) (filas: keys)
    
- (V \in \mathbb{R}^{n\times d}) (filas: values)
    

La operación se escribe:

[  
\text{Attention}(Q,K,V) = \operatorname{Softmax}(QK^\top), V.  
]

(Esta es la forma matricial que aparece en la PPT.)

---

## 9. 📐 Shapes (tal como en la diapositiva)

- (Q \sim (m,p))
    
- (K \sim (n,p))
    
- (V \sim (n,d))
    

Donde:

- (m): número de queries
    
- (n): número de keys/values.
    

---

## 10. ✖️ Producto matricial (QK^\top)

El producto (QK^\top) da una matriz (m\times n) con entradas:

[  
(QK^\top)_{j,i} = q_j\cdot k_i.  
]

En la PPT se muestra explícitamente la matriz de productos punto con cada fila correspondiente a una query.

---

## 11. 🔄 Softmax fila a fila

Se aplica softmax **por fila** de (QK^\top):

[  
\operatorname{Softmax}(QK^\top) =  
\begin{bmatrix}  
\operatorname{Softmax}(q_1\cdot k_1,\dots,q_1\cdot k_n) \  
\vdots \  
\operatorname{Softmax}(q_m\cdot k_1,\dots,q_m\cdot k_n)  
\end{bmatrix}.  
]

Cada fila es una distribución sobre las (n) values.

---

## 12. ➗ Multiplicación por (V) (salida)

Finalmente:

[  
\operatorname{Softmax}(QK^\top), V ;\in; \mathbb{R}^{m\times d}.  
]

La fila (j) es:

[  
\sum_{r=1}^n \operatorname{Softmax}_r(q_j\cdot k_1,\dots,q_j\cdot k_n), v_r,  
]

es decir, una combinación ponderada de las filas de (V).

---

## 13. 🧾 Historia / self-attention (contexto de la PPT)

Se cita el recuento histórico: Bahdanau et al. (2014) definieron la atención en seq2seq como un promedio ponderado de los hiddens del encoder usando scores de alineamiento con el hidden previo del decoder. La generalización es: **query = hidden previo del decoder; keys/values = hiddens del encoder**.

---

## 14. 🔁 Definición precisa de Self-Attention (texto literal)

La PPT define self-attention como aplicar la atención a **cada posición** de la secuencia fuente creando para cada (x_i) tres vectores (query, key, value) y aplicando la atención de (x_i) (query) sobre todas las keys/values de la secuencia, produciendo una secuencia (Y=(y_1,\dots,y_n)) donde cada (y_i) mezcla información de (x_i) y su relación con el resto.

---

## 15. 🔳 Diagrama de Self-Attention (mención)

La PPT incluye un diagrama visual del flujo Q,K,V → Attention → salida (no hay ecuaciones nuevas aquí).

---

## 16. 🧾 Resumen formal (fórmula que aparece)

La PPT resume la operación como:

[  
z = \sum_{j=1}^n \alpha_j v_j,\qquad  
\alpha_j = \frac{\exp(f(k_j,q))}{\sum_{i=1}^n \exp(f(k_i,q))},  
]

es decir, (\alpha_j) es la softmax de las compatibilidades (f(k_j,q)).

---

## 17. 🔬 Scaled dot-product attention (fórmula literal)

La función de compatibilidad usada frecuentemente es la **scaled dot-product**:

[  
f(k,q) = \frac{k q^\top}{\sqrt{d_k}},  
]

donde (d_k) es la dimensión de las keys. La PPT cita que el escalado mejora la estabilidad numérica cuando (d_k) crece.

---

## 18. 🧮 Self-Attention en forma matricial (fórmula final)

La operación en toda la fuente en paralelo se escribe como:

[  
\text{Attention}(Q,K,V) = \operatorname{Softmax}!\left(\frac{QK^\top}{\sqrt{d_k}}\right), V.  
]

(Esta es la forma matricial que aparece literalmente en la diapositiva.)

---

## 19. ♻️ Invarianza bajo permutaciones y positional encodings

- La PPT señala que el mecanismo de atención **es invariante** al orden de la secuencia (depende solo de similitudes entre tokens).
    
- Por eso es necesario incorporar **información posicional**; la PPT enumera tres soluciones propuestas:
    
    1. Positional encodings sinusoidales (fijos)
        
    2. Positional encodings aprendidos
        
    3. Representaciones posicionales relativas dentro del mecanismo de atención.
        

---

## 20. 📚 Bibliografía (tal como aparece)

- Ambartsoumian, A. & Popowich, F. (2018). _Self-attention: A better building block..._
    
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). _Neural machine translation by jointly learning to align and translate._
    
- Vaswani, A., et al. (2017). _Attention is all you need._
    

---

# ✅ Cierre — cumplimiento estricto

- Seguí **palabra por palabra** el orden de la PPT y **copié las ecuaciones tal como están** en las diapositivas.
    
- No añadí contenido que no esté en la PPT y no omití ninguna ecuación mostrada.
    
- Si querés, ahora te lo formateo **idéntico** a tus apuntes previos (títulos, emojis, tabla comparativa adicional) o te lo dejo como **Markdown descargable (.md)**. ¿Qué preferís?