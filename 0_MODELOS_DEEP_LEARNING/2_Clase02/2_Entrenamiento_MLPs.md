# 📘 Apuntes - Deep Learning: Entrenamiento de un MLP (GD y SGD)

---
## 1) 🎯 Funciones de pérdida

Para medir qué tan bien predice la red usamos una **función de pérdida** $L(\hat y, y)$.

- **Regresión (MSE):**
$$
L(\hat y, y) = (\hat y - y)^2
$$
👉 Penaliza cuadráticamente el error. Derivada clave (se usará luego):
$$
\frac{\partial L}{\partial \hat y} = 2(\hat y - y)
$$

- **Clasificación binaria (BCE)** con **salida sigmoidea** $\hat y \in (0,1)$:
$$
L(\hat y, y) = -\Big[\, y \log \hat y + (1-y)\log(1-\hat y)\,\Big]
$$
👉 Maximiza la probabilidad de la clase correcta (log-likelihood negativa).

- **Clasificación multiclase (CCE)** con **softmax**:
$$
L(\hat y, y) = -\sum_{k=1}^{O} y_k \log \hat y_k,
\quad y \in \{0,1\}^O \text{ (one-hot)}
$$
👉 Empuja probabilidad hacia la clase verdadera.

---

## 2) 🧮 Costo empírico y ERM

Dada una pérdida por ejemplo $L_i = L(f(x_i;W,b), y_i)$, el **costo empírico** (promedio en el dataset) es:
$$
J(W,b) = \frac{1}{N}\sum_{i=1}^N L_i
$$

**Entrenar** = resolver ERM (minimización del riesgo empírico):
$$
(\hat W, \hat b) = \arg\min_{W,b} J(W,b)
$$

💡 **Por qué promedio:** estabiliza la escala del objetivo y hace comparable $J$ entre datasets/tamaños de batch.

---

## 3) ⬇️ Descenso por gradiente (GD)

Algoritmo (batch completo):
1. Inicializar $(W_0,b_0)$ aleatoriamente.
2. Repetir:
$$
(W_{k+1}, b_{k+1}) = (W_k, b_k) - \eta \,\nabla J(W_k, b_k)
$$
donde $\eta$ es el **learning rate**.

**Linealidad del gradiente** sobre el promedio:
$$
\nabla J(W,b) = \frac{1}{N}\sum_{i=1}^N \nabla L(f(x_i;W,b), y_i)
$$

💡 **Por qué funciona:** tomamos el paso en la dirección de **mayor descenso local**; la linealidad permite promediar gradientes por ejemplo.

---

## 4) 🧱 Ejemplo de MLP (configuración)

Arquitectura (1D → 2 hidden → 1 salida):
$$
\hat y = \underbrace{\text{softplus}(xW^{(1)} + b^{(1)})}_{a}\, W^{(2)} + b^{(2)}
$$

**Shapes:**
- $W^{(1)}\!\sim (1,2)$, $b^{(1)}\!\sim (1,2)$  
- $W^{(2)}\!\sim (2,1)$, $b^{(2)}\!\sim (0)$ (escalar)
- Pérdida: **MSE** $\;L=\frac{1}{N}\sum_i(\hat y_i-y_i)^2$

**Nombres intermedios (por claridad):**
- $z = xW^{(1)} + b^{(1)} \in \mathbb{R}^{1\times 2}$
- $a = \text{softplus}(z) \in \mathbb{R}^{1\times 2}$
- $\hat y = aW^{(2)} + b^{(2)} \in \mathbb{R}$

Recordatorio: $\dfrac{d}{dz}\text{softplus}(z) = \sigma(z)$ (sigmoidea).

---

## 5) 🔁 Cálculo de derivadas (paso a paso, un ejemplo)

Objetivo: $\nabla L$ para un par fijo $(x,y)$.

### 5.1) Salida y capa 2
- Derivada de la pérdida respecto a la salida:
$$
\frac{\partial L}{\partial \hat y} = 2(\hat y - y)
$$

- Sesgo de la capa 2:
$$
\frac{\partial \hat y}{\partial b^{(2)}} = 1
\;\Rightarrow\;
\frac{\partial L}{\partial b^{(2)}} = 2(\hat y - y)
$$

- Pesos de la capa 2:
$$
\frac{\partial L}{\partial W^{(2)}} 
= \frac{\partial L}{\partial \hat y}\,\frac{\partial \hat y}{\partial W^{(2)}}
= 2(\hat y - y)\, a^\top
$$
(da matriz $2\times 1$; componente a componente: $2(\hat y-y)a_j$).

- Activación intermedia:
$$
\frac{\partial L}{\partial a} 
= \frac{\partial L}{\partial \hat y}\,\frac{\partial \hat y}{\partial a}
= 2(\hat y - y)\,(W^{(2)})^\top
$$

### 5.2) No linealidad y capa 1
- Pre-activación $z$ (softplus):
$$
\frac{\partial L}{\partial z}
= \frac{\partial L}{\partial a}\;\odot\;\sigma(z)
= 2(\hat y - y)\,(W^{(2)})^\top \odot \sigma(z)
$$
(el $\odot$ es Hadamard / elemento a elemento).

- Sesgo de la capa 1:
$$
\frac{\partial L}{\partial b^{(1)}}
= \frac{\partial L}{\partial z} \cdot \underbrace{\frac{\partial z}{\partial b^{(1)}}}_{=\,\mathbf{1}}
= 2(\hat y - y)\,(W^{(2)})^\top \odot \sigma(z)
$$

- Pesos de la capa 1:
$$
\frac{\partial L}{\partial W^{(1)}}
= \frac{\partial L}{\partial z}\;\frac{\partial z}{\partial W^{(1)}} 
= \Big(2(\hat y - y)\,(W^{(2)})^\top \odot \sigma(z)\Big)\, x
$$
(ajustar la orientación para obtener shape $(1\times 2)$ apropiada: típicamente $x^\top$ multiplica al vector de arriba).

📌 **Por qué cada paso:** es **regla de la cadena** pura. En cada nodo del grafo computacional, derivamos respecto a sus entradas y multiplicamos por la derivada acumulada que llega “desde arriba”.

---

## 6) 🧾 Resumen (cálculo directo, por componente)

- $\,\displaystyle \frac{\partial L}{\partial b^{(2)}} = 2(\hat y - y)$  

- $\,\displaystyle \frac{\partial L}{\partial W^{(2)}} = 2(\hat y - y)\, a^\top$  

- $\,\displaystyle \frac{\partial L}{\partial a} = 2(\hat y - y)\,(W^{(2)})^\top$  

- $\,\displaystyle \frac{\partial L}{\partial z} = 2(\hat y - y)\,(W^{(2)})^\top \odot \sigma(z)$  

- $\,\displaystyle \frac{\partial L}{\partial b^{(1)}} = 2(\hat y - y)\,(W^{(2)})^\top \odot \sigma(z)$  

- $\,\displaystyle \frac{\partial L}{\partial W^{(1)}} = \Big(2(\hat y - y)\,(W^{(2)})^\top \odot \sigma(z)\Big)\, x$

💡 **Chequeo de shapes:**  
- $\partial L/\partial W^{(2)}$ debe ser $(2\times 1)$ → $a^\top$ es $(2\times 1)$.  
- $\partial L/\partial W^{(1)}$ debe ser $(1\times 2)$ → escalar $\cdot$ vector $(1\times 2)$ o $x^\top$ $(1\times 1)$ “inyecta” la entrada correcta.

---

## 7) 📦 Versión en batch (útil para implementar)

Para $N$ ejemplos, apilá:
- $X \in \mathbb{R}^{N\times 1}$ entradas
- $Z, A \in \mathbb{R}^{N\times 2}$ (por fila: $z_i^\top, a_i^\top$)
- $\hat y, y \in \mathbb{R}^{N\times 1}$

Entonces:
- $\,\displaystyle \frac{\partial J}{\partial \hat y} = \frac{2}{N}(\hat y - y)^\top \in \mathbb{R}^{1\times N}$

- $\,\displaystyle \frac{\partial J}{\partial b^{(2)}} = \frac{2}{N}(\hat y - y)^\top\,\mathbf{1}_N \in \mathbb{R}$

- $\,\displaystyle \frac{\partial J}{\partial W^{(2)}} = \frac{2}{N}(\hat y - y)^\top A \in \mathbb{R}^{1\times 2}$

  > ⚠️ ¡Ojo transponer al implementar! Equivalente a $A^\top(\hat y-y)/N$ con shapes convencionales.

- Definí el **término intermedio** (error en la capa 1):
$$
E \;=\; \Big[(\hat y - y)\,(W^{(2)})^\top\Big]\;\odot\;\sigma(Z)
\quad\in \mathbb{R}^{N\times 2}
$$

- $\,\displaystyle \frac{\partial J}{\partial b^{(1)}} = \frac{2}{N}\,\mathbf{1}_N^\top E \in \mathbb{R}^{1\times 2}$

- $\,\displaystyle \frac{\partial J}{\partial W^{(1)}} = \frac{2}{N}\, X^\top E \in \mathbb{R}^{1\times 2}$

💡 **Por qué funciona:** vectorizamos la regla de la cadena para todos los ejemplos simultáneamente → eficiencia en GPU.  
💡 **Truco práctico:** usar `keepdim=True` al reducir ejes para que el broadcasting posterior sea correcto.

---

## 8) 🔙 Backpropagation en forma “delta”

Definí **errores (deltas)**:
- $\,\displaystyle \delta_2 = \frac{\partial L}{\partial \hat y} = 2(\hat y - y)$

- $\,\displaystyle \delta_1 = \frac{\partial L}{\partial z}= \Big(2(\hat y - y)\,(W^{(2)})^\top\Big)\odot \sigma(z)$

Entonces:
- **Capa 2:** 
  - $\,\displaystyle \frac{\partial L}{\partial b^{(2)}} = \delta_2$

  - $\,\displaystyle \frac{\partial L}{\partial W^{(2)}} = \delta_2\, a^\top$

- **Capa 1:** 
  - $\,\displaystyle \frac{\partial L}{\partial b^{(1)}} = \delta_1$

  - $\,\displaystyle \frac{\partial L}{\partial W^{(1)}} = x^\top\,\delta_1$  *(ajustá transpuestas según convención)*

📝 *Nota:* si ves escrito $\partial L/\partial W^{(2)} = \delta_1 x$ en el bloque “En la capa 1”, es un **typo**: debe ser $\partial L/\partial W^{(1)}$.

---

## 9) 🧠 Backpropagation general para un MLP de $K$ capas

Definiciones por capa $k$:
- **Forward:**
  - $z_k = a_{k-1}^\top W_k + b_k^\top$
  - $a_k = A(z_k)$
  - $a_0 = x$

- **Backward (errores):**
  - $\delta_K = A'(z_K)\,\nabla_{a_K} L$
  - Para $k=K-1,\dots,1$:
    $$
    \delta_k = \big(W_{k+1}^\top \delta_{k+1}\big)\;\odot\; A'(z_k)
    $$

- **Gradientes:**
  $$
  \nabla_{W_k} L = a_{k-1}\,\delta_k^\top
  \quad\text{o}\quad
  \delta_k\,a_{k-1}^\top \;(\text{según convención de shapes}),
  \qquad
  \nabla_{b_k} L = \delta_k
  $$

- **Promedio (linealidad) y actualización:**
  $$
  \nabla J = \frac{1}{N}\sum_{i=1}^N \nabla L_i,
  \qquad
  W_k \leftarrow W_k - \eta\,\nabla_{W_k}J,\quad
  b_k \leftarrow b_k - \eta\,\nabla_{b_k}J
  $$

💡 **Por qué:** la regla de la cadena “propaga” el error desde la salida hacia las capas previas, reusando productos matriciales → complejidad lineal en #parámetros.

---

## 10) 🎲 SGD y mini-batches

**SGD básico (estocástico):**
- En cada *step* $t$, tomar un $(x,y)$ al azar y actualizar:
$$
(W_{t+1}, b_{t+1}) = (W_t, b_t) - \eta\,\nabla L\big(f(x;W_t,b_t),y\big)
$$

**Mini-batch SGD (práctico):**
- Partir el dataset en *mini-batches* de tamaño $n$.
- Para cada batch $B$:
  $$
  J_B = \frac{1}{n}\sum_{(x_i,y_i)\in B} L_i,
  \qquad
  (W,b)\leftarrow (W,b) - \eta\,\nabla J_B
  $$
- Reordenar (shuffle) en cada **epoch**.

![[Pasted image 20251018142308.png]]

💡 **Por qué mini-batch:**  
- Compromiso entre **ruido** (mejora generalización, “explora” mínimos) y **eficiencia** (aprovecha GPU).  
- Gradientes más **estables** que en puramente estocástico y más **baratos** que el batch completo.

---

# ✅ Conclusiones y consejos

- La **pérdida** define el objetivo; el **costo empírico** promedia sobre datos.  
- **GD/SGD** actualizan en la dirección del **descenso** del costo.  
- **Backprop** aplica la **regla de la cadena** eficientemente en MLPs.  
- **Mini-batches** equilibran costo computacional y estabilidad.  
- **Chequeá shapes y transpuestas** (fuente #1 de bugs): confirmá dimensiones en cada multiplicación y reducción.
