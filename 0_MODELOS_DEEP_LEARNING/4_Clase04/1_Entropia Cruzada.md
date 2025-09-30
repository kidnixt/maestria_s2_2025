# 📘 Apuntes - Deep Learning: Entropía Cruzada y Clasificación

---

## 1) 🧠 Clasificación binaria con redes neuronales

Tenemos un dataset $T = (X, y)$:
- $X = \{x_1, \dots, x_N\}$ con $x_i \in \mathbb{R}^d$
- $y = \{y_1, \dots, y_N\}$ con $y_i \in \{0,1\}$

Queremos entrenar una red que devuelva probabilidades:
$$
\hat p = f(X; W, B) = \Pr[y = 1 \mid X; W, B]
$$

La predicción final es:
$$
\hat y_i = 
\begin{cases}
1 & \text{si } \hat p_i \geq 0.5\\
0 & \text{si } \hat p_i < 0.5
\end{cases}
$$

👉 Esto equivale a usar **sigmoide** en la última capa para obtener probabilidades y luego **umbral 0.5** para clasificar.

---

## 2) 🧮 Binary Cross Entropy (BCE)

La función de pérdida usada es la **Entropía Cruzada Binaria**:
$$
\text{BCE}(\hat p, y) = -\Big[y \ln \hat p + (1 - y)\ln(1 - \hat p)\Big]
$$

- Si $y=1$ → penaliza $-\ln(\hat p)$ → queremos $\hat p$ cerca de 1.  
- Si $y=0$ → penaliza $-\ln(1-\hat p)$ → queremos $\hat p$ cerca de 0.

💡 **Motivación:** esta pérdida corresponde a **maximizar la verosimilitud** de los datos bajo un modelo Bernoulli.

![[Pasted image 20250930163256.png]]

---

## 3) 📈 Verosimilitud y Máxima Verosimilitud

La verosimilitud de los parámetros $(W,B)$ dado el dataset $T$ es:
$$
V_T(W,B) = \prod_{(x,y)\in T} \Pr[y \mid x, W, B]
$$

Queremos:
$$
\hat W, \hat B = \arg\max_{W,B} V_T(W,B)
$$

Separando por clases:
$$
V_T(W,B) 
= \prod_{(x,1)\in T} \Pr[1 \mid x, W, B]
\cdot \prod_{(x,0)\in T} \Pr[0 \mid x, W, B]
$$

---

## 4) 📉 Menos Log-Verosimilitud = BCE Total

Aplicando $-\ln$ a la verosimilitud:
$$
-\ln V_T(W,B)
= -\sum_{(x,1)\in T} \ln \hat p(x) 
\;-\sum_{(x,0)\in T} \ln(1 - \hat p(x))
$$

Usando $y\in\{0,1\}$:
$$
-\ln V_T(W,B)
= -\sum_{(x,y)\in T} 
\big[y\ln\hat p(x) + (1 - y)\ln(1 - \hat p(x))\big]
$$

👉 Esto es exactamente la **suma de BCE** en todo el dataset:
$$
-\ln V_T(W,B) = \sum_{(x,y)\in T} \text{BCE}(\hat p(x), y)
$$

---

## 5) 📝 Resumen binario

Minimizar la **entropía cruzada binaria** equivale a **maximizar la verosimilitud** del modelo bajo Bernoulli:
$$
\hat W, \hat B 
= \arg\min_{W,B} 
-\sum_{(x,y)\in T} \Big[y\ln \hat p(x) + (1 - y)\ln(1 - \hat p(x))\Big]
$$

O en promedio:
$$
\hat W, \hat B 
= \arg\min_{W,B} 
\mathbb{E}_T \big[\text{BCE}(\hat p(x), y)\big]
$$

💡 **Clave:** BCE no es una elección arbitraria → surge naturalmente de la estadística.

---

## 6) 🌈 Clasificación multiclase y Softmax

Sea $C = \{c_1, \dots, c_r\}$ el conjunto de clases.  
Cada clase se representa como **one-hot vector** $y_i \in \{0,1\}^r$:
$$
y_{i,k} =
\begin{cases}
1 & \text{si } i = k\\
0 & \text{otro caso}
\end{cases}
$$

La red produce **logits** $z \in \mathbb{R}^r$ y aplica **softmax**:
$$
\text{softmax}(z)_i 
= \frac{e^{z_i}}{\sum_{k=1}^r e^{z_k}}
$$

Propiedades:
- $\text{softmax}(z)_i \in (0,1)$
- $\sum_i \text{softmax}(z)_i = 1$

👉 Interpretable como **distribución de probabilidad** sobre clases.

---

## 7) 🧠 Modelo multiclase con softmax

Dado un dataset $T = (X, Y)$ con:
- $X \sim (N, d)$
- $Y \sim (N, r)$ (codificación one-hot)

La red devuelve:
$$
\hat P = \text{softmax}(Z) \in \mathbb{R}^{N\times r}
$$
donde cada $\hat P_{i,c} = \Pr[c \mid x_i; W,B]$.

Predicción final:
$$
\hat Y = \arg\max_c \hat P_{i,c}
$$

---

## 8) 🧮 Ejemplo (AND multiclase)

Ejemplo de tabla (X, W → logits → softmax → predicción):

$$
X =
\begin{pmatrix}
0 & 0 & 1\\
0 & 1 & 1\\
1 & 0 & 1\\
1 & 1 & 1
\end{pmatrix},
\quad
W =
\begin{pmatrix}
-1 & 1\\
-1 & 1\\
2 & -1
\end{pmatrix}
$$

$$
Z = XW =
\begin{pmatrix}
2 & -1\\
1 & 0\\
1 & 0\\
0 & 1
\end{pmatrix}
\quad
\Rightarrow\quad
\hat P = \text{softmax}(Z) =
\begin{pmatrix}
0.95 & 0.05\\
0.73 & 0.27\\
0.73 & 0.27\\
0.27 & 0.73
\end{pmatrix}
$$

Predicción $\hat Y$ = argmax por fila.

---

## 9) 📊 Categorical Cross Entropy (CCE)

Para una observación con $r$ clases, $y$ one-hot:
$$
\text{CCE}(\hat p, y) = -\sum_{k=1}^r y_k \ln \hat p_k
$$

Si la clase verdadera es $c$ (es decir $y_c=1$):
$$
\text{CCE}(\hat p, y) = -\ln \hat p_c
$$

👉 Minimizar CCE = **maximizar la probabilidad de la clase correcta** en softmax.

---

# ✅ Conclusiones

- **BCE** y **CCE** derivan directamente de la **máxima verosimilitud**.  
- BCE = caso binario (Bernoulli), CCE = caso multiclase (Categorical).  
- **Softmax** transforma logits en probabilidades válidas.  
- **Clasificar** = elegir la clase con mayor probabilidad (argmax).  
- Estas pérdidas son **estándar en DL** para tareas de clasificación.

