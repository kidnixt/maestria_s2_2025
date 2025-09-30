# 📘 Apuntes - Deep Learning: Regularización

---

## 1) 🧠 Motivación

En redes neuronales profundas, es común que el modelo tenga **muchos parámetros** y pueda **ajustarse perfectamente** a los datos de entrenamiento → *overfitting*.  
👉 La **regularización** busca **restringir el espacio de soluciones** para favorecer modelos más simples y con mejor generalización.

---

## 2) 📝 Regularización sobre los pesos

Queremos evitar soluciones con pesos enormes.  
Sea:
- $W$ = tensor de pesos del modelo (o de una capa)
- $J_T(W)$ = costo empírico (por ej. entropía cruzada)
- $R(W)$ = término de regularización
- $\lambda \ge 0$ = hiperparámetro que controla la fuerza de la regularización

La función de costo regularizada es:
$$
J_{\text{reg}}(W) = J_T(W) + \lambda R(W)
$$

- $\lambda = 0$ → no hay regularización  
- $\lambda \to \infty$ → los pesos quedan fuertemente restringidos (prior muy fuerte)

👉 Intuición: *Soluciones con $R(W)$ grande son penalizadas*.

---

## 3) 🧮 Regularización como inferencia MAP

Otra forma de ver la regularización es como **inferencia bayesiana**.  
En lugar de hacer Máxima Verosimilitud (MLE), hacemos Máximo a Posteriori (MAP):

$$
\hat W 
= \arg\min_W \big\{-\ln [p(T \mid W) \, p(W)]\big\}
= \arg\min_W \{-\ln V_T(W) - \ln p(W)\}
$$

- $p(T|W)$ = verosimilitud de los datos  
- $p(W)$ = prior sobre los pesos

👉 Regularizar equivale a **imponer una distribución a priori $p(W)$** sobre los pesos.

---

## 4) 🟦 Ridge (ℓ2) = Prior Gaussiano

**Regularización Ridge / L2**:
$$
R(W) = \|W\|_2^2 = \sum_{i,j} W_{ij}^2
$$

**Prior gaussiano isotrópico** sobre los pesos:
$$
W \sim \mathcal{N}(0, \sigma^2 I)
\quad \Rightarrow \quad
p(W) = \text{cte} \cdot \exp\left(-\frac{1}{2\sigma^2}\|W\|_2^2\right)
$$

El estimador MAP:
$$
-\big[\ln V_T(W) + \ln p(W)\big]
= J_T(W) + \frac{1}{2\sigma^2}\|W\|_2^2 + \text{cte}
$$

Por lo tanto:
$$
\lambda = \frac{1}{2\sigma^2}
$$

👉 **Minimizar $J_T(W) + \lambda\|W\|_2^2$ es equivalente a MAP con prior Gaussiano**.

**Efecto geométrico:** Las soluciones se restringen a una *bola* centrada en el origen → tiende a achicar todos los pesos suavemente.

![[Pasted image 20250930164016.png]]

---

## 5) 🟨 LASSO (ℓ1) = Prior Laplaciano

**Regularización LASSO / L1**:
$$
R(W) = \|W\|_1 = \sum_{i,j} |W_{ij}|
$$

**Prior Laplaciano i.i.d.**:
$$
p(W) = \text{cte} \cdot \exp\left(-\frac{1}{b}\|W\|_1\right)
\quad \Rightarrow \quad
\ln p(W) = \text{cte} - \frac{1}{b}\|W\|_1
$$

El estimador MAP:
$$
-\big[\ln V_T(W) + \ln p(W)\big]
= J_T(W) + \frac{1}{b}\|W\|_1 + \text{cte}
$$

Entonces:
$$
\lambda = \frac{1}{b}
$$

👉 **LASSO induce esparsidad**: muchos pesos quedan exactamente en cero → útil para seleccionar características.

**Efecto geométrico:** La región factible es un *rombo* en 2D → las soluciones tienden a estar en vértices (esparsas).

![[Pasted image 20250930164112.png]]

---

## 6) 📐 Interpretación geométrica

La regularización se puede ver como:
$$
\min_W J_T(W) \quad \text{sujeto a } R(W) \le \mu
$$

- ℓ2 → región = esfera centrada en el origen.  
- ℓ1 → región = politopo (rombo en 2D).  

La solución regularizada cae dentro de estas regiones → controla la magnitud y estructura de $W$.

---

## 7) ⏸️ Early Stopping

**Idea:** detener el entrenamiento *antes* de que el modelo sobreajuste.  

Algoritmo típico:

1. Inicializar $W_\text{best}$ y $J_\text{best}$
2. Entrenar por epochs
3. Evaluar en validación
4. Si mejora → guardar pesos y resetear contador  
5. Si no mejora durante $p$ epochs → detener entrenamiento

**Efecto:** Early stopping actúa como una forma de regularización parecida a ℓ2, **pero sin modificar directamente los pesos**.  
Además, determina automáticamente *cuánta regularización aplicar*.

---

## 8) 🧢 Dropout

### ✨ Idea:
Durante entrenamiento, **apagamos aleatoriamente neuronas** con probabilidad $1 - r$.

Original:
$$
y = (\text{FC} \circ \text{FC})(x)
$$

Con Dropout:
$$
y = (\text{FC} \circ \text{Dropout} \circ \text{FC} \circ \text{Dropout})(x)
$$

### 📌 Definición:

Sea $X \sim (n,c)$ (mini-batch de activaciones).  
Muestreamos una **máscara binaria** $M \sim \text{Bernoulli}(p)$ con misma forma que $X$.

Salida:
$$
\text{Dropout}(X) = M \odot X
$$

- $r$ = probabilidad de *mantener* una neurona (hiperparámetro)
- No hay parámetros entrenables.

### 🧠 Entrenamiento vs Inferencia

- **Entrenamiento:** aplicamos máscaras aleatorias en cada batch.  
- **Inferencia:** no muestreamos → usamos el valor esperado:

$$
\mathbb{E}_M[\text{Dropout}(X)] = r X
$$

👉 En práctica, se multiplican los pesos por $r$ en inferencia para compensar.

---

## 9) 📊 Batch Normalization (BN)

### 🧮 Estadísticos del mini-batch

Sea $X \sim (n, c)$ (n=batch size, c=features).  
Para cada característica $j$:

$$
\mu_j = \frac{1}{n} \sum_i X_{ij}
\qquad
\sigma_j^2 = \frac{1}{n} \sum_i (X_{ij} - \mu_j)^2
$$

Normalización columna a columna:
$$
X'_j = \frac{X_j - \mu_j}{\sqrt{\sigma_j^2 + \varepsilon}}
$$

### 📌 Capa BN

$$
\text{BN}(X) = \alpha \cdot \frac{X - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
$$

- $\alpha, \beta \in \mathbb{R}^c$ → parámetros entrenables.
- Se suele aplicar entre la capa lineal y la activación:
$$
H = (\text{ReLU} \circ \text{BN} \circ \text{Linear})(X)
$$

👉 Ayuda a estabilizar y acelerar el entrenamiento.

### 🧠 Inferencia

Durante inferencia no usamos estadísticas del mini-batch → se emplean **promedios móviles** acumulados durante el entrenamiento.

---

## 10) 🧰 Otras técnicas de regularización

- **Aumento de datos (Data Augmentation):**  
  - Transformaciones de imágenes: flips, traslaciones, deformaciones, ruido, cambios de color, etc.  
  - Aumenta efectivamente el dataset → reduce overfitting.

- **Ejemplos adversarios:**  
  Se agregan pequeñas perturbaciones intencionales:
  $$
  x' = x + \epsilon \, \text{sign}(\nabla_x J(\theta, x, y))
  $$
  → mejora robustez y generalización.

- **Gradient Clipping:**  
  Se acotan los gradientes para evitar explosiones:
  $$
  \text{clip}(g) = 
  \begin{cases}
  g & \|g\| < \nu\\
  \frac{\nu}{\|g\|}g & \text{caso contrario}
  \end{cases}
  $$
  Evita updates demasiado grandes en zonas con gradientes enormes.

- **Ruido en coeficientes / Label smoothing:**  
  Se agrega ruido a los targets o a los pesos para suavizar decisiones del modelo.

---

# ✅ Conclusiones

- La regularización es **clave para generalizar** y evitar overfitting.  
- Puede entenderse desde:
  - 📊 Estadística (MAP con priors)  
  - 📐 Geometría (regiones factibles)  
  - 🧠 Entrenamiento (métodos prácticos como dropout / early stopping)
- **ℓ2 (Ridge)** → achica pesos suavemente  
- **ℓ1 (LASSO)** → induce esparsidad  
- **Early stopping**, **Dropout**, **BN**, **Data Augmentation**, **Gradient Clipping** → técnicas prácticas muy usadas en DL modernos.

