# 📘 Apuntes - Deep Learning: Convoluciones (Parte 2)

---

## 1. 🧠 Capa Convolucional 2D: idea general

- Cada **ventana local** $(s \times s \times c)$ se multiplica por un **kernel** (filtro) para producir una salida.
- El **mismo kernel se aplica en todas las posiciones** de la imagen → equivarianza traslacional.
- Este mecanismo generaliza muchos kernels clásicos del procesamiento de imágenes (e.g. Sobel, Gaussiano, etc.).

---

## 2. 🧱 Definición formal de la capa convolucional 2D

- Entrada:  
  $$ X \sim (h, w, c) $$
- Kernel size:  
  $$ s = 2k + 1 $$
- Número de canales de salida: $c'$

Para cada posición $(i,j)$:

$$
H_{ij} = \text{vect}(P_k(i,j))^\top W + b^\top
$$

- $W \sim (s^2 c, c')$ → pesos compartidos.  
- $b \sim (c')$ → bias por filtro.  

Hiperparámetros:
- $k$ (o $s$)
- $c'$ (número de filtros)
- uso de zero-padding.

Shape de salida:

$$
H \sim
\begin{cases}
(h, w, c') & \text{con padding de } k \\
(h - 2k, w - 2k, c') & \text{sin padding}
\end{cases}
$$

---

## 3. 🧍 Padding en convoluciones

**Propósito:**  
Evitar que las dimensiones de la imagen se reduzcan tras aplicar convoluciones repetidas.

- Se agregan **filas y columnas de ceros** alrededor de la entrada.
- Permite conservar el **tamaño espacial** original $(h,w)$ tras la convolución.
- Es especialmente útil cuando se desea **apilar múltiples capas convolucionales** sin alterar la resolución.

---

## 4. 🧮 Convolución con un kernel: ejemplo visual

Una ventana $(s \times s)$ se desplaza sobre la imagen.  
En cada posición, se realiza:

1. Producto elemento a elemento entre patch y kernel.  
2. Suma de todos los productos.  
3. Se agrega bias → salida para ese píxel.

👉 Este proceso se repite para **todas las posiciones** y **todos los filtros**.

---

## 5. 📊 Múltiples canales

Cuando la imagen tiene múltiples canales ($c > 1$):
- Cada filtro tiene **profundidad igual a $c$**.
- Se realiza la convolución en cada canal y luego se **suman los resultados**.
- Cada filtro produce **un solo mapa de activación** → $c'$ filtros → $c'$ canales de salida.

---

## 6. 🧭 Stride en convoluciones

**Stride** = paso con el que se desliza la ventana.

- Stride = 1 → se recorren todas las posiciones → salida más grande.  
- Stride > 1 → se **subsamplea** la salida (downsampling), reduciendo ancho y alto.

📌 Stride controla la **resolución espacial** de las features.  
Stride altos → menos superposición entre patches.

---

## 7. 🏊 Max / Avg Pooling

Pooling = operación **fija, no aprendida**, que resume información local.

- **Max pooling:** toma el valor máximo en la ventana.  
- **Avg pooling:** toma el promedio.

Hiperparámetros típicos:
- Tamaño: $2 \times 2$
- Stride: $2$

Efectos:
- **Downsampling** (reduce resolución espacial).  
- **Resalta patrones relevantes**.  
- **Aumenta el campo receptivo** de capas posteriores.

---

## 8. 📝 Resumen: Shape de una capa convolucional

Entrada: $(W_1, H_1, C_1)$

Hiperparámetros:
- $K$: número de filtros
- $F$: tamaño espacial (kernel size)
- $S$: stride
- $P$: padding

Salida: $(W_2, H_2, C_2)$ con:

$$
W_2 = \frac{W_1 - F + 2P}{S} + 1
\quad , \quad
H_2 = \frac{H_1 - F + 2P}{S} + 1
\quad , \quad
C_2 = K
$$

Número de parámetros:

$$
\text{params} = (F \cdot F \cdot C_1) \cdot K + K
$$

👉 Cada canal de salida corresponde a un filtro aplicado con stride $S$ y desplazado por un bias.

---

## 9. 🧠 Bloques convolucionales

En la práctica no se usan capas aisladas, sino **bloques** que combinan varias convoluciones y operaciones de pooling:

$$
\text{ConvBlock}(X) = (\text{MaxPool} \circ A \circ \text{Conv} \circ \cdots \circ A \circ \text{Conv})(X)
$$

Y redes completas se construyen apilando bloques:

$$
H = (\text{ConvBlock} \circ \text{ConvBlock} \circ \cdots \circ \text{ConvBlock})(X)
$$

👉 En arquitecturas como **VGG**, se:
- Mantiene **constante** el tamaño de filtro dentro de cada bloque.  
- Mantiene **constante** el número de canales dentro del bloque.  
- Se **duplican** los canales entre bloques consecutivos.

---

## 10. 🏗️ Arquitectura típica de una CNN

Una red convolucional puede dividirse en **3 componentes** principales:

1. **Extracción de features** (backbone):

   $$
   H = (\text{ConvBlock} \circ \cdots \circ \text{ConvBlock})(X)
   $$

2. **Global Average Pooling** (opcional):

   $$
   h = \frac{1}{h' w'} \sum_{i,j} H_{ij}
   $$

   → resume espacialmente los mapas de activación en un vector.

3. **Clasificador final (MLP):**

   $$
   y = \text{MLP}(h)
   $$

   o bien flatten → fully connected layers.

---

## 11. 🧱 Ejemplo de arquitectura CNN

**Input:** $(64, 64, 3)$

1. Convolution (32 filtros) → **shape:** $(64, 64, 32)$  
2. Max-pooling (2×2) → **shape:** $(32, 32, 32)$  
3. Convolution (64 filtros) → **shape:** $(32, 32, 64)$  
4. Max-pooling (2×2) → **shape:** $(16, 16, 64)$  
5. Global pooling → **shape:** $(64)$  
6. Fully-connected layer (10 unidades) → **shape:** $(10)$

- Las primeras capas aprenden **features locales**.
- Las últimas capas capturan **patrones globales** para clasificación.

---

## 12. 🧪 Arquitecturas históricas

### 🟦 LeNet-5 (Y. LeCun, 1998)
- Primera red convolucional exitosa.
- Aplicada a MNIST → superó el SOTA de la época.
- Patrón típico:
  - $(H, W)$ ↓ después de conv/pool
  - Canales $C$ ↑
  - Al final: Fully Connected + Softmax
- Conv 5×5, stride 1 + avg pooling stride 2
- ∼ 60k parámetros

---

### 🟩 AlexNet (2012)
- Red mucho más profunda y ancha → entrenada en GPU.
- Uso intensivo de ReLU, dropout y normalización.
- Ganó ImageNet 2012.

---

### 🟧 VGG (2014)
- Diseño modular con bloques convolucionales homogéneos.
- Filtros 3×3, stride 1, padding 1 → forma sencilla pero profunda.
- Duplica canales después de cada bloque → red muy grande pero estructurada.

---

# ✅ Conclusiones

- **Padding**, **stride** y **pooling** controlan la **resolución espacial** y **campo receptivo**.  
- El **número de parámetros** depende del kernel, canales y número de filtros, **no** de la resolución de la imagen.  
- Los **bloques convolucionales** permiten construir arquitecturas profundas de manera eficiente.  
- CNN típicas:
  - $(H, W)$ ↓ progresivamente  
  - Canales ↑  
  - Al final: global pooling o flatten → clasificador denso.
- Arquitecturas históricas (LeNet, AlexNet, VGG) establecieron patrones que aún se usan hoy.

---
