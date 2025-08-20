# 📘 Apuntes - Deep Learning: Derivadas

## 1. 🔎 Repaso: definición de derivada

Para una función $y = f(x)$ con entrada y salida **escalar**, su derivada es:

$$
f'(x) = \lim_{\Delta x \to 0} \frac{\Delta f}{\Delta x}
$$

![[Pasted image 20250820164033.png]]

Otra notación: $\partial f(x)$.

* Si $\partial f(x) > 0$: la función **crece** en ese punto.
* Si $\partial f(x) < 0$: la función **decrece** en ese punto.

**💡 Intuición:** la derivada mide la **pendiente de la recta tangente** a la curva en un punto.

---

## 2. 🔬 Haciendo zoom en la derivada

Las diapositivas muestran zoom progresivo en la función $\cos(x)$.

👉 A medida que nos acercamos a un punto $x_0$, la curva se parece cada vez más a una **recta**. Esa recta es la **aproximación lineal** de $f$ cerca de $x_0$.

---

## 3. 📏 Aproximación lineal (Taylor de primer orden)

Para $x \approx x_0$:

$$
f(x) \approx f(x_0) + \partial f(x_0) \cdot (x - x_0)
$$

* Se reemplaza la función por su **tangente** en un punto.
* Útil en optimización: es la base del **descenso por gradiente**.

![[Pasted image 20250820164100.png]]

**Ejemplo:** si $f(x) = \sin(x)$, en $x_0=0$:

* $f(0) = 0$,
* $f'(0) = \cos(0) = 1$.
* Aproximación: $f(x) \approx x$.

---

## 4. 🛠️ Propiedades básicas de derivadas

* **Linealidad**

  $$
  \partial \big[a f(x) + b g(x)\big] = a \partial f(x) + b \partial g(x)
  $$

* **Regla del producto**

  $$
  \partial \big[f(x) g(x)\big] = \partial f(x)\, g(x) + f(x)\, \partial g(x)
  $$

* **Regla de la cadena**

  $$
  \partial \big[f(g(x))\big] = \partial f(g(x)) \cdot \partial g(x)
  $$

**💡 En DL:** la **regla de la cadena** es esencial → es la base de **backpropagation**.

---

## 5. 🧩 Derivadas parciales y gradiente

Para $y = f(x)$, donde $x \sim (d)$ es un vector de dimensión $d$:

* Derivada parcial respecto a $x_i$:

  $$
  \partial_i f(x) = \lim_{\Delta x_i \to 0} \frac{\Delta f}{\Delta x_i}
  $$

  (manteniendo fijos los otros $x_j, j\neq i$).

* El **gradiente** es el vector de todas las derivadas parciales:

  $$
  \nabla f(x) =
  \begin{bmatrix}
  \partial_1 f(x) \\
  \vdots \\
  \partial_d f(x)
  \end{bmatrix}
  $$

**💡 Intuición:** el gradiente apunta en la dirección de **máximo crecimiento** de la función.

---

## 6. 🌍 Curvas de nivel

Ejemplo:

$$
f(x,y) = 2x^2 + 2xy + y^2
$$
![[Pasted image 20250820164131.png]]

* Las curvas de nivel muestran dónde la función toma el mismo valor.
* El **gradiente** en un punto es siempre **perpendicular** a la curva de nivel.

Esto explica por qué en optimización nos movemos en la dirección contraria al gradiente (**descenso por gradiente**).

---

## 7. 📏 Aproximación lineal en varias variables

Para $x \approx x_0$:

$$
f(x) \approx f(x_0) + \langle \nabla f(x_0), x - x_0 \rangle
$$

**💡 Significado:** reemplazamos la función en un entorno de $x_0$ por el **plano tangente**.

![[Pasted image 20250820164201.png]]

Ejemplo: en imágenes, esto se usa para entender cómo cambia una función de pérdida cuando modificamos ligeramente los **pesos** de la red.

---

## 8. 🧮 El Jacobiano

Sea $y = f(x)$, con $x \sim (d)$ y salida $y \sim (k)$.

El **Jacobiano** es la matriz de derivadas parciales:

$$
\partial f(x) =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_k}{\partial x_1} & \cdots & \frac{\partial y_k}{\partial x_d}
\end{bmatrix}
\sim (k,d)
$$

* Cada fila es el gradiente de una salida $f_i$.
* Si $f: \mathbb{R}^d \to \mathbb{R}^k$, entonces el Jacobiano es de shape $(k,d)$.

**Regla de la cadena (multivariable):**

$$
\partial \big[f(g(x))\big] = \partial f(g(x)) \cdot \partial g(x)
$$

---

## 9. 📝 Ejemplo: función lineal

Sea $y = Wx$:

* Vista como función de $x$:

  $$
  \partial_x [Wx] = W
  $$

* Vista como función de $W$:

  $$
  y_i = \sum_j W_{ij} x_j
  $$

  Entonces:

  $$
  \frac{\partial y_i}{\partial W_{r\ell}} =
  \begin{cases}
  x_\ell & \text{si } r=i \\
  0 & \text{si } r \neq i
  \end{cases}
  $$

  Lo que da como Jacobiano:

  $$
  \partial_W [Wx] =
  \begin{bmatrix}
  x^\top & 0 & \cdots & 0 \\
  0 & x^\top & \cdots & 0 \\
  \vdots & & \ddots & \vdots \\
  0 & \cdots & 0 & x^\top
  \end{bmatrix}
  $$

**💡 En DL:** esta es la base de cómo calculamos gradientes de capas lineales respecto a sus **pesos**.

---

# ✅ Conclusión

* La **derivada** mide cómo cambia una función respecto a sus variables.
* En varias dimensiones, se generaliza al **gradiente** y al **Jacobiano**.
* La **aproximación lineal** (tangente o plano tangente) es la base de los métodos de optimización.
* La **regla de la cadena** conecta funciones compuestas → es el corazón del **backpropagation** en redes neuronales.

---

