	# 📘 Apuntes - Deep Learning: Tensores

## 1. 🎯 Objeto de estudio

Las redes neuronales pueden verse como **funciones** con estas características:

* 🟦 **Input y output** representados mediante **tensores** (arreglos n-dimensionales).
* 🟩 **Parámetros** (pesos y sesgos) también son tensores.
* 🟨 Son **componibles**: capas = composiciones de funciones.
* 🟪 Son **diferenciables**: permiten backpropagation.
* 🔧 Se pueden **optimizar end-to-end** con descenso por gradiente.

**💡 Intuición:** todo lo que “entra”, “sale” y “se aprende” en una red se codifica como tensores. Optimizar “end-to-end” = ajustar todos los parámetros a la vez para minimizar una pérdida.

---

## 2. 🔢 Tensores

**Tipos de datos fundamentales**

* **Escalar (0D)**: un número real.
* **Vector (1D)**: lista ordenada de números.
* **Matriz (2D)**: tabla (filas × columnas).
* **Tensor nD**: generalización a más ejes/dimensiones.

👉 A estos objetos los llamamos **tensores**.
El número de dimensiones (n) se llama **orden del tensor**.

**📌 Ejemplos de shapes en DL**

* Imagen RGB: `(H, W, C)` o `(C, H, W)`.
* Batch de imágenes: `(N, H, W, C)` o `(N, C, H, W)`.
* Secuencias (NLP): `(T, d)` o `(N, T, d)`.

---

## 3. 📐 Índices y Shape

* Un tensor `X` es un arreglo n-dimensional de elementos del mismo tipo.
* El **shape** (forma) se denota:

  $X \sim (s_1, s_2, \dots, s_n)$

**Ejemplo (n = 3):**

* `X ∼ (h, w, c)` → tensor 3D con alto, ancho y canales.

* Elemento específico:

  $X_{i,j,k} \quad \text{o bien} \quad [X]_{i,j,k}$

* **Slicing** (rebanado):

  $X[:, i, :] \sim (h, c)$

**💡 Nota práctica (convención de ejes):**

* Visión: muchos frameworks usan **channels-first** `(N, C, H, W)` (p.ej., PyTorch).
* Otros usan **channels-last** `(N, H, W, C)` (p.ej., TensorFlow por defecto).
  Verificá la convención para evitar errores de shape.

---

## 4. ⚪ Escalares y Vectores

* **Escalar (0D):** admite operaciones como `+, -, ·, sin, cos, sqrt(x), exp, |·|, …`
* **Vector (1D):** columna de valores

$$
x =
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_m
\end{pmatrix},
\quad
x^\top = (x_1, x_2, \dots, x_m)
$$

**💡 Intuición:** un vector representa una dirección y magnitud en un espacio. En DL, un **embedding** es un vector que codifica información (palabras, imágenes, usuarios, etc.).

---

## 5. 🖥️ Notación matemática vs código

⚠️ Diferencia importante:

* En **matemática**, los vectores son **columna** por defecto.
* En **código**, puede ser **fila o columna**:

  * `(d,)` → tensor 1D (vector “plano”).
  * `(d,1)` o `(1,d)` → tensor 2D con 1 columna o 1 fila.

Esto afecta al **broadcasting** (cómo se alinean shapes automáticamente).

**Ejemplo en PyTorch:**

```python
import torch
x = torch.randn((4,1))  # Vector columna, shape (4,1)
y = torch.randn((4,))   # Vector 1D, shape (4,)
print((x+y).shape)      # torch.Size([4,4])  <-- broadcasting "expande" y
```

**Errores comunes:**

* Sumar `(d,)` con `(d,1)` produce una **matriz (d,d)** por broadcasting.
* Si querés sumar componente a componente, hacé `y[:, None]` o `x.squeeze(1)` para alinear shapes.

---

## 6. ➕ Operaciones con vectores

* **Combinación lineal:**

  $$
  z = a x + b y \quad \Rightarrow \quad [z]_i = a x_i + b y_i
  $$

* **Norma euclídea (ℓ2):**

  $$
  \|x\|_2 = \sqrt{\sum_i x_i^2}
  $$

* **Producto interno (punto/escalar):**

  $$
  \langle x, y \rangle = \sum_i x_i y_i = x^\top y
  $$

**💡 Intuición:**

* La norma mide “tamaño” del vector.
* El producto interno mide **alineación** entre dos vectores: positivo si apuntan parecido, negativo si opuesto.

---

## 7. 📏 Producto interno y similaridad coseno

* Relación con el ángulo $\theta$:

  $\cos(\theta) = \frac{\langle x, y \rangle}{\|x\|\,\|y\|}$

* Si $\langle x, y \rangle = 0$ → vectores **ortogonales**.

* Rango de **cosine similarity**: $[-1, 1]$.

* **Distancia euclídea via productos internos:**

  $$
  \|x-y\|_2^2 = \langle x, x \rangle + \langle y, y \rangle - 2\langle x, y \rangle
  $$

**💡 En DL:** cosine similarity se usa para comparar embeddings (búsqueda semántica, retrieval, métricas de recomendación).

---

## 8. 🟦 Matrices

* Tensores de dimensión 2:

  $$
  X =
  \begin{bmatrix}
  X_{1,1} & \dots & X_{1,n} \\
  \vdots  & \ddots & \vdots \\
  X_{m,1} & \dots & X_{m,n}
  \end{bmatrix}
  $$

* Interpretaciones:

  * **Filas:** cada fila = un vector fila $x_i^\top$.
  * **Columnas:** cada columna = un vector columna $c_j$.

**💡 Intuición:** una matriz puede verse como una **colección de vectores** (por filas o por columnas) o como un **operador lineal** que transforma vectores.

---

## 9. 🔄 Transformación lineal y multiplicación

* Una matriz representa una **aplicación lineal**:

  $$
  x \mapsto y = W x, \quad W \sim (m,n), \quad x \sim (n), \quad y \sim (m)
  $$

* **Multiplicación de matrices** (si $X \sim (a,b)$ e $Y \sim (b,c)$):

  $$
  (XY)_{ij} = \langle X_{i,:}, Y_{:,j} \rangle = \sum_{z=1}^b X_{iz} Y_{zj}
  $$

* Interpretación: **composición de funciones** (aplicar $Y$ y luego $X$ equivale a $XY$).

**📌 Nota PyTorch (densas totalmente conectadas):**
`nn.Linear(in, out)` recibe un `x` de shape `(N, in)` y produce `(N, out)`. Internamente aplica $y = x W^\top + b$, donde $W$ se guarda con shape `(out, in)`.

---

## 10. 📊 Batch de operaciones

**Ejemplo (multiplicar todas las filas por la misma matriz $W$):**

$$
XW =
\begin{bmatrix}
X_{1,:}W \\
\vdots \\
X_{m,:}W
\end{bmatrix}
$$

* Producto interno entre todas las filas:

  $$
  XX^\top = \big[\langle X_{i,:}, X_{j,:} \rangle\big]_{i,j}
  $$

**💡 En la práctica:** organizar datos en **batches** permite paralelismo en GPU. Operaciones matriciales vectorizan cálculos repetidos.

---

## 11. 📦 Batch Matrix Multiplication (BMM)

Operaciones en dimensiones mayores son variantes “batched”:

* Si $X \sim (n, a, b)$ e $Y \sim (n, b, c)$:

  $$
  [BMM(X,Y)]_i = X_{i,:,:}\, Y_{i,:,:} \quad \sim (n, a, c)
  $$

**Ejemplo en PyTorch:**

```python
import torch
X = torch.randn((4,5,2)) # batch de 4 matrices 5x2
Y = torch.randn((4,2,3)) # batch de 4 matrices 2x3
Z = torch.matmul(X, Y)   # también: torch.bmm(X, Y) si son 3D
print(Z.shape)           # torch.Size([4,5,3])
```

**🔎 matmul vs bmm**

* `torch.matmul` generaliza a más dims y usa broadcasting cuando aplica.
* `torch.bmm` es específicamente para **batch de matrices 3D** `(N, a, b) @ (N, b, c)`.

---

## 12. ✖️ Otras operaciones

* **Producto de Hadamard (elemento a elemento):**

  $$
  (X \odot Y)_{ij} = X_{ij}\, Y_{ij}
  $$

* **Broadcasting:** operar tensores de distinto shape “expandiendo” dims de tamaño 1.

  **Reglas de broadcasting (resumen):**

  1. Alinear desde la **derecha** (últimas dimensiones).
  2. Dos dims son compatibles si son **iguales** o alguna es **1**.
  3. Si una dim “falta”, se asume como **1**.
  4. El resultado tiene el **máximo** a lo largo de cada eje.

**Ejemplo:**

$$
Y_{(n,m)} = X_{(n,m)} + a_{(m)} \quad \Rightarrow \quad Y_{i,:} = X_{i,:} + a
$$

**⚠️ Pitfall:** cuidado al sumar bias con shape incorrecto; confirmá que el eje a “expandir” sea el esperado (p.ej., usar `unsqueeze`).

---

## 13. 🌐 Broadcasting

⚡ El broadcasting permite “expandir” tensores para operar sin copiarlos explícitamente.
**Ejemplo típico:** sumar un vector a **cada fila** de una matriz (o a cada canal de una imagen).

**En PyTorch (tips):**

* `x.unsqueeze(dim)` agrega una dimensión de tamaño 1 para alinear.
* `x.squeeze(dim)` elimina dimensiones de tamaño 1 cuando estorban.
* `x.view`/`x.reshape` reacomodan sin copiar (cuando se puede).

![[Pasted image 20251018141612.png]]

---

## 14. 📉 Operaciones de reducción

* **Reducción** a lo largo de ejes (sum, mean, max, min, prod, …). Ejemplo:

  $$
  H_{(b,c)} = \sum_i X_{i,:,:}, \quad X \sim (a,b,c)
  $$

* **Producto interno generalizado:**

  $$
  y = \sum_{i,j,k} \big[X_1 \odot X_2\big]_{i,j,k}
  $$

* **Vectores como caso particular:**

  $$
  \sum_i x_i = \langle x, \mathbf{1} \rangle
  $$

**En PyTorch (útil para prácticas):**

```python
# Suma por ejes
X.sum(dim=0)      # reduce eje 0
X.sum(dim=(0,2))  # reduce ejes 0 y 2
X.mean(dim=-1)    # promedio en último eje
X.sum(dim=1, keepdim=True)  # mantiene dimensión para broadcasting posterior

# Norma L2 de un batch de vectores (N, d)
x = torch.randn(32, 128)
l2 = torch.sqrt((x**2).sum(dim=1))  # shape: (32,)
```

**🔧 einsum (opcional pero poderoso):**

```python
# Producto interno generalizado: suma de X1*X2 en todas las dims
y = torch.einsum('ijk,ijk->', X1, X2)

# Matmul clásico con einsum
Y = torch.einsum('ab,bc->ac', A, B)
```

---

# ✅ Conclusión

* En Deep Learning, **todo** se expresa con **tensores**: datos, parámetros, activaciones.
* Operaciones clave: **producto interno**, **normas**, **multiplicación matricial**, **BMM**, **broadcasting** y **reducciones**.
* Dominar **shapes e índices** evita bugs y habilita implementaciones eficientes en GPU/TPU.
* **Regla de oro:** antes de operar, **verificá shapes** y **ejes** (¡te ahorra horas de debug! 😅)
