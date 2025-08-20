# 📘 Apuntes - Deep Learning: Tensores

  
## 1. 🎯 Objeto de estudio

Las redes neuronales pueden verse como **funciones** con estas características:

- 🟦 **Input y output** representados mediante **tensores**.
- 🟩 **Parámetros** también son tensores.
- 🟨 Son **componibles** (podemos apilar funciones en capas).
- 🟪 Son **diferenciables**, lo que permite el cálculo de gradientes.
- 🔧 Se pueden **optimizar numéricamente end-to-end**.

---
## 2. 🔢 Tensores

Tipos de datos fundamentales:
- **Escalar (0D)**: número real.
- **Vector (1D)**: lista de valores.
- **Matriz (2D)**: tabla con filas y columnas.
- **Array n-dimensional (nD)**: generalización.

👉 A estos objetos los llamamos **tensores**.  

El número de dimensiones (n) se llama **orden del tensor**.

---
## 3. 📐 Índices y Shape

- Un tensor `X` es un arreglo n-dimensional de elementos del mismo tipo.
- El **shape** (forma) se denota:

$$X \sim (s_1, s_2, \dots, s_n)$$
Ejemplo:  
- Si `n = 3`, entonces `X ∼ (h, w, c)` → tensor 3D con alto, ancho y canales.  
- Elemento específico:  

$$X_{i,j,k} \quad \text{o bien} \quad [X]_{i,j,k}$$

- **Slicing**:  
$$X[:, i, :] \sim (h, c)$$
---
## 4. ⚪ Escalares y Vectores

- **Escalar (0D):** un número real. Se puede operar con `+, -, ·, sin, cos, sqrt(x), exp, |·|, ...`
- **Vector (1D):** columna de valores

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
---

## 5. 🖥️ Notación matemática vs código

⚠️ Diferencia importante:
- En matemática, por defecto, los vectores son **columna**.
- En código, puede ser **fila o columna**:
  - `(d,)` → vector 1D (tensor unidimensional).
  - `(d,1)` o `(1,d)` → tensor 2D con 1 columna o 1 fila.

Esto afecta al **broadcasting**.

Ejemplo en PyTorch:

```python
import torch
x = torch.randn((4,1))  # Vector columna, shape (4,1)
y = torch.randn((4,))   # Vector 1D, shape (4,)
print((x+y).shape)      # torch.Size([4,4])  <-- broadcasting
```


---

  

## 6. ➕ Operaciones con vectores

- **Combinación lineal:**
$$
z = a x + b y \quad \Rightarrow \quad [z]_i = a x_i + b y_i
$$

- **Norma euclídea (ℓ2):**
$$
\|x\|_2 = \sqrt{\sum_i x_i^2}
$$
- **Producto interno:**  
$$
\langle x, y \rangle = \sum_i x_i y_i = x^\top y
$$
---
## 7. 📏 Producto interno y similaridad coseno

- Relación con ángulo θ:
$$\cos(\theta) = \frac{\langle x, y \rangle}{\|x\|\|y\|} $$
- Si ⟨x, y⟩ = 0 → vectores ortogonales.
- Valores de coseno: entre -1 (opuestos) y +1 (alineados).


- **Distancia euclídea en términos de productos internos:**
$$
\|x-y\|_2^2 = \langle x, x \rangle + \langle y, y \rangle - 2\langle x, y \rangle

$$
---

## 8. 🟦 Matrices

- Tensores de dimensión 2.

$$

X =

\begin{bmatrix}

X_{1,1} & \dots & X_{1,n} \\

\vdots  & \ddots & \vdots \\

X_{m,1} & \dots & X_{m,n}

\end{bmatrix}

$$

- Interpretaciones:
  - **Filas:** cada fila = un vector fila $(x_i^\top)$.
  - **Columnas:** cada columna = un vector columna $(c_j)$.

---
## 9. 🔄 Transformación lineal y multiplicación

- Una matriz representa una aplicación lineal:

$$

x \mapsto y = W x, \quad W \sim (m,n), \quad x \sim (n), \quad y \sim (m)

$$

- **Multiplicación de matrices**:
$$
(XY)_{ij} = \langle X_{i,:}, Y_{:,j} \rangle = \sum_{z=1}^b X_{iz} Y_{zj}

$$

- Interpretación: **composición de funciones**.

---

  

## 10. 📊 Batch de operaciones

Ejemplo:
$$

XW =

\begin{bmatrix}

X_{1,:}W \\

\vdots \\

X_{m,:}W

\end{bmatrix}
$$


- Producto interno entre todas las filas:
$$
XX^\top = [\langle X_{i,:}, X_{j,:} \rangle]_{i,j}
$$
---
## 11. 📦 Batch Matrix Multiplication (BMM)

Operaciones de mayor dimensión:  

- Si $(X \sim (n, a, b)$ e $(Y \sim (n, b, c)$:
$$
[BMM(X,Y)]_i = X_{i,:,:} Y_{i,:,:} \quad \sim (n, a, c)
$$
Ejemplo en PyTorch:

```python
X = torch.randn((4,5,2)) # batch de 4 matrices 5x2
Y = torch.randn((4,2,3)) # batch de 4 matrices 2x3
Z = torch.matmul(X,Y)    # o X @ Y
print(Z.shape)           # torch.Size([4,5,3])
```

---
## 12. ✖️ Otras operaciones

- **Producto de Hadamard (elemento a elemento):**
$$
(X \odot Y)_{ij} = X_{ij} Y_{ij}
$$
- **Broadcasting:** operaciones con tensores de distinto shape.

Ejemplo:
$$
Y_{(n,m)} = X_{(n,m)} + a_{(m)}
$$
---
## 13. 🌐 Broadcasting

⚡ El broadcasting permite “expandir” tensores para operar sin copiarlos explícitamente.  

Ejemplo: sumar un vector a cada fila de una matriz.

---
## 14. 📉 Operaciones de reducción

- Reducción a lo largo de ejes:
$$
H_{(b,c)} = \sum_i X_{i,:,:}, \quad X \sim (a,b,c)
$$

- Producto interno generalizado:

  

\[

y = \sum_{i,j,k} [X_1 \odot X_2]_{i,j,k}

\]

  

- Ejemplo en vectores:

  

\[

\sum_i x_i = \langle x, \mathbf{1} \rangle

\]

  

---

  

# ✅ Conclusión

- Todo en Deep Learning se formula con **tensores**.

- Operaciones clave: **producto interno, normas, multiplicaciones matriciales, BMM, broadcasting, reducciones**.

- Estas operaciones son la base para redes neuronales y cómputo eficiente en GPUs.