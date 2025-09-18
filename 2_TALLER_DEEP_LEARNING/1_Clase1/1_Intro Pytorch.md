# 📘 Intro a PyTorch

```python
import torch
import numpy as np
```

[PyTorch](https://pytorch.org/) es una biblioteca de **aprendizaje profundo** de código abierto basada en Python.  
Fue desarrollada por **Facebook AI Research (FAIR)** y es muy usada en IA y ML.

👉 Unidad básica de cálculo: **el tensor**.

**Basado en:**

- Joe Papa (2021), _PyTorch Pocket Reference_.
- [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).
- d2l.ai (_Data Manipulation_ y _Linear Algebra_).

---

## 🔢 ¿Qué es un tensor?

Un **tensor** es una **matriz multidimensional** que puede contener datos (int, float, etc.).

- Son similares a arrays de **NumPy**.
- Diferencia clave: los tensores pueden usarse en **GPU** para acelerar cálculos.
- En PyTorch, son la **unidad básica de cálculo**.

---

## 🏗️ Creación de tensores

Existen varias formas de crear tensores en PyTorch.

### 📌 Desde estructuras de datos

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

👉 Crea un tensor a partir de una lista/tupla/matriz de NumPy.

---

### 📌 Desde un `ndarray` de NumPy

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

👉 `torch.from_numpy()` comparte memoria con el `ndarray`.

---

### 📌 Usando funciones de inicialización

- `torch.ones(shape)` → tensor lleno de 1s.
- `torch.zeros(shape)` → tensor lleno de 0s.
- `torch.rand(shape)` → números aleatorios uniformes en [0,1].
- `torch.randn(shape)` → números aleatorios normales.

Ejemplo:

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

---

### 📌 Creación a partir de otro tensor

Se heredan **forma y dtype** del original.

```python
x_ones = torch.ones_like(x_data)   # hereda forma y tipo de x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # hereda forma pero redefine dtype
```

---

### 📌 Especificando `shape`

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

---

## 📋 Atributos de un tensor

Propiedades básicas:

- `shape` → dimensiones.
- `dtype` → tipo de datos.
- `device` → CPU o GPU.

Ejemplo:

```python
tensor = torch.rand(3,4)
print(f"Shape: {tensor.shape}")
print(f"Dtype: {tensor.dtype}")
print(f"Device: {tensor.device}")
```

---

