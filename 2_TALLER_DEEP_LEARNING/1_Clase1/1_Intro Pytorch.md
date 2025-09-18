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

# ⚙️ Operaciones con tensores

Los tensores soportan muchas operaciones: indexación, slicing, algebra lineal, etc.

---
## 📌 Indexación y slicing

Similar a NumPy:

```python
tensor = torch.ones(4,4)
print("Primera fila:", tensor[0])        # toda la primera fila
print("Primera columna:", tensor[:,0])   # toda la primera columna
print("Última columna:", tensor[:,-1])   # toda la última columna
```

👉 Muy útil para acceder a subconjuntos de datos.

---

## 📌 Concatenación de tensores

```python
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1.shape)
```

👉 `torch.cat` concatena tensores a lo largo de un eje (`dim`).

---

## 📌 Multiplicación de tensores

- **Elemento a elemento (Hadamard product):**

```python
print(tensor * tensor)
print(torch.mul(tensor, tensor))
```

- **Multiplicación matricial:**

```python
print(tensor @ tensor.T)           # usando @
print(torch.matmul(tensor, tensor.T)) # usando función
```

👉 `@` es un atajo para multiplicación de matrices.

---

## 📌 Otras operaciones útiles

- `torch.sum()` → suma de todos los elementos.
- `torch.mean()` → promedio.
- `torch.max()` / `torch.min()` → máximo/mínimo.
- `torch.argmax()` / `torch.argmin()` → índice del máximo/mínimo.

Ejemplo:

```python
agg = tensor.sum()
print(agg)
print(agg.item())  # pasar de tensor escalar a número Python
```

---

## 📌 In-place operations

Algunas operaciones modifican el tensor directamente (ahorran memoria, pero son peligrosas).

```python
print(tensor)
tensor.add_(5)   # suma 5 en el mismo tensor
print(tensor)
```

👉 Las operaciones in-place terminan con `_`.

⚠️ **Cuidado**: pueden sobrescribir valores necesarios para cálculos de gradientes.

---

# 🔄 Interoperabilidad con NumPy

Los tensores de PyTorch y arrays de NumPy **comparten memoria** si se convierten entre sí.  
Esto significa que cambiar uno **cambia también al otro** (sin copiar).

---

## 📌 De tensor a NumPy

```python
t = torch.ones(5)
print(t)

n = t.numpy()
print(n)
```

Si modificamos el tensor:

```python
t.add_(1)
print(t)
print(n)  # también cambia!
```

---

## 📌 De NumPy a tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)  # modificamos el array NumPy
print(t)             # el tensor también cambia
```

---

# 🖥️ Dispositivos: CPU vs GPU

PyTorch permite mover tensores entre **CPU y GPU**.  
Esto es crucial para **acelerar cálculos**, especialmente en redes neuronales grandes.

---

### 📌 Crear un tensor en un dispositivo específico

```python
# CPU
tensor_cpu = torch.ones(3,3, device="cpu")

# GPU (si está disponible)
if torch.cuda.is_available():
    tensor_gpu = torch.ones(3,3, device="cuda")
```

---

### 📌 Mover un tensor entre dispositivos

```python
tensor = torch.ones(2,2)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")  # de CPU a GPU
    tensor = tensor.to("cpu")   # de GPU a CPU
```

---

### 📌 Propiedades importantes

```python
print(tensor.device)  # indica el dispositivo actual
```

💡 Tip: operaciones entre tensores **deben estar en el mismo dispositivo**.  
Si intentás operar entre CPU y GPU → error.

---

# 🧮 Gradientes y Autograd

PyTorch tiene un sistema automático de cálculo de gradientes: **Autograd**.  
Esto permite **diferenciar funciones de manera automática**, esencial para el entrenamiento de redes neuronales.

---

### 📌 Activar cálculo de gradientes

```python
x = torch.ones(2,2, requires_grad=True)
print(x)
```

- `requires_grad=True` indica que **PyTorch debe rastrear todas las operaciones** sobre este tensor.
    

---

### 📌 Operaciones y gradientes

```python
y = x + 2
z = y * y * 3
out = z.mean()
print(out)
```

- `out` es una función escalar de `x`.
- Para calcular el gradiente:

```python
out.backward()
print(x.grad)
```

💡 `x.grad` ahora contiene $\frac{\partial \text{out}}{\partial x}$.

---

### 📌 Detalles importantes

- `grad` sólo existe para tensores **con `requires_grad=True`**.
- `backward()` calcula **gradientes automáticamente** siguiendo la cadena de operaciones.
- Si `out` no es escalar (ej: matriz), se debe pasar `gradient`:

```python
v = torch.tensor([[1.0,0.1],[0.01,1.0]])
out.backward(v)
```

---

### 📌 Desactivar gradientes temporalmente

Cuando no necesitamos gradientes (ej: inferencia), usar `torch.no_grad()` para ahorrar memoria:

```python
with torch.no_grad():
    y = x + 2
    print(y)
```

---

### 📌 Detener seguimiento de gradientes en un tensor específico

```python
x.requires_grad_(False)
```

Esto puede ser útil cuando queremos **congelar pesos** de una red durante entrenamiento.

---
