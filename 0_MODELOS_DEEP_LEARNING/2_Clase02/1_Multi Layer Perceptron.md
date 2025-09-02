# 📘 Apuntes - Deep Learning: Multi-Layer Perceptron (MLP)

---

## 1. 🧠 Modelo de una neurona

- Modelo paramétrico:
$$
\hat{y} = f(x;\theta)
$$

donde la función $f$ está parametrizada por $\theta$.

- **Regresión lineal** (modelo base de regresión):
$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_D x_D + b = x^\top w + b
$$

- **Regresión logística** (modelo base de clasificación):
$$
\hat{y} = \text{Sigmoid}(x^\top w + b)
$$

- En ambos casos:
$$
\theta = (w, b)
$$

💡 **Intuición:** una neurona combina entradas linealmente y luego aplica una función (identidad o sigmoide).

---

## 2. 🔩 Perceptrón: modelo de una neurona

Con notación matricial:
$$
\hat{y} = A(x^\top w + b)
$$

Donde:
- $w = [w_1, \dots, w_D]^\top$
- $x = [x_1, \dots, x_D]^\top$
- $A: \mathbb{R} \to \mathbb{R}$ es la **función de activación**.

💡 **Diferencia con regresión lineal:** gracias a $A$ aparece la **no linealidad**.

---

## 3. ⚡ Funciones de activación

Funciones más comunes:

- **Sigmoid:** salida en $[0,1]$.
- **Tanh:** salida en $[-1,1]$.
- **ReLU:** $A(z) = \max(0, z)$.
- **Leaky ReLU:** pendiente pequeña cuando $z<0$.
- **ELU:** suaviza la parte negativa.
- **SoftPlus:** $A(z) = \ln(1+e^z)$.
- **GELU:** $A(z) = z \Phi(z)$ (muy usada en Transformers).

💡 **Intuición:** la activación define qué tan “activo” está un nodo y permite que el modelo represente **no linealidades**.

---

## 4. 🔗 Componente básica: Capa Densa / Lineal

La operación central de un MLP es:
$$
a = D(x;W,b) = A(x^\top W + b^\top)
$$

- Entrada: $x \sim (I)$
- Salida: $a \sim (O)$
- Pesos: $W \sim (I,O)$
- Bias: $b \sim (O)$

La operación lineal es:
$$
z = x^\top W + b^\top
$$

La activación aplica:
$$
a = A(z)
$$

💡 **Cada capa densa = linealidad + activación.**

---

## 5. 🧮 Parámetros de una capa densa

La cantidad de parámetros es:
$$
\text{params}(D) = I \times O + O = (I+1)O
$$

- $I$: número de entradas  
- $O$: número de salidas  

Ejemplo: si $I=100$, $O=50$:  
$(100+1)\times 50 = 5050$ parámetros.

---

## 6. 📦 Operando en batch

Con $N$ ejemplos en paralelo:

- Entrada: $X \sim (N, I)$ (cada fila un input)
- Pesos: $W \sim (I, O)$
- Bias: $b \sim (O)$
- Salida:
$$
a = A(XW + b) \sim (N,O)
$$

💡 **Broadcasting:** el bias $b$ se expande a todas las filas.

Ejemplo en PyTorch:
```python
import torch
X = torch.randn(32, 100)   # batch de 32, input de dim 100
W = torch.randn(100, 50)   # pesos
b = torch.randn(50)        # bias
a = torch.relu(X @ W + b)  # salida (32, 50)

