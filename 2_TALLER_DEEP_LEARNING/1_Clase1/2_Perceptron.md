
# 🚀 Mi primera red neuronal


``` python
import torch
import torch.nn as nn
import torch.optim as optim
```

En esta notebook, se explora el **perceptrón**, el bloque de construcción fundamental de las **redes neuronales**. Se usa para resolver problemas de **clasificación binaria**, como las funciones lógicas.

👉 La clave es el uso de la función `sgn` para decidir la clase de salida.

**Basado en:**

- Implementación paso a paso de perceptrones con PyTorch.
- Resolución del problema XOR con un Perceptrón Multicapa (MLP).
- [Manejo de módulos y optimizadores en `torch.nn`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/nn.html%5D\(https://pytorch.org/docs/stable/nn.html\)).

---

## ⚙️ El Perceptrón

Un perceptrón clasifica entradas usando la función `sgn(x * w + b)`.

### 📌 La Función `sgn`


``` python
def sgn(x):
    return torch.sign(x) + (x == 0).float()
```

👉 Implementación personalizada para asegurar que `sgn(0) = 1.0`, a diferencia de `torch.sign(0) = 0`.

---

### 📌 Implementación con `matmul`


``` python
X = torch.tensor([[0,0,1],[0,1,1],[1,0,1],[1,1,1]], dtype=torch.float32)
W = torch.tensor([[0.5],[0.5],[-0.5]], dtype=torch.float32)
y_pred = sgn(torch.matmul(X, W))
```

👉 Incluye una columna extra de `1` en la matriz de entrada `X` para el sesgo. La operación se reduce a una simple multiplicación de matrices.

---

### 📌 Implementación con Sesgo


``` python
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
W = torch.tensor([[0.5],[0.5]], dtype=torch.float32)
b = torch.tensor([-0.5], dtype=torch.float32)
y_pred = sgn(torch.matmul(X, W) + b)
```

👉 El sesgo (`b`) se añade por separado y se aprovecha el **broadcasting** de PyTorch para sumarlo a todas las filas.

---

## 🏗️ Perceptrón Multicapa (MLP)

Un perceptrón simple no puede resolver el problema **XOR**. Para esto, se necesita una red neuronal con al menos una capa oculta, un **MLP**.

### 📌 Arquitectura del Modelo

El modelo se define como una clase que hereda de `nn.Module`.


``` python
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.input = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.input(x))
        x = torch.sigmoid(self.output(x))
        return x
```

- `nn.Linear`: capa lineal que aplica `y = xA^T + b`.
- `torch.sigmoid`: función de activación para introducir **no-linealidad**.

---

## ⚙️ Bucle de Entrenamiento

El proceso de entrenamiento sigue un ciclo estándar de PyTorch.

### 📌 Configuración del Entrenamiento


``` python
criterion = nn.BCELoss() # Función de pérdida (Binary Cross-Entropy)
optimizer = optim.SGD(model.parameters(), lr=0.5) # Optimizador (Stochastic Gradient Descent)
```

- `nn.BCELoss`: ideal para problemas de **clasificación binaria**.
- `optim.SGD`: ajusta los pesos del modelo para minimizar la pérdida.

---

### 📌 Pasos del Bucle


``` python
for epoch in range(10000):
    model.train() # Modo de entrenamiento
    optimizer.zero_grad() # Limpia gradientes
    outputs = model(X) # Forward pass
    loss = criterion(outputs, y) # Calcula la pérdida
    loss.backward() # Backpropagation
    optimizer.step() # Actualiza los pesos
```

💡 `loss.backward()` calcula los gradientes del modelo, y `optimizer.step()` los aplica.

⚠️ **Cuidado**: `optimizer.zero_grad()` es esencial. Si se omite, los gradientes se **acumulan** en cada iteración.

---

## 📋 Evaluación del Modelo

### 📌 Modo de Evaluación



``` python
model.eval()
with torch.no_grad():
    y_pred = model(X)
```

- `model.eval()`: pone el modelo en modo de evaluación.
- `with torch.no_grad()`: desactiva el cálculo de gradientes.

👉 Es una buena práctica para la **inferencia** para **ahorrar memoria y tiempo de cálculo**.

---

## 🎯 Conclusión de la notebook

1. **Perceptrón**: unidad simple para clasificación lineal.
2. **MLP**: necesario para problemas no lineales como XOR, introduciendo capas ocultas y funciones de activación.
3. **Módulos clave de `torch.nn`**: `nn.Linear`, `nn.Module` y funciones de pérdida como `nn.BCELoss`.
4. **Proceso de entrenamiento**: `zero_grad()`, `forward pass`, `backward()`, `step()`.
5. **Prácticas clave**: uso de `model.train()` y `model.eval()` para la gestión del modelo.