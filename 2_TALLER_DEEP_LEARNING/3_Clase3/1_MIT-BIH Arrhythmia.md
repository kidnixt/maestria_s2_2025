# 💓 Clasificación de Arritmias con PyTorch



```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
```

Esta notebook se enfoca en la implementación de técnicas de **regularización** en redes neuronales, como **`Early Stopping`** y **`Dropout`**, para la tarea de clasificación de arritmias cardíacas. Utiliza un subconjunto del **MIT-BIH Arrhythmia Database** para entrenar una red neuronal en **PyTorch**, con el objetivo de prevenir el **sobreajuste** y mejorar la capacidad del modelo para generalizar a datos no vistos.

👉 El enfoque es el uso de la **regularización** para asegurar que la red no se "memorice" los datos de entrenamiento, sino que aprenda patrones útiles.

**Basado en:**

- Dataset de arritmias cardíacas del [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
- Módulos `nn.Module`, `nn.Linear`, `nn.Dropout` de PyTorch.
- [Manejo de datos: `Dataset` y `DataLoader`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/data.html%5D\(https://pytorch.org/docs/stable/data.html\)).
- Métricas de evaluación de `scikit-learn`.

---

## 💾 Carga y Preparación de Datos

### 📌 La clase `ECGDataset`

Para una gestión eficiente de los datos, se crea una clase personalizada que hereda de `torch.utils.data.Dataset`. Esta clase se encarga de cargar los datos de entrada y las etiquetas, convirtiéndolos en tensores de PyTorch con los tipos de datos correctos.



```Python
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

- **`__init__`**: Convierte el arreglo de NumPy `X` a `torch.float` (para operaciones con punto flotante) y las etiquetas `y` a `torch.long`, que es el tipo requerido por la función de pérdida `nn.CrossEntropyLoss` para las etiquetas de clase.
- **`__len__`**: Devuelve la cantidad total de muestras en el dataset, lo que es necesario para que el `DataLoader` funcione correctamente.
- **`__getitem__`**: Permite acceder a una muestra específica del dataset a través de un índice, retornando el par `(característica, etiqueta)`.

---

### 📌 `DataLoader`

Una vez que el `Dataset` está definido, se utilizan los `DataLoader` para crear iteradores que devuelven los datos en mini-lotes.


```Python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

- **`batch_size`**: Define el número de muestras por mini-lote (en este caso, 64).
- **`shuffle=True`**: Se utiliza en el conjunto de entrenamiento para aleatorizar el orden de las muestras en cada época, lo que evita que el modelo aprenda el orden de los datos.

---

## 🏗️ Arquitectura de la Red y Regularización

### 📌 El Modelo MLP

El modelo de clasificación es un **Perceptrón Multicapa (MLP)**. Se define como una clase que hereda de `nn.Module` y utiliza múltiples capas lineales y una técnica de regularización clave.


```Python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(187, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5) 
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```

- **`nn.Linear`**: Se utilizan para crear las capas completamente conectadas. El tamaño de entrada de la primera capa es **187**, que es el número de características en cada muestra de ECG.
- **`nn.Dropout(p=0.5)`**: Esta es la técnica de regularización clave. Durante el **entrenamiento**, el `Dropout` apaga aleatoriamente el 50% de las neuronas de la capa anterior. Esto reduce la **co-adaptación** de las neuronas, obligándolas a aprender de forma más robusta y evitando el sobreajuste.

---

## ⚙️ Técnicas de Regularización en Acción

### 📌 `Dropout` y los Modos del Modelo

El comportamiento de `Dropout` depende del modo en el que se encuentre el modelo.


```Python
model.train() # Activa el comportamiento del dropout
# ... bucle de entrenamiento
model.eval() # Desactiva el dropout y otras capas de regularización
```

- **`model.train()`**: Se debe llamar antes del bucle de entrenamiento. Habilita las capas como `nn.Dropout`.
- **`model.eval()`**: Se debe llamar antes de la evaluación o inferencia. Desactiva el `Dropout` para que todas las neuronas contribuyan a la predicción final.

---

### 📌 `Early Stopping`

Esta técnica es una forma de **regularización por detención temprana**. Detiene el entrenamiento del modelo cuando el rendimiento en el conjunto de validación deja de mejorar.


```Python
# Variables para Early Stopping
patience = 5
counter = 0
best_val_loss = float('inf')

# Dentro del bucle de entrenamiento, después de calcular la pérdida de validación
if val_loss < best_val_loss:
    best_val_loss = val_loss
    counter = 0 # Reinicia el contador si la pérdida mejora
    torch.save(model.state_dict(), 'best_model.pth') # Guarda el mejor modelo
else:
    counter += 1
    if counter >= patience:
        print("Early stopping!")
        break # Detiene el bucle
```

- **`patience`**: El número de épocas consecutivas sin mejora en la pérdida de validación antes de detener el entrenamiento.
- **Ventaja**: Evita el sobreajuste, ahorra tiempo de cómputo y permite guardar el mejor modelo en lugar del último.

---

## 🚀 Proceso de Entrenamiento y Evaluación

### 📌 Configuración



```Python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

- **`nn.CrossEntropyLoss()`**: Es la función de pérdida estándar para la **clasificación multiclas** en PyTorch. Es ideal porque internamente combina las operaciones necesarias para la clasificación, como `LogSoftmax` y `NLLLoss`.
- **`optim.Adam()`**: El optimizador utilizado para ajustar los pesos de la red.

### 📌 Bucle de Entrenamiento

El bucle de entrenamiento es el proceso iterativo donde el modelo aprende. Por cada `epoch`:

1. **Entrenamiento**: Se itera sobre el `train_loader`, se realizan los `forward` y `backward` passes, y se actualizan los pesos.
2. **Validación**: Se itera sobre el `val_loader` en modo `eval`, se calcula la pérdida de validación, y se utiliza la lógica de `Early Stopping` para decidir si continuar o no.

### 📌 Evaluación del Modelo

Una vez finalizado el entrenamiento, el rendimiento del mejor modelo se evalúa en un conjunto de prueba.


```Python
model_classification_report(ej3_model, test_loader)
```

- La función `model_classification_report` utiliza `sklearn.metrics.classification_report` para mostrar métricas detalladas por clase, como **precisión**, **recall** y **f1-score**. Esto proporciona una visión más completa del rendimiento del modelo que solo la precisión global.

---

## 🎯 Conclusión de la notebook

1. **Clasificación Multiclas**: La notebook demuestra cómo abordar un problema de clasificación de varias clases con PyTorch.
2. **Regularización**: La implementación de **`Dropout`** y **`Early Stopping`** es fundamental para prevenir el **sobreajuste** y asegurar que el modelo sea robusto y generalice bien.
3. **Flujo de Trabajo Completo**: Se muestra un ciclo completo de machine learning: preparación de datos, definición de la arquitectura, entrenamiento con optimizador y función de pérdida, y evaluación detallada con métricas de clasificación.