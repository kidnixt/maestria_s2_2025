
# 🏡 Regresión: Predicción de Precios con PyTorch


``` python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from sklearn.datasets import fetch_california_housing
```

En esta notebook, se desarrolla un modelo de **regresión** para predecir los valores medianos de las casas en California. Para lograrlo, se construye una red neuronal en **PyTorch** y se utiliza la herramienta **Weights & Biases (W&B)** para el seguimiento de métricas y la visualización de experimentos.

👉 El enfoque se centra en un flujo de trabajo de **Machine Learning de principio a fin**, desde la carga de datos hasta el entrenamiento y el seguimiento.

**Basado en:**

- Dataset de California Housing de `sklearn.datasets`.
- Implementación de un modelo `nn.Module` para regresión.
- [Módulos de PyTorch para datos: `Dataset` y `DataLoader`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/data.html%5D\(https://pytorch.org/docs/stable/data.html\)).
- [Documentación oficial de Weights & Biases](https://docs.wandb.ai/).

---

## 💾 Carga y Preparación de Datos

### 📌 Carga del Dataset

El conjunto de datos de California Housing es cargado directamente desde la biblioteca `scikit-learn`. El objetivo es predecir la columna **`MedHouseVal`** (`Median House Value`) a partir de 8 características demográficas y geográficas, como la mediana de ingresos (`MedInc`), la edad de la casa (`HouseAge`) y la latitud y longitud.



```python
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame
print(df.info())
```

👉 Un primer análisis muestra que el dataset no tiene valores nulos y todas las características son de tipo `float64`.

### 📌 Escalamiento de Características

El modelo de regresión puede beneficiarse del escalado de las características de entrada para que tengan una media de cero y una desviación estándar de uno. Esto es importante para el buen desempeño de la red.


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('MedHouseVal', axis=1))
y = df['MedHouseVal'].values.reshape(-1, 1)
```

- Se utiliza `StandardScaler` para normalizar las características.
- La variable objetivo `y` se mantiene sin escalar, pero se ajusta su forma con `.reshape(-1, 1)` para que sea una matriz de una columna.

---

### 📌 Clase `CaliforniaHousingDataset`

Para una gestión eficiente de los datos, se crea una clase `Dataset` personalizada.



```python
class CaliforniaHousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

- **`__init__`**: Convierte los arreglos de NumPy de características (`X`) y etiquetas (`y`) directamente a tensores de PyTorch con el tipo `float`. Es crucial que los tensores tengan el mismo tipo de dato.
- **`__len__`**: Devuelve la cantidad total de muestras en el conjunto de datos.
- **`__getitem__`**: Permite acceder a una muestra específica y su etiqueta por índice.

---

### 📌 División y `DataLoader`

Se divide el `dataset` en conjuntos de entrenamiento (80%) y validación (20%) de forma aleatoria. Luego, se utilizan los `DataLoader` para crear iteradores de mini-lotes.


```python
train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
```

- **`shuffle=True`**: Aleatoriza el orden de los datos en cada época, lo que ayuda a evitar sesgos en el entrenamiento.
- **`num_workers=4`**: Permite cargar los datos en paralelo, acelerando el proceso de entrenamiento.


---

## 🏗️ Modelo y Entrenamiento

### 📌 Definición del Modelo

El modelo de regresión se define como una clase `HousingModel` que hereda de `nn.Module`. Es una red neuronal de tres capas completamente conectadas.


```python
class HousingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU() # o 'torch.relu'

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

- La capa de entrada (`self.fc1`) tiene `input_dim` (8) neuronas y 64 salidas.
- Las capas ocultas usan la función de activación **`ReLU`** (`nn.ReLU` o `torch.relu`) que introduce no-linealidad.
- La capa de salida (`self.fc3`) tiene 1 neurona, adecuada para la tarea de regresión.


### 📌 Configuración y Bucle de Entrenamiento

Se define la función de pérdida y el optimizador antes de iniciar el bucle de entrenamiento.



```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
```

- **`nn.MSELoss()`**: La función de pérdida de **Error Cuadrático Medio** (`Mean Squared Error`), ideal para problemas de regresión.
- **`optim.Adam()`**: El optimizador Adam, que ajusta la tasa de aprendizaje para cada peso de la red. `lr` es un hiperparámetro clave.

---

## 📊 Integración con W&B

**Weights & Biases (W&B)** es una herramienta esencial para el seguimiento y la visualización de los experimentos de Machine Learning.

### 📌 Inicialización del Experimento

Antes del entrenamiento, se inicia una sesión de W&B (`wandb.init`) para registrar el experimento.



``` python
wandb.init(project="California-Housing", config={
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 100
})
```

- El parámetro **`project`** organiza los experimentos.
- El diccionario **`config`** almacena los **hiperparámetros** y metadatos relevantes del experimento. Estos son accesibles y modificables desde la interfaz de W&B.


### 📌 Registro de Métricas y Visualizaciones

Durante el bucle de entrenamiento, se llama a `wandb.log()` para enviar las métricas a la plataforma en tiempo real.


``` python
wandb.log({"train_loss": train_loss_avg, "val_loss": val_loss_avg, "epoch": epoch})
```

- Esto genera **gráficos de evolución** de la pérdida de entrenamiento y validación, permitiendo monitorear el progreso del modelo y detectar overfitting.

---

## 🎯 Conclusión de la notebook

1. **Datos**: La preparación incluye la carga, escalado de características y la creación de un `Dataset` personalizado y `DataLoaders`.
2. **Modelo**: Se implementa una red neuronal de 3 capas (`nn.Linear`) con activaciones `ReLU` para la regresión.
3. **Entrenamiento**: El proceso sigue la secuencia estándar de PyTorch: `zero_grad`, `forward`, `loss`, `backward` y `step`.
4. **W&B**: La integración de **Weights & Biases** permite un seguimiento y una visualización eficientes de las métricas clave, como la pérdida de entrenamiento y validación.