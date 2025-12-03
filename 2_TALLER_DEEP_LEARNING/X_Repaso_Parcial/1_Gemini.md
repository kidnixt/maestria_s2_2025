
## 💻 1. Fundamentos de Tensores y PyTorch

### Propiedades Esenciales

Un tensor en PyTorch tiene varias propiedades clave. El **`dtype`** especifica el tipo de datos de los elementos que contiene, como `torch.float32`. El **`device`** indica si el tensor está almacenado en la CPU o en la GPU. El **`ndim`** es el número de dimensiones del tensor, y **`shape`** es una tupla que indica el tamaño de cada dimensión. La utilidad del `device` radica en permitir mover un tensor a la GPU para aprovechar el procesamiento paralelo. Si intentas operar tensores en diferentes dispositivos (CPU y GPU), se producirá un error; la solución es mover ambos al mismo dispositivo.

### Manipulación de Formas

El **Broadcasting** es un mecanismo útil que permite realizar operaciones entre tensores con diferentes formas, estirando las dimensiones compatibles implícitamente. Por ejemplo, al sumar un tensor de forma $(4, 3)$ con uno de forma $(3,)$, el tensor de forma $(3,)$ se copia a lo largo de la dimensión 0 para hacerlo compatible, resultando en una forma $(4, 3)$.

Para manipular tensores:

- **`torch.cat` (Concatenar):** Une tensores a lo largo de una dimensión **existente**. Las formas deben ser idénticas en todas las demás dimensiones.
    
- **`torch.stack` (Apilar):** Une tensores a lo largo de una **nueva dimensión** (aumenta el número de dimensiones). Todos los tensores de entrada deben tener la misma forma.
    
- **`squeeze`:** Elimina las dimensiones del tensor que tienen un tamaño de 1.
    
- **`unsqueeze(dim=0)`:** Agrega una dimensión de tamaño 1 en la posición especificada.
    
- **`reshape`:** Permite cambiar la forma de un tensor siempre que el número total de elementos se mantenga constante.
    

---

## ⚙️ 2. Entrenamiento y Evaluación

### Data Loading

Al crear un **`Dataset`** personalizado, debes implementar las funciones `__len__` (para devolver el tamaño del dataset) y `__getitem__` (para obtener la muestra y etiqueta de un índice). El **`DataLoader`** utiliza el parámetro **`batch_size`** para determinar la cantidad de muestras procesadas antes de actualizar los pesos. Un `batch_size` grande puede acelerar el entrenamiento, pero uno pequeño puede ayudar a escapar de mínimos locales. **`shuffle`** se activa para reordenar los datos en cada época, previniendo que el modelo aprenda el orden de las muestras. **`num_workers`** especifica el número de procesos paralelos para la carga de datos, lo que puede mejorar el rendimiento.

### Modos del Modelo y Gradientes

Se utiliza **`model.train()`** para poner el modelo en modo de entrenamiento, lo cual activa capas como **`nn.Dropout`** y **`nn.BatchNorm`**. Por el contrario, **`model.eval()`** pone el modelo en modo de evaluación, **desactivando el `Dropout`**  y congelando las estadísticas de `BatchNorm`. Es crucial usar este modo durante la evaluación, ya que olvidarlo puede llevar a resultados inconsistentes debido al `Dropout` activo. Además, se usa el contexto **`torch.no_grad()`** durante la evaluación para deshabilitar el cálculo del gradiente, ahorrando memoria y tiempo.

### Pérdidas y Regularización

La **Train Loss** representa la pérdida calculada sobre los datos de entrenamiento, y la **Val Loss** es la pérdida calculada sobre los datos de validación, que mide la capacidad de generalización.

Si la **Train Loss disminuye** pero la **Val Loss aumenta**, se está produciendo **sobreajuste (overfitting)**.

El **Early Stopping** es una técnica que ayuda a detener este fenómeno. Su implementación se basa en la **Val Loss**: el entrenamiento se detiene si la `val_loss` no mejora (o comienza a aumentar) después de un número predefinido de época.

---

## 🧠 3. Capas y Arquitecturas

### Capas Comunes

La capa **`nn.Linear`** realiza una transformación lineal, mapeando una entrada de forma $(N, \text{in\_features})$ a una salida de forma $(N, \text{out\_features})$. El uso de **`bias=True`** permite a la capa aprender un desplazamiento constante en la salida.

La capa **`nn.Dropout`** es una forma de regularización que, durante el entrenamiento, establece una fracción $p$ de las entradas a cero. Esto evita la co-adaptación de neuronas.

La capa **`nn.Embedding`** mapea índices enteros (tokens discretos) a vectores densos (embeddings) y es esencial para procesar lenguaje natural.

Una capa de **Pooling** (ej. `MaxPool2d`) reduce la dimensionalidad espacial de los _feature maps_ después de la convolución. Esto ayuda a reducir parámetros, la complejidad computacional y a generar cierta invarianza traslacional.

En cuanto a las capas recurrentes, **`nn.LSTM`** se diferencia de **`nn.RNN`** al introducir un estado de celda de memoria ($C_t$) y varias puertas (input, forget, output) que controlan explícitamente el flujo de información, solucionando el problema del gradiente desvanecido/explosivo.

### Arquitecturas CNN y Regularización

Las **convoluciones** son ventajosas sobre las capas lineales para clasificación de imágenes porque explotan la localidad y la invarianza traslacional, aprendiendo patrones locales que se pueden aplicar en cualquier parte de la imagen.

En **DenseNet**, el concepto de **conexiones densas** significa que cada capa en un bloque recibe como entrada los _feature maps_ de **todas las capas anteriores** en ese bloque. Esto se diferencia de las **conexiones residuales de ResNet**, donde la salida de una capa simplemente se **suma** a la entrada de una capa anterior (skip-connection). El beneficio de la concatenación en DenseNet es una mayor reutilización de características y un flujo de gradiente mejorado. El parámetro **`growth_rate`** determina cuántos _feature maps_ añade cada capa al _state_ colectivo del bloque.

Los **Bloques de Transición** en DenseNet se colocan entre los bloques densos y tienen el propósito de reducir el tamaño de los _feature maps_ mediante convoluciones 1x1 y pooling, controlando el tamaño del modelo y mejorando la eficiencia computacional. Si se eliminaran, el modelo sería mucho más grande e ineficiente, ya que los _feature maps_ crecerían en cada capa.

---

## 🌐 4. NLP y Arquitecturas de Atención

### Pre-procesamiento y Vocabulario

El objetivo de **normalizar texto** es estandarizarlo (ej. minúsculas) para que el modelo no trate la misma palabra escrita de manera diferente como _tokens_ separados. Un **token** es la unidad de texto (ej. palabra) que se mapea a un **vocabulario** (el conjunto de todos los tokens únicos). Limitar el **tamaño del vocabulario** es crucial en datasets grandes para reducir la complejidad, generalmente descartando las palabras menos frecuentes. Las palabras no pueden usarse directamente en una red neuronal; deben representarse numéricamente, por ejemplo, mediante `nn.Embedding`.

### Seq2Seq y Teacher Forcing

En un modelo **Seq2Seq**, el **Codificador** procesa la secuencia de entrada para crear un **vector de contexto** (estado oculto). Este vector de contexto pasa al **Decodificador**, que lo usa para generar la secuencia de salida, generalmente token por token. El **Teacher Forcing** es una técnica de entrenamiento donde se usa el token de salida real (el "profesor") del paso anterior como entrada para el decodificador en el paso actual, lo que acelera y estabiliza el entrenamiento. Durante la inferencia, la predicción se detiene cuando el decodificador predice un token de fin de secuencia (`<EOS>`).

### Transformers

Los **Transformers** resuelven los problemas de las arquitecturas recurrentes (como la dependencia a largo plazo) al usar el mecanismo de atención y el procesamiento paralelo en lugar de pasos secuenciales. Simulan la secuencialidad inyectando **Codificación Posicional (_Positional Encoding_)** a los _embeddings_ de entrada. Esta codificación es una señal sinusoidal y cosinusoidal que proporciona al modelo información sobre la posición absoluta y relativa de los tokens.

En el Transformer original (Attention Is All You Need), el **Codificador** tiene un mecanismo de **Self-Attention** (Auto-atención). El **Decodificador** tiene **dos** mecanismos: uno de **Masked Self-Attention** (para ver solo tokens anteriores) y uno de **Encoder-Decoder Attention** (para mirar la salida del codificador).

El mecanismo de atención funciona así:

1. **Cálculo de Scores:** $\text{scores} = \text{torch.matmul}(\text{query}, \text{key.transpose}(-2, -1)) / \text{math.sqrt}(d_k)$ calcula la similitud entre el vector _Query_ y todos los vectores _Key_.
2. **Softmax:** $\text{attention} = \text{torch.softmax}(\text{scores}, \text{dim}=-1)$ convierte estas puntuaciones en pesos de atención probabilísticos.
3. **Contexto:** $\text{context} = \text{torch.matmul}(\text{attention}, \text{value})$ realiza una suma ponderada de los vectores _Value_ para obtener el vector de contexto.

Las **máscaras** se utilizan para evitar que el Transformer preste atención a posiciones no deseadas (como los tokens de padding o los tokens futuros durante la generación, lo cual es vital para el _Masked Self-Attention_ del decodificador).

Durante la **inferencia en modelos de generación de texto** (como el decodificador del Transformer), no se puede generar toda la secuencia de salida a la vez porque la predicción de un token depende del token generado en el paso anterior60. Por lo tanto, el proceso se maneja **token por token**, alimentando la salida predicha como entrada para generar el siguiente token.

