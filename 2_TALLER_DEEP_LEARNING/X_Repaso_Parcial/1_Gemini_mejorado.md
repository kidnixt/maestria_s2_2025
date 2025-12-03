¡Absolutamente! Entiendo la urgencia y la necesidad de tener los conceptos **al detalle** para el parcial de hoy. He reformulado el resumen en un formato redactado y exhaustivo, concentrándome en la mecánica y el propósito de cada concepto clave.

---

## 💻 1. Fundamentos de Tensores y PyTorch

### Propiedades Esenciales de un Tensor

Un tensor es la estructura de datos fundamental en PyTorch. Sus propiedades más solicitadas son:

1. **`dtype`**: Define el tipo de dato subyacente de los elementos (ej. `torch.float32`, `torch.int64`). Esto es vital para la precisión y la eficiencia.
    
2. **`device`**: Indica la ubicación física del tensor, ya sea en la **CPU** o en la **GPU** (`cuda`). La **utilidad** es aprovechar el procesamiento paralelo de la GPU para acelerar cálculos pesados. La **regla de oro** es que **todas las operaciones binarias** requieren que los tensores estén en el mismo `device`, o se producirá un error.
    
3. **`ndim`** o **`dim()`**: El número de dimensiones (ejes o _rank_) del tensor.
    
4. **`shape`** o **`size()`**: Una tupla que indica el tamaño de cada dimensión, por ejemplo, `(4, 3)` para un tensor 2D de 4 filas y 3 columnas.
    

### Mecanismo de Broadcasting (Al Detalle)

El **Broadcasting** es un mecanismo que permite a PyTorch (y NumPy) realizar operaciones aritméticas entre tensores con **formas diferentes** de manera implícita, sin tener que duplicar datos explícitamente. Es útil porque simplifica el código y conserva memoria.

#### Reglas de Compatibilidad

Para que dos tensores sean compatibles para el _broadcasting_, deben cumplir estas tres reglas, comenzando por las dimensiones finales (de derecha a izquierda):

1. **Igual Tamaño:** Las dimensiones son iguales.
    
2. **Dimensión Unitaria:** Una de las dimensiones es de tamaño 1.
    
3. **Dimensión Ausente:** La dimensión no existe en el tensor de forma más pequeña (se añade implícitamente un 1 a la izquierda).
    

**Ejemplo Clave:** Si sumas un tensor $X$ de forma **(4, 3)** con un tensor $Y$ de forma **(3,)**, son compatibles. El tensor $Y$ es _broadcasted_ (estirado) a lo largo de la dimensión 0. Internamente, $Y$ se trata como si fuera de forma $(1, 3)$ y luego se copia 4 veces para coincidir con la forma $(4, 3)$ de $X$, permitiendo la suma elemento por elemento.

### Manipulación de Tensores

- **`torch.cat` (Concatenación):** Une tensores a lo largo de una dimensión **existente**. El número total de dimensiones del tensor de salida es el mismo que el de las entradas. La forma de las entradas debe coincidir en todas las dimensiones _excepto_ en la dimensión de concatenación.
    
- **`torch.stack` (Apilamiento):** Une tensores creando una **nueva dimensión** (aumenta el número de dimensiones). Todos los tensores de entrada _deben_ tener la misma forma. Por ejemplo, apilar dos tensores de forma $(3, 4)$ con `dim=0` resulta en un tensor de forma **(2, 3, 4)**.
    
- **`squeeze()` y `unsqueeze()`:** Se utilizan para gestionar dimensiones de tamaño 1. **`squeeze()`** elimina estas dimensiones (ej. de `(1, 10)` a `(10,)`). **`unsqueeze(dim=n)`** añade una dimensión de tamaño 1 en el índice `n` (ej. de `(10,)` a `(1, 10)`).
    
- **`reshape()`:** Cambia la forma del tensor (ej. de `(8, 1)` a `(2, 4)`), manteniendo el número total de elementos.
    

---

## ⚙️ 2. Entrenamiento, Evaluación y Regularización

### El Bucle de Entrenamiento

- **`model.train()`:** Pone el modelo en modo de entrenamiento. Esto es esencial porque activa el comportamiento dinámico de capas como **`nn.Dropout`** y **`nn.BatchNorm`**.
    
- **`model.eval()`:** Pone el modelo en modo de evaluación. Desactiva **`nn.Dropout`** y congela las estadísticas de **`nn.BatchNorm`** (utiliza las estadísticas globales acumuladas durante el entrenamiento).
    
- **`nn.Dropout` (Impacto de `p`):** El parámetro $p$ es la probabilidad de que una neurona de entrada sea _puesta a cero_ durante el **entrenamiento**. Para mantener la magnitud esperada de la salida, las activaciones restantes se **escalan** por un factor de $1 / (1 - p)$. Durante la **evaluación**, `Dropout` está **inactivo**: no se eliminan neuronas y no hay escalamiento.
    
- **`torch.no_grad()`:** Se utiliza para deshabilitar el cálculo y seguimiento del gradiente. Debe usarse siempre durante la **evaluación/validación** y la **inferencia** para ahorrar memoria de la GPU y acelerar el proceso.
    

### Data Loading y Pérdida

- **`Dataset` y `DataLoader`:** Un `Dataset` personalizado requiere `__len__` (tamaño total) y `__getitem__` (obtener muestra). El `DataLoader` utiliza **`batch_size`** (muestras por actualización de peso), **`shuffle`** (reordenar los datos cada época para evitar aprender el orden) y **`num_workers`** (procesos paralelos de carga).
    
- **Overfitting (Sobreajuste):** Ocurre cuando el modelo comienza a **memorizar** el _ruido_ de los datos de entrenamiento. Se detecta cuando la **Train Loss (pérdida de entrenamiento) disminuye consistentemente**, pero la **Val Loss (pérdida de validación) comienza a aumentar**.
    
- **Early Stopping:** Es una técnica de regularización que previene el sobreajuste. Su implementación se basa en el **`val_loss`**: el entrenamiento se detiene si la pérdida de validación no mejora (es decir, disminuye) durante un número predefinido de épocas (`patience`).
    

---

## 🧠 3. Arquitecturas y Flujo de Información

### Capas Convolucionales y Densas

- **Ventajas de CNN vs. Capas Lineales:** Las capas convolucionales explotan la **localidad espacial** y la **invarianza traslacional** en imágenes. Usan menos parámetros porque los filtros se comparten a través de toda la imagen (peso compartido), lo que permite aprender características locales (bordes, texturas) de manera más eficiente que una capa lineal completamente conectada.
    
- **`nn.Conv2d` (Cálculo de Salida):** La forma de la salida se determina por la fórmula: $H_{\text{out}} = \lfloor (H_{\text{in}} + 2 \times \text{padding} - \text{kernel\_size}) / \text{stride} \rfloor + 1$. El número de _feature maps_ de salida es igual a `out_channels`.
    

### DenseNet vs. ResNet (Conexiones Skip-Connections)

Las **skip-connections** (o conexiones de salto) son vías que permiten que la información y el gradiente fluyan directamente de una capa anterior a una capa posterior. Su utilidad principal es combatir el problema del **gradiente desvanecido** en redes muy profundas.

- ResNet (Residual): Utiliza una operación de suma (aditiva) para combinar la salida de la capa actual con la entrada original (función de identidad).
    
    $$y = F(x) + x$$
    
- **DenseNet (Densa):** Utiliza una operación de **concatenación** para combinar la salida de la capa actual con _todas_ las salidas de las capas anteriores en el bloque. Esto fomenta la **reutilización de características** y es más eficiente en el uso de parámetros gracias al `growth_rate` pequeño.
    

**Bloques de Transición (DenseNet):** Se encuentran entre los bloques densos. Su propósito es **reducir la dimensionalidad espacial** (usando _pooling_) y la **cantidad de _feature maps_** (usando convoluciones 1x1). Si se eliminaran, el número de _feature maps_ crecería exponencialmente con la profundidad, haciendo que el modelo sea **enormemente grande** e **ineficiente** computacionalmente.

---

## 📝 4. Procesamiento de Lenguaje Natural (NLP) y Transformers

### Pre-procesamiento y `nn.Embedding`

- **Tokenización y Normalización:** Un **token** es la unidad de texto (ej. una palabra). La **normalización** busca estandarizar el texto (ej. pasar a minúsculas, remover puntuación) para que el modelo no trate variaciones de la misma palabra como _tokens_ separados.
    
- **Representación Numérica:** Las redes neuronales requieren entradas numéricas, por lo que las palabras no se pueden usar directamente.
    
- **`nn.Embedding` (Al Detalle):** Esta capa convierte un índice discreto (un token) en un **vector denso de punto flotante** (el _embedding_), que codifica su significado semántico. Si una entrada es de forma $(B, L)$, donde $B$ es el tamaño del lote y $L$ es la longitud de la secuencia, y el _embedding_ tiene un tamaño de $E$, la salida será de forma **$(B, L, E)$**.
    
- **Padding y Truncamiento:** **Padding** (relleno) añade un token especial (`<pad>`) a las secuencias más cortas para que todas tengan la misma longitud máxima. **Truncamiento** recorta las secuencias que exceden la longitud máxima. Ambos sirven para uniformizar la entrada al modelo.
    

### Seq2Seq

- **Encoder y Decoder:** El **Codificador** procesa la secuencia de entrada completa y la comprime en un **vector de contexto** (el estado oculto final). El **Decodificador** toma este vector de contexto y genera la secuencia de salida, token por token.
    
- **`Teacher Forcing`:** Es una estrategia de entrenamiento que consiste en pasar el **token real** de la secuencia objetivo (el _ground truth_) como entrada al decodificador en el siguiente paso de tiempo, en lugar de la predicción del modelo. Esto acelera la convergencia.
    
- **Token `<SOS>`:** El token _Start of Sequence_ se agrega como la primera entrada al decodificador para señalar el **comienzo de la generación** de la secuencia de salida.
    
- **Inferencia / Fin de Predicción:** Durante la inferencia (generación), el proceso termina cuando el decodificador predice el token especial de **fin de secuencia** (`<EOS>`).
    

### Arquitectura Transformer (Al Detalle)

Los Transformers solucionan el problema de la dependencia secuencial de las arquitecturas recurrentes (RNN/LSTM) gracias al mecanismo de atención. **Simulan la secuencialidad** inyectando el **Positional Encoding**.

- **Positional Encoding (Al Detalle):** Los Transformers procesan todos los tokens de entrada simultáneamente, por lo que no tienen información inherente sobre el orden. El _Positional Encoding_ es un conjunto de vectores (basados en funciones sinusoidales/cosinusoidales) que se **suman** a los _embeddings_ de entrada. Esto dota a cada token de una representación única que codifica su **posición absoluta y relativa** dentro de la secuencia.
    
- **Mecanismos de Atención:**
    
    - **Codificador:** Contiene **un** mecanismo de **Self-Attention** (Auto-atención). Permite que un token atienda a todos los demás tokens de la secuencia de entrada para calcular mejor su propia representación.
        
    - **Decodificador:** Contiene **dos** mecanismos:
        
        1. **Masked Self-Attention (Auto-atención Enmascarada):** Solo permite que un token atienda a los tokens que ya han sido generados (incluido él mismo), **impidiendo que vea tokens futuros**.
            
        2. **Encoder-Decoder Attention (Atención Cruzada):** Permite que el decodificador atienda a la secuencia de salida del codificador (el vector de contexto), usando sus _queries_ para buscar información en las _keys_ y _values_ del codificador.
            
- **Cálculo de Atención (Q, K, V) - Línea por Línea:**
    
    1. `scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)`: Calcula la **similitud** entre el vector **Query** de un token y todos los vectores **Key** de la secuencia. Un puntaje alto significa alta relevancia. Se divide por $\sqrt{d_k}$ para estabilizar los gradientes (escalamiento).
        
    2. `attention = torch.softmax(scores, dim=-1)`: Convierte las puntuaciones de similitud en **pesos de atención probabilísticos** que suman 1 (en la dimensión de las claves).
        
    3. `context = torch.matmul(attention, value)`: Realiza una **suma ponderada** de los vectores **Value** utilizando los pesos de atención. Esto produce el **vector de contexto** final para el token actual, que es una mezcla de información relevante de toda la secuencia.
        
- **Uso de Máscaras:** Las máscaras (`mask`) se utilizan para evitar la atención a ciertas posiciones. Esto se logra estableciendo los _scores_ de atención para esas posiciones a un valor muy bajo (ej. $-\infty$) antes de la operación **Softmax**, lo que resulta en una probabilidad de atención de 0. Son esenciales para:

    1. **Paddings:** Evitar atender a los tokens de relleno (`<pad>`).
    2. **Masked Attention:** Evitar que el decodificador atienda a tokens **futuros** en la secuencia de salida.
- **Inferencia Token por Token:** Durante el **entrenamiento**, las secuencias objetivo están disponibles, y la generación puede ser en paralelo (con la máscara). Durante la **inferencia**, el decodificador **debe generar la secuencia token por token**. Esto se debe a que la entrada para el siguiente paso (`token_{t+1}`) es el token que el modelo acaba de predecir (`token_t`). No se puede generar toda la secuencia a la vez porque la entrada de cada paso depende de la salida anterior.
