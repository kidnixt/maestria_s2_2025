
# 📘 Apuntes - Deep Learning: Procesamiento de Lenguaje Natural (NLP)

---

## 1. 🎯 NLP: Contexto y Aplicaciones

El **Procesamiento de Lenguaje Natural (NLP)** es un campo de la Inteligencia Artificial (AI) que permite a las máquinas comprender el lenguaje humano.

### 🛠️ Usos Prácticos

El NLP se aplica en diversas tareas, facilitando la interacción entre humanos y computadoras:

- **Sentiment Analysis** (Análisis de Sentimiento).
- **Machine Translation** (Traducción Automática).
- **Question Answering** (Respuesta a Preguntas).
- **Text Summarization** (Resumen de Texto).
- **Text Classification** (Clasificación de Texto).

### 🧠 Comprensión del Lenguaje

La comprensión requiere extraer el significado contextual de las palabras. Por ejemplo, un texto se puede analizar para identificar la Emoción (Frustrado), el Tono (Negativo, Subjetivo), el Producto (Messenger App) y el Lenguaje (Inglés, Informal).

👉 Un sistema puede, por ejemplo, transformar un input como "Cynthia sold the bike to Bob for $200" en una transacción comercial estructurada, identificando al vendedor, comprador, bien y precio.

---

## 2. ⚠️ Desafíos Fundamentales del NLP

El lenguaje humano está lleno de complejidades que los modelos deben resolver:

### 🎭 Ambigüedad Semántica

- El **Contexto es la matriz del significado**.
- Una misma frase tiene significados radicalmente distintos según el contexto, lo que obliga a los modelos a interpretar la situación para desambiguar.

### 🔄 Sinónimos

- Múltiples palabras pueden tener el mismo significado (ej: _cultivation_, _civilization_, _refinement_ para _culture_).
- El modelo debe entender la **equivalencia semántica** de palabras distintas.

### 🔗 Dependencia

- Las relaciones semánticas dependen de palabras que pueden estar distantes en el texto.
- Esto obliga a los modelos a mantener el **contexto a largo plazo**.

---

## 3. ⚙️ Pipeline de Preprocesamiento

El texto debe ser preparado antes de ser usado por un modelo de _Deep Learning_ mediante un _pipeline_:

### 🧮 3.1. Tokenización

- Proceso de descomponer el texto en las unidades más pequeñas, generalmente **palabras individuales o _tokens_**.
- Estos tokens son los bloques básicos para el análisis posterior.

### ✂️ 3.2. Remoción de Stop Words

- Las _Stop Words_ son **palabras comunes** que suelen eliminarse para enfocar el análisis en el contenido relevante.

### 🌿 3.3. Reducción a la Raíz

Se busca reducir las distintas formas de una palabra (flexión) a una raíz.

- **Stemming:**
    
    - Reduce a una raíz que **no necesariamente es una palabra válida** del idioma.
    - Se elige cuando se busca **simplicidad y velocidad**.
    - Ejemplo: _Dancing_ → _danc_.
- **Lematización:**
    
    - Reduce las palabras a su **forma base correcta** (_lema_), asegurando que la raíz pertenezca al idioma.
    - Se elige cuando es **crítico mantener la validez de las palabras** y comprender el contexto.
    - Ejemplo: _Dancing_ → _dance_.


---

## 4. 🔢 Vectorización del Texto

La vectorización es la conversión de palabras en vectores numéricos, un requisito fundamental.

### 💼 4.1. Bag of Words (BoW)

- Se construye una tabla donde las **filas** son los documentos ($d$) y las **columnas** son los _tokens_ ($t$) del vocabulario.
- La entrada $f(t, d)$ es el **número de ocurrencias** del token $t$ en el documento $d$.

### ⚖️ 4.2. TFIDF

El **Term Frequency-Inverse Document Frequency** (TFIDF) pondera la importancia de una palabra:

$$\text{tfidf}(t, d, D) = \text{tf}(t, d) \cdot \text{idf}(t, D)$$

- **TF (Token Frequency):** Normaliza la frecuencia de la palabra en el documento.
- **IDF (Inverse Document Frequency):** Penaliza las palabras que aparecen en muchos documentos (palabras comunes).

### 🧠 4.3. One-hot vs. Word Embeddings

|**Característica**|**🟥 One-hot Word Vectors**|**🟦 Word Embeddings**|
|---|---|---|
|**Representación**|**Sparse** (dispersa) y de **alta dimensión**|**Densas** y de **menor dimensión**|
|**Relaciones**|**Ortogonales** (no capturan semántica)|Capturan la **estructura semántica**|
|**Origen**|Hardcoded|**Aprendidos** desde los datos|
|**Similitud**|Cada palabra es independiente|Palabras similares tienen **vectores cercanos**|

---

## 5. 🌍 Word Embeddings: Word2Vec

### 📐 5.1. Estructura y Semántica

Los _Word Embeddings_ son vectores densos que capturan la estructura del lenguaje, permitiendo **relaciones geométricas significativas**.

- **Ejemplo:** La diferencia vectorial entre "Queen" y "King" es similar a la diferencia entre "Woman" y "Man" (relación semántica de género).
- Los vectores reflejan relaciones **semánticas** (Rey vs Reina) y **sintácticas** (Grande vs Más Grande).

### 🏗️ 5.2. Modelo Word2Vec

**Word2Vec** es una arquitectura de red neuronal simple y eficiente para **aprender los _embeddings_**. Se basa en la hipótesis de que el significado de una palabra se puede inferir de su contexto.

#### 5.2.1. Arquitectura

El modelo utiliza una arquitectura básica de **dos capas** (input, hidden layer/embedding, output layer):

- **Input:** Codificado como _one-hot_ con dimensión igual al tamaño del vocabulario ($\text{vocab\_size}$).
- **Hidden Layer:** Es la capa de proyección, cuya matriz de pesos entre el input y ella **es la matriz de _Word Embeddings_**.
- **Output Layer:** Dimensión $\text{vocab\_size}$, que produce las probabilidades de las palabras de salida.

#### 5.2.2. Modos de Entrenamiento

Existen dos arquitecturas principales para entrenar los _embeddings_:

- **CBOW (Continuous Bag of Words):**
    
    - **Input:** La suma o promedio de los vectores _one-hot_ de las **palabras del contexto** circundante.
    - **Output:** Predice la **palabra central**.
    - **Lógica:** Dado un contexto, ¿cuál es la palabra que falta en el medio?.
        
- **Skip-gram:**
    - **Input:** El vector _one-hot_ de la **palabra central**.
    - **Output:** Predice las **palabras del contexto** circundante (se hace una predicción para cada palabra del contexto).
    - **Lógica:** Dada una palabra, ¿cuáles son las palabras que probablemente la rodean?.

---

# ✅ Conclusiones

- **NLP** resuelve desafíos como la **ambigüedad** y la **dependencia** del lenguaje para la comprensión automática.
    
- El **preprocesamiento** incluye **Tokenización**, eliminación de **Stop Words**, y **Reducción a la Raíz** (Lematización es más precisa que _Stemming_).
    
- La **Vectorización** tradicional usa **BoW** o **TFIDF**, pero no captura relaciones semánticas.
    
- Los **Word Embeddings** son la representación moderna: **vectores densos** que codifican el significado y las relaciones semánticas/sintácticas en un espacio geométrico.
    
- **Word2Vec** es el modelo base para aprender estos _embeddings_ a través de dos modos: **CBOW** (predice la palabra central dado el contexto) y **Skip-gram** (predice el contexto dada la palabra central).