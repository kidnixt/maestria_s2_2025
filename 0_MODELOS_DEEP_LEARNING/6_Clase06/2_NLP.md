# 📘 Apuntes - Deep Learning: Procesamiento del Lenguaje Natural (NLP)

---

## 1. 🧠 Introducción al NLP

El **Procesamiento del Lenguaje Natural (NLP)** busca que las computadoras **comprendan, interpreten y generen lenguaje humano**.

Principales aplicaciones:

- 🗣️ **Sentiment Analysis**
- 🌍 **Machine Translation**
- ❓ **Question Answering**
- 📰 **Text Summarization**
- 🧾 **Text Classification**
- 🔊 **Text-to-Speech**
- 🎧 **Speech Recognition**

👉 Todas estas tareas dependen de la **representación del lenguaje** en una forma numérica que los modelos puedan procesar.

---

## 2. 💬 Comprender el lenguaje natural

El lenguaje natural es **ambiguo, contextual y dependiente** del orden y de las relaciones gramaticales entre palabras.  
Los modelos de NLP deben ser capaces de:

- Capturar el **significado** más allá de las palabras individuales.
- Entender **dependencias** entre términos distantes.
- Manejar **polisemia, sinónimos y variaciones gramaticales**.

---

## 3. ⚠️ Dificultades del NLP

### 🌀 Ambigüedad

Una palabra puede tener múltiples significados según el contexto.

> Ejemplo: “banco” puede ser un asiento o una institución financiera.

### 🧩 Sinónimos

Distintas palabras pueden expresar el mismo concepto.

> Ejemplo: “feliz” y “contento”.

### 🔗 Dependencia

El significado de una palabra depende de las anteriores o siguientes.

> Ejemplo: “No me gusta el café” ≠ “Me gusta el café”.

---

## 4. 🎯 Enfoque del curso

Este teórico se centra en las **etapas fundamentales del preprocesamiento y representación del texto**, previas a entrenar modelos de Deep Learning:

1. Tokenización
2. Eliminación de _stop words_
3. Reducción a la raíz (stemming/lematización)
4. Vectorización del texto
5. Representaciones semánticas: embeddings

---

## 5. ✂️ Tokenización

Proceso de **dividir un texto en unidades básicas** llamadas _tokens_ (palabras, signos o subpalabras).

Ejemplo:
**Input:**

> “Los perros son animales muy leales y se llevan bien con las personas.”

**Output:**

> [“Los”, “perros”, “son”, “animales”, “muy”, “leales”, “y”, “se”, “llevan”, “bien”, “con”, “las”, “personas”, “.”]

📌 Estos _tokens_ son los bloques básicos para análisis posteriores (frecuencias, embeddings, etc.).

---

## 6. 🚫 Eliminación de _Stop Words_

Las _stop words_ son **palabras frecuentes y poco informativas** (artículos, preposiciones, pronombres).

Objetivo → eliminar ruido y centrarse en los términos relevantes.

**Ejemplo:**

Input:

> [“Los”, “perros”, “son”, “animales”, “muy”, “leales”, “y”, “se”, “llevan”, “bien”, “con”, “las”, “personas”, “.”]

Stop words: “Los”, “son”, “y”, “se”, “con”, “las”, “.”

Output:

> [“perros”, “animales”, “muy”, “leales”, “llevan”, “bien”, “personas”]

---

## 7. 🌱 Reducción a la raíz

Proceso que busca **unificar las variaciones gramaticales** de una palabra bajo una forma base.

### 🔹 Stemming

- Recorta sufijos/prefijos de manera **sintáctica**.
- Puede producir “raíces” que no son palabras reales.
- ✅ Rápido, ❌ menos preciso.

### 🔹 Lematización

- Usa **diccionarios y análisis morfológico**.
- Devuelve palabras válidas del idioma.
- ✅ Preciso, ❌ más costoso.

|Técnica|Ventaja|Desventaja|Ejemplo (“running”)|
|---|---|---|---|
|Stemming|Rápido|“raíces” inválidas|“runn”|
|Lemmatización|Correcta gramaticalmente|Más lento|“run”|

---

## 8. 🧪 Ejemplos de Stemming vs Lemmatization

|Original|Stemming|Lemmatization|
|---|---|---|
|Dancing, dancer, danced|[“danc”, “dancer”, “danc”]|[“dance”, “dancer”, “dance”]|
|Organization, organized|[“organ”, “organ”]|[“organization”, “organize”]|
|Happiness, happier, happiest|[“happi”, “happier”, “happiest”]|[“happiness”, “happy”, “happy”]|

📌 Lematización mantiene el sentido semántico correcto.

---

## 9. 🧮 Vectorización: Bag of Words (BoW)

Transforma un conjunto de documentos en **vectores numéricos** que representan las frecuencias de palabras.

- Filas → documentos
- Columnas → tokens del vocabulario
- Valor $(t, d)$ → número de ocurrencias de la palabra $t$ en documento $d$

Ejemplo:

|Documento|the|quick|brown|fox|dog|
|---|---|---|---|---|---|
|d1: “The quick brown fox jumped over the lazy dog.”|2|1|1|1|1|
|d2: “The dog hunts a fox.”|1|0|0|1|1|

👉 Es simple, pero **no considera el orden ni el contexto** de las palabras.

---

## 10. 📊 TF-IDF (Term Frequency – Inverse Document Frequency)

Pondera la frecuencia de una palabra según su **importancia relativa** en el corpus.

### Fórmulas:

- **Frecuencia del término (TF):**  
    $$  
    tf(t, d) = \frac{f(t, d)}{\sum_{t' \in d} f(t', d)}  
    $$
    
- **Frecuencia inversa del documento (IDF):**  
    $$  
    idf(t, D) = \log_2 \frac{|D|}{|{d \in D : t \in d}|}  
    $$
    
- **Combinación:**  
    $$  
    tfidf(t, d, D) = tf(t, d) \cdot idf(t, D)  
    $$
    

📌 TF-IDF reduce el peso de palabras muy frecuentes (_the, and, of, ..._) y resalta términos distintivos.

---

## 11. 📚 Ejemplo práctico de TF-IDF

Corpus:

```text
1. This is the first document.
2. This is the second second document.
3. And the third one.
4. Is this the first document?
```

→ Se genera una matriz donde cada fila es un documento y cada columna un término, con los valores ponderados por su relevancia TF-IDF.

Resultado: las palabras **únicas o menos frecuentes** (p. ej. “third”) tienen mayor peso.

---

## 12. 🧱 Limitaciones de BoW / TF-IDF

- Ignoran el **orden** de las palabras.
- No capturan **relaciones semánticas** (p. ej. “king” y “queen” son tratados como diferentes).
- Representaciones **sparse** y de alta dimensión.

💡 Solución: usar **Word Embeddings** → vectores densos que capturan significado.

---

## 13. 🧩 One-Hot Encoding vs Word Embeddings

|Aspecto|One-Hot Encoding|Word Embeddings|
|---|---|---|
|Tipo de vector|Disperso (sparse)|Denso|
|Dimensión|Igual al tamaño del vocabulario|Pequeña (50–300 típicamente)|
|Entrenamiento|Manual / estático|Aprendido automáticamente|
|Relaciones semánticas|❌ No capta similitud|✅ Palabras similares → vectores cercanos|

---

## 14. 🧭 Word Embeddings

Representan palabras como **vectores densos en un espacio continuo** donde la **distancia refleja relaciones semánticas**.

- Palabras similares → vectores cercanos
- Relaciones pueden representarse **geométricamente**:

$$  
\text{king} - \text{man} + \text{woman} \approx \text{queen}  
$$

Ejemplo de direcciones semánticas:

- “gender”: hombre ↔ mujer
- “pluralidad”: gato ↔ gatos
- “tiempo verbal”: correr ↔ corrió

![[Pasted image 20251103173027.png]]

---

## 15. 🧠 Formas de obtener Word Embeddings

1. **Entrenados desde cero:**  
    Se aprenden junto con la tarea principal (p. ej. clasificación de texto).  
    ➕ Adaptados a la tarea, ➖ requieren más datos.
    
2. **Pre-entrenados:**  
    Derivados de modelos entrenados en grandes corpus.  
    ➕ Reutilizables y eficientes (ej. Word2Vec, GloVe, FastText).
    

---

## 16. 🔬 Word2Vec (Mikolov et al., 2013)

Paper: _“Efficient Estimation of Word Representations in Vector Space”_  
👉 Propone un método simple y efectivo para aprender representaciones vectoriales a partir de texto.

Dos arquitecturas principales:

|Modelo|Idea|Objetivo|
|---|---|---|
|**CBOW** (_Continuous Bag of Words_)|Predice una palabra a partir del contexto|Maximizar $P(w_t \mid w_{t-m}, ..., w_{t+m})$|
|**Skip-Gram**|Predice el contexto a partir de una palabra|Maximizar $P(w_{t-m}, ..., w_{t+m} \mid w_t)$|

![[Pasted image 20251103173039.png]]

📈 Word2Vec se entrena sobre grandes corpus y produce **espacios semánticos coherentes**.

---

## 17. 🧩 CBOW vs Skip-Gram

|Característica|CBOW|Skip-Gram|
|---|---|---|
|Predicción|Palabra central|Palabras del contexto|
|Datos requeridos|Corpus grande|Corpus pequeño|
|Velocidad|Más rápido|Más lento|
|Ejemplo|Contexto: “el ___ come huesos” → “perro”|Palabra: “perro” → predice “el”, “come”, “huesos”|

---

## 18. 🧭 Aplicaciones de Word Embeddings

- Inicialización de modelos de NLP (LSTM, Transformers, etc.)
- Clasificación de texto y sentimientos
- Búsqueda semántica
- Detección de sinónimos y analogías
- Sistemas de recomendación basados en texto

---

# ✅ Conclusiones

- El **preprocesamiento del texto** (tokenización, limpieza, stemming/lematización) es esencial para representar lenguaje en forma numérica.
- **BoW y TF-IDF** fueron los primeros enfoques efectivos, pero no capturan **semántica ni contexto**.
- Los **Word Embeddings** introducen una representación **densa y continua**, que refleja relaciones semánticas entre palabras.
- **Word2Vec (CBOW/Skip-Gram)** marcó un hito en NLP al permitir aprender estas representaciones de manera no supervisada.
- Estas técnicas sentaron las bases para modelos más avanzados, como **RNNs**, **LSTMs** y **Transformers**, que utilizan embeddings como entrada para procesar secuencias.

