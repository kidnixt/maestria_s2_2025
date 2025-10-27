# 📘 Apuntes - Deep Learning: Natural Language Processing (NLP)

---

## 1. 🧠 Introducción al Procesamiento del Lenguaje Natural

El **Procesamiento del Lenguaje Natural (NLP)** busca que las máquinas puedan **entender, generar y razonar con texto humano**.

Ejemplos de tareas:
- ✍️ **Clasificación de texto:** análisis de sentimientos, detección de spam, etc.
- 💬 **Modelado del lenguaje:** predecir la próxima palabra o token.
- 🧩 **Traducción automática:** convertir texto de un idioma a otro.
- ❓ **Preguntas y respuestas:** sistemas tipo ChatGPT o asistentes virtuales.

---

## 2. 🔠 Representación del texto

El texto es discreto, pero los modelos neuronales operan con **vectores continuos**.  
Por eso, el primer paso es representar palabras como **vectores numéricos** (embeddings).

---

## 3. 📦 One-Hot Encoding

Representación más simple:  
Cada palabra se codifica como un vector binario con un `1` en la posición correspondiente.

Ejemplo con vocabulario de tamaño $V = 5$:
| Palabra | Vector |
|----------|--------|
| gato     | [1,0,0,0,0] |
| perro    | [0,1,0,0,0] |
| pez      | [0,0,1,0,0] |

💡 Problemas:
- Alta **dimensionalidad** (uno por palabra del vocabulario).
- No hay **relaciones semánticas** entre palabras.
  - Ejemplo: `gato` y `perro` son totalmente ortogonales, aunque sean semánticamente similares.

---

## 4. 🌌 Word Embeddings

Solución: representar las palabras en un **espacio vectorial continuo** de baja dimensión (por ejemplo, 100 o 300).

Cada palabra se asocia a un vector $w_i \in \mathbb{R}^d$.

Ventajas:
- Palabras con significado similar tienen **vectores cercanos**.
- Las relaciones semánticas se pueden expresar de forma **geométrica**:
  $$
  \text{Rey} - \text{Hombre} + \text{Mujer} \approx \text{Reina}
  $$

Modelos clásicos:
- **Word2Vec** (Mikolov et al., 2013)
- **GloVe** (Pennington et al., 2014)
- **FastText** (Bojanowski et al., 2016)

---

## 5. 🧩 Word2Vec: Skip-Gram y CBOW

### 🟢 Skip-Gram
Predice el **contexto** a partir de la palabra central.  
Ejemplo:
> “El gato duerme en la cama”  
Palabra central: “gato”  
Contexto: “El”, “duerme”, “en”

Función objetivo:
$$
\max_\theta \sum_{t} \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} | w_t)
$$

### 🔵 CBOW (Continuous Bag of Words)
Predice la **palabra central** a partir del contexto.

$$
\max_\theta \sum_{t} \log P(w_t | w_{t-c}, \dots, w_{t+c})
$$

Ambos modelos aprenden embeddings útiles para representar significado semántico.

---

## 6. ⚙️ Modelado del lenguaje (Language Modeling)

Un **modelo de lenguaje** estima la probabilidad de una secuencia de palabras:

$$
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t | w_1, \dots, w_{t-1})
$$

Objetivo: aprender una función que **prediga la próxima palabra** dada la secuencia previa.

Ejemplo:
> “El gato duerme en la” → “cama”

---

## 7. 🧮 Modelos N-gramas

Aproximan el contexto usando solo las últimas *n-1* palabras:

$$
P(w_t | w_1, \dots, w_{t-1}) \approx P(w_t | w_{t-n+1}, \dots, w_{t-1})
$$

Ejemplo:  
Trigrama ($n=3$):
$$
P(\text{"hace"} | \text{"hoy"}, \text{"no"}) \approx P(\text{"hace"} | \text{"hoy"}, \text{"no"})
$$

💡 Problemas:
- No generalizan bien a secuencias nuevas.
- El número de combinaciones crece **exponencialmente** con *n*.
- Dificultad para capturar dependencias largas.

---

## 8. 🧱 Redes Neuronales para lenguaje

Para superar las limitaciones de los *n-gramas*, se usan modelos neuronales que aprenden representaciones continuas del contexto.

Tipos principales:
- 🔁 **RNN (Recurrent Neural Networks)**
- 🔄 **LSTM / GRU** (versiones mejoradas de RNN)
- ⚡ **Transformers**

---

## 9. 🔁 Recurrent Neural Networks (RNN)

Las RNN procesan texto **secuencialmente**, manteniendo un **estado oculto** que resume la información previa.

Ecuaciones básicas:

$$
h_t = f(W_{xh}x_t + W_{hh}h_{t-1})
$$
$$
y_t = W_{hy}h_t
$$

Donde:
- $x_t$ = entrada (palabra en forma de embedding)
- $h_t$ = estado oculto en el tiempo $t$
- $y_t$ = salida (predicción)

💡 Permiten capturar dependencias en secuencia, pero presentan problemas de *vanishing gradient*.

---

## 10. ⚙️ LSTM y GRU

Extensiones de las RNN que incorporan **puertas de control** para manejar dependencias largas.

Ejemplo (LSTM):
- **Input gate**: decide qué nueva información almacenar.
- **Forget gate**: decide qué olvidar del estado previo.
- **Output gate**: decide qué parte del estado usar para la salida.

Resultado:
- Mejor capacidad para aprender **relaciones a largo plazo** en texto.

---

## 11. ⚡ Transformers

Modelo introducido por Vaswani et al. (2017):  
**“Attention Is All You Need”**

Elimina la recurrencia y utiliza un **mecanismo de atención** para modelar relaciones entre todas las palabras de la secuencia **en paralelo**.

💡 Cada palabra puede “atender” a cualquier otra palabra de la oración.

Ventajas:
- Computación paralela → más rápido.
- Capta **dependencias largas** de forma más directa.
- Escala bien a modelos enormes (GPT, BERT, etc).

---

## 12. 🧭 Mecanismo de Atención

Dada una secuencia de vectores $x_1, x_2, \dots, x_n$:

1. Se calculan tres matrices:
   - **Query (Q)**  
   - **Key (K)**  
   - **Value (V)**  

2. La atención se define como:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

Esto asigna **pesos de importancia** a cada palabra según su relación con las demás.

---

## 13. 🌍 Modelos de lenguaje preentrenados

Los grandes modelos modernos (GPT, BERT, etc.) son **preentrenados** en grandes corpus de texto y luego **ajustados (fine-tuned)** para tareas específicas.

Ejemplos:
- **BERT:** entrenamiento *bidireccional* con enmascaramiento.
- **GPT:** entrenamiento *autoregresivo* (predice la siguiente palabra).
- **T5:** modelo *encoder-decoder* para tareas de texto a texto.

---

## 14. 🧩 Tokenización

El texto debe dividirse en **tokens** antes de ser procesado.

Tipos comunes:
- Por palabra (“gato”, “perro”)
- Por subpalabra (“gato” → “ga”, “to”)
- Por carácter

Ejemplo de tokenización subpalabra (BPE, SentencePiece):
> “jugando” → “jug”, “ando”

Ventajas:
- Reduce el tamaño del vocabulario.
- Maneja mejor palabras raras o nuevas.

---

## 15. 📚 Aplicaciones del NLP moderno

- 💬 **Chatbots y asistentes virtuales**
- 🧾 **Resumen automático de texto**
- 🌐 **Traducción automática**
- 🕵️ **Análisis de sentimientos**
- 🔍 **Búsqueda semántica**
- 🧠 **Modelos generativos de texto (GPT, LLaMA, etc.)**

---

## 16. 🧠 Concepto de contextualización

En los modelos modernos (como BERT o GPT), el embedding de una palabra **depende del contexto**.

Ejemplo:
> “El **banco** del parque” vs. “El **banco** de inversión”

Antes: mismo vector (“banco”)  
Ahora: diferentes embeddings según el contexto → **representaciones contextualizadas**.

---

## 17. 🧩 Representaciones finales

- **Word2Vec / GloVe:** un vector por palabra → *no contextual*.
- **Transformers:** un vector distinto por palabra según contexto → *contextual*.

Estas representaciones son la base de todo el procesamiento moderno del lenguaje.

---

# ✅ Conclusiones

- El NLP busca representar y procesar el lenguaje natural con redes neuronales.  
- La representación del texto pasó de **vectores one-hot** a **embeddings continuos**.  
- Los modelos evolucionaron de **n-gramas → RNN → LSTM → Transformers**.  
- El **mecanismo de atención
