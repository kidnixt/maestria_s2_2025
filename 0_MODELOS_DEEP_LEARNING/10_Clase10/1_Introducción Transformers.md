# 📘 Apuntes – Deep Learning: **Transformers (Parte 1)**

---

# 1. 🚀 De la Atención a los Transformers

- En 2017, Vaswani et al. publican **“Attention Is All You Need”**, introduciendo el **Transformer**.
- Descubren que **la atención por sí sola** puede reemplazar a las RNNs:
    - Capta dependencias largas sin recurrencia.
    - Permite **procesamiento paralelo** → entrenamiento mucho más rápido.
    - Escala de manera eficiente a grandes datasets.
- Resultado: revolución en NLP → luego visión, audio, multimodal, genAI.
    

👉 El Transformer es hoy la **base de todos los modelos modernos**, desde BERT hasta GPT-4/5, LLaMA y T5.

---

# 2. 🧠 Motivación del Self-Attention

Los **word embeddings clásicos** asignan un único vector fijo por palabra.  
Pero el significado es **contextual**:

- “go on a **date**” ≠ “mark the **date**”
- “see you **soon**” ≠ “see what you **mean**”

✦ Necesitamos representaciones **dependientes del contexto**.  
✦ El **self-attention** lo resuelve: cada palabra ajusta su representación en función de todas las demás del enunciado.

---

# 3. 🔎 ¿Cómo funciona el Self-Attention?

Para cada palabra:

1. Calcula **scores de relevancia** respecto a todas las otras (producto punto).
2. Esos scores indican qué tan relacionadas están.
3. Se aplica **softmax** para normalizar los pesos (suman 1).
4. Se realiza una **suma ponderada** de todos los vectores → nuevo vector contextual.

Ejemplo:

> En “The train left the station on time”, la palabra “station” incorpora información de “train”.

Resultado: **representaciones context-aware**.

---

# 4. 🔬 Intuición del mecanismo

Cada palabra "mira" al resto:

- Combina su información original + señales relevantes de otras palabras.
- Produce una representación dinámica, adaptada al contexto.

Esto permite captar **relaciones semánticas complejas**, incluso con distancia arbitraria.

---

# 5. ⚙️ Ejemplo Formal del Self-Attention

Cada token se proyecta a:

- **Query** $q$
- **Key** $k$
- **Value** $v$

Cálculo:
1. **Scores:**  
 $$ s_{ij} = q_i^\top k_j $$
2. **Normalización:**  
$$ \alpha_{ij} = \text{softmax}(s_{ij}) $$
3. **Salida contextual:**  
$$ z_i = \sum_j \alpha_{ij} v_j $$

🎯 El vector final **zᵢ** incorpora información de toda la oración.

---

# 6. 🎯 Dot-Product Attention

Objetivo: identificar qué partes del **input fuente** son relevantes para cada **palabra destino**.

Pasos:

- Para cada palabra de salida, se calcula similitud con cada palabra fuente.
- Softmax → distribución de atención.
- Suma ponderada de los vectores fuente.

Ejemplo:  
Al traducir “te traeré la bolsa”, el modelo asigna alto peso a “I”, “bring” al generar “I will bring”.

---

# 7. 📌 Queries, Keys y Values — Intuición

Analogía con un buscador:

- **Key = tags de las imágenes**
- **Query = texto de búsqueda**
- **Value = imágenes recuperadas**

El modelo busca qué keys son relevantes para un query y combina sus values.

---

# 8. 🧩 Multi-Head Attention

Un único head capta un solo tipo de relación.

Los Transformers usan **múltiples cabezas en paralelo**, cada una especializada en un patrón:

- Sujeto-verbo
- Dependencia larga
- Sentimiento
- Estructura sintáctica
- Resolución de coreferencia
- etc.

El proceso:

1. Cada head calcula su atención.
2. Se **concatenan** todas las salidas.
3. Se proyectan con una capa lineal.

👉 Esto produce representaciones **mucho más ricas y expresivas**.

---

# 9. 🏗️ Bloque **Encoder** del Transformer

El encoder usa **self-attention completo** (cada token atiende a todos).

Estructura interna:

1. **Multi-Head Self-Attention**
    - con máscara de **padding**
2. **Residual + LayerNorm**
3. **Feed-Forward Network** (dos capas densas y activación)
4. **Residual + LayerNorm**
    

Características clave:

- La salida tiene la **misma dimensión** que la entrada.
- Se pueden apilar **N encoders** para aumentar capacidad.
    

---

# 10. 🏛️ Bloque **Decoder** del Transformer

Similar al encoder, pero agrega mecanismos esenciales para generación autoregresiva.

Componentes:

1. **Masked Self-Attention**
    
    - Máscara causal → evita mirar tokens futuros.
    - Fundamental para tareas de generación (GPT-like).
        
2. **Cross-Attention**
    
    - Usa las representaciones del encoder como keys/values.
        
3. **Feed-Forward + Residual + LayerNorm**
    

El decoder genera la secuencia **token por token**, condicionándose en los tokens previos.

---

# 11. 🧱 Componentes Clave del Transformer

|Componente|Función|
|---|---|
|**Multi-Head Attention**|Modela relaciones ricas y paralelas.|
|**Feed-Forward Network**|Procesa cada token individualmente; agrega no linealidad.|
|**Residual Connections**|Evitan el problema del gradiente; facilitan redes profundas.|
|**Layer Normalization**|Estabiliza y acelera entrenamiento.|
|**Positional Encoding**|Introduce información de orden (fundamental en un modelo sin secuencialidad).|

---

# 12. 📚 Bibliografía Principal

- Vaswani et al. (2017). **Attention Is All You Need**.  
    NeurIPS 30.
    

---

# ✅ Conclusiones

- El Transformer elimina la recurrencia y usa **solo atención** para procesar secuencias.
    
- Self-attention produce representaciones **contextuales** superiores a los embeddings fijos.
    
- Multi-head attention permite capturar múltiples relaciones en paralelo.
    
- Encoder y decoder tienen estructuras similares, pero el decoder incorpora:
    
    - **masked attention** (causal)
        
    - **cross-attention** (usa la información del encoder)
        
- Es la arquitectura base de todos los modelos modernos de NLP y más allá (visión, audio, multimodal).
    

---

Si querés, seguimos con el siguiente PDF cuando lo tengas listo 📘🔥