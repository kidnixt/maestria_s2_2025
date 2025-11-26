# 📘 Apuntes – Deep Learning: Modelos de Lenguaje (LMs)

---

## 1. 🧠 ¿Qué es un Modelo de Lenguaje?

Un **Modelo de Lenguaje (Language Model, LM)** es un modelo que intenta **predecir la siguiente palabra** dada una secuencia previa.

Ejemplos intuitivos:

- “Necesito un vaso de …” → _agua_, _cerveza_, _vino_, etc.
- El LM asigna **probabilidades** a cada posible palabra siguiente.

Formalmente, dado:

$$  
x^{(1)}, x^{(2)}, \ldots, x^{(t)},  
$$

queremos modelar:

$$  
P(x^{(t+1)} \mid x^{(t)}, x^{(t-1)}, \ldots, x^{(1)}),  
$$

donde:

- $x^{(i)} \in V$
- $V$ = vocabulario de tamaño $|V|$.

Este problema es la base de **autocompletado**, **traducción**, **chatbots**, **speech recognition**, etc.

---

## 2. 📚 N-Gramas: El enfoque clásico en NLP

Antes de Deep Learning, los LMs estaban dominados por métodos **estadísticos**.

### 2.1 📏 Definición

Un **N-grama** es una secuencia de **N tokens consecutivos**.

Ejemplo con “I need a glass of …”

- Unigramas: I | need | a | glass | of
- Bigramas: I need | need a | a glass | glass of
- Trigramas: I need a | need a glass | a glass of

Idea fundamental:

> “La próxima palabra depende de las últimas N−1 palabras”.

---

## 3. 📊 Probabilidad con N-Gramas

Aproximamos:

$$  
P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})  
\approx  
P(x^{(t+1)} \mid x^{(t)}, x^{(t-1)}, \ldots, x^{(t-N+2)}).  
$$

Ejemplo para un trigram:

$$  
P(x^{(t+1)} \mid x^{(t)}, x^{(t-1)}).  
$$

Estimación a partir de frecuencias:
 
$$  
P(x^{(t+1)} | x^{(t)}, \ldots, x^{(t-N+2)})

\frac{\text{count}(x^{(t-N+2)}, \ldots, x^{(t)}, x^{(t+1)})}  
{\text{count}(x^{(t-N+2)}, \ldots, x^{(t)})}.  
$$

---

## 4. 🧪 Ejemplo numérico de 4-gramas


![[Pasted image 20251126142604.png]]
Supongamos:

- count(a glass of) = 1000
- count(a glass of water) = 400
- count(a glass of milk) = 100

Entonces:

$$  
P(\text{water} \mid \text{a glass of}) = 0.4  
$$

$$  
P(\text{milk} \mid \text{a glass of}) = 0.1  
$$

---

## 5. ⚠️ Problemas de los N-Gramas

1. **N pequeño → modelo limitado**
    - Solo captura dependencias cortas.
    - “Corto de vista”.
2. **N grande → explosión combinatoria**
    - Costo exponencial en $|V|^N$.
    - La mayoría de los n-gramas **sparse**: casi nunca aparecen.
3. **Lenguas flexibles → peor desempeño**
    - Idiomas con orden de palabras variable generan combinaciones muy dispersas.
4. **Oraciones largas inrepresentables**
    - No importa cuán grande sea N, siempre existe una oración que supera esa ventana.


---

## 6. 🧠 LMs con Redes: MLPs (Ventanas fijas + Embeddings)

Para mejorar los n-gramas, se introduce un **MLP** que recibe una ventana fija de tamaño N.

### Arquitectura general:

Cada palabra se representa como un embedding $e$:

$$  
e = [e_1; e_2; \ldots; e_N]  
$$

Luego:

**Capa oculta:**

$$  
h = f(We + b)  
$$

**Distribución de salida:**

$$  
\hat{y} = \text{softmax}(Uh + b_2) \in \mathbb{R}^{|V|}  
$$

![[Pasted image 20251126142729.png]]

### ✔️ Ventajas:

- Ya no necesitamos almacenar todos los n-gramas.
- No sufre de sparsity severo.
- Aprende representación vectorial de palabras (**embeddings**).

### ❌ Problemas:

- La ventana sigue siendo **chica y fija**.
- Aumentar ventana → aumenta mucho W.
- La red aprende funciones repetidas para posiciones distintas.

---

## 7. 🔄 De MLPs a RNNs

Los **Modelos Recurrentes de Lenguaje (RNN-based LMs)** resuelven las limitaciones del MLP con ventana fija.

### ✔️ Ventajas:

- **Escalan mejor**: número de parámetros no crece con la longitud de la oración.
- Pueden capturar **dependencias largas** (especialmente LSTM y GRU).
- Comparten pesos en cada paso temporal → una única función recurrente para todos los t.
- Más eficientes que n-gramas grandes.
- Dominaron el campo del NLP por una década.

### Funcionamiento general:

En cada paso $t$:

1. Reciben $x^{(t)}$
2. Actualizan estado oculto $h^{(t)}$
3. Producen distribución:

$$  
P(x^{(t+1)} \mid h^{(t)})  
$$

---

## 8. 🎯 Predicción y Entrenamiento en RNNs

La salida del modelo es una distribución sobre el vocabulario:

- La palabra predicha es aquella con mayor probabilidad.
- Se acumulan los errores de todas las posiciones para entrenar.

### 🧪 Teacher Forcing

Durante entrenamiento:

- Se le da al modelo la palabra **correcta** como siguiente input.
- Acelera el aprendizaje al evitar que los errores acumulativos “destruyan” la secuencia generada.

Ejemplo:

- En “the quick brown fox…”
- El modelo predice $P(\text{brown} \mid \text{the, quick})$.
- Si se equivoca, igualmente se le da “brown” como siguiente input.

---

# ✅ Conclusiones

- Los **Modelos de Lenguaje** buscan estimar  $$ P(x^{(t+1)} | x^{(1)}, \ldots, x^{(t)}) $$  
    y son el corazón del NLP moderno.
    
- Los **N-gramas** fueron la primera solución efectiva, pero sufren de sparsity y ventana rígida.
- Los **MLPs con embeddings** mejoran la representación, pero mantienen ventana limitada.
- Las **RNNs** (incluyendo LSTM/GRU):
    
    - escalan mejor,
    - capturan dependencias largas,
    - comparten pesos a través del tiempo,
    - dominaron el NLP hasta los Transformers.
        
- Conceptos clave:
    
    - Ventanas fijas vs. dependencia larga
    - Sparsity
    - Embeddings
    - Teacher forcing
    - Softmax sobre el vocabulario
        
