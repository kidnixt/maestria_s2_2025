
# 📘 Apuntes – Deep Learning: Modelos Secuencia a Secuencia (Seq2Seq)

---

## 1. 🎯 Motivación: ¿qué es un modelo Seq2Seq?

Los **modelos Secuencia-a-Secuencia (Seq2Seq)** buscan aprender una función que transforme una secuencia de entrada:

$$(x_1, x_2, \ldots, x_T) $$

en una secuencia de salida potencialmente distinta en longitud:

$$  
(y_1, y_2, \ldots, y_{T'})  
$$

📌 Es una generalización de los language models:  
no predicen solo la siguiente palabra, sino una **secuencia completa**.

### 🔧 Aplicaciones

- Traducción automática
- Resumen automático
- Subtitulados
- Chatbots
- Sistemas de pregunta–respuesta
- Cualquier tarea input→output basada en secuencias

---

## 2. ⚠️ Desafíos en Seq2Seq

### 2.1 🧩 Longitudes diferentes

Las secuencias de input y output pueden no coincidir:

- “Where is the train station?”  
    → “¿Dónde está la estación de trenes?”

### 2.2 🔄 Diferencias en el orden

- “The red house”  
    → “La casa roja”

El orden relevante cambia entre idiomas.

### 2.3 🧠 Necesidad de memoria a largo plazo

Preguntas como:  
“¿De quién era el cumpleaños?”  
requieren recordar elementos anteriores.

### 2.4 🔁 Evitar loops infinitos

Si no definimos inicio y final:  
la red puede generar secuencias interminables.

---

## 3. 🧰 Soluciones iniciales

### 3.1 🔖 Tokens especiales

Agregar:

- **SOS** → _inicio de secuencia_
- **EOS** → _fin de secuencia_

Permiten saber cuándo empezar y cuándo terminar de generar.

### 3.2 🔁 Uso de RNNs

Las RNN pueden:

- Recordar información del pasado
- Manejar secuencias de longitud variable
- Capturar patrones secuenciales complejos

👉 pero **NO** pueden manejar por sí solas input y output de longitud diferente.  
Necesitamos más estructura…

![[Pasted image 20251126143041.png]]

---

## 4. 🏗️ Arquitectura Encoder–Decoder

La solución estándar: **dividir el problema en dos redes**.

```
Entrada → ENCODER → vector de contexto → DECODER → salida
```

### 4.1 🧱 Encoder

Objetivo: **comprimir** toda la información relevante del input en un **vector de contexto**.

- Este vector suele ser el **estado oculto final** de una RNN (p. ej. LSTM/GRU).
- También podrían usarse outputs intermedios, pero en el modelo básico se usa solo el último estado.

![[Pasted image 20251126143119.png]]

### 4.2 🧱 Decoder

Objetivo: **generar la secuencia output paso a paso**.

- El estado inicial del decoder = el estado final del encoder  
 $$ 
    h^{(dec)}_0 = h^{(enc)}_T  
    $$
- Durante el entrenamiento, el input del decoder es:  
$$SOS, y₁, y₂, …, yₙ$$
- En inferencia:  
    el modelo se autoalimenta con sus propias predicciones, hasta llegar a **EOS**.



---

## 5. 🧠 Funcionamiento del Decoder

### 5.1 🔁 Entrenamiento

En cada paso temporal:

$$  
\text{input} =  
\begin{cases}  
\text{SOS} & t=1 \\  
\text{palabra real}; y_{t-1} & t>1  
\end{cases}  
$$

Luego el modelo predice $$ \hat{y}_t$$
![[Pasted image 20251126143152.png]]

### 5.2 🔮 Inferencia

El decoder recibe:

$$ \text{input}_1 = \text{SOS}$$

Genera una palabra → se realimenta → genera la siguiente → … hasta **EOS**.

Así produce secuencias de longitud arbitraria.

---

## 6. ⚔️ Training vs. Predicting

Durante **training**:

- recibe siempre la palabra correcta previa → aprendizaje estable.

Durante **predicting**:

- recibe sus propias predicciones → más realista, pero más difícil.

Esta diferencia provoca una brecha entre entrenamiento e inferencia.

---

## 7. 🎓 Teacher Forcing

Técnica clave en Seq2Seq.

### 📌 Definición

El decoder **SIEMPRE** recibe la **palabra real** del dataset como input, aunque su predicción anterior haya sido incorrecta.

### ⭐ Ventajas

- Entrenamiento más rápido
- Convergencia más estable

### ❗ Problema

Si se usa al 100%, el modelo:

- NO aprende a corregir sus propios errores
- Falla en inferencia porque nunca vio sus propios outputs

### 🔧 Solución práctica

**Teacher forcing parcial** (scheduled sampling):

- A veces usar la palabra real
- A veces usar la predicción del modelo
- Elegido aleatoriamente en cada paso

---

## 8. 🧮 Entrenamiento End-to-End

Aunque encoder y decoder sean redes separadas:

- se entrenan **juntas**, con una misma loss (normalmente cross entropy)
- la backpropagation fluye desde los outputs del decoder hacia **ambos** modelos

El costo total = suma de pérdidas en todos los pasos del decoder.

---

## 9. 🧩 Limitaciones de Seq2Seq básico (sin atención)

El modelo “clásico" encoder–decoder tiene un problema estructural:

> ❌ **Toda la información del input queda comprimida en un solo vector.**

Esto limita la capacidad de recordar secuencias largas.  
(Spoiler: esta limitación es la motivación del mecanismo de **Atención**, que aparece en el siguiente teórico.)

---

# 🧪 Ejemplo visual (resumen del PDF)

### Entrenamiento (Teacher Forcing)

```
[SOS, y₁, y₂, ..., yₙ] → Decoder → Predicciones → Loss
```

### Inferencia

```
SOS → predicción → realimentar → predicción → ... → EOS
```

---

# 🧷 Comparación: Encoder vs Decoder

|Componente|Rol|Estado inicial|Input en t|Output en t|
|---|---|---|---|---|
|🟦 Encoder|Procesar secuencia de entrada|h₀ = 0|x_t|h_t final (contexto)|
|🟩 Decoder|Generar secuencia de salida|h₀ = contexto del encoder|y_{t-1} (o pred. previa)|y_t|

---

# ✅ Conclusiones

- Los modelos **Seq2Seq** permiten transformar secuencias de entrada en secuencias de salida de **longitud variable**.
- Combinan dos redes:
    - **Encoder:** comprime la información del input en un vector.
    - **Decoder:** genera el output paso a paso.
        
- El mecanismo fundamental de entrenamiento es **teacher forcing**, pero debe combinarse con autoregresión para evitar problemas en inferencia.
- Seq2Seq se entrena **end-to-end** usando solo la loss del decoder.
- Sin atención, el vector de contexto limita la capacidad de manejar secuencias largas → lo cual motiva el siguiente tema: **mecanismos de Atención** y luego **Transformers**.
