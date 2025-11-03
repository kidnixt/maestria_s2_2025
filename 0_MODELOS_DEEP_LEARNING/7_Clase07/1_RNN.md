
# 📘 Apuntes - Deep Learning: Redes Neuronales Recurrentes (RNNs)

---

## 1. 🎯 Motivación

Las **Redes Neuronales Recurrentes (RNNs)** están diseñadas para procesar **datos secuenciales**, donde **el orden de los elementos importa**:

- Texto  
- Voz  
- Series temporales  

A diferencia de las redes *feedforward*, las RNNs incorporan **conexiones recurrentes**, lo que significa que:

- La **salida en un instante de tiempo** se retroalimenta como **entrada en el siguiente**.  
- Esto les permite **capturar dependencias temporales** y **mantener una forma de memoria** a lo largo del tiempo.

---

## 2. 🧩 Unidad recurrente y estado oculto

El componente básico de una RNN es la **unidad recurrente**, que mantiene un **estado oculto** (*hidden state*) — una forma de memoria que se actualiza en cada paso temporal:

$$
h_t = f(x_t, h_{t-1})
$$

La actualización depende de:
- la **entrada actual** $x_t$, y  
- el **estado oculto anterior** $h_{t-1}$.

Esta retroalimentación permite que la red **aprenda de entradas pasadas** y use ese conocimiento para interpretar la entrada presente.

---

## 3. ⚙️ Definición formal de una RNN

Una RNN se representa como una función:

$$
f_\theta : (x_t, h_t) \mapsto (y_t, h_{t+1})
$$

donde:
- $x_t$: vector de entrada,  
- $h_t$: vector de estado oculto,  
- $y_t$: vector de salida,  
- $\theta$: parámetros de la red neuronal.

En cada paso:
- La red transforma $x_t$ en $y_t$,  
- usando $h_t$ como **memoria**,  
- y **actualiza su estado interno** $h_{t+1}$.

![[Pasted image 20251103170709.png]]

---

## 4. 🔁 Ecuaciones de una RNN

Las actualizaciones en el tiempo se definen por:

$$
h_t = \text{Activation}(U x_t + V h_{t-1} + b_h)
$$

$$
y_t = \text{Activation}(W h_t + b_y)
$$

donde:
- $U$: pesos de la entrada al estado oculto,  
- $V$: pesos recurrentes (de $h_{t-1}$ a $h_t$),  
- $W$: pesos del estado oculto a la salida,  
- $b_h$, $b_y$: vectores de bias.

En conjunto:

$$
f_\theta : (x_t, h_{t-1}) \mapsto (y_t, h_t), \quad \theta = (W, U, V, b_h, b_y)
$$

---

## 5. ⚖️ Ventajas y desventajas de las RNNs

| Ventajas ✅ | Desventajas ❌ |
|-------------|----------------|
| Procesan **entradas de cualquier longitud** | Entrenamiento **lento** |
| **Pesos compartidos** a lo largo del tiempo | Difícil capturar dependencias muy largas |
| Consideran **información histórica** | No pueden usar información **futura** |
Como ventaja, el tamaño del modelo no aumenta con el tamaño de la entrada.

---

## 6. 📉 Función de pérdida en RNNs

Cada paso $t$ produce una predicción $\hat{y}_t$ y se calcula una **pérdida local**:

$$
\text{Loss}(\hat{y}_t, y_t)
$$

La pérdida total combina los errores de todos los pasos temporales:

$$
\text{Loss}(\hat{y}, y) = \sum_{t=1}^{T_y} \text{Loss}(\hat{y}_t, y_t)
$$

Esto implica que la red debe **mantener coherencia** en sus predicciones a lo largo de toda la secuencia, no solo en el último paso.

---

## 7. 🔙 Retropropagación a través del tiempo (BPTT)

La **Backpropagation Through Time (BPTT)** consiste en:

1. **Desplegar** la red a lo largo de varios pasos temporales (copias del mismo modelo con pesos compartidos).  
2. Aplicar **retropropagación estándar** sobre el grafo desplegado.  
3. Sumar las contribuciones de los gradientes de cada instante:

$$
\frac{\partial \text{Loss}(T)}{\partial W} = \sum_{t=1}^T \frac{\partial \text{Loss}(T)}{\partial W} \bigg|_{(t)}
$$

De esta forma, los gradientes fluyen hacia atrás a lo largo del tiempo.

![[Pasted image 20251103170828.png]]


---

## 8. ⏳ Horizonte temporal (k)

El **horizonte temporal $k$** define cuántos pasos hacia atrás se propaga el error.

- **BPTT completa**: $k = n$ (toda la secuencia).  
  → Preciso pero costoso e inestable.  
- **BPTT truncada**: $k < n$.  
  → Más eficiente, evita *vanishing/exploding gradients*.

📌 El valor de $k$ controla la **profundidad temporal del aprendizaje**.

---

## 9. 🧮 Algoritmo de BPTT (esquema)

1. Inicializar estado oculto $h_0 = 0$.  
2. Para cada posición $t$ en la secuencia:  
   - Propagar hacia adelante en $k$ pasos:  
     $(\hat{y}_{t+i}, h_{t+i}) = f(x_{t+i}, h_{t+i-1})$  
   - Retropropagar el error a través de las $k$ copias.  
   - Acumular gradientes y actualizar pesos:  
     $(W,U,V) \leftarrow (W,U,V) + \eta (\Delta W, \Delta U, \Delta V)$  


![[Pasted image 20251103170927.png]]

---

## 10. ⚠️ Problemas de las RNNs

### 🔺 Exploding gradients
Los gradientes pueden **crecer exponencialmente**, haciendo que el entrenamiento sea inestable.

### 🔻 Vanishing gradients
Los gradientes pueden **desaparecer** (tienden a 0), impidiendo aprender dependencias largas.

👉 Las RNNs *vanilla* tienden a **no retener memoria a largo plazo**, lo que limita su desempeño en tareas secuenciales extensas.

---

## 11. 🧠 Ejemplo: *Copying Task*

Una tarea diseñada para probar la **capacidad de memoria pura** de las RNNs.

### Objetivo:
- La red debe **memorizar una secuencia** de $S$ símbolos.  
- Luego de un retardo de $T$ pasos, debe **reproducirla exactamente**.

### Entrada:
1. **Secuencia inicial:** $S$ símbolos del alfabeto principal.  
2. **Brecha (delay):** $T-1$ pasos con símbolo *blank*.  
3. **Delimitador:** indica cuándo empezar a copiar.  
4. **Salida esperada:** repetir los $S$ símbolos originales.

---

## 12. ⚙️ Parámetros críticos

| Parámetro | Descripción | Valores típicos |
|------------|--------------|----------------|
| $K$ | Alfabeto principal | 8–10 símbolos |
| $S$ | Longitud de secuencia | 10 |
| $T$ | Longitud de brecha (retardo) | 500–1000 |

Si la red falla, la pérdida converge al valor aleatorio:

$$
L_{\text{rand}} = \ln(K + 2)
$$

Por ejemplo, para $K=8$,  
$$
L_{\text{rand}} = \ln(10) \approx 2.30
$$

---

## 13. 📊 Resultados típicos (T = 1000)

| Arquitectura | Pérdida            | Conclusión                                      |
| ------------ | ------------------ | ----------------------------------------------- |
| RNN Vanilla  | ≈ ln(K + 2) (≈2.3) | Falla → olvida la secuencia                     |
| LSTM         | ≈ 0.0              | Éxito → conserva la memoria perfectamente       |
| GRU          | ≈ 0.0              | Éxito → conserva la memoria de manera eficiente |

👉 Las LSTM y GRU logran **preservar dependencias a largo plazo**, donde las RNN tradicionales fallan.

---

## 14. 🧱 Variantes de RNN

### 🔹 Stacked RNN
- Varias capas recurrentes apiladas.  
- Permite aprender **representaciones jerárquicas** de secuencias.

![[Pasted image 20251103171147.png]]

### 🔹 Bidirectional RNN
- Procesan la secuencia **en ambas direcciones** (hacia adelante y hacia atrás).  
- Cada salida combina información **pasada y futura**.  
- Útil para tareas donde se conoce toda la secuencia (p. ej. NLP).

![[Pasted image 20251103171159.png]]

---

## 15. 🧩 LSTM (Long Short-Term Memory)

Las **LSTM** (Hochreiter & Schmidhuber, 1997) introducen **compuertas** que controlan el flujo de información:

| Gate | Función | Descripción |
|------|----------|-------------|
| Forget gate | $F_t$ | Decide qué información olvidar. |
| Input gate | $I_t$ | Determina cuánta nueva información agregar. |
| Update gate | — | Controla actualización de memoria. |
| Output gate | $O_t$ | Decide cuánta info exponer como salida. |

💡 La celda LSTM mantiene dos estados:
- **Cell state ($c_t$):** memoria a largo plazo.  
- **Hidden state ($h_t$):** salida momentánea.


![[Pasted image 20251103171237.png]]

---

## 16. 📐 Ecuaciones de la LSTM

$$
F_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

$$
I_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

$$
c_t = F_t \odot c_{t-1} + I_t \odot \tilde{c}_t
$$

$$
O_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

$$
h_t = O_t \odot \tanh(c_t)
$$

Cada gate regula dinámicamente el flujo de información y evita el *vanishing gradient*.

---

## 17. ⚙️ GRU (Gated Recurrent Unit)

La **GRU** simplifica la LSTM, reduciendo parámetros y mejorando eficiencia.

| Componente | Descripción |
|-------------|-------------|
| **Reset gate ($R_t$)** | Controla cuánta info pasada se ignora. |
| **Update gate ($Z_t$)** | Controla cuánto del estado anterior se mantiene. |

![[Pasted image 20251103171315.png]]

Ecuaciones:

$$
Z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
R_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
\tilde{h}_t = \tanh(W_h x_t + U_h (R_t \odot h_{t-1}) + b_h)
$$

$$
h_t = (1 - Z_t) \odot h_{t-1} + Z_t \odot \tilde{h}_t
$$

---

## 18. 💡 Intuición del GRU

GRU simplifica LSTM manteniendo solo dos compuertas:

- **Reset Gate:** Decide cuánta información del pasado debe ignorarse al generar el nuevo estado candidato.
- **Update Gate:** controla el balance entre mantener la memoria previa y agregar nueva información.

- Si $R_t$ es **pequeño**, el modelo “resetea” su memoria (ignora el pasado).  
- Si $Z_t$ es **grande**, el modelo **actualiza fuertemente** con nueva información.  
- Si $Z_t$ es **pequeño**, mantiene la memoria previa $h_{t-1}$.

🔁 En resumen:  
GRU = equilibrio entre **recordar** y **actualizar** memoria de forma más simple que LSTM.

---

## 19. 🧠 Ejemplo de uso: Clasificación de texto

1. **Word embeddings**: cada palabra → vector $x_t \in \mathbb{R}^d$.  
2. **LSTM cell**: procesa $(x_t, h_{t-1}, c_{t-1}) \to (h_t, c_t)$.  
3. **Representación global**: $h_T$ resume el contexto completo.  
4. **Capa densa**: $y = W h_T + b$.  
5. **Salida**: $\arg\max(y)$ → clase de sentimiento (positivo/negativo, etc.).

![[Pasted image 20251103171452.png]]

---

# ✅ Conclusiones

- Las **RNNs** son esenciales para modelar **secuencias** con dependencias temporales.  
- Sin embargo, las RNNs **vanilla** sufren de *vanishing/exploding gradients*.  
- La **BPTT truncada** permite un entrenamiento más estable.  
- **LSTM** y **GRU** introducen mecanismos de *gating* para preservar memoria a largo plazo.  
- **Bidirectional** y **Stacked RNNs** extienden su capacidad de representación.  
- Estas variantes son la base de modelos modernos en **NLP, voz y series temporales**.

👉 Las RNNs fueron un paso clave en el camino hacia arquitecturas más avanzadas como **Transformers**, que también buscan modelar dependencias secuenciales, pero con mecanismos de atención.

---

¿Querés que te lo deje también en formato **Markdown descargable (.md)** como los anteriores apuntes? Puedo generarlo enseguida.