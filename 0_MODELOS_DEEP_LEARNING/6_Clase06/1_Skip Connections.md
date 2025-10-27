# 📘 Apuntes - Deep Learning: Skip Connections (ResNet y DenseNet)

---

## 1. 🎯 Motivación

Supongamos tres secuencias de capas $f_1$, $f_2$, $f_3$, y definimos dos modelos:

$$
g_1(x) = (f_3 \circ f_1)(x), \quad g_2(x) = (f_3 \circ f_2 \circ f_1)(x)
$$

Si $f_2(x) \approx x$, entonces $g_2(x) \approx g_1(x)$.

👉 En teoría, un modelo más profundo **debería poder rendir tan bien como uno más superficial**, ajustando $f_2$ a la identidad.

❌ En la práctica, **esto no ocurre**: redes más profundas a menudo tienen **mayor error de entrenamiento y validación**, incluso con más capacidad.

---

## 2. 📉 Mayor profundidad ≠ menor error

Experimento en CIFAR-10:

- Red “plana” de **20 capas**
- Red “plana” de **56 capas**

Resultado:
- La red de 56 capas tiene **mayor error de entrenamiento** y también **mayor error de validación**.

👉 El aumento de profundidad **no garantiza** mejor desempeño, y puede empeorar el entrenamiento.

---

## 3. ⚠️ Problemas de redes profundas

### 🧩 Vanishing gradients
- En redes profundas, los gradientes se propagan multiplicando derivadas muchas veces (regla de la cadena).
- Si los gradientes son pequeños (<1), se **van haciendo cada vez más pequeños**.
- Esto hace que las **primeras capas aprendan muy lentamente**.

### 🌀 Consecuencias
- Las capas iniciales tienen **poca influencia** sobre la salida final.
- Entrenamiento ineficiente o directamente imposible para redes muy profundas.

👉 Necesitamos una forma de **facilitar el flujo de gradientes** hacia capas anteriores.

---

## 4. 🧱 Solución: conexiones "skip" o residuales

**Idea:** permitir que la información “salte” capas intermedias conectando directamente una capa con otra más adelante.

Dos tipos principales:

1. **Concatenación:** la salida se une apilando canales.  
2. **Suma:** la salida se suma directamente al input original.

Estas conexiones ayudan a:
- **Propagar gradientes** más fácilmente.
- **Acelerar el aprendizaje**.
- **Evitar el estancamiento** en redes profundas.

---

## 5. 🔁 Conexión residual: idea básica

En lugar de aprender una función compleja $H(x)$ directamente, aprendemos una **función residual**:

$$
F(x) = H(x) - x
$$

Luego la salida del bloque es:

$$
H(x) = F(x) + x
$$

🎯 El modelo aprende **la desviación respecto de la identidad**.  
Esto facilita el entrenamiento porque:
- Si el bloque no aprende nada útil, puede aproximar $F(x) = 0$ → $H(x) = x$ (identidad).
- Esto elimina la necesidad de "aprender la identidad" desde cero.

---

## 6. 💡 Idea clave: sesgo hacia la identidad

Al estructurar el modelo como:

$$
H(x) = F(x) + x
$$

la red está **sesgada hacia funciones identidad**.

- $F(x)$ modela **pequeñas correcciones** a la identidad.
- Esto simplifica el proceso de optimización y permite entrenar redes con cientos de capas.

👉 Así nacen las **Redes Residuales (ResNets)**, basadas en bloques residuales.

---

## 7. ⚙️ Compatibilidad de dimensiones y normalización

Las conexiones residuales funcionan bien junto con **Batch Normalization** dentro del camino residual.

Condiciones para poder sumar:

$$
\text{shape}(F(x)) = \text{shape}(x)
$$

Si no coinciden las dimensiones (por ejemplo, cambia el número de canales), se puede **ajustar $x$ con una convolución 1×1**:

$$
H(x) = F(x) + \text{Conv2D}_{1×1}(x)
$$

Este truco ajusta los canales sin afectar la estructura espacial.

---

## 8. 🧩 Diseño básico del bloque residual

Versión simple con BatchNorm:

$$
h = ( \text{ReLU} \circ \text{BN} \circ \text{Conv2D} )(x) + x
$$

Sin embargo, como ReLU es no negativa, $h \ge x$ elemento a elemento.  
Esto introduce una **asimetría**: los valores pueden crecer pero no decrecer.

🔧 Para corregirlo, se **elimina la última activación**, quedando:

$$
h = (\text{BN} \circ \text{Conv2D} \circ \text{ReLU} \circ \text{BN} \circ \text{Conv2D})(x) + x
$$

Así, el bloque puede producir tanto incrementos como decrementos respecto a $x$.

---

## 9. 🧱 Ejemplo: Bloque original de ResNet

Estructura típica:

Entrada (64 canales)  
│  
├── Conv 1×1, 64 → ReLU  
├── Conv 3×3, 64 → ReLU  
├── Conv 1×1, 256  
│  
└── Suma con la entrada (skip connection)  
↓  
Salida (256 canales)


Este bloque es el componente básico de ResNet (He et al., 2016).

---

## 10. ➕ Suma de caminos residuales

Consideremos dos bloques residuales:

$$
h_1 = f_1(x) + x, \quad h_2 = f_2(h_1) + h_1
$$

Desenrollando:

$$
h_2 = f_2(f_1(x) + x) + f_1(x) + x
$$

🔍 Esto equivale a **sumar varios caminos**:
- Camino directo (x pasa sin cambios)  
- Camino con solo $f_1$  
- Camino con solo $f_2$  
- Camino con ambos $f_1$ y $f_2$

Cada bloque añade rutas adicionales por las que puede fluir información.

---

## 11. 🌲 Ensamble implícito

El número de caminos posibles **crece exponencialmente** con la cantidad de bloques residuales.

Las ResNets pueden interpretarse como un **ensamble implícito** de muchos modelos más pequeños, todos compartiendo parámetros.

👉 Esto explica parte de su capacidad de generalización y robustez.

---

## 12. 🧩 Efecto regularizador

- Las **superficies de costo** de una ResNet son más **suaves y estables** que las de una red plana.  
- Las conexiones residuales **suavizan la optimización** y actúan como una forma de regularización estructural.  
- Resultado: redes más profundas **entrenan mejor y generalizan mejor**.

---

## 13. 🧱 Arquitecturas ResNet originales

Características de ResNet (He et al., 2016):

- Bloques residuales con:
  - Convoluciones 3×3  
  - BatchNorm + ReLU  
  - Skip connections aditivas  
- Estructura jerárquica de bloques:
  - Bloques con 64, 128, 256, 512 canales  
  - Reducción de resolución (stride 2) al pasar de un bloque a otro  
- Clasificador final: global average pooling + fully connected.

Modelos clásicos:
- ResNet-18, 34 → versiones “simples”
- ResNet-50, 101, 152 → versiones profundas con bloques “bottleneck”

---

## 14. 📉 Ejemplo de desempeño

Entrenamiento en ImageNet:

- Comparación entre redes planas (18 y 34 capas) y ResNets equivalentes.

Resultados:
- Las **ResNets** logran **menor error de entrenamiento y validación**.
- Las redes planas sufren de saturación o divergencia.
- Con ResNet se pueden entrenar **modelos más profundos sin degradación**.

---

## 15. 🧬 Skip Connections por concatenación

Además de la suma, existe otro tipo de conexión: **concatenación**.

En lugar de sumar tensores, se **apilan a lo largo del eje de canales**:

$$
H(x) = [x, F(x)]
$$

- Incrementa el número de canales en la salida.
- La información de $x$ se mantiene explícitamente, no se mezcla aditivamente.

Este enfoque da origen a **DenseNet**.

---

## 16. 🌿 DenseNet: conexión residual por concatenación

DenseNet (Huang et al., 2017):

- Cada capa recibe **como entrada todas las salidas anteriores**, concatenadas:
  $$ x_l = [x_0, x_1, \dots, x_{l-1}] $$
- Permite una **reutilización sistemática de features**.
- Promueve una **mejor propagación de gradientes**.
- Reduce la cantidad de parámetros (por reutilización), aunque aumenta el costo de memoria.

---

## 17. ⚖️ ResNet vs. DenseNet: diferencias clave

| Aspecto | 🟦 ResNet | 🟩 DenseNet |
|----------|-----------|-------------|
| Tipo de conexión | Suma (+) | Concatenación ([·]) |
| Flujo de información | Aditivo | Acumulativo (explícito) |
| Canales | Se mantienen | Crecen capa a capa |
| Estructura | Bloques residuales | Bloques densos + transición |
| Propósito | Facilitar gradientes | Reutilizar características |
| Costo de memoria | Moderado | Mayor |
| Parámetros por capa | Más | Menos (por reutilización) |
| Año | 2016 (He et al.) | 2017 (Huang et al.) |
| Ejemplos | ResNet-18/34/50/101 | DenseNet-121/169/201 |

---

# ✅ Conclusiones

- **Skip connections** resuelven el problema del *vanishing gradient* y facilitan el entrenamiento de redes muy profundas.  
- Las **ResNets** modelan la *residual function* $F(x)$ y suman la entrada original ($x$).  
- Las **DenseNets** propagan información mediante *concatenación* de features.  
- En ambos casos:
  - Se mejora la propagación del gradiente.  
  - Se facilita la optimización.  
  - Se logra mejor generalización.  

👉 Las skip connections son una de las ideas más influyentes del Deep Learning moderno.

---
