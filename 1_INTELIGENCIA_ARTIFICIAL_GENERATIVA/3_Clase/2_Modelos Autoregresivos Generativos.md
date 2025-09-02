# 📘 Apuntes - Modelos Autorregresivos Generativos

📅 2 de Septiembre, 2025

---
## 1. 🧠 ¿Qué son los modelos autorregresivos?

Los modelos autorregresivos 📈 son un tipo de **modelo generativo**. Su característica principal es que la probabilidad de un elemento en una secuencia de datos depende de los elementos que lo preceden. La idea es que el modelo aprende a predecir el siguiente dato basándose en el historial de los datos que ya ha visto.

- Se enfocan en **variables discretas**, por ejemplo, donde cada variable $X_i$ puede ser un valor binario (0 o 1).
- La probabilidad de cada variable $X_i$ se modela como una distribución de Bernoulli.
$$p(x_{i} | x_{<i}) = \mathcal{B}(f_i(x_{<i}; W_i))$$
- La función $f_i$ es una función que toma como entrada todos los elementos anteriores $x_{<i}$ (es decir, $[x_1, \dots, x_{i-1}]$) y devuelve la probabilidad de que $x_i$ sea 1.
- **$W_i$** representa los parámetros del modelo.

---
## 2. 🧱 Modelos Lineales (Belief Network)

Los modelos lineales son una de las formas más simples de implementar un modelo autorregresivo.

- La función $f_i$ es una **función sigmoide** aplicada a una combinación lineal de los datos de entrada.
$$f_i(x_{<i}) = \text{sigmoid}(w_i \cdot x_{<i} + b_i)$$
- **$w_i$** es un vector de pesos y **$b_i$** es un sesgo. Ambos son parámetros que el modelo aprende durante el entrenamiento.
- El producto punto $w_i \cdot x_{<i}$ pondera la importancia de cada elemento anterior ($x_1, \dots, x_{i-1}$) para predecir el siguiente.

---
## 3. 🕸️ MLP (Multilayer Perceptron)

Una forma más avanzada es usar un **MLP** para la función $f_i$. Esto le da al modelo más capacidad para capturar relaciones no lineales en los datos.

- En el caso de un MLP de una sola capa, la función $f_i$ se define en dos pasos:
1.  Se calcula una capa oculta ($h_i$) como una combinación lineal y una activación sigmoide de las entradas anteriores.
$$h_i = \text{sigmoid}(V_i \cdot x_{<i} + v_i)$$
2.  La salida final es otra sigmoide de una combinación lineal de la capa oculta.
$$f_i(x_{<i}) = \text{sigmoid}(a_i \cdot h_i + b_i)$$
- **$V_i$**, **$v_i$**, **$a_i$** y **$b_i$** son los parámetros que el modelo aprende.

---
## 4. 🚀 Neural Autoregressive Distribution Estimator (NADE)

**NADE** es un modelo autorregresivo que mejora la eficiencia computacional de los MLPs tradicionales para este tipo de tarea.

- En lugar de usar parámetros diferentes ($V_i, a_i$) para cada variable $x_i$ (como en el MLP simple), NADE usa un **conjunto compartido de parámetros**.
- La capa oculta $h_i$ para la i-ésima variable se calcula utilizando solo las primeras $i-1$ columnas de una matriz de pesos global **V**.
$$h_i = \text{sigmoid}(V_{(\cdot, <i)} \cdot x_{<i} + v)$$
- La salida $f_i(x_{<i})$ sigue siendo la misma que en el MLP:
$$f_i(x_{<i}) = \text{sigmoid}(a_i \cdot h_i + b_i)$$
- Esto hace que el modelo sea **más eficiente** porque no necesita aprender un nuevo conjunto de pesos para cada elemento en la secuencia, sino que los "reutiliza".

---
## ✅ Conclusión

- Los **modelos autorregresivos** ✍️ son una clase de modelos generativos que aprenden la distribución de probabilidad de los datos de forma secuencial, prediciendo cada elemento en función de los anteriores.
- La **regla de la cadena de probabilidad** es el principio matemático detrás de su funcionamiento.
- Pueden ser implementados con **modelos lineales** (simples) o **redes neuronales** como **MLPs** y **NADE** para capturar relaciones más complejas.
- **NADE** representa una mejora al usar parámetros compartidos para la capa oculta, lo que lo hace más eficiente para el cómputo.