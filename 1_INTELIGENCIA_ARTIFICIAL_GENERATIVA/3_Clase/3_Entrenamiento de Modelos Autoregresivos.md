# 📘 Apuntes - Entrenamiento de Modelos Generativos

📅 2 de Septiembre, 2025

---
## 1. 🧠 Divergencia de Kullback-Leibler (KL)

La divergencia de Kullback-Leibler, o **KL divergence** ($KL(p||q)$), es una medida de la diferencia o "distancia" entre dos distribuciones de probabilidad. Nos dice qué tan distinta es la distribución $q$ de la distribución $p$.

- La fórmula se define como una sumatoria (o integral) sobre todos los posibles valores de $x$:
$$KL(p||q) = \sum_{x} p(x) \log\left[\frac{p(x)}{q(x)}\right]$$
- También se puede expresar como una **esperanza** (valor promedio):
$$KL(p||q) = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)]$$
- **Propiedades clave**:
    - Es siempre no negativa: $KL(p||q) \ge 0$.
    - Es cero si y solo si las dos distribuciones son idénticas: $KL(p||q) = 0 \iff p=q$.
    - **No es simétrica**: $KL(p||q) \ne KL(q||p)$. Esto es importante porque el orden importa al medir la divergencia.
    - **No satisface la desigualdad triangular**: No es una distancia métrica en el sentido estricto.

---
## 2. 📉 Función de Pérdida

En el entrenamiento de modelos generativos, el objetivo es que la distribución que el modelo aprende ($p_\theta$) sea lo más parecida posible a la distribución real de los datos ($p$). Para lograr esto, usamos la **KL divergence** como una **función de pérdida**.

- La función de pérdida se define como la divergencia entre la distribución de los datos ($p$) y la distribución estimada por el modelo ($p_\theta$).
$$KL(p||p_\theta) = \mathbb{E}_{x \sim p}[\log p(x) - \log p_\theta(x)]$$
- Para entrenar el modelo, queremos **minimizar esta divergencia**. Esto se reduce a minimizar el término que depende de los parámetros del modelo, $\theta$.
$$ \min_{\theta} KL(p||p_\theta) = \min_{\theta} \mathbb{E}_{x \sim p}[-\log p_\theta(x)]$$
- El término $-\log p_\theta(x)$ se conoce como la **pérdida de probabilidad logarítmica negativa** (negative log-likelihood) y es la función de pérdida más común para este tipo de modelos. Minimizarla equivale a maximizar la probabilidad de los datos observados bajo el modelo.

---
## 3. 📊 Minimización del Riesgo Empírico

El problema anterior es teórico, ya que no conocemos la distribución real de los datos $p$. En la práctica, usamos un conjunto de datos (dataset) de tamaño $m$, $T \sim p(x)^m$.

- La **función de pérdida** para una única muestra $x$ es:
$$l(\theta, x) = -\log p_\theta(x)$$
- El **riesgo empírico** es la pérdida promedio sobre todo el conjunto de datos $T$. Es una aproximación de la esperanza teórica.
$$\mathcal{L}(\theta, T) = \frac{1}{m}\sum_{x \in T} l(\theta, x)$$
- El entrenamiento del modelo consiste en minimizar este riesgo empírico, es decir, encontrar los parámetros $\theta$ que minimicen la pérdida promedio en el conjunto de datos.

---
## 4. ✍️ Entrenamiento de Modelos Autorregresivos

Para los modelos autorregresivos, el entrenamiento se basa en la **regla de la cadena de probabilidad**. La probabilidad conjunta de una secuencia se descompone en el producto de las probabilidades de cada elemento, condicionado a los anteriores.

- La función de pérdida para una secuencia completa se puede descomponer usando la regla de la cadena. El logaritmo del producto se convierte en una suma de logaritmos.
$$\log p_\theta(x_1, \dots, x_n) = \log \prod_{i=1}^{n} p_{\theta_i}(x_i | x_{<i}) = \sum_{i=1}^{n} \log p_{\theta_i}(x_i | x_{<i})$$
- La pérdida para cada elemento de la secuencia es:
$$l_i(\theta, x) = -\log p_{\theta_i}(x_i | x_{<i})$$
- El **riesgo empírico** para un dataset de secuencias se calcula sumando las pérdidas individuales para cada elemento en cada secuencia y luego promediando.
$$\mathcal{L}(\theta, T) = \frac{1}{m}\sum_{x \in T}\sum_{i=1}^{n} l_i(\theta, x)$$

---
## ✅ Conclusión

- El entrenamiento de modelos generativos se basa en **minimizar la divergencia de KL** entre la distribución del modelo y la distribución real de los datos.
- En la práctica, esto se logra minimizando el **riesgo empírico** sobre un dataset.
- Para los modelos **autorregresivos**, el riesgo empírico se descompone en una suma de las pérdidas de probabilidad logarítmica negativa para cada elemento de la secuencia.