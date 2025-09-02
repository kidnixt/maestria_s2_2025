# 📘 Apuntes - Variables Aleatorias Continuas

📅 2 de Septiembre, 2025

---
## 1. 🤓 Propiedades Fundamentales

Para entender cómo funcionan los modelos generativos, es crucial repasar las propiedades de la probabilidad.

- **Regla de la suma (o marginalización)**: Se utiliza para encontrar la distribución de probabilidad de una variable ($p_X(x)$) a partir de una distribución conjunta ($p_{XY}(x,y)$), sumando sobre todos los posibles valores de la otra variable ($y$). En el caso discreto, se expresa como:
$$p_{X}(x) = \sum_{y} p_{XY}(x,y)$$
- **Regla de la cadena (o del producto)**: Permite descomponer una distribución de probabilidad conjunta en el producto de una distribución marginal y una distribución condicional. Esta regla es fundamental en los modelos generativos para descomponer problemas complejos.
$$p_{XY}(x,y) = p_{Y}(y)p_{X|Y}(x|y)$$

---
## 2. 🧐 Observación Importante

Combinando las reglas anteriores, podemos expresar la distribución de una variable como una suma ponderada de distribuciones condicionales. La **regla de la suma** nos dice que la distribución marginal de $X$ puede calcularse sumando la distribución conjunta $p_{XY}(x,y)$ sobre todos los valores de $y$.
$$p_{X}(x) = \sum_{y} p_{XY}(x,y)$$
Si reemplazamos $p_{XY}(x,y)$ con la **regla de la cadena**, obtenemos una fórmula que relaciona la distribución marginal con una expectativa:
$$p_{X}(x) = \sum_{y} p_{Y}(y)p_{X|Y}(x|y) = \mathbb{E}_{y \sim p_{Y}}[p_{X|Y}(x|y)]$$
En otras palabras, la distribución de $X$ es el promedio de las distribuciones de $X$ condicionadas a cada valor posible de $Y$. Esta es una idea clave para los **modelos generativos mixtos**, que representan distribuciones complejas como una combinación de distribuciones más simples.

---
## 3. 📉 Distribuciones Continuas Comunes

En Deep Learning, se usan varias distribuciones de probabilidad continuas como bloques de construcción para modelos más complejos.

- **Distribución Uniforme** $\mathcal{U}(a,b)$: Todos los valores en un intervalo $[a,b]$ son igualmente probables.
$$p(x) = \frac{1}{b-a}\Pi_{a\le x\le b}$$
Donde $\Pi$ es la función indicatriz (vale 1 si $a \le x \le b$ y 0 en cualquier otro caso).
- **Distribución Normal (Gaussiana)** $\mathcal{N}(\mu, \sigma)$: Es la más común y se utiliza para modelar muchos fenómenos naturales. Está definida por su media ($\mu$) y desviación estándar ($\sigma$).
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]$$
- **Distribución de Laplace** $Lap(\mu, b)$: Similar a la Normal pero con picos más agudos en la media y colas más pesadas, lo que la hace útil para datos con outliers.
$$p(x) = \frac{1}{2b} \exp\left[-\frac{|x-\mu|}{b}\right]$$

---
## 4. 🔗 Distribuciones Conjuntas en la Práctica

La **regla de la cadena** es la herramienta principal para construir distribuciones conjuntas más complejas a partir de distribuciones más simples.

### ➡️ Ejemplo 1: Variable discreta y continua
- **Y discreta uniforme**, $Y \sim \mathcal{B}(p)$ (Dist. Bernoulli).
- **X continua normal**, condicionada por Y, $X|(Y=b) \sim \mathcal{N}(\mu_b, \sigma_b)$.
En este caso, la distribución de $X$ se aproxima por una **mezcla de dos distribuciones normales**. La fórmula sería:
$$X \sim p\mathcal{N}(\mu_1, \sigma_1) + (1-p)\mathcal{N}(\mu_0, \sigma_0)$$

### ➡️ Ejemplo 2: Variable categórica y continua
- **Y categórica**, $Y \sim \mathcal{C}(p_1, \dots, p_K)$.
- **X continua normal**, condicionada por Y, $X|(Y=k) \sim \mathcal{N}(\mu_k, \sigma_k)$.
La distribución de $X$ es una **mezcla de K distribuciones normales**, una por cada categoría.
$$X \sim \sum_k p_k\mathcal{N}(\mu_k, \sigma_k)$$

---
## 5. 📦 VAEs y Distribuciones Continuas

El **Variational Autoencoder (VAE)** es un ejemplo práctico de cómo se usan las distribuciones conjuntas en la IA generativa. En un VAE, tanto la variable latente $Y$ como la variable de salida $X|Y$ son **continuas y normales**.
- La variable latente $Y$ se asume que sigue una **distribución normal estándar**: $Y \sim \mathcal{N}(0,1)$.
- La variable de salida $X$, condicionada a la variable latente $Y$, también sigue una **distribución normal**: $X|(Y=y) \sim \mathcal{N}(M(y), S(y))$.
- Aquí, $M(y)$ y $S(y)$ son redes neuronales 🧠 que aprenden la media y la desviación estándar de la distribución de salida, respectivamente.
- Los VAEs usan esta estructura para aprender una representación comprimida de los datos y luego generar nuevas muestras a partir de esa representación latente.

---
## ✅ Conclusión

- La **regla de la cadena y la de la suma** son la base matemática para construir modelos generativos.
- Las **distribuciones continuas** (Normal, Uniforme, Laplace) son los bloques de construcción.
- Los modelos generativos complejos, como los VAEs, utilizan **distribuciones conjuntas** para modelar la relación entre datos y variables latentes.