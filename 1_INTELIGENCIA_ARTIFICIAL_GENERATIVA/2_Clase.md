# 📘 Apuntes - Inteligencia Artificial Generativa: Redes Bayesianas

📅 25 de Agosto, 2025  
👨‍🏫 Franz Mayr – Universidad ORT Uruguay  

---

## 1. 📊 Propiedades de las distribuciones (caso de 2 variables)

Sean $X, Y$ variables aleatorias:

- **Regla del producto:**  

$$
p(x, y) = p(x) \, p(y|x)
$$

👉 La probabilidad conjunta se puede descomponer en una **marginal** y una **condicional**.

- **Independencia:**  

$$
X \perp Y \quad \Longleftrightarrow \quad p(x,y) = p(x)\,p(y)
$$

👉 Dos variables son independientes si conocer una **no cambia** la probabilidad de la otra.

**Ejemplo simple:**  
- Sea $X$ = “tirar un dado” y $Y$ = “lanzar una moneda”.  
  Como son procesos independientes:  

$$
p(X=3, Y=H) = p(X=3)\,p(Y=H) = \tfrac{1}{6}\cdot\tfrac{1}{2}
$$

---

## 2. 📐 Tres variables y condicionales

Sean $X, Y, Z$ variables aleatorias:

- **Regla del producto:**  

$$
p(x,y,z) = p(x)\,p(y|x)\,p(z|x,y)
$$

- **Independencia condicional:**  

$$
X \perp Z \mid Y \quad \Longleftrightarrow \quad p(x,z|y) = p(x|y)\,p(z|y)
$$

- De esto se deduce:  

$$
X \perp Z \mid Y \;\;\Longrightarrow\;\; p(x,y,z) = p(x)\,p(y|x)\,p(z|y)
$$

**Ejemplo intuitivo:**  
- $X$ = “nublado”, $Y$ = “llueve”, $Z$ = “llevo paraguas”.  
- Dado $Y=\text{llueve}$, el hecho de que esté nublado o no ya no afecta a si llevo paraguas → independencia condicional.

---

## 3. 🧮 Generalización a $n$ variables

Para $X_1, \dots, X_n$:

- **Regla general del producto:**  

$$
p(x_1,\dots,x_n) = \prod_{i=1}^n p(x_i \mid x_1,\dots,x_{i-1})
$$

- Si cada variable depende **solo de la inmediata anterior**:  

$$
p(x_1,\dots,x_n) = \prod_{i=1}^n p(x_i \mid x_{i-1})
$$

👉 Este es un caso **Markoviano**: cada variable depende solo de la previa.  

**Ejemplo:** cadenas de Markov en modelado de lenguaje (cada palabra depende de la anterior).

---

## 4. 🕸️ Red Bayesiana

Una **red bayesiana** es:

- 📌 Un **grafo dirigido acíclico (DAG)**.  
- Cada nodo = una variable aleatoria $X_i$.  
- Un arco $i \to j$ indica que $X_i$ influye en $X_j$.  
- Cada nodo tiene una **distribución condicional** asociada:  

$$
p(x_i \mid \{x_j : j \to i\})
$$

La distribución conjunta se factoriza como:  

$$
p(x_1,\dots,x_n) = \prod_{i=1}^n p(x_i \mid \text{padres}(X_i))
$$

**💡 Ventaja:** Representa dependencias locales → evita calcular la distribución conjunta completa (exponencial en tamaño).

---

## 5. 📞 Ejemplo clásico: *Sally’s Alarm*

Variables:  
- $B$: Ladrón (Burglar)  
- $E$: Terremoto (Earthquake)  
- $A$: Alarma (Alarm)  
- $R$: Radio reporta terremoto  
- $J$: John llama  
- $M$: Mary llama  

**Tablas de probabilidad condicional (ejemplos):**  

- Probabilidad de que John llame depende de la alarma:  

$$
p(J=1|A=0)=0.05, \quad p(J=1|A=1)=0.90
$$  

- Probabilidad de que Mary llame depende de la alarma:  

$$
p(M=1|A=0)=0.01, \quad p(M=1|A=1)=0.70
$$  

- La alarma depende de ladrón y terremoto.  
- El radio depende solo de terremoto.

**Explicación:**  
- El grafo conecta causas → efectos.  
- La alarma puede sonar por un robo o un terremoto.  
- John y Mary reaccionan a la alarma, pero no directamente al robo o terremoto.  

👉 Esto permite **razonamiento probabilístico**:  
- Ejemplo: si suena la alarma y John llama, ¿cuál es la probabilidad de que haya un robo?  
- Se calcula con **inferencia bayesiana** usando la red.

---

## 6. 📚 Bibliografía clave

- Barber, D. *Bayesian Reasoning and Machine Learning*. Cambridge University Press, 2012 (Cap. 3).  

---

# ✅ Conclusión

- Las **redes bayesianas** representan dependencias probabilísticas entre variables usando un grafo.  
- Permiten **factorizar distribuciones conjuntas** en términos de condicionales simples.  
- Capturan conceptos de **independencia condicional**.  
- Ejemplo práctico: el caso de la alarma de Sally, usado en IA para mostrar cómo razonar con causas y efectos.  
- Son muy útiles en **IA generativa y explicativa**, ya que permiten **simulación, inferencia y aprendizaje** de relaciones probabilísticas.  
