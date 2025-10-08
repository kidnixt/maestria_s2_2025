# 📘 Apuntes - Deep Learning: Convoluciones (Parte 1)

---

## 1. 🧠 Por qué las capas densas no alcanzan (I)

Las imágenes se representan como tensores tridimensionales:

$$
X \sim (h, w, c)
$$

- $h$ = altura (rows)  
- $w$ = ancho (columns)  
- $c$ = número de canales (1 en escala de grises, 3 en RGB, etc.)

👉 Los ejes $h$ y $w$ tienen **estructura espacial** (vecindad entre píxeles), mientras que los canales **no tienen orden natural**.

Para aplicar una capa densa:
1. Se aplana la imagen:
   $$ \text{vect}(X) \in \mathbb{R}^{h\cdot w\cdot c} $$
2. Se aplica una transformación lineal:
   $$ a = A\big(\text{vect}(X)^\top W + b\big) $$

Esto **rompe la estructura espacial** original → cada píxel es tratado como una entrada independiente sin relación con su entorno.

---

## 2. 📈 Por qué las capas densas no alcanzan (II)

### 🧱 Desventaja 1: pérdida de estructura composicional

Aunque después se pueda "reformatear" la salida en forma de tensor, la **estructura espacial ya se perdió** en la capa densa.

### 📊 Desventaja 2: demasiados parámetros

Supongamos:
- Entrada: $X \sim (h,w,c)$  
- Salida: $H \sim (h,w,c')$

Una capa densa necesita:

$$
\text{parámetros} = (h w c) \times (h w c') = (h w)^2 c c'
$$

📌 Ejemplo:  
Si $h = w = 1024$, $c=3$, $c'=3$:

$$
(1024^2)^2 \times 3 \times 3 \approx 10^{13} \text{ parámetros}
$$

👉 Cada salida depende de todos los píxeles → ineficiente y costoso.

---

## 3. 🧪 Ejemplo 1D

Para 4 “píxeles” y 1 canal, la salida es:

$$
\begin{bmatrix} h_1 & h_2 & h_3 & h_4 \end{bmatrix}
=
\begin{bmatrix} x_1 & x_2 & x_3 & x_4 \end{bmatrix}
\begin{bmatrix}
W_{11} & W_{12} & W_{13} & W_{14} \\
W_{21} & W_{22} & W_{23} & W_{24} \\
W_{31} & W_{32} & W_{33} & W_{34} \\
W_{41} & W_{42} & W_{43} & W_{44}
\end{bmatrix}
$$

👉 Cada $h_i$ depende de **todos** los $x_j$.  
Esto motiva:
- Explorar **localidad espacial**
- **Reducir parámetros**
- **Compartir pesos**

---

## 4. 📍 Capas locales: intuición

En imágenes hay una **métrica natural**:

$$
d\big((i,j),(i',j')\big) = \max\big(|i - i'|, |j - j'|\big)
$$

La influencia entre píxeles **disminuye con la distancia**.

Definimos un **patch** centrado en $(i,j)$ con radio $k$:

$$
P_k(i,j) = X_{i-k:i+k,\, j-k:j+k,\, :}
$$

- Tamaño del patch: $(s, s, c)$ con $s = 2k + 1$ (kernel size).  
- Bien definido si $(i,j)$ está al menos a $k$ del borde.

👉 Un patch es la “ventana” que observa la red alrededor de un píxel.

---

## 5. 🧠 Capas locales: definición formal

Una capa $f$ con entrada $X \sim (h,w,c)$ y salida $H=f(X)\sim(h,w,c')$ es **$k$-local** si:

$$
H_{ij} = f\big(P_k(i,j)\big)
$$

👉 Cada activación depende solo de un **patch local**.

---

## 6. 🏗️ De capa densa a capa local

Original (densa, sin bias):

$$
a = A\big(\text{vect}(X)^\top W\big)
$$

Forzando localidad → anulamos pesos que conectan $(i,j)$ con píxeles fuera del patch:

$$
H_{ij} = A\big(\text{vect}(P_k(i,j))^\top W_{ij}\big)
$$

- Dimensión de $\text{vect}(P_k(i,j))$: $s^2 c$
- Si hay $c'$ salidas por píxel:
  $$ W_{ij} \sim (s^2 c, c') $$

Total de parámetros:

$$
\text{params} = h w s^2 c c'
$$

Comparado con la densa → reducción por un factor $\frac{s^2}{h w}$ ✅

---

## 7. 📏 Ejemplo con bordes (Zero-padding)

Ejemplo 1D, $k=1$ (kernel size $s=3$):

Entrada extendida con padding:

$$
\begin{bmatrix} 0 & x_1 & x_2 & x_3 & x_4 & 0 \end{bmatrix}
\cdot
\text{(matriz estructurada)}
$$

👉 El **zero-padding** permite aplicar el mismo patrón de pesos incluso en los bordes, manteniendo el tamaño de salida igual al de entrada.

---

## 8. 🔁 Equivarianza por traslaciones

Una capa local $f$ es **equivariante por traslaciones** si:

$$
P_k(i,j) = P_k(i',j') 
\quad \Rightarrow \quad 
f(P_k(i,j)) = f(P_k(i',j'))
$$

👉 Si trasladamos un patch en la imagen, la activación se traslada igual.  
Esto permite **detectar patrones independientemente de la posición**.

---

## 9. 🧠 Pesos compartidos ⇒ Convoluciones

**Idea clave:** usar **los mismos pesos $W$ en todas las posiciones $(i,j)$**:

$$
H_{ij} = A\big(\text{vect}(P_k(i,j))^\top W + b\big)
$$

- $W \sim (s^2 c, c')$ (independiente de $h$ y $w$)  
- $b \sim (c')$

👉 Número de parámetros **no crece con la resolución**.  
👉 **Peso compartido + localidad** = **convolución**.

---

## 10. 🧮 Ejemplo 1D con pesos compartidos

Para $k=1$, la matriz de pesos tiene **patrón repetido**: cada posición aplica el mismo kernel desplazado.  
Esto equivale a aplicar una **convolución discreta 1D** clásica (estructura Toeplitz en forma matricial).

---

## 11. 🧱 Capa convolucional 2D: definición completa

Entrada: $X \sim (h,w,c)$  
Kernel size: $s = 2k+1$  
Número de canales de salida: $c'$

Para cada $(i,j)$:

$$
H_{ij} = \text{vect}(P_k(i,j))^\top W + b^\top
$$

- $W \sim (s^2 c, c')$  
- $b \sim (c')$

**Hiperparámetros**:
- $k$ (o $s$),  
- $c'$ (número de filtros),  
- uso o no de padding.

**Shape de salida**:
- Con padding $k$: $H \sim (h, w, c')$  
- Sin padding: $H \sim (h - 2k, w - 2k, c')$

👉 Las convoluciones combinan:
- **Localidad**  
- **Equivarianza por traslación**  
- **Peso compartido**  
→ Alta eficiencia y buen sesgo inductivo.

---

# ✅ Conclusiones

- Las **capas densas** no aprovechan la estructura espacial → parámetros excesivos.  
- Las **capas locales** reducen parámetros usando patches, pero aún tienen pesos distintos por posición.  
- Las **convoluciones** agregan **compartición de pesos**:
  - Eficientes en parámetros  
  - Equivariantes a traslaciones  
  - Independientes de la resolución  
- El **zero-padding** permite aplicar kernels en bordes sin perder tamaño.
