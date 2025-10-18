	# 📘 Apuntes - Deep Learning: Optimización (GD, Momentum, Nesterov y Adam)

---

## 1) 🎯 Problema de optimización en DL

- El **entrenamiento** de una red busca minimizar el **costo empírico**:
$$
J(\theta) = \frac{1}{N}\sum_{i=1}^N L(f(x_i;\theta), y_i)
$$
- Usamos **Gradient Descent (GD)** y variantes.
- El **learning rate** $\eta$ controla el tamaño de paso:
  - Si es muy chico → convergencia lenta.
  - Si es muy grande → divergencia u oscilaciones.

![[Pasted image 20250918120505.png]]

---

## 2) 🔒 Regularización

Objetivo: reducir **error de validación** (no solo entrenamiento) penalizando modelos complejos.

- Forma general:
$$
J_\lambda(\theta) = J(\theta) + \lambda \times \text{Penalización}(\theta)
$$
- $\lambda \geq 0$ controla la importancia.

### 📌 Regularización L2 (Weight Decay, Ridge)
$$
J_\lambda(\theta) = J(\theta) + \frac{\lambda}{2}\|\theta\|_2^2
$$
- Gradiente:
$$
\nabla J_\lambda(\theta) = \nabla J(\theta) + \lambda\theta
$$

### 🛠️ Actualización con L2 acoplado
Con gradiente $g_t = \nabla J(\theta_t)$:
$$
\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta g_t
$$

👉 En `torch.optim.SGD`, el argumento `weight_decay` implementa exactamente este término.

---

## 3) ⚙️ SGD completo (con extras)

Entrada: learning rate $\gamma$, parámetros iniciales $\theta_0$, weight decay $\lambda$, momentum $\mu$, dampening $\tau$, nesterov.

1. $g_t \leftarrow \nabla J(\theta_{t-1})$
2. Weight decay: $g_t \leftarrow g_t + \lambda\theta_{t-1}$
3. Momentum:
   - $v_t \leftarrow \mu v_{t-1} + (1-\tau)g_t$
   - Si Nesterov:
     $g_t \leftarrow g_t + \mu v_t$
     (sino, $g_t \leftarrow v_t$)
4. Actualización: $\theta_t \leftarrow \theta_{t-1} - \gamma g_t$

---

## 4) 🚀 Momentum: intuición

### 📌 Forma acumulada
Si $\lambda=0$, $\tau=0$ y Nesterov=False:
$$
\theta_{t+1} = \theta_t - \gamma \sum_{i=1}^t \mu^{t-i} g_i
$$
👉 Es un promedio ponderado exponencial de gradientes pasados.

![[Pasted image 20250918120627.png]]

### ⚖️ Interpretación física (Heavy-ball)
- Partícula sobre el paisaje $f(\theta)$:
$$
\dot\theta = v, 
\quad m \dot v = -\nabla f(\theta) - c v
$$
- Discretización:
$$
v_{t+1} = \mu v_t - \eta \nabla f(\theta_t), 
\quad \theta_{t+1} = \theta_t + v_{t+1}
$$
- Donde:
  - $\mu = 1 - \frac{c}{m}\Delta t$ (inercia)
  - $\eta = \frac{\Delta t}{m}$ (escala del empuje)

💡 **Interpretación:**
- Momentum suaviza zig-zag y acelera en direcciones coherentes.
- Fricción (amortiguamiento) evita oscilaciones.

---

## 5) 📐 Geometría en valles elípticos

- En la dirección “empinada” → gradiente cambia de signo → zig-zag.
- En la dirección del valle → gradiente coherente → se acelera.

Ecuación con dampening $\tau$:
$$
v_{t+1} = \mu v_t + (1-\tau)g_t
$$

- Efecto:
  - $\mu$: memoria de gradientes (~$\frac{1}{1-\mu}$ pasos).
  - $\tau$: recorta la inyección de gradiente nuevo → pasos más contenidos.

---

## 6) 🔮 Nesterov Accelerated Gradient (NAG)

Idea: **look-ahead** → calcular el gradiente en un punto adelantado por momentum.

1. $y_t = \theta_t - \eta \mu v_t$ (punto adelantado)
2. $\tilde g_t = \nabla f(y_t)$
3. $v_{t+1} = \mu v_t - \eta \tilde g_t$
4. $\theta_{t+1} = \theta_t + v_{t+1}$

💡 Diferencia: primero se “mira” el gradiente en el punto adelantado, luego se corrige con él.

![[Pasted image 20250918120733.png]]

---

## 7) 🐍 Nesterov en PyTorch

Regla usada en `torch.optim.SGD`:
- $g_t \leftarrow \nabla J(\theta_t) + \lambda\theta_t$
- $v_{t+1} \leftarrow \mu v_t + (1-\tau) g_t$
- $d_t = g_t + \mu v_{t+1}$ (si Nesterov=True)
- Actualización: $\theta_{t+1} = \theta_t - \eta d_t$

---

## 8) ⚡ Adam (Adaptive Moment Estimation)

Adam combina:
- Momentum (promedio de gradientes).
- Adaptación por coordenada (escala de gradientes al cuadrado).

Reglas:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Corrección de sesgo:
$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, 
\quad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

Actualización:
$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
$$

Hiperparámetros típicos:
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

---

## 9) 🛠️ Adam en la práctica

- Paso adaptativo **por coordenada** → cada parámetro tiene su propio learning rate efectivo.
- Costo: requiere guardar $m_t, v_t$ (más memoria).
- En Adam clásico, el L2 es acoplado ($+ \lambda\theta$).
- Pros: converge rápido, poco tuning.
- Contras: a veces se cambia luego a SGD+Momentum para mejor generalización.

---

# ✅ Conclusiones

- **Weight Decay (L2):** evita parámetros grandes, mejora generalización.
- **Momentum:** acelera aprendizaje en direcciones coherentes, reduce zig-zag.
- **Nesterov:** anticipa el gradiente, mejora la dirección de descenso.
- **Adam:** combina momentum + escalado adaptativo, muy usado en la práctica.
- En frameworks (PyTorch, TensorFlow) estos métodos ya están implementados y listos para usar.
