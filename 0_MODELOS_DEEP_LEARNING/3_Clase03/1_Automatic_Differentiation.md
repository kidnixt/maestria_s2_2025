# 📘 Apuntes - Deep Learning: Automatic Differentiation (AD)

---

![[Pasted image 20250918115547.png]]

## 1) 🎯 Idea central de AD

- Toda función en un modelo se compone de **operaciones elementales**:
  - suma, producto, exponencial, logaritmo, trigonométricas, etc.
- Las derivadas de estas operaciones son **conocidas**.
- Por la **regla de la cadena**, podemos obtener la derivada de toda la composición.
- AD trabaja sobre:
  - La **traza de evaluación** (orden en que se ejecutan operaciones).
  - El **grafo computacional** (dependencias entre operaciones).

💡 **Intuición:** AD no deriva “fórmulas” como cálculo simbólico, ni hace aproximaciones como las derivadas numéricas → sigue el flujo de operaciones y aplica la regla de la cadena de forma exacta.

---

## 2) 🕸️ Grafo computacional (ejemplo)

Sea:
$$
y = f(x_1, x_2) = \ln(x_1) + x_1x_2 - \sin(x_2)
$$

![[Pasted image 20250918115618.png]]

Definimos variables intermedias:
- $v_{-1} = x_1$, $v_0 = x_2$
- $v_1 = \ln v_{-1}$
- $v_2 = v_{-1} \cdot v_0$
- $v_3 = \sin v_0$
- $v_4 = v_1 + v_2$
- $v_5 = v_4 - v_3$
- $y = v_5$

💡 **Idea:** descomponemos la función en pasos atómicos → cada nodo del grafo tendrá su regla de derivada.

---

## 3) ➡️ AD Forward Mode

Queremos $\frac{\partial y}{\partial x_1}$.

- Asociamos a cada variable $v_i$ una tangente $\dot v_i = \partial v_i/\partial x_1$.
- Propagamos las derivadas usando la regla de la cadena, en el **mismo orden en que se evalúan**.

Inicialización:
- $\dot x_1 = 1$ (porque derivada de $x_1$ respecto a sí mismo es 1).
- $\dot x_2 = 0$ (independiente de $x_1$).

Resultado:
- $\dot y = \partial y/\partial x_1$.

💡 **Intuición:** se propaga “junto” al valor normal una especie de “valor sombra” que representa la derivada respecto a $x_1$.

![[Pasted image 20250918115714.png]]

---

## 4) 🔢 Números duales

Una forma elegante de implementar AD Forward es con **números duales**:

- Representación: $v + \dot v \,\varepsilon$, donde $\varepsilon^2 = 0$ pero $\varepsilon \neq 0$.
- Evaluar una función:
$$
f(v + \dot v \varepsilon) = f(v) + f'(v)\,\dot v \,\varepsilon
$$

Ejemplo de producto:
$$
[f(v) + f'(v)\dot v \varepsilon] \cdot [g(v) + g'(v)\dot v \varepsilon]
= f(v)g(v) + \big(f'(v)g(v) + f(v)g'(v)\big)\dot v \varepsilon
$$

💡 **Regla de la cadena** se conserva naturalmente:
$$
f(g(v + \dot v \varepsilon)) = f(g(v)) + f'(g(v))g'(v)\dot v \varepsilon
$$

👉 Así, al evaluar con duales, las derivadas se “propagan automáticamente”.

---

## 5) ⬅️ AD Reverse Mode

- Se usa mucho en Machine Learning (incluye **backpropagation**).
- Tiene **dos fases**:
  1. **Forward pass:** se evalúa la función, guardando $v_i$ y dependencias.
  2. **Backward pass:** se calculan **adjuntos** $\bar v_i = \partial y/\partial v_i$, y se propagan de la salida a las entradas.

Inicialización:
- $\bar y = 1$

Resultado:
- $\bar x_1 = \partial y/\partial x_1$
- $\bar x_2 = \partial y/\partial x_2$

💡 **Diferencia con Forward:** ahora la propagación va **hacia atrás** en el grafo.

![[Pasted image 20250918115746.png]]

---

## 6) ⚖️ Forward vs Reverse

Para $f: \mathbb{R}^n \to \mathbb{R}$, una sola pasada **reverse** produce $\nabla f = (\partial y / \partial x_i)^n_{i=1}$. 

Para $f: \mathbb{R}^n \to \mathbb{R}^m$: si $\text{ops}(f)$ es el costo de evaluar $f$, el Jacobiano cuesta 

* $n \cdot c \cdot \text{ops}(f)$ en **AD forward mode**. * $m \cdot c \cdot \text{ops}(f)$ en **AD reverse mode**.
💡 **Conclusión:**
- Si $m \ll n$ (muchas entradas, pocas salidas → típico en ML), conviene **reverse mode**.
- Desventaja: requiere mucha memoria (hay que guardar los $v_i$ intermedios).




---

## 7) 🔙 Backpropagation como AD

- El **algoritmo de backpropagation** en redes neuronales es un caso particular de **AD Reverse Mode**.
- Se propaga el error desde la salida hasta las entradas, aplicando la regla de la cadena de manera eficiente.

---

## 8) ⚙️ AD en PyTorch

En PyTorch, cada `Tensor` tiene:
- `data`: valor numérico.
- `grad`: gradiente acumulado.
- `grad_fn`: puntero al nodo del grafo que lo creó.

Ejemplo:
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.log(x) + x**2 - torch.sin(x)
y.backward()   # Calcula dy/dx
print(x.grad)  # gradiente en x=2
