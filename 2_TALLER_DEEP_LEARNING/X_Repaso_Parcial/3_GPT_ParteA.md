
# PARTE A — Fundamentos de PyTorch, DataLoaders, Entrenamiento y Capas

## 1) Tensores — propiedades y transformaciones esenciales

**Idea clave:** un _tensor_ es un array N-D con `dtype`, `device`, `shape` y `ndim`. Es la unidad básica de datos en PyTorch.

Qué saber para el parcial:

- `dtype` (float32, int64, etc.) afecta precisión/memoria.
    
- `device` → `"cpu"` o `"cuda:n"`; mover con `.to(device)`.
    
- `shape` es una tupla (ej. `(B,C,H,W)`), `ndim = len(shape)`.
    
- `view` vs `reshape`: `view` necesita tensor _contiguous_; `reshape` intenta devolver view o copia.
    
- `squeeze(dim)` elimina dims de tamaño 1; `unsqueeze(dim)` inserta dim=1.
    
- `permute` y `transpose` cambian orden de ejes (pueden crear no-contiguos). Usar `.contiguous()` antes de `view` si hace falta.
    

Snippet rápido:

```python
x = torch.randn(8,1,28,28)    # shape (8,1,28,28)
x2 = x.squeeze(1)             # (8,28,28)
x3 = x2.unsqueeze(1)          # (8,1,28,28)
y = x.view(8,28*28)           # falla si no es contiguo
z = x.reshape(8,28*28)        # ok (view o copy)
```

Pitfalls:

- `view` en tensor no contiguo → RuntimeError.
    
- Inplace ops con broadcasting pueden fallar (`a += b`).
    

---

## 2) Broadcasting — reglas, ejemplos y consejos

**Idea clave:** broadcasting permite operar tensores con shapes distintas repitiendo dimensiones de tamaño 1 según reglas deterministas.

Reglas (concrete):

1. Alinear shapes por la **derecha**.
    
2. Para cada dimensión comparada: si son iguales → OK; si una es `1` → se repite; si una no existe → se considera `1`; si incompatibles → error.
    
3. Resultado: máximo por dimensión.
    

Ejemplo:

- `(4,3)` + `(3,)` → `(4,3)` porque `(3,)` se interpreta como `(1,3)` y se replica sobre dim0.
    

Uso común:

- Añadir bias `out + bias` donde `bias` es `(C,)` y `out` es `(N,C,H,W)`.
    
- Normalizaciones: restar `mean` con shape `(C,)`.
    

Operaciones útiles:

- `expand` (view sin copiar) vs `repeat` (copia). `expand` solo funciona si dimension a expandir es 1.
    

Pitfalls:

- Confundir orden al alinear (siempre por la derecha).
    
- Operaciones in-place con broadcasting pueden dar errores de asignación.
    

---

## 3) Dataset y DataLoader — diseño y performance

**Idea clave:** `Dataset` entrega ítems; `DataLoader` agrupa, paraleliza y ordena.

Qué implementar:

- `class MyDataset(Dataset): __len__(), __getitem__()`.
    

Parámetros esenciales de `DataLoader`:

- `batch_size`: tradeoff memoria/estabilidad.
    
- `shuffle`: mezclar para train.
    
- `num_workers`: procesos paralelos para cargar/preprocesar. Aumenta throughput; en Windows cuidado con pickling y `if __name__=="__main__"`.
    

Consejos:

- `pin_memory=True` para acelerar transferencias GPU.
    
- Usar `worker_init_fn` si necesitas seeds reproducibles.
    

Pitfalls:

- Lambdas o objetos no picklables en dataset -> fallos con `num_workers>0`.
    
- Demasiados workers saturan CPU / I/O.
    

---

## 4) Train / Eval / torch.no_grad — modos de modelo

**Idea clave:** `model.train()` activa comportamiento de entrenamiento (dropout y batchnorm), `model.eval()` los desactiva; `torch.no_grad()` apaga cálculo de gradiente.

Puntos para parcial:

- **Dropout** y **BatchNorm** se comportan distinto en `train()` vs `eval()`.
    
- En validación siempre usar `model.eval()` + `with torch.no_grad()`.
    
- Olvidar `eval()` produce métricas ruidosas (dropout activo, estadísticas BN cambiadas).
    

Snippet:

```python
model.train()
# training step...

model.eval()
with torch.no_grad():
    val_out = model(x)
```

---

## 5) Train loss, Val loss y Early Stopping

**Idea clave:** `train_loss` mide ajuste al set de entrenamiento; `val_loss` mide generalización. Early stopping previene overfitting.

Implementación típica:

- Mantener `best_val = inf`, `patience = k`, `counter = 0`.
    
- Tras cada epoch: si `val_loss < best_val - delta`: guardar y `counter=0`; else `counter +=1`; si `counter>=patience` → parar.
    

Consejos:

- `delta` evita paradas por ruido.
    
- Guardar checkpoint del mejor modelo.
    

Pitfalls:

- Usar `train_loss` para decidir detener — mal criterio.
    
- No resetear contador tras mejora leve por ruido.
    

---

## 6) Capas básicas: Linear, Dropout, Embedding

**nn.Linear(in, out)**

- `y = x @ W^T + b`. Input `(B, in)` → output `(B, out)`. `bias=True` añade vector `b`.
    

**nn.Dropout(p)**

- En training aplica máscara Bernoulli(1-p) y escala por `1/(1-p)`. En eval no cambia salida.
    
- `p` controla fuerza de regularización (0.5 bastante fuerte).
    

**nn.Embedding(num_embeddings, emb_dim)**

- Mapea enteros `[0..num_embeddings-1]` a vectores. Input `(B, seq_len)` → output `(B, seq_len, emb_dim)`.
    
- `padding_idx` permite fijar embedding del padding.
    

Snippet embedding pre-trained:

```python
emb = nn.Embedding(vocab_size, d)
emb.weight.data.copy_(torch.from_numpy(pretrained_weights))
```

Pitfalls:

- Olvidar `.eval()` → dropout activo en evaluación.
    
- No usar `padding_idx` cuando se hace padding -> embeddings de padding se entrenan y contaminan.
    

