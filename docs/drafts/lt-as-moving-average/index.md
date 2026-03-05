# Hyperbolic Structure of Time-Averaged Presence

This note summarizes several related observations about the structure of the time-average of presence

$$
L(T) = \frac{H(T)}{T}.
$$

These observations explain why $L(T)$ exhibits its characteristic *mean-seeking* behavior and reveal the geometric structure underlying flow metrics.

---

# 1. Hyperbolic decomposition of $L(T)$

Between events the instantaneous presence is constant.

Let the last event before time $T$ occur at $T_i$. Then

$$
N(t) = N_i \quad \text{for } T \in [T_i, T_{i+1})
$$

and cumulative presence evolves linearly:

$$
H(T) = H_i + N_i (T - T_i).
$$

Substituting into the definition of $L(T)$:

$$
L(T) = \frac{H_i + N_i (T - T_i)}{T}.
$$

Rearranging gives

$$
L(T) = N_i + \frac{H_i - N_i T_i}{T}.
$$

Define

$$
C_i = H_i - N_i T_i.
$$

Then

$$
L(T) = N_i + \frac{C_i}{T}.
$$

This shows that **between events $L(T)$ follows a hyperbolic trajectory** whose horizontal asymptote is the current instantaneous presence $N_i$.

As $T$ increases,

$$
L(T) \to N_i.
$$

---

# 2. Interpretation of the memory constant

The constant

$$
C_i = H_i - N_i T_i
$$

can be written using the definition of cumulative presence:

$$
H_i = \int_0^{T_i} N(t)\,dt.
$$

Substituting gives

$$
C_i =
\int_0^{T_i} N(t)\,dt - N_i T_i.
$$

Rewriting,

$$
C_i =
\int_0^{T_i} (N(t) - N_i)\,dt.
$$

Thus $C_i$ measures the **historical deviation of the process from its current state**.

- If earlier presence was larger than $N_i$, then $C_i > 0$.
- If earlier presence was smaller than $N_i$, then $C_i < 0$.

The term

$$
\frac{C_i}{T}
$$

therefore represents a **time-normalized memory of past imbalance**.

As time advances this memory decays proportionally to

$$
\frac{1}{T}.
$$

---

# 3. Mean-seeking behavior of $L(T)$

The decomposition

$$
L(T) = N_i + \frac{C_i}{T}
$$

makes the dynamics of the moving average transparent.

If

$$
L(T_i) > N_i
$$

then $C_i > 0$ and $L(T)$ decreases toward $N_i$.

If

$$
L(T_i) < N_i
$$

then $C_i < 0$ and $L(T)$ increases toward $N_i$.

Thus the time average always moves toward the current instantaneous state.

Taking the derivative,

$$
\frac{dL}{dT} = -\frac{C_i}{T^2}.
$$

But

$$
N_i - L(T) = -\frac{C_i}{T}.
$$

Substituting yields

$$
\frac{dL}{dT} = \frac{N_i - L(T)}{T}.
$$

This is the classical sensitivity equation for a moving average.

---

# 4. Residual work / total age interpretation

Cumulative presence also has a natural interpretation in terms of **ages of items in the system**.

At time $T$, let

- $a_j(T)$ be the age of item $j$ currently present,
- $s_k$ be the sojourn time of completed item $k$.

Then cumulative presence can be written as

$$
H(T) =
\sum_{j=1}^{N(T)} a_j(T)
+
\sum_{k=1}^{D(T)} s_k.
$$

This identity states that cumulative presence equals the **total age accumulated by all items that have ever been present in the process**.

Substituting into the constant

$$
C_i = H_i - N_i T_i
$$

yields

$$
C_i =
-\sum_{j=1}^{N_i} \tau_j
+
\sum_{k=1}^{D_i} s_k
$$

where $\tau_j$ are arrival times of currently present items.

Thus $C_i$ reflects the **historical imbalance between arrivals and completed work**.

Equivalently, it measures how much accumulated age in the system differs from what would be expected if the system had always operated at its current state.

The quantity

$$
\sum_{j=1}^{N(T)} a_j(T)
$$

is often called **total age** or **residual work** in queueing theory (e.g. Kim & Whitt).

Therefore the correction term

$$
L(T) - N(T)
$$

can be interpreted as a **time-normalized residual age imbalance** carried forward from the history of the process.

---

# 5. Geometric interpretation

Between events the time-average can therefore be viewed as

$$
L(T) = \text{current state} + \text{decaying memory}.
$$

More precisely,

$$
L(T) = N(t) + \frac{1}{T}\int_0^{T_i} (N(t)-N_i)\,dt.
$$

The first term represents the **current process state**, while the second term represents a **hyperbolically decaying memory of past deviations**.

This explains several properties of flow metrics:

- why $L(T)$ tends to move toward $N(t)$,
- why early history gradually becomes less influential,
- and why the dynamics of $L(T)$ are hyperbolic even though the underlying process evolves through piecewise linear segments.

The characteristic behavior of the time-average therefore arises directly from the identity

$$
L(T) = \frac{H(T)}{T}.
$$

The curvature of the trajectory is introduced entirely by **time normalization**.
