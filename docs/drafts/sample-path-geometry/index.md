# Geometry of the Finite–Window Identity in Real Space and Log Space

This note summarizes the geometric structure implied by the finite–window Little identity  
$$
L(T) = \Lambda(T)\,w(T)
$$
for all observation windows $T>0$.  
We examine the argument in two domains:

1. **Real space** $(L,\Lambda,w)$ using the exact finite–difference identity and the differential constraint.  
2. **Log space** $(\ell,a,r) = (\log L,\log\Lambda,\log w)$ where the identity becomes linear and all increments behave additively.


---

## 1. Real–space geometry

Assume  
$$
L(T)=\Lambda(T)\,w(T)
$$
for all $T>0$.  
Fix any two endpoints $0<T_1<T_2$ and write  
$$
L_i=L(T_i),\qquad \Lambda_i=\Lambda(T_i),\qquad w_i=w(T_i)
$$
with finite increments  
$$
\Delta L=L_2-L_1,\qquad \Delta\Lambda=\Lambda_2-\Lambda_1,\qquad \Delta w = w_2-w_1.
$$

### 1.1 Exact finite–difference relation

Using  
$$
L_1=\Lambda_1 w_1,\qquad L_2=\Lambda_2 w_2,
$$
subtracting gives the exact identity  
$$
\Delta L=\Lambda_1\,\Delta w \;+\; w_1\,\Delta\Lambda \;+\; \Delta\Lambda\,\Delta w.
$$

This shows that $(\Delta L,\Delta\Lambda,\Delta w)$ cannot vary independently.  
The coupling term $\Delta\Lambda\,\Delta w$ reflects the curvature of the surface  
$$
L = \Lambda w.
$$

### 1.2 First–order (tangent–plane) approximation

When $T_2$ is close to $T_1$, the product term is second order, yielding  
$$
\Delta L \approx \Lambda_1\,\Delta w + w_1\,\Delta\Lambda.
$$

This is the equation of the **tangent plane** to the nonlinear surface  
$$
L = \Lambda w
$$
at the point $(\Lambda_1,w_1,L_1)$.  
Small finite–window steps therefore move approximately within this plane.

### 1.3 Differential form

For smooth trajectories, taking $T_2\to T_1$ gives the exact differential constraint  
$$
\frac{dL}{dT}
=
w(T)\,\frac{d\Lambda}{dT}
+
\Lambda(T)\,\frac{dw}{dT}.
$$

This is the total derivative of $L=\Lambda w$.  
In real space the geometry is therefore **curved**, with linear behavior obtained only locally via the tangent plane.

---

## 2. Log–space geometry

Define the log–coordinates  
$$
\ell = \log L,\qquad a = \log\Lambda,\qquad r = \log w.
$$

Applying $\log$ to the finite–window identity gives  
$$
\ell = a + r.
$$

This is the equation of a **plane** in $(a,r,\ell)$–space.

### 2.1 Exact finite–difference relation for any two points

Let  
$$
P_1=(a_1,r_1,\ell_1),\qquad P_2=(a_2,r_2,\ell_2)
$$
be any two points satisfying $\ell=a+r$, and define increments  
$$
\Delta a=a_2-a_1,\qquad \Delta r=r_2-r_1,\qquad \Delta\ell=\ell_2-\ell_1.
$$

Because both points lie on the plane, we have  
$$
\Delta\ell=\Delta a+\Delta r.
$$

This holds for **arbitrary** finite steps: large windows, small windows, stable or unstable processes.  
The geometry is globally linear.

### 2.2 Differential form

Differentiating $\ell=a+r$ gives  
$$
\frac{d\ell}{dT}
=
\frac{da}{dT}
+
\frac{dr}{dT}.
$$

Equivalently, in terms of the original variables,  
$$
\frac{1}{L}\frac{dL}{dT}
=
\frac{1}{\Lambda}\frac{d\Lambda}{dT}
+
\frac{1}{w}\frac{dw}{dT}.
$$

This expresses relative (logarithmic) changes as a **simple additive decomposition**, in contrast to the nonlinear real–space identity.

### 2.3 Geometric implication

In log space:

- The entire constraint $L=\Lambda w$ becomes a **flat 2D surface**.  
- Every pair of points satisfies the linear relation $\Delta\ell=\Delta a+\Delta r$.  
- Finite increments behave as exact vector additions.

Thus the curved geometry of real space collapses into a globally linear structure in log space.

---

## Summary

- In **real space**, $L=\Lambda w$ defines a curved surface and the increments satisfy  
  $$
  \Delta L=\Lambda_1\,\Delta w+w_1\,\Delta\Lambda+\Delta\Lambda\,\Delta w,
  $$
  with the differential equation  
  $$
  \frac{dL}{dT}=w\,\frac{d\Lambda}{dT}+\Lambda\,\frac{dw}{dT}.
  $$

- In **log space**, $\ell=a+r$ defines a plane and all increments satisfy the exact identity  
  $$
  \Delta\ell=\Delta a+\Delta r,
  $$
  with the differential version  
  $$
  \frac{d\ell}{dT}=\frac{da}{dT}+\frac{dr}{dT}.
  $$

Real space is nonlinear, curved, and locally constrained.  
Log space is globally linear, additive, and geometrically flat.


-----

## Deriving the differential equation for the co-evolution of $L(T)$, $\Lambda(T)$, and $w(T)$

We assume the finite-window identity holds for every prefix length $T>0$:
$$
L(T) = \Lambda(T)\,w(T).
$$
Here $L(T)$, $\Lambda(T)$, and $w(T)$ are functionals of the same underlying sample path: for each $T$ they are computed deterministically from the events in $[0,T]$.

### 1. Starting from the finite-window identity

For each $T$, define
$$
f(T) := L(T),\quad g(T) := \Lambda(T),\quad h(T) := w(T).
$$
The finite-window identity is simply
$$
f(T) = g(T)\,h(T)\quad\text{for all }T>0.
$$

### 2. Differentiate the product

Assume $f,g,h$ are differentiable at some $T$. Then we can apply the ordinary product rule to $f(T) = g(T)\,h(T)$:
$$
\frac{df}{dT}(T) = \frac{d}{dT}\bigl(g(T)\,h(T)\bigr) = g(T)\,\frac{dh}{dT}(T) + h(T)\,\frac{dg}{dT}(T).
$$
Now substitute back $f=L$, $g=\Lambda$, and $h=w$:
$$
\frac{dL}{dT}(T)
=
\Lambda(T)\,\frac{dw}{dT}(T)
+
w(T)\,\frac{d\Lambda}{dT}(T).
$$

### 3. Derivation via finite differences (optional, more “sample-path” flavored)

Fix $T$ and a small increment $\Delta T>0$. Using the identity at $T$ and $T+\Delta T$:
$$
L(T) = \Lambda(T)\,w(T),\qquad
L(T+\Delta T) = \Lambda(T+\Delta T)\,w(T+\Delta T).
$$
Define finite differences
$$
\Delta L = L(T+\Delta T)-L(T),\quad
\Delta\Lambda = \Lambda(T+\Delta T)-\Lambda(T),\quad
\Delta w = w(T+\Delta T)-w(T).
$$
Then
$$
\Delta L
=
\Lambda(T)\,\Delta w
+
w(T)\,\Delta\Lambda
+
\Delta\Lambda\,\Delta w.
$$
Divide by $\Delta T$:
$$
\frac{\Delta L}{\Delta T}
=
\Lambda(T)\,\frac{\Delta w}{\Delta T}
+
w(T)\,\frac{\Delta\Lambda}{\Delta T}
+
\frac{\Delta\Lambda}{\Delta T}\,\Delta w.
$$
If $L,\Lambda,w$ are differentiable at $T$, then as $\Delta T\to 0$ we have $\Delta w\to 0$ and
$$
\frac{\Delta L}{\Delta T}\to\frac{dL}{dT}(T),\quad
\frac{\Delta\Lambda}{\Delta T}\to\frac{d\Lambda}{dT}(T),\quad
\frac{\Delta w}{\Delta T}\to\frac{dw}{dT}(T),
$$
while the mixed term vanishes:
$$
\frac{\Delta\Lambda}{\Delta T}\,\Delta w \;\longrightarrow\; 0.
$$
Taking the limit $\Delta T\to 0$ yields
$$
\frac{dL}{dT}(T)
=
\Lambda(T)\,\frac{dw}{dT}(T)
+
w(T)\,\frac{d\Lambda}{dT}(T),
$$
which is the desired governing differential equation for the co-evolution of the three functionals.



## derivation of the finite difference formula.

We start from the identities at two window endpoints:
$$
L_1 = \Lambda_1 w_1,\qquad L_2 = \Lambda_2 w_2.
$$

Define the finite differences:
$$
\Delta L = L_2 - L_1,\qquad \Delta\Lambda = \Lambda_2 - \Lambda_1,\qquad \Delta w = w_2 - w_1.
$$

Write the second point in terms of the first point and increments:
$$
\Lambda_2 = \Lambda_1 + \Delta\Lambda,\qquad w_2 = w_1 + \Delta w.
$$

Substitute into $L_2 = \Lambda_2 w_2$:
$$
L_2 = (\Lambda_1 + \Delta\Lambda)(w_1 + \Delta w).
$$

Expand the product:
$$
(\Lambda_1 + \Delta\Lambda)(w_1 + \Delta w)
=
\Lambda_1 w_1
+ \Lambda_1 \Delta w
+ w_1 \Delta\Lambda
+ \Delta\Lambda\,\Delta w.
$$

Subtract $L_1 = \Lambda_1 w_1$ to obtain the finite-difference identity:
$$
\Delta L
= \Lambda_1\,\Delta w
+ w_1\,\Delta\Lambda
+ \Delta\Lambda\,\Delta w.
$$

# Tensor form

Let the state of the finite–window metrics at time $T$ be represented by the vector
$$
X(T) =
\begin{bmatrix}
L(T)
\Lambda(T)
w(T)
\end{bmatrix}.
$$

For any two window endpoints $T_1 < T_2$, define the increment tensor
$$
\Delta X =
\begin{bmatrix}
\Delta L
\Delta\Lambda
\Delta w
\end{bmatrix}
=
\begin{bmatrix}
L_2 - L_1
\Lambda_2 - \Lambda_1
w_2 - w_1
\end{bmatrix}.
$$

The finite–window identity
$$
L = \Lambda\,w
$$
can be expressed as the bilinear form
$$
F(X) = X_2\,X_3 - X_1 = 0,
$$
with $X_1=L$, $X_2=\Lambda$, $X_3=w$.

The first–order expansion (tensor form of the differential) is
$$
DF(X_1)\,\Delta X
=
\frac{\partial F}{\partial X_1}\Delta X_1
+
\frac{\partial F}{\partial X_2}\Delta X_2
+
\frac{\partial F}{\partial X_3}\Delta X_3.
$$

Compute the gradient:
$$
\nabla F(X_1)
=
\begin{bmatrix}
-1
w_1
\Lambda_1
\end{bmatrix}.
$$

Thus the constraint on increments is
$$
\nabla F(X_1)^{\!\top}\,\Delta X
=
-\,\Delta L
+
w_1\,\Delta\Lambda
+
\Lambda_1\,\Delta w.
$$

Because both $X_1$ and $X_2$ satisfy $F(X)=0$, the linearized constraint must equal the second–order remainder:
$$
\nabla F(X_1)^{\!\top}\,\Delta X
=
-\,\Delta\Lambda\,\Delta w.
$$

Rearranging yields the finite-difference tensor identity:
$$
\Delta L
=
\Lambda_1\,\Delta w
+
w_1\,\Delta\Lambda
+
\Delta\Lambda\,\Delta w.
$$

This expression decomposes the increment $\Delta L$ into a first–order multilinear term
$$
(\nabla F(X_1))^\top \Delta X
=
\Lambda_1\,\Delta w + w_1\,\Delta\Lambda
$$
and a symmetric second–order tensor correction
$$
\Delta\Lambda\,\Delta w.
$$


-----

# Hamiltonian Dynamics

# Hamiltonian Dynamics of the Finite-Window Identity

This document derives the Hamiltonian mechanics for a flow system governed by the finite-window Little's Law. We treat the system as a dynamic particle constrained to the manifold defined by the identity.

## 1. The Configuration Space

We define the generalized coordinates $q$ in **Log Space**, where the geometry is globally linear and additive.

Let the independent coordinates be:
$$
q_1 = a = \log \Lambda \quad (\text{Arrival Intensity})
$$
$$
q_2 = r = \log w \quad (\text{Service Intensity})
$$

The dependent coordinate is $\ell = \log L$. The system is subject to the holonomic constraint:
$$
\ell(a, r) = a + r
$$

This defines a flat 2D plane in the 3D configuration space $(a, r, \ell)$.

## 2. The Lagrangian

We assume a "free particle" model where the system has inertia (resistance to infinite volatility) but no external potential ($V=0$) acting on it yet. The Lagrangian is the kinetic energy of the system.

Using the differential form $\dot{\ell} = \dot{a} + \dot{r}$, the kinetic energy $T$ is:

$$
T = \frac{1}{2}m \left( \dot{a}^2 + \dot{r}^2 + \dot{\ell}^2 \right)
$$

Substituting the constraint $\dot{\ell} = \dot{a} + \dot{r}$:

$$
\mathcal{L}(q, \dot{q}) = \frac{m}{2} \left[ \dot{a}^2 + \dot{r}^2 + (\dot{a} + \dot{r})^2 \right]
$$

Expanding the square term yields the Lagrangian with an inertial coupling term:

$$
\mathcal{L} = m \left( \dot{a}^2 + \dot{r}^2 + \dot{a}\dot{r} \right)
$$

**Physical Interpretation:** The cross-term $m\dot{a}\dot{r}$ arises because the geometry of the finite-window identity couples the fluctuations of arrival rates and service times.

## 3. The Conjugate Momenta

We derive the generalized momenta $p_i = \frac{\partial \mathcal{L}}{\partial \dot{q}_i}$:

$$
p_a = \frac{\partial \mathcal{L}}{\partial \dot{a}} = m(2\dot{a} + \dot{r})
$$

$$
p_r = \frac{\partial \mathcal{L}}{\partial \dot{r}} = m(2\dot{r} + \dot{a})
$$

In matrix form $p = M v$:

$$
\begin{bmatrix} p_a \\ p_r \end{bmatrix} = m \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} \dot{a} \\ \dot{r} \end{bmatrix}
$$

## 4. The Hamiltonian

The Hamiltonian $H$ represents the total energy of the flow. It is obtained via the Legendre transform:

$$
H(q, p) = p^\top \dot{q} - \mathcal{L}
$$

First, we invert the mass matrix $M$ to express velocities in terms of momenta. The inverse of $M$ is:

$$
M^{-1} = \frac{1}{3m} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}
$$

The Hamiltonian for a quadratic kinetic energy is given by $H = \frac{1}{2} p^\top M^{-1} p$:

$$
H(p_a, p_r) = \frac{1}{6m} \left[ \begin{matrix} p_a & p_r \end{matrix} \right] \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} p_a \\ p_r \end{bmatrix}
$$

Expanding this yields:

$$
H = \frac{1}{3m} \left( p_a^2 + p_r^2 - p_a p_r \right)
$$

## 5. Hamilton's Equations of Motion

The evolution of the system in phase space is governed by the symplectic gradients.

### A. Dynamics of the Metrics (Velocities)
$$
\dot{a} = \frac{\partial H}{\partial p_a} = \frac{1}{3m}(2p_a - p_r)
$$
$$
\dot{r} = \frac{\partial H}{\partial p_r} = \frac{1}{3m}(2p_r - p_a)
$$

This highlights the **coupling effect**: The rate of change of the arrival intensity ($\dot{a}$) is dampened by the momentum of the service duration ($p_r$).

### B. Conservation Laws (Forces)
Since the Hamiltonian is translationally invariant in log-space (no dependence on $a$ or $r$ explicitly):

$$
\dot{p}_a = -\frac{\partial H}{\partial a} = 0 \implies p_a = \text{constant}
$$
$$
\dot{p}_r = -\frac{\partial H}{\partial r} = 0 \implies p_r = \text{constant}
$$

## 6. Conclusion

By mapping the finite-window identity to log-space, we have successfully formulated the system as a free particle on a constrained plane. The Hamiltonian $H = \frac{1}{3m} (p_a^2 + p_r^2 - p_a p_r)$ reveals that the "queueing fluid" has a non-diagonal mass tensor, implying that fluctuations in arrival rates and service times are inertially entangled.

------------

# Presence Mass as Hamiltonian Mass

This note clarifies the relationship between *presence mass* from Presence Calculus and the *mass* parameter that appears in the Hamiltonian formulation of the finite-window identity. The result is that they coincide naturally: presence mass is the correct choice for Hamiltonian mass because both quantities encode **inertia of change** along the prefix window.

---

## 1. Mass in Hamiltonian Mechanics

In Hamiltonian and Lagrangian mechanics, *mass* represents **inertia**:

- resistance to rapid changes in velocity  
- smoothing of motion over time  
- a scaling factor in the kinetic energy metric

For generalized coordinates $q = (q_1, q_2)$, the kinetic energy is
$$
T = \frac{1}{2}\,\dot{q}^\top M\,\dot{q},
$$
where $M$ is the mass matrix. Larger mass means **slower response** to forces and perturbations.

---

## 2. Presence Mass in Flow Systems

Presence mass in Presence Calculus measures:

- the cumulative time weight carried by the prefix window  
- the "historical inertia" embedded in the averaged quantities  
- the resistance of the finite-window metrics to instantaneous fluctuations  

As the window $T$ grows, presence mass increases; consequently:

- $a(T)=\log\Lambda(T)$ changes more slowly  
- $r(T)=\log w(T)$ changes more slowly  
- $\ell(T)=\log L(T)$ becomes less sensitive to short-term shifts  

This is *exactly* the property of mass in mechanics.

---

## 3. Why Presence Mass and Hamiltonian Mass Coincide

The finite-window identity in log space,
$$
\ell = a + r,
$$
defines a holonomic constraint on the system. The Lagrangian for a free particle moving on this constraint manifold is
$$
\mathcal{L} = \frac{1}{2}m \left( \dot{a}^2 + \dot{r}^2 + \dot{\ell}^2 \right),
$$
with $\dot{\ell} = \dot{a} + \dot{r}$. Substituting gives the reduced Lagrangian
$$
\mathcal{L}
= m\left(\dot{a}^2 + \dot{r}^2 + \dot{a}\dot{r}\right).
$$

Here the parameter $m$ governs:

- how hard it is for the log-arrival intensity to change  
- how hard it is for the log-residence-time to change  
- how strongly the two changes couple through the metric tensor  

These effects correspond exactly to the inertia induced by accumulated presence.

Thus, the mathematical role played by $m$ is identical to the operational meaning of presence mass.

---

## 4. Interpretation

Under this identification:

- **Large presence mass**  
  - prefix metrics evolve slowly  
  - the system is “heavy”  
  - historical data dominates instantaneous changes  

- **Small presence mass**  
  - prefix metrics are highly reactive  
  - the system is “light”  
  - evolution is volatile and sensitive to small perturbations  

This matches the classical mechanical interpretation of mass as resistance to acceleration.

---

## 5. Natural Conclusion

Presence mass **is** Hamiltonian mass.

It determines:

- the geometry of the kinetic term  
- the metric on the log-space constraint manifold  
- the inertia of the system’s evolution under the finite-window identity  

This equivalence is not metaphorical. It follows directly from:

1. the definition of presence as accumulated time weight,  
2. the structure of the kinetic energy in log space, and  
3. the co-evolution constraint $\dot{\ell} = \dot{a} + \dot{r}$.

Presence mass therefore provides the canonical physical interpretation of the mass parameter in the Hamiltonian formulation of finite-window Little dynamics.
