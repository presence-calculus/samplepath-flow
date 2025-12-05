---
toc: false
number-sections: false
---

## Technical Note: Sample path analysis is not statistics.

It works with real-valued time-varying functions of a flow process derived from its
observed history. We study the behavior of these continuous functions using the
machinery of real analysis: integrals, derivatives, limits, convergence.

This makes sample path analysis fundamentally different from statistical approaches to
measuring flow. These rely on sampling and summarize behavior through averages,
variances, percentiles, and similar measures. Those methods work well when the
underlying statistical distributions are well defined and stationary: ie their moments
of the underlying process (average, variance, skew etc.) don’t drift over time.

The common implementations of flow metrics based on Little’s Law is a good example. It
is usually framed as a theorem about statistical averages and flow metrics measure these
averages over a sampling window. But this approach requires stationary distributions for
the law to be applied. Yet most real-world processes in complex adaptive systems are
non-stationary.

Being able to reason about non-stationary processes is essential to analyze stochastic
processes in complex adaptive systems. Here state, history, and feedback loops drive
behavior. Such processes may never settle into a stable regime, or may stabilize only
briefly. Software development is a canonical example.

Sample path analysis, by contrast, works with a finite-window version of Little’s Law.
It directly manifests conservation laws that underlie Little’s Law, and these apply
without any pre-conditions, at all times, over _any_ finite observation window.

This gives us a stable invariant that constrains process behavior even when
distributions shift continuously or lack well-defined moments. It lets us reason
rigorously about nominally distributional properties of non-stationary processes even
when the underlying distributions may not be well defined or exist in a statistical
sense.

In this framing, stability becomes an empirical property of the process: the convergence
of these time varying functions toward finite limits over long observation horizons.
When these limits exist, they coincide with the stationary averages required by the
classical statistical interpretation of Little’s Law. At that point, statistical and
probabilistic techniques can be applied with much greater confidence.

But even if those limits never exist, sample path analysis lets us reason about the
internal dynamics of flow process starting from any point in time, without knowing
_anything_ about the internal mechanisms or its underlying statistical properties. Even
without any of this information we can assess how close or far the process is from
stability—and therefore when standard statistical or probabilistic inference can be done
safely.

This makes it the ideal analytical tool to reason about processes that operate and
remain far from equilibrium and stability, as often the case in software development.
