# pypcalc
# The Presence Calculus Toolkit
_A modern foundation for reasoning about flow and presence in non-linear systems._

[![PyPI](https://img.shields.io/pypi/v/pypcalc.svg)](https://pypi.org/project/pypcalc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://py.pcalc.org)

---

## 🔍 Overview

`pypcalc` provides a formal foundation and practical tools for modeling, measuring, and analyzing *presence* — 
a unifying concept for modeling and measuring "flow" in systems ranging from simple, linear systems to complex, 
stochastic and non-linear systems. 

The toolkit includes implementations for key concepts of The Presence Calculus, including

- A `Presence` class representing the fundamental unit of analysis. 
- The `PresenceMatrix` for computing over presences.
- The `PresenceInvariant` that expresses the fundamental invariant of The Presence Calculus. 

In addition to base implementations we also provide 

- Support for optional export to `pandas` or `polars` and imports from csv files for barebones integration with external models.
- Examples showing how to use the concepts in different contexts, including simulations and real-time sources, as well as how to use it for causal analysis, and modeling flow in complex adaptive systems

---

## 📦 Installation

Install the core package:

```bash
pip install pypcalc
```

Optional extras:

```bash
pip install pypcalc[pandas,polars]
```

---

## 🚀 Quickstart

```python
from pcalc.presence import Presence

# A presence record indicating an element existed in a boundary from t=0 to t=10
p = Presence(element="job-42", boundary="system-a", onset_time=0.0, reset_time=10.0)

print(p.duration())  # Outputs: 10.0
```

For more advanced usage, see examples in the repository.

---

## 📚 Documentation

- 📘 [API Reference](https://py.pcalc.org)
- 🧠 [Concepts and Blog](https://www.polaris-flow-dispatch.com)



## 🛠 Development

Clone the repo and install using Poetry:

```bash
poetry install
poetry shell
```

Run tests:

```bash
pytest
```

Code quality:

```bash
black pcalc/
isort pcalc/
mypy pcalc/
```

---

## 📝 License

This project is licensed under the MIT License.  
See [LICENSE](./LICENSE) for full details.
