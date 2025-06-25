# pypcalc
# The Presence Calculus Toolkit
_A modern foundation for reasoning about flow and presence in non-linear systems._

[![PyPI](https://img.shields.io/pypi/v/pypcalc.svg)](https://pypi.org/project/pypcalc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://py.pcalc.org)

---

## 🔍 Overview

`pypcalc` is a python library that provides a reference implementation for the concepts
in [The Presence Calculus](https://docs.pcalc.org)

The toolkit includes implementations for key concepts of The Presence Calculus, including

- A `Presence` class representing the fundamental unit of analysis. 
- The `PresenceMatrix` for computing over presences.
- The `PresenceInvariant` that expresses the fundamental invariant of The Presence Calculus. 

In addition to base implementations we plan to provide 

- Support for optional export to `pandas` or `polars` and imports from csv files for barebones integration with external models.
- Integration APIs to use the calculus in different contexts, including simulations and real-time sources, as well as how to use it for causal analysis, and modeling flow in complex adaptive systems

STATUS: The current implementation should be considered pre-alpha quality and not ready for production use. 
When a releasable library is available, we will publish it as a package on PyPi. 

For now, it is best to think of the code in this repository as a concrete, but evolving implementation of the 
high level concepts we talk about in the documentation. 

---

## 📚 Documentation

- 🧠 [Concepts](https://docs.pcalc.org)
- 📘 [API Reference](https://docs.pcalc.org/api/pcalc.html)
- 



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
