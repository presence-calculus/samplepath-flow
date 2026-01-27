# Coding Style Guide

## General function signature rules
- Group related keyword args together in signatures: `show_title`, `title`, `show_derivations` (feature toggle first, then feature controls).
- Maintain a consistent order for calling functions with keyword args: they should be in the same order they are declared in the function signature.

## Creating charts and plots
- Review ADR in decisions/006-plotting-architecture-and-conventions.md
- After implementing a change to the plot review that the it is compliant with the conventions and report any issues with compliance.
