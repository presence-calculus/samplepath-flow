# Release Notes

## v0.2.0

**Released 2026-02-02** | Since v0.1.4

### New Features
- **Data Export**: Added `--export-data` and `--export-only` flags to export processed metrics and sample path data to CSV format (Task 29)
- **Calendar-Indexed Charts**: New flow metrics indexed by calendar dates (Task 26)
- **Process Time Convergence**: Convergence visualization for process time metrics (Tasks 21-23)
- **Duration Scales**: Support for configurable duration units in analysis and visualization (Task 27)
- **SVG Output Support**: Added SVG export capability for plots (Task 13)

### Internal Improvements
- **Core Plotting Architecture**: Major refactor of plotting infrastructure with improved panel abstractions and simplified plot API (Tasks 5, 12, 24)
- **Scatter Plots**: Enhanced core scatter plot functionality with support for overlays and event markers (Task 25)
- **Metrics Module**: Reorganized with advanced calculations moved to dedicated module; enhanced with metric derivations (Tasks 10, 20)
- **CLI**: Enhanced with data export capabilities, improved help documentation, and better sampling frequency controls (Task 28)
- **Documentation**: Expanded CLI reference, added Architectural Decision Records (ADRs), and new tools analysis documentation

### Summary
47 commits with major architectural improvements, new export capabilities, and expanded metrics support.

---

## v0.1.4

Previous release.
