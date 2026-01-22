# Plotly Backend Spike (Step Chart)

**Purpose:** Validate the proposed backend-agnostic `ChartSpec`/`RenderBackend` protocol by rendering a step chart with Plotly, mirroring the Matplotlib path.

## What I tried

- Installed Plotly + Kaleido for static image export: `uv pip install plotly kaleido`.
- Implemented a minimal `ChartSpec` (series_type, x, y, title, y_label, legend_label, unit, color) and a `render_step_plotly` that maps `series_type == "step"` to a `go.Scatter` with `line.shape="hv"` (horizontal-then-vertical), sets titles/labels, and calls `fig.write_image(out_path)`.

## Results

- Matplotlib parity mapping is straightforward: `line.shape="hv"` produces the expected step profile for N(t).
- Static export failed in the sandbox: `plotly.io.write_image` (via Kaleido) aborted because the embedded browser could not start (`BrowserFailedError: The browser seemed to close immediately after starting. ... located at /Applications/Google Chrome.app`). Kaleido needs a working Chromium; the sandboxed environment blocked it.

## Implications for the design

- The `RenderBackend` contract should allow backends to choose an output mode beyond static images:
  - Support HTML export (`write_html`) as a fallback when static export engines (Kaleido) are unavailable.
  - Allow caller to specify desired format (png/svg/html) so the backend can pick the viable path.
- The backend should surface capability checks/errors cleanly (e.g., raise a backend-specific exception or return a status) instead of failing deep inside Kaleido.
- The `ChartSpec` itself is fine; no Plotly-specific fields are needed for steps. Backend adapts `series_type="step"` to `line.shape="hv"` internally.
- For overlays, Plotly supports scatter markers and annotations; the generic `Overlay` protocol remains compatible.

## Proposed adjustments

1) Extend the renderer API to accept an output format enum (`"png" | "svg" | "html"`) or a save target that encodes format; let each backend pick the feasible path.
2) Add a capability check to the Plotly backend: if Kaleido is unavailable or fails, fall back to HTML and emit a warning.
3) Keep `ChartSpec` unchanged; keep all backend-specific export concerns inside the backend implementation.

## Next steps if we proceed

- Implement a minimal `PlotlyBackend` that supports HTML output by default and optional static export when Kaleido is available.
- Wire the backend selection into the `ChartRenderer` for parity tests with Matplotlib (mocked save targets in unit tests).
