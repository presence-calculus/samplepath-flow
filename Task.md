---
ID: 14
Task: Misc contract cleanups
Branch: misc-contract-cleanups
---

Spec: Here are some more cleanups we need to do.

1. The contract between plot and figure context still involves too much boilerplate code. Here is an example:

    def plot(
        self,
        df: pd.DataFrame,
        chart_config: ChartConfig,
        filter_result: Optional[FilterResult],
        metrics: FlowMetricsResult,
        out_dir: str,
    ) -> None:
        del df
        out_path = resolve_chart_path(
            os.path.join(out_dir, "core"),
            "sample_path_N",
            chart_config.chart_format,
        )
        ...and then

        with figure_context(
            out_path,
            nrows=1,
            ncols=1,
            unit=unit,
            caption=caption,
            save_kwargs=save_kwargs,
        ) as (
            _,
            axes,
        ):

Each caller to save_format is responsible for calling and assemmbling the path and resolving the save format to pass onto figure_context. instead the figure context should just take chart_config and out path and file name and do this logic inside the context manager block before the same. Its a cleaner separatation of responsibility and allows us to extend figure_context behavior with more things from chart_config later if we choose to do so.

2. the fact that you are calling del df shows we have problem with the signature of plot. In general these plot methods dont require df in general (some others might for example in convergence, but maybe not). But it shows that df is an optional bit of data for plotting in general, so it should not be added to any plot method that does not need it. The most important, required argument is metrics. So we should do the following here:
   a. remove df from the signature of any plot method that does not use it.
   b. Reorder the arguments of the plot methods as follows:
    def plot(metrics, chart_config, filter_result, out_dir) if it does not require df and
    def plot(dr, metrics, chart_config, filter_result, out_dir) if it does

We can do this in two separate steps in different commits.

----
Implementation Status:

Step 1 Review Feedback:

The various calls to figure context call the arguments to figure context in random order. I want to standardize the order of arguments in the declaration and ensure all callers call this in the same order (including tests).

I have organized it logically into groups below using comments and have a question on why save_kwargs is required in th signature at all.

def figure_context(
    out_path: Optional[str] = None,
    *,
    chart_config: Optional[ChartConfig] = None,
    # plot sizing
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    tight_layout: bool = True,
    # chart decorators
    caption: Optional[str] = None,

    # x-axis formatting
    unit: Optional[str] = "timestamp",
    sharex: bool = False,
    format_axis_fn: Callable[[plt.Axes, Optional[str]], None] = _format_axis_label,

    # save configruration
    out_dir: Optional[str] = None,
    subdir: Optional[str] = None,
    base_name: Optional[str] = None,

    # why is this necessary in the signature? We have chart_config
    save_kwargs: Optional[dict] = None,

) -> Iterator[Tuple[plt.Figure, Union[plt.Axes, np.ndarray], str]]:

For layout_context here is the canonical order of arguments.

def layout_context(
    out_path: Optional[str] = None,
    *,
    chart_config: Optional[ChartConfig] = None,
    layout: LayoutSpec,
    decor: Optional[FigureDecorSpec] = None,
    # x-axis-formatting
    unit: Optional[str] = "timestamp",
    format_targets: Literal[
        "all", "bottom_row", "left_col", "bottom_left"
    ] = "bottom_row",
    format_axis_fn: Callable[[plt.Axes, Optional[str]], None] = _format_axis_label,

    # save configuration
    out_dir: Optional[str] = None,
    subdir: Optional[str] = None,
    base_name: Optional[str] = None,
    # why is this needed anymore since we have chart_config passed.
    save_kwargs: Optional[dict] = None,
) -> Iterator[Tuple[plt.Figure, Union[plt.Axes, np.ndarray], str]]:

----
Implementation Status:

- Step 1 (in progress):
  - Centralized output path + save kwargs resolution inside figure/layout contexts.
  - Contexts now yield `(fig, axes, resolved_out_path)` and callers collect paths.
  - Removed `save_kwargs` from context signatures; output flows through `ChartConfig`.
  - Standardized context argument ordering and updated core callers/tests.
  - Updated all core panel `plot(...)` APIs to `(df, chart_config, filter_result, metrics, out_dir)` and resolved fields inside plot.
  - Updated `plot_core_flow_metrics_charts(...)` and core plot tests to match the unified contract.
  - Migrated `LLWPanel.plot(...)` to use `figure_context` with a square figsize.

- Step 2 (in progress):
  - Reordered core plot signatures to `(metrics, filter_result, chart_config, out_dir)` and removed unused `df` parameters.
