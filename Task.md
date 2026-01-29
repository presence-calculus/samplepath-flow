---
ID: 21
Task: Sojourn time panel
Branch: sojourn-time-panel
---

Spec: Add Sojourn time panel to core panels. This is the first of metrics that we will pull from Elementwise Empirical Metrics.
The sojourn time is stored in the w* field of the Empirical flow metrics. We will need to extend the api of plot_core_flow_metrics_charts to include this.


The arg order should be:
plot_core_flow_metrics_charts(
    metrics: FlowMetricsResult,
    empirical_metics: ElementWiseEmpiricalMetrics,
    filter_result: Optional[FilterResult],
    chart_config: ChartConfig,
    out_dir: str,
