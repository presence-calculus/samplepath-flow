---
ID: 24
Task: Convergence module clean-up
Branch: convergence-clean-up
---

Spec: Bring convergence.py fully up to the new panel/plot architecture.
1. There are two methods in this module

 - samplepath.plots.convergence.draw_dynamic_convergence_panel_with_errors
 - samplepath.plots.convergence.draw_dynamic_convergence_panel_with_errors_and_endeffects

that write to plots that are stored under the advanced directory. Move these methods and their associated calls so that they are driven by the top level driver in the advanced module rather than from the convergence module. No need to rewrite the charts yet, we can keep them in teh old format and clean them up later. But there should be no dependence on these charts in the convergence.py module.
2. What remains are the arrival/departure convergence panels and stacks. Migrate these to the new architecture and wire them up to the driver. These must be done using all the plot/render conventions.

3. Lets finish the cleanup by migrating the last remaining legacy chart samplepath.plots.convergence.plot_sample_path_convergence into a standard Panel structure with plot and render. It will also allow us to remove the args argument from the top level driver signature.
