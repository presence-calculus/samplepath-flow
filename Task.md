---
ID: 28
Task: cli cleanup
Branch: cli-cleanup
---

Spec: The cli currently has 3 parameters that have inconsistent names compared to the rest of the params.

--start_column and --end_column use underscores instead of dashes.
--dayfirst should be --day-first to stay consistent with the rest of the parameters.

Rename the arguments in the cli and update all documentation that refereences those arguments. In th top level README,
in ADRS and in the cli reference and chart reference documents under docs.

There is no need to retain the old form for backwards compatibility.
