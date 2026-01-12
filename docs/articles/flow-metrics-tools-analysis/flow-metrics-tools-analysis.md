---
title: "<strong>Why do we need a new way to measure flow?</strong>"
author: |
  <a href=""><em>Dr. Krishna Kumar</em></a>

document-root: "../.."
header-image: "-root/assets/"
---


Flow metrics today are largely used for visibility and reporting of process metrics like Throughput and Process Time (Lead/Cycle/Flow Time etc).  The more sophisticated flow metrics tools also report on Work In Process levels and Work Item Age, but as we have seen in our series, the way you measure these has a lot to do with how useful they are in providing actionable diagnostics for improving flow.

Today  these metrics are reported as independent statistics measured across some arbitrarily chosen business reporting period: we pick a window and report the number of items that have finished in the window and how long those items took to complete, the average age of the items that are in progress in that window. If they do report WIP it is the instantaneous WIP

By and large, thes measurement approaches look at flow metrics through the lens of sample statistics and statistical distributions. So the average process time is assumed to be the statistical average of some underlying distribution  of sampled cycle time over the reporting window.

What Little’s Law shows us is that flow metrics provide true diagnostic value only when all the key metrics - Arrivals/Departure Rate, time-average of WIP, and process time averages are measured over the same observation window.
