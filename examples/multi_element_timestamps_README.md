# Multi-Element Timestamps Test Dataset

This dataset demonstrates the multi-element timestamp feature in data export.

## Dataset Overview

- **File:** `multi_element_timestamps.csv`
- **Records:** 20 work items (tasks, bugs, features, spikes)
- **Time span:** January 1-15, 2024
- **Attributes:** id, start_ts, end_ts, class, priority

## Multi-Element Timestamp Scenarios

This dataset includes all scenarios for multi-element timestamps:

### 1. Multiple Arrivals (3 items at same time)
**2024-01-01 09:00:00** - Items 101, 102, 103 all arrive
- element_id: `A:101;102;103`

### 2. Multiple Arrivals (2 items)
**2024-01-02 10:30:00** - Items 104, 105 arrive
- element_id: `A:104;105`

### 3. Mixed Arrivals and Departures
**2024-01-03 11:15:00** - Items 106, 107, 108 arrive; Item 102 departs
- element_id: `A:106;107;108|D:102`

### 4. Zero-Duration Element Among Multiple
**2024-01-04 14:00:00** - Items 109, 110 arrive; Item 110 also completes (spike)
- element_id: `A:109;110|D:110`
- Item 110 has zero duration (arrives and departs at same time)

### 5. Multiple Departures
**2024-01-13 11:00:00** - Items 115, 116 depart together
- element_id: `D:115;116`

### 6. Complex Mixed Events
**2024-01-08 10:00:00** - Items 115, 116 arrive; Items 106, 107 depart
- element_id: `A:115;116|D:106;107`

## Usage

```bash
# Export event-indexed metrics with multi-element IDs
uv run samplepath examples/multi_element_timestamps.csv --export-only

# View the element_id column formatting
uv run samplepath examples/multi_element_timestamps.csv --export-data
```

## Expected Output Format

In `event_indexed_metrics.csv`, the `element_id` column will show:

- Single element: `"101"` (plain ID)
- Multiple arrivals: `"A:101;102;103"` (semicolon-separated)
- Multiple departures: `"D:115;116"`
- Mixed: `"A:115;116|D:106;107"` (arrivals before departures, pipe-separated)
- No elements: empty/null

## Verification

Total events: 20 items with carefully orchestrated timestamps to create:
- 3 arrivals at 09:00 (items 101-103)
- 2 arrivals at 10:30 (items 104-105)
- 3 arrivals + 1 departure at 11:15 (items 106-108 arrive, 102 departs)
- 1 zero-duration event at 14:00 (item 110)
- Multiple other mixed scenarios throughout the timeline
