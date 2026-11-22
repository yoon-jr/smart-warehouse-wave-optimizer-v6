# Changes in v4

## Data

- Removed Hugging Face dataset integration.
- Removed submission-time synthetic task fallback.
- Added robust Footwear dataset discovery and CSV parsing.
- Added storage-location Cartesian coordinate parsing and grid mapping.
- Added timestamp detection with explicit wave-duration/fixed-order fallback.
- Added sequence-proportional release of pick lines within each wave.
- Added provenance and modeling-assumption manifest.

## Control policy

- Added `AdaptivePolicyController`.
- Reactive allocation remains the base behavior.
- Predictive idle-robot repositioning is enabled only after hard safety gates and suitability thresholds pass.
- Added enter/exit hysteresis and a minimum hold time.
- Limited proactive repositioning to two robots per forecast cycle.
- Added immediate Reactive fallback during backlog overload or low idle capacity.

## User interface

- Renamed contender from Predictive to Adaptive Hybrid.
- Added active-policy badge.
- Added Zone labels with waiting and forecast values.
- Aggregated overlapping task markers and displayed counts.
- Added NORMAL/BUSY/OVERLOADED live situation status.
- Added plain-language policy reason and current action.
- Added map legend and Order Surge countdown explanation.
- Updated Streamlit dashboard with threshold signals and data provenance.

## Evaluation

- Updated fair comparison to Reactive vs Adaptive Hybrid.
- Updated experiment outputs with policy-use percentages.
- Added Footwear-specific surge templates.
- Added tests for coordinate parsing, preprocessing, adaptive gates, fairness, and explanations.
