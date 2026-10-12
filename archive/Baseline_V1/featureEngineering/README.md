# Handwriting Feature Engineering

`handwriting_features.py` reads the cleaned, ordered trajectory table and aggregates by `participant_id` and `test_id`. The output has one row per participant/test, never one row per coordinate.

The initial model-independent feature set includes path length, velocity, acceleration, jerk, pressure level and variability, grip-angle variation, movement duration, pen-up fraction, spatial variation, and bounding-box size. No scaling is performed here. Fit any scaler later on training participants only.