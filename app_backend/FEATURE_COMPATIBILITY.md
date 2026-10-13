# Feature compatibility and endpoint status

## Saved artifacts

The backend loads the saved speech, handwriting, and tremor classifiers from `experiments/regularized_svm_rf/results/artifacts/`. The bundles contain fitted sklearn pipelines, preprocessing, calibrated probabilities, and feature schemas. The adapter checks exact feature order and class order `[0, 1]`, and performs no fitting. Baseline V1 is unchanged.

## Speech

The cleaned speech table has 240 rows and 48 columns: identifiers/target (`id`, `recording`, `status`), `gender`, and 44 acoustic descriptors. The model consumes 45 fields including numeric `gender`: Jitter variants, Shimmer variants, HNR values, RPDE, DFA, PPE, GNE, 13 MFCC values, and 13 Delta values.

`cleaningPipeline/speech_cleaning.py` reads and cleans an already-featured CSV; it does not derive descriptors from audio. No local implementation was found for waveform extraction or its parameters, units, segmentation, MFCC/Delta definition, or gender coding contract. Existing project dependencies also lacked Praat/Parselmouth, openSMILE, or Librosa extraction. Therefore WAV validation is functional, the saved model can be loaded and tested with schema-compatible rows, but `/api/speech/predict` returns `501 raw_audio_feature_extractor_unavailable` without invoking the classifier. The `gender` model field additionally needs a defined input/metadata contract.

## Spiral / handwriting

The trained feature order is `n_points`, `path_length`, `mean_velocity`, `max_velocity`, `mean_acceleration`, `max_acceleration`, `mean_jerk`, `max_jerk`, `pressure_mean`, `pressure_std`, `pressure_range`, `grip_angle_std`, `movement_duration`, `pen_up_fraction`, `spatial_std`, `bounding_box_diagonal`.

The cleaning code reads tablet trajectories with x/y/z, pressure, grip angle, timestamp, test ID, and point order. `featureEngineering/handwriting_features.py` safely defines the exact training extractor. The backend reuses it through a guarded tablet trajectory adapter, tested against a saved training feature row. Camera/video does not provide validated pressure, grip-angle, pen-up signals, or equivalent trajectory units. Media validation is functional, video temporary files are cleaned on success and failure, and `/api/spiral/predict` returns `501 camera_to_tablet_feature_mapping_unvalidated` without calling the classifier.

## Tremor and sessions

The legacy manual `location` and 0–10 intensity endpoint remains a prototype and returns no prediction. Authenticated assessments can classify the exact seven UPDRS III item 20/21 features in the saved tremor model schema. Inputs are validated as integer ratings 0–4; the output is a research probability, not a diagnosis or severity estimate. Sessions are process-local with a one-hour default TTL and enforce speech → spiral → tremor. Failed steps are recorded and retryable; results are hidden until every step has succeeded.

## Functional status

| Endpoint | Status |
|---|---|
| `GET /api/health` | Functional: reports saved model readiness without fitting. |
| `GET /api/spiral/status` | Functional: reports model readiness and camera mapping blocker. |
| `POST /api/speech/predict` | WAV validation functional; classification blocked by missing verified raw-audio extraction and gender input contract. |
| `POST /api/spiral/predict` | Image/video validation functional; classification blocked by camera/tablet feature incompatibility. |
| `POST /api/assessments/<id>/tremor/predict` | Authenticated classification from exact seven trained UPDRS III item ratings; persists probability in assessment. |
| `POST /api/tremor/submit` | Functional as a manual prototype only; no classification or regression. |
| Assessment session routes | State machine/order enforcement functional; a real session cannot complete until speech and spiral prediction are supported. |

No Baseline V1 files, raw datasets, `Demographics.csv`, classification fusion weights, or trained artifacts are modified by the backend.
