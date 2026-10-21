# Feature compatibility and endpoint status

## Saved artifacts

The backend loads the saved speech, handwriting, and tremor classifiers from `experiments/regularized_svm_rf/results/artifacts/`. The bundles contain fitted sklearn pipelines, preprocessing, calibrated probabilities, and feature schemas. The adapter checks exact feature order and class order `[0, 1]`, and performs no fitting. Baseline V1 is unchanged.

## Speech

The cleaned speech table has 240 rows and 48 columns: identifiers/target (`id`, `recording`, `status`), `gender`, and 44 acoustic descriptors. The model consumes 45 fields including numeric `gender`: Jitter variants, Shimmer variants, HNR values, RPDE, DFA, PPE, GNE, 13 MFCC values, and 13 Delta values. The UCI documentation encodes gender as 0=man and 1=woman. The source dataset task is a sustained /a/ vowel for at least five seconds, repeated three times, recorded at 44.1 kHz/16-bit; the app's current 15–20 second reading passage is a different task and is not a compatible input protocol.

`cleaningPipeline/speech_cleaning.py` reads and cleans an already-featured CSV; it does not derive descriptors from audio. The historic article does not resolve every implementation setting, so its exact schema cannot be recreated from these files. The app now uses the clearly separate 45-column `speech_audio_v2` extractor documented in `experiments/speech_audio_v2/README.md`, implemented with Parselmouth and custom SciPy signal processing. It accepts sustained vowel and reading mode independently and requires explicit gender metadata; no value is inferred. Neither route sends v2 values to the historical SVM. No labeled raw WAV recordings are available, so the v2 model-training script exists but no mode model has been trained; valid audio currently extracts then returns HTTP 503 with no prediction.

The active saved bundle's exact input order is:

```text
gender, jitter_rel, jitter_abs, jitter_rap, jitter_ppq,
shim_loc, shim_db, shim_apq3, shim_apq5, shi_apq11,
hnr05, hnr15, hnr25, hnr35, hnr38,
rpde, dfa, ppe, gne,
mfcc0, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7,
mfcc8, mfcc9, mfcc10, mfcc11, mfcc12,
delta0, delta1, delta2, delta3, delta4, delta5, delta6, delta7,
delta8, delta9, delta10, delta11, delta12
```

The original study specifies relative jitter as a percentage, uses rough F0 from normalized autocorrelation over 80 ms frames followed by waveform matching, and specifies cepstral HNR bands at 0–500/1500/2500/3500/3800 Hz averaged over voiced frames using VoiceSauce. It identifies shimmer/APQ measures, MFCCs and their temporal derivatives, and RPDE/DFA/PPE, but the article and repository do not give enough implementation settings to recreate every input exactly. Exact amplitude-measure formulas/units and thresholds, full MFCC/Delta settings, GNE implementation, preprocessing/normalization, and the gender metadata acquisition contract remain unresolved. Parselmouth/Praat, Librosa, and openSMILE are candidate tools, not proven reproductions; their defaults cannot establish compatibility. The repository has only precomputed descriptors, no paired original WAV recordings for extractor comparison, and no extractor code. Preserve this model as a baseline; do not send it approximate output. A replacement needs a labeled raw-audio dataset and its own documented, consistently extractable schema and evaluation.

The actual code intentionally does not claim that the v2 Praat or custom measurements match the old dataset. See the mode-specific algorithm, units, thresholds, limitations, and subject-level training protocol in the v2 README.

## Spiral / handwriting

The trained feature order is `n_points`, `path_length`, `mean_velocity`, `max_velocity`, `mean_acceleration`, `max_acceleration`, `mean_jerk`, `max_jerk`, `pressure_mean`, `pressure_std`, `pressure_range`, `grip_angle_std`, `movement_duration`, `pen_up_fraction`, `spatial_std`, `bounding_box_diagonal`.

The cleaning code reads tablet trajectories with x/y/z, pressure, grip angle, timestamp, test ID, and point order. `featureEngineering/handwriting_features.py` defines the training extractor. The authenticated `/api/assessments/<id>/spiral/predict` route reuses it and calls the saved classifier when complete measured tablet telemetry is supplied. It rejects missing sensor fields and does not synthesize pressure or grip angle. The canvas captures browser pointer telemetry but does not provide compatible calibrated tablet measurements; camera/video likewise does not provide validated equivalents. The separate image/video validation endpoint returns `501 camera_to_tablet_feature_mapping_unvalidated` without calling the classifier.

The canvas telemetry prototype records pointer event coordinates/times, stroke boundaries, browser-reported pressure when available, and pen tilt when available. It derives distance/speed/acceleration/jerk in canvas pixel/time units for inspection only. Those values are not passed to the saved model. Canvas pressure is normalized browser input rather than calibrated tablet pressure, tilt is not the dataset's grip angle, and stroke boundaries do not provide the training digitizer's `z` sensor. The prototype therefore reports model-feature coverage as approximate or unavailable rather than claiming schema compatibility.

## Tremor and sessions

The legacy manual `location` and 0–10 intensity endpoint remains a prototype and returns no prediction. Authenticated assessments can classify the exact seven UPDRS III item 20/21 features in the saved tremor model schema. Inputs are validated as integer ratings 0–4; the output is a research probability, not a diagnosis or severity estimate. Sessions are process-local with a one-hour default TTL and enforce speech → spiral → tremor. Failed steps are recorded and retryable; results are hidden until every step has succeeded.

## Functional status

| Endpoint | Status |
|---|---|
| `GET /api/health` | Functional: reports saved model readiness without fitting. |
| `GET /api/spiral/status` | Functional: reports model readiness and camera mapping blocker. |
| `POST /api/speech/predict` | Extracts `speech_audio_v2`; returns HTTP 503 after extraction until a mode-specific trained v2 model is available. Never uses the legacy SVM with v2 values. |
| `POST /api/spiral/predict` | Image/video validation functional; classification blocked by camera/tablet feature incompatibility. |
| `POST /api/assessments/<id>/spiral/predict` | Authenticated classification from complete measured tablet telemetry; rejects browser canvas coordinates without sensor data. |
| `POST /api/assessments/<id>/speech/predict` | Authenticated multipart WAV extraction using separate mode-specific v2 schema; audio is not stored. Returns HTTP 503 after extraction until a matching trained model exists. |
| `POST /api/assessments/<id>/tremor/predict` | Authenticated classification from exact seven trained UPDRS III item ratings; persists probability in assessment. |
| `POST /api/tremor/submit` | Functional as a manual prototype only; no classification or regression. |
| Database-backed assessment routes | Can complete with at least one verified model prediction; unsupported modalities are omitted and fusion weights renormalized. |
| Legacy session routes | State machine/order enforcement functional; requires all three steps, so cannot currently complete with supported inputs. |

No Baseline V1 files, raw datasets, `Demographics.csv`, classification fusion weights, or trained artifacts are modified by the backend.
