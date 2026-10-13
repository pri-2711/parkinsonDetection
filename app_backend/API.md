# Assessment API

The backend is a Flask JSON API used by the React frontend. It loads tuned speech, handwriting, and tremor classifier bundles from `experiments/regularized_svm_rf/results/artifacts`; it never fits a model. Preprocessing and calibrated probability estimators are in the saved bundles. Baseline V1 is not loaded or modified by the backend.

## Start locally

Install project dependencies, set environment values from `.env.example`, then run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -c "import secrets; from pathlib import Path; p=Path('.env'); p.exists() or p.write_text('HOST=127.0.0.1\\nPORT=5000\\nAPP_SECRET_KEY='+secrets.token_urlsafe(48)+'\\nLOG_LEVEL=INFO\\n', encoding='utf-8')"
python -m app_backend
```

The server binds to `127.0.0.1:5000` with debug mode off. App creation reloads the repository-root `.env` when `python-dotenv` is installed, including when a blank process-level `APP_SECRET_KEY` would otherwise mask the file. The file is ignored by Git. `APP_SECRET_KEY` signs authentication tokens and is required for registration and login. It is not used by the process-local session store.

## Health and readiness

`GET /api/health` reports liveness, saved-model loading, and modality capabilities. Tremor classification is available from compatible UPDRS ratings; speech and camera-to-spiral prediction remain blocked until compatible input features are available. `GET /api/spiral/status` reports the handwriting model and the tablet signals it expects.

The response includes a `models` object with each artifact's readiness, feature count, and prediction availability. Missing artifacts are reported as unavailable; health status is degraded if a model artifact fails to load.

## Speech

`POST /api/speech/predict` accepts multipart form data with a `file` field and optional `session_id`.

- Only uncompressed PCM WAV, mono or stereo, 8–48 kHz is accepted.
- Audio is limited to 16 MiB and 1–30 seconds by default.
- The WAV is validated and not application-persisted. Werkzeug may temporarily spool larger multipart requests and removes its request data afterward. After validation the route returns HTTP `501 raw_audio_feature_extractor_unavailable`; no prediction or probability is returned.
- The saved speech model expects 45 columns, including numeric `gender` and Jitter, Shimmer, HNR, RPDE, DFA, PPE, GNE, MFCC, and Delta features. The repository contains precomputed descriptors but no verified raw-waveform extractor or gender metadata contract.

Example error:

```json
{"error":{"code":"raw_audio_feature_extractor_unavailable","message":"Prediction is disabled: the repository contains precomputed acoustic features but no verified raw-audio feature extractor matching the saved model schema.","request_id":"..."}}
```

## Spiral

`POST /api/spiral/predict` accepts multipart `file` and optional `session_id`.

- Images: PNG/JPEG. Videos: MP4, MOV, AVI, or WebM when OpenCV can decode them.
- Uploads are limited to 20 MiB. Image dimensions are 16–4096 pixels; video is limited to 30 seconds and 1200 frames by default.
- Images are validated in memory. Video uses a randomized temporary file removed in `finally` on both success and failure.
- Valid media receives HTTP `501 camera_to_tablet_feature_mapping_unvalidated`; the route does not call the handwriting model.

Example error for a valid but incompatible image/video:

```json
{"error":{"code":"camera_to_tablet_feature_mapping_unvalidated","message":"The upload was validated, but prediction is disabled: camera input cannot currently reproduce the tablet pressure, grip-angle, pen-up, and trajectory features required by the saved handwriting model.","request_id":"..."}}
```

The 16 handwriting features come from tablet x/y/z, pressure, grip angle, timestamps, and test ID. A guarded tablet-data service reuses the existing `featureEngineering.handwriting_features.extract_features`, but camera imagery cannot supply validated pressure and grip-angle measurements or equivalent trajectory units.

## Manual tremor prototype

`POST /api/tremor/submit` accepts JSON and does not invoke the tremor classifier or UPDRS regression.

```json
{"location":"left_hand","intensity":4}
```

`location` must be in `TREMOR_ALLOWED_LOCATIONS`; `intensity` must be numeric from 0 through `TREMOR_MAX_INTENSITY` (default 10). A valid standalone request returns HTTP 201 with `prototype: true`, `classification: null`, and `updrs_estimate: null`. With `session_id`, tremor is accepted only after speech and spiral are complete.

Example response:

```json
{"status":"submitted","prototype":true,"classification":null,"updrs_estimate":null,"data":{"location":"left_hand","intensity":4.0}}
```

## Tremor classification from UPDRS ratings

`POST /api/assessments/<id>/tremor/predict` requires a bearer token and an in-progress assessment owned by that user. The JSON body must contain exactly the seven feature names from `tremor_svm_tuned.joblib`; each value must be an integer rating from 0 through 4. The response includes and persists the model's probability and predicted class as a research output, not a clinical diagnosis.

## Handwriting & spiral trajectory classification

`POST /api/assessments/<id>/spiral/predict` requires a bearer token. Accepts JSON with a `trajectory` or `points` array (point_index, x, y, z, pressure, grip_angle, timestamp, test_id) generated by the HTML5 drawing canvas or tablet telemetry CSV. Extracts the 16 kinematic features via `extract_tablet_trajectory_features` and executes `handwriting_svm_tuned.joblib`. Persists probability and predicted class in the assessment.

## Speech assessment and passage validation

`POST /api/assessments/<id>/speech/predict` accepts JSON metadata with `passage_version` ("1.0"), `passage_text`, `duration_seconds`, and `sample_rate`. Validates the standardized reading passage, logs metadata, and transparently returns HTTP 501 `raw_audio_feature_extractor_unavailable` until a full 45-acoustic-feature live pipeline is available.

## PDF assessment reports

`GET /api/assessments/<id>/report.pdf` downloads a professional ReportLab PDF report containing assessment metadata, 7-item UPDRS rating breakdown, handwriting telemetry metrics, speech passage record, late-fusion composite scores, methodology summary, and academic research disclaimers.


## Assessment sessions

The authenticated database-backed assessment workflow uses `/api/assessments` endpoints. It can complete after at least one verified model prediction; unsupported modalities are omitted and eligible weights are renormalized. Client-supplied probabilities cannot be saved through step-update endpoints. Classification probability is stored separately from severity; no severity model is currently exposed.

1. `POST /api/assessment/sessions` creates an in-memory session with `next_step: speech`.
2. Submit speech, spiral, then tremor, passing `session_id` in each multipart form or JSON body.
3. `GET /api/assessment/sessions/{session_id}` returns step state, explicit failures, and the retryable current step.
4. `POST /api/assessment/sessions/{session_id}/complete` succeeds only after all three steps complete.
5. `GET /api/assessment/sessions/{session_id}/results` only returns completed results.

These legacy sessions are process-local, expire after `SESSION_TTL_SECONDS`, and are cleared on restart. They are separate from the database-backed workflow above. The legacy flow still requires all three steps; because speech and camera feature mapping are blocked, it cannot currently reach results.

Example session creation response:

```json
{"session_id":"...","status":"in_progress","next_step":"speech","steps":{"speech":{"status":"pending"},"spiral":{"status":"pending"},"tremor":{"status":"pending"}}}
```

## Errors and security

Errors use `{"error":{"code":"...","message":"...","request_id":"..."}}`. Responses contain no stack traces or server paths. `X-Request-ID` is returned. Logs record the route and status only; they omit upload contents, session IDs, and tremor values. The app does not persist audio; Werkzeug may spool multipart bodies temporarily and cleans them when the request ends. Backend-created video temp files are deleted. Configure limits, temp path, session TTL, artifact paths, and tremor values through environment variables.

## Manual smoke test

Start the server, then from another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
Invoke-RestMethod http://127.0.0.1:5000/api/spiral/status
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/tremor/submit -ContentType 'application/json' -Body '{"location":"left_hand","intensity":4}'
```

Sending a valid WAV or PNG/JPEG/video to its prediction route should return the documented 501 blocker. This confirms upload validation and compatibility reporting; it is not a classifier smoke test.

Automated tests:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s app_backend/tests -v
```
