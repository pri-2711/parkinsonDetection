# Assessment API

The backend is a Flask JSON API with no frontend. It loads selected tuned speech and handwriting SVM bundles from `experiments/regularized_svm_rf/results/artifacts`; it never fits a model. Median imputation, speech scaling, SVC, and sigmoid calibration are already in each saved model bundle. Baseline V1 is not loaded or modified by the backend.

## Start locally

Install project dependencies, set environment values from `.env.example`, then run from the repository root:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m app_backend
```

The server binds to `127.0.0.1:5000` with debug mode off. Flask CLI and `python -m app_backend` load `.env` when `python-dotenv` is installed. `APP_SECRET_KEY` is not used by the process-local session store.

## Health and readiness

`GET /api/health` reports liveness, saved-model loading, and modality capabilities. The classifiers provide calibrated probabilities, but live speech and camera-to-spiral prediction remain blocked until compatible input features are available. `GET /api/spiral/status` reports the handwriting model and the tablet signals it expects.

Example health response (readiness can show `model_loaded: false` if an artifact is missing):

```json
{"status":"ok","capabilities":{"speech_prediction":false,"spiral_prediction":false,"manual_tremor_prototype":true,"updrs_regression":false}}
```

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

## Assessment sessions

1. `POST /api/assessment/sessions` creates an in-memory session with `next_step: speech`.
2. Submit speech, spiral, then tremor, passing `session_id` in each multipart form or JSON body.
3. `GET /api/assessment/sessions/{session_id}` returns step state, explicit failures, and the retryable current step.
4. `POST /api/assessment/sessions/{session_id}/complete` succeeds only after all three steps complete.
5. `GET /api/assessment/sessions/{session_id}/results` only returns completed results.

Sessions are process-local, expire after `SESSION_TTL_SECONDS`, and are cleared on restart. No database is used. Because speech and camera feature mapping are blocked, a real session cannot currently reach results. The backend does not claim an end-to-end assessment.

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
