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

`GET /api/health` reports liveness, saved-model loading, and modality capabilities. Tremor classification is available from compatible UPDRS ratings; handwriting classification is available only from complete measured tablet telemetry. Speech extraction uses a separate versioned schema, but speech prediction requires a mode-specific model trained for it. `GET /api/spiral/status` reports model readiness, browser camera status, and the tablet signals the classifier expects.

The response includes a `models` object with each artifact's readiness, feature count, and prediction availability. Missing artifacts are reported as unavailable; health status is degraded if a model artifact fails to load.

## Speech

`POST /api/speech/predict` accepts multipart form data with `file`, `mode` (`sustained_vowel` or `reading`), `gender_code` (0 or 1), and optional `session_id`.

- Only uncompressed PCM WAV, mono or stereo, 8–48 kHz is accepted.
- Audio is limited to 16 MiB and 1–30 seconds by default.
- Sustained-vowel mode requires at least five seconds. The WAV is processed in memory and is not application-persisted.
- Extraction returns the separate 45-column `speech_audio_v2` schema documented in `experiments/speech_audio_v2/README.md`. Assessment workflow requests save those feature values and return HTTP 200 with `prediction_status: unavailable` because this repository has no labeled raw WAV training dataset. The legacy model is never called with v2 features.

Example error:

```json
{"error":{"code":"speech_audio_v2_model_not_trained","message":"Audio features were extracted with speech_audio_v2, but no model has been trained for this recording mode.","request_id":"..."}}
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

`POST /api/assessments/<id>/spiral/predict` requires a bearer token. Accepts JSON with a `trajectory` or `points` array containing measured tablet fields `x`, `y`, `z`, `pressure`, `grip_angle`, and `timestamp` (with optional `point_index` and `test_id`). It extracts the 16 kinematic features using the shared training extractor and calls `handwriting_svm_tuned.joblib`, then persists the model probability and class. Missing sensor fields return HTTP 422 `tablet_telemetry_required`; values are never filled with defaults. The browser canvas records only 2D coordinates and timing, so it cannot currently supply a compatible classifier request. The UI keeps the canvas and offers a separate authenticated video-analysis upload; no tablet telemetry CSV upload is provided.

`POST /api/assessments/<id>/spiral/video/analyze` requires a bearer token and multipart field `file` containing MP4, MOV, AVI, or WebM video. It validates file size, decoding, duration, frame count, and dimensions; creates an OpenCV annotated MP4; and stores the original and annotation privately under the authenticated assessment. The response contains pixel-space trajectory observations, candidate red movement and blue appearance events, and limitations. This is an explainable visual baseline only: it does not call the handwriting classifier or return a Parkinson's probability. Retrieve media with the authenticated `GET /api/assessments/<id>/spiral/video/annotated` and `/original` endpoints. The annotated sidecar is included in assessment history and PDF reporting.

## Speech assessment and passage validation

`POST /api/assessments/<id>/speech/predict` requires authentication and multipart form data: a PCM WAV `file`, `mode`, and `gender_code`; passage metadata is optional for reading mode. The backend saves extracted v2 feature values and metadata, but does not persist audio. Until a matching v2 model is trained, it returns HTTP 200 with `prediction_status: unavailable` and no probability. The process-local legacy session endpoint remains separate and may return 503 when no matching mode model exists.

`POST /api/assessments/<id>/spiral/canvas/observe` requires authentication and JSON containing `samples`, `canvas_width`, and `canvas_height`. It saves a 2D canvas telemetry summary and marks classifier prediction unavailable because the values do not match the digitizer-trained handwriting model schema.

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

Errors use `{"error":{"code":"...","message":"...","request_id":"..."}}`. Responses contain no stack traces or server paths. `X-Request-ID` is returned. Logs record the route and status only; they omit upload contents, session IDs, and tremor values. The app does not persist audio; Werkzeug may spool multipart bodies temporarily and cleans them when the request ends. Video-analysis intermediates are deleted; source and annotated videos remain in the private `SPIRAL_MEDIA_DIR` (default `data/spiral_media`) for authenticated assessment access. Configure limits, temp path, session TTL, artifact paths, and tremor values through environment variables.

## Manual smoke test

Start the server, then from another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
Invoke-RestMethod http://127.0.0.1:5000/api/spiral/status
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/tremor/submit -ContentType 'application/json' -Body '{"location":"left_hand","intensity":4}'
```

Sending a valid speech WAV without a trained v2 artifact returns 503 after extraction. This confirms the extraction/model gate; it is not a prediction smoke test.

Automated tests:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s app_backend/tests -v
```
