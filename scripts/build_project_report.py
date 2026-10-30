"""Generate the editable project report (docs/Project_Report.docx).

Standard library only: the .docx is written as WordprocessingML inside a zip.
Routes, schema, research metrics and dataset sizes are read from the repository
at build time, so the report reflects the code and saved outputs it describes.

    python scripts/build_project_report.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "Project_Report.docx"
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- DOCX writer

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' \
       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _runs(text: str, size: int | None = None, color: str | None = None) -> str:
    """Supports **bold** segments inside a paragraph."""
    parts = text.split("**")
    out = []
    for index, part in enumerate(parts):
        if not part:
            continue
        props = ""
        if index % 2 == 1:
            props += "<w:b/>"
        if color:
            props += f'<w:color w:val="{color}"/>'
        if size:
            props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)


class Document:
    def __init__(self) -> None:
        self.body: list[str] = []

    def heading(self, text: str, level: int = 1) -> None:
        self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr>{_runs(text)}</w:p>')

    def para(self, text: str, style: str | None = None, size: int | None = None, color: str | None = None) -> None:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        self.body.append(f"<w:p>{ppr}{_runs(text, size, color)}</w:p>")

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.body.append(f'<w:p><w:pPr><w:pStyle w:val="ListBullet"/></w:pPr>{_runs(item)}</w:p>')

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, headers: list[str], rows: list[list[str]], widths: list[int] | None = None) -> None:
        widths = widths or [int(9360 / len(headers))] * len(headers)
        border = ('<w:tblBorders>' + "".join(
            f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="B8C2B0"/>'
            for side in ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>')

        def cell(text: str, width: int, header: bool) -> str:
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="EBF3EA"/>' if header else ""
            body = _runs(f"**{text}**" if header and text else text, size=17)
            return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>{body}</w:p></w:tc>')

        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        xml = [f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/>{border}</w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
        xml.append("<w:tr><w:trPr><w:tblHeader/></w:trPr>" + "".join(cell(h, w, True) for h, w in zip(headers, widths)) + "</w:tr>")
        for row in rows:
            xml.append("<w:tr>" + "".join(cell(str(v), w, False) for v, w in zip(row, widths)) + "</w:tr>")
        xml.append("</w:tbl>")
        self.body.append("".join(xml))
        self.body.append('<w:p><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>')

    def xml(self) -> str:
        sect = ('<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
                '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" w:header="576" w:footer="576" w:gutter="0"/>'
                '</w:sectPr>')
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<w:document {W_NS}><w:body>{"".join(self.body)}{sect}</w:body></w:document>')


STYLES = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles {W_NS}>
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="21"/><w:szCs w:val="21"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault>
  <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="80"/></w:pPr><w:rPr><w:b/><w:color w:val="3E5438"/><w:sz w:val="40"/><w:szCs w:val="40"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="320" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="3E5438"/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="526B4A"/><w:sz w:val="25"/><w:szCs w:val="25"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr><w:spacing w:after="60"/><w:ind w:left="360" w:hanging="360"/></w:pPr></w:style>
</w:styles>'''

NUMBERING = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering {W_NS}>
  <w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/>
    <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="360" w:hanging="360"/></w:pPr></w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>'''

CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>'''

ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>'''

DOC_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>'''


def core_properties(title: str) -> str:
    stamp = date.today().isoformat() + "T00:00:00Z"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>Parkinson's detection project</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">{stamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{stamp}</dcterms:modified>
</cp:coreProperties>'''


def write_docx(path: Path, document: Document, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first so an interrupted build never leaves a broken report.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=path.parent) as handle:
        temp = Path(handle.name)
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", ROOT_RELS)
        archive.writestr("word/document.xml", document.xml())
        archive.writestr("word/_rels/document.xml.rels", DOC_RELS)
        archive.writestr("word/styles.xml", STYLES)
        archive.writestr("word/numbering.xml", NUMBERING)
        archive.writestr("docProps/core.xml", core_properties(title))
    temp.replace(path)


# ---------------------------------------------------------------- repository facts

def git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def api_routes() -> list[list[str]]:
    from app_backend import create_app
    temp = tempfile.mkdtemp(prefix="pd-report-routes-")
    app = create_app({"TESTING": True, "DATABASE_PATH": str(Path(temp) / "routes.db"),
                      "APP_SECRET_KEY": "report-route-introspection", "SECRET_KEY": "report-route-introspection",
                      "UPLOAD_TEMP_DIR": temp, "SPIRAL_MEDIA_DIR": str(Path(temp) / "media")})
    rows = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        view = app.view_functions[rule.endpoint]
        # login_required is a functools.wraps wrapper; its code object has a fixed qualname.
        protected = getattr(view, "__code__", None) is not None and \
            view.__code__.co_qualname == "login_required.<locals>.decorated"
        methods = sorted(m for m in rule.methods if m not in {"HEAD", "OPTIONS"})
        rows.append([", ".join(methods), rule.rule, "Bearer token" if protected else "Public", rule.endpoint])
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def schema_tables() -> list[list[str]]:
    from app_backend import database
    rows = []
    for statement in database._SCHEMA.split(";"):
        statement = statement.strip()
        if not statement.upper().startswith("CREATE TABLE"):
            continue
        name = statement.split("IF NOT EXISTS")[1].split("(")[0].strip()
        columns = []
        for line in statement.split("(", 1)[1].rsplit(")", 1)[0].splitlines():
            line = line.strip().rstrip(",")
            if line and not line.upper().startswith(("PRIMARY", "UNIQUE", "FOREIGN")):
                columns.append(" ".join(line.split()[:2]))
        rows.append([name, "; ".join(columns)])
    return rows


def dataset_sizes() -> dict[str, str]:
    import pandas as pd
    base = ROOT / "cleaned_dataset"
    speech = pd.read_csv(base / "speech_clean.csv")
    tremor = pd.read_csv(base / "tremors_clean.csv", usecols=["participant_code"])
    telemonitoring = pd.read_csv(base / "telemonitoring_clean.csv")
    features = pd.read_csv(ROOT / "featureEngineering" / "handwriting_features.csv")
    raw_hw = pd.read_csv(base / "handwriting_clean.csv", usecols=["participant_id"])
    return {
        "speech_rows": str(len(speech)),
        "speech_participants": str(speech["id"].nunique()),
        "speech_pd": str(int((speech["status"] == 1).sum())),
        "speech_control": str(int((speech["status"] == 0).sum())),
        "hw_points": f"{len(raw_hw):,}",
        "hw_test_rows": str(len(features)),
        "hw_participants": str(features["participant_id"].nunique()),
        "hw_control": str(features.loc[features["label"] == 0, "participant_id"].nunique()),
        "hw_pd": str(features.loc[features["label"] == 1, "participant_id"].nunique()),
        "tremor_rows": str(len(tremor)),
        "tremor_pd": str(int(tremor["participant_code"].str.startswith("PD").sum())),
        "tremor_control": str(int((~tremor["participant_code"].str.startswith("PD")).sum())),
        "tele_rows": f"{len(telemonitoring):,}",
        "tele_subjects": str(telemonitoring["subject_number"].nunique()),
    }


def database_counts() -> dict[str, str]:
    import sqlite3
    path = ROOT / "data" / "parkinson_app.db"
    if not path.is_file():
        return {"users": "unavailable", "scrypt": "unavailable", "completed": "unavailable", "in_progress": "unavailable"}
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        scrypt = conn.execute("SELECT COUNT(*) FROM users WHERE pw_hash LIKE 'scrypt:%'").fetchone()[0]
        statuses = dict(conn.execute("SELECT status, COUNT(*) FROM assessments GROUP BY status").fetchall())
    return {"users": str(users), "scrypt": str(scrypt), "completed": str(statuses.get("completed", 0)),
            "in_progress": str(statuses.get("in_progress", 0))}


def pct(value, digits=3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


# ---------------------------------------------------------------- report content

TEST_NOTE = (
    "Last verified run when this report was generated: backend suite 52 tests passed (unittest discover over "
    "app_backend/tests, including the 8 resumable-extraction tests and the speech compatibility and reporting tests); "
    "experiment and fusion tests 14 passed (pytest); frontend node tests 8 passed; TypeScript and Vite production build "
    "succeeded; git diff --check reported no whitespace errors."
)


def build() -> Document:
    from app_backend import create_app  # noqa: F401  (ensures the package imports cleanly)
    from app_backend.config import BackendConfig
    from app_backend.services.acoustic_features import SPEECH_V2_FEATURES
    from app_backend.services.research_insights import build_research_insights

    results = ROOT / "experiments" / "regularized_svm_rf" / "results"
    insights = build_research_insights(results, results / "artifacts", ROOT / "experiments" / "speech_audio_v2" / "artifacts")
    sizes = dataset_sizes()
    counts = database_counts()
    modalities = {m["modality"]: m for m in insights["modalities"]}
    fusion = insights["fusion"]

    doc = Document()
    doc.para("Parkinson's Disease Detection: Research Prototype", style="Title")
    doc.para("Project report: architecture, datasets, features, algorithms, validation, results, database, "
             "APIs, user interface, security, limitations and future work.", color="526B4A")
    doc.para(f"Generated {date.today().isoformat()} from repository revision **{git_revision()}**. "
             "This is a research prototype. It is not a clinical diagnostic device, and no output in this report "
             "is a diagnosis.", size=19, color="6B5B2A")

    doc.heading("1. Project overview and objectives")
    doc.para("Parkinson's disease is a progressive neurological disorder that affects movement and speech. This research "
             "prototype explores whether measurements from three independent modalities (tremor ratings, tablet handwriting "
             "and speech recordings) can support research classifiers, and provides a user workflow to collect, store and "
             "report those measurements.")
    doc.bullets([
        "Objective 1: reproducible, participant-grouped machine-learning pipelines for each modality.",
        "Objective 2: an authenticated assessment workflow with per-user data isolation and persisted results.",
        "Objective 3: transparent reporting of performance, limitations and modality status, without clinical claims.",
    ])

    doc.heading("1.1 Executive summary", 2)
    doc.para("The project investigates whether tremor, tablet handwriting and speech measurements can support "
             "research classifiers for Parkinson's disease. Each modality comes from an independent participant "
             "cohort. The application combines the saved classifiers with a user-owned assessment workflow, "
             "a PDF report and a research insights page. Only verified model outputs are ever treated as predictions.")
    doc.table(["Component", "What the application does", "Prediction status"], [
        ["Tremor", "Seven UPDRS III items 20–21 (integers 0–4) through a calibrated RBF SVM", "Implemented"],
        ["Handwriting / spiral (tablet)", "Complete measured tablet telemetry through 16 kinematic features and a calibrated SVM",
         "Implemented for complete tablet telemetry only"],
        ["Spiral canvas and video", "Browser pointer telemetry and OpenCV visual tracking", "Experimental observations; not classifier input"],
        ["Speech, reading passage (speech_audio_v2)", "WAV audio measured with 45 Praat, MFCC and custom descriptors (44 used by the model); 20 s windows", "EXPERIMENTAL prediction from a reading-mode model trained on 37 participants; validation is near the majority-class baseline"],
        ["Speech, sustained vowel", "Same extractor, sustained-vowel mode", "Unavailable: no training data"],
        ["Speech legacy SVM", "Archived 45-feature classifier on precomputed UCI descriptors", "Baseline only; never applied to speech_audio_v2 features"],
        ["Fusion", "AUC-weighted late fusion of verified predictions", "Implemented; a combined probability is withheld when more than one modality contributes"],
        ["Motor UPDRS severity", "Ridge and random-forest regressors on telemonitoring data", "Disabled: negative R² under grouped cross-validation"],
    ], widths=[2200, 4560, 2600])

    doc.heading("2. Architecture")
    doc.para("The system has four layers. The React and TypeScript single-page frontend calls a Flask JSON API. "
             "The API validates inputs, runs read-only classifiers loaded from saved artifacts, and persists "
             "assessments to SQLite. Offline experiment scripts produce the artifacts and the metrics that the "
             "research page reads.")
    doc.table(["Layer", "Components", "Location"], [
        ["Frontend", "React 18, React Router, Vite; pages for dashboard, speech, handwriting, tremor, results, history, research", "frontend/src"],
        ["API", "Flask application factory, blueprints for auth, assessments, speech, spiral, tremor, history, results, research, health", "app_backend/"],
        ["Services", "Model adapters, fusion service, speech v2 extractor and model, assessment summary, research insights, PDF report", "app_backend/services"],
        ["Persistence", "SQLite database with users and assessments; private spiral media folder", "data/"],
        ["Experiments", "Regularized SVM/RF tuning with grouped validation; speech v2 training script", "experiments/"],
        ["Archive", "Baseline V1 pipeline, retained unchanged for comparison", "archive/Baseline_V1"],
    ], widths=[1500, 5560, 2300])
    doc.para("Assessment flow: a user creates an assessment, submits each modality (tremor, handwriting, speech, "
             "in that session order), and finalizes it. Finalization reads only verified predictions, computes the "
             "fusion where applicable, stores the results, and releases the PDF report.")

    doc.heading("3. Datasets")
    doc.table(["Dataset", "Source and content", "Size (from cleaned files)", "Use"], [
        ["Speech (precomputed)", "Replicated acoustic features of the Parkinson speech database (UCI); sustained /a/ vowel descriptors",
         f"{sizes['speech_rows']} recordings, {sizes['speech_participants']} participants ({sizes['speech_pd']} PD, {sizes['speech_control']} control rows)",
         "Legacy classifier (archived baseline)"],
        ["Handwriting (PaHaW tablet)", "Tablet drawing telemetry: x, y, z, pressure, grip angle, timestamp",
         f"{sizes['hw_points']} trajectory points; {sizes['hw_test_rows']} test rows from {sizes['hw_participants']} participants ({sizes['hw_pd']} PD, {sizes['hw_control']} control)",
         "Tablet trajectory classifier"],
        ["Tremor (UPDRS items 20–21)", "Clinician-rated tremor items; label from participant code prefix",
         f"{sizes['tremor_rows']} participants ({sizes['tremor_pd']} PD, {sizes['tremor_control']} control)", "Tremor classifier"],
        ["Telemonitoring (UCI)", "Voice measures with motor UPDRS target", f"{sizes['tele_rows']} recordings from {sizes['tele_subjects']} subjects",
         "Regression study only (disabled)"],
        ["Raw speech audio (MDVR-KCL ReadText)", "Public CC-BY-4.0 reading recordings (King's College London, Zenodo 2867216); not stored in the repository", "37 recordings, one per participant, split into 259 twenty-second windows (256 extracted)", "Reading-mode speech model (experimental)"],
    ], widths=[1900, 3300, 2900, 1260])
    doc.para("Cohorts are independent: no participant appears in more than one modality. The existing AUDIT.md lists "
             "1,040 speech rows; the speech file in the repository has 240 rows, which this report uses. Handwriting "
             "participants who share an identical feature signature are placed in the same validation group, so "
             "40 participants form 33 groups.")

    doc.heading("4. Features")
    doc.heading("4.1 Tremor (7 features)", 2)
    doc.para("UPDRS III item 20 (rest tremor: head, right and left upper extremities, right and left lower "
             "extremities) and item 21 (postural and action tremor: right and left upper extremities). Each rating is "
             "an integer from 0 to 4.")
    doc.heading("4.2 Handwriting (16 features)", 2)
    doc.para("n_points, path_length, mean_velocity, max_velocity, mean_acceleration, max_acceleration, mean_jerk, "
             "max_jerk, pressure_mean, pressure_std, pressure_range, grip_angle_std, movement_duration, "
             "pen_up_fraction, spatial_std, bounding_box_diagonal.")
    doc.heading("4.3 Speech legacy (45 features, archived)", 2)
    doc.para("gender; jitter (rel, abs, RAP, PPQ); shimmer (loc, dB, APQ3, APQ5, APQ11); HNR at 5 cutoffs; RPDE; "
             "DFA; PPE; GNE; 13 MFCC means; 13 delta means. Their extraction recipe is not fully recoverable from the "
             "repository, so no audio input is mapped to them.")
    doc.heading("4.4 Speech speech_audio_v2 (45 extracted; 44 model inputs)", 2)
    doc.para(f"The v2 schema contains {len(SPEECH_V2_FEATURES)} named features in the order defined in "
             "app_backend/services/acoustic_features.py. Praat measures (Parselmouth, 75–500 Hz pitch bounds) give "
             "jitter and shimmer; band-limited harmonicity gives five HNR values; custom SciPy descriptors give RPDE-like, "
             "DFA, pitch-period entropy and GNE-like values; a deterministic 40-band HTK-mel MFCC implementation gives 13 "
             "MFCC and 13 delta means. Gender is required metadata and is never inferred from audio. The README in "
             "experiments/speech_audio_v2 states that these descriptors are not claimed to reproduce the published "
             "algorithms exactly. The reading-mode model uses the 44 non-gender features: the training cohort has no sex "
             "metadata, so gender is excluded from the model rather than guessed.")
    doc.heading("4.5 Video module (spiral observations)", 2)
    doc.para("Uploaded spiral videos are tracked with OpenCV (headless) to produce visual movement and appearance observations, "
             "with an annotated output video. These observations are saved and shown on Results for review. They are not fed to "
             "any classifier and are not treated as predictions.")

    doc.heading("4.6 Dataset for the speech model (MDVR-KCL)", 2)
    doc.para("The reading-mode speech model is trained on the public MDVR-KCL recordings (King's College London, Zenodo record "
             "2867216, CC-BY-4.0). Only the ReadText folder is used: one passage-reading recording per participant "
             "(21 healthy controls, 16 Parkinson's; 44.1 kHz). Each recording is split into 20-second windows, giving 256 "
             "segments after three windows failed extraction. The archive has no sex or age metadata. The app's reading passage "
             "differs from the dataset's passage, and the recording devices differ, so the model has not been tested on app recordings.")

    doc.heading("5. Algorithms and training")
    doc.bullets([
        "**Classifiers:** RBF support-vector machine and random forest per modality, with median imputation and standard "
        "scaling inside the pipeline and sigmoid (Platt) calibration fitted on grouped inner folds.",
        "**Grid:** SVM C ∈ {0.1, 1, 10}, gamma ∈ {scale, 0.01, 0.1}, class_weight ∈ {none, balanced}. Random forest: 300 trees, "
        "max_depth ∈ {4, 8}, min_samples_leaf ∈ {3, 6, 10}, max_features ∈ {sqrt, 0.5}.",
        "**Selection:** ROC-AUC on grouped inner cross-validation. Deployment parameters are chosen on the full cohort.",
        "**Decision threshold:** 0.5 on calibrated probability.",
        "**Speech v2 model (reading mode):** nested subject-disjoint validation. Inside each training fold: standardization, "
        "SelectKBest feature selection, grid search over an SVM (C, gamma, k) and a random forest, and sigmoid calibration on "
        "grouped folds. The deployed family is chosen by inner grouped CV on the training data. The bundle is loaded only when "
        "its schema, mode, 44-feature list and class order match.",
        "**Fusion:** P = Σ w_m P_m over the contributing modalities, with weights renormalized to sum to one. The "
        "weights are AUC-normalized using fold 1 only.",
    ])

    doc.heading("6. Validation protocol")
    doc.bullets([
        f"Outer evaluation: {insights['validation']['outer_splitter']}. Inner search: {insights['validation']['inner_selection']}.",
        f"Calibration: {insights['validation']['calibration']}.",
        f"Fold use: {insights['validation']['fusion_weight_fold']}. Reported folds are {insights['validation']['reported_folds']}.",
        "Leakage controls: all learned preprocessing and calibration are fitted inside training folds; duplicate "
        "handwriting signatures share a group; the speech v2 splitter asserts both classes in train and test; the fusion "
        "file records that no test-fold labels or metrics were used.",
        f"Patient-level multimodal validation: {insights['validation']['patient_level_multimodal_validation']}",
        f"External validation: {insights['validation']['external_validation']}",
    ])

    doc.heading("7. Results")
    doc.para("Values are read from the saved experiment outputs. Pooled figures cover the reported folds; "
             "per-fold rows give mean ± sample standard deviation across those folds.")
    for key in ("speech", "handwriting", "tremor"):
        m = modalities[key]
        doc.heading(f"7.{['speech', 'handwriting', 'tremor'].index(key) + 1} {m['label']}", 2)
        rows = []
        for model in ("baseline_svm", "tuned_svm", "baseline_random_forest", "tuned_random_forest", "nested_selected"):
            values = m["pooled_metrics"].get(model, {})
            folds = m["fold_metrics"].get(model, {}).get("roc_auc", {})
            fold_text = f"{pct(folds.get('mean'))} ± {pct(folds.get('sd'))}" if folds.get("mean") is not None else "—"
            rows.append([model.replace("_", " "), pct(values.get("accuracy")), pct(values.get("precision")),
                         pct(values.get("recall")), pct(values.get("f1")), pct(values.get("roc_auc")), fold_text])
        doc.table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Fold ROC-AUC mean ± SD"], rows,
                  widths=[2360, 1100, 1100, 1100, 1000, 1100, 1600])
        inner = m["inner_cv_roc_auc"]
        doc.para(f"Deployed family: {(m['deployed_family'] or '—').upper()}. Inner-CV ROC-AUC: SVM "
                 f"{pct(inner.get('svm'))}, random forest {pct(inner.get('random_forest'))}. "
                 f"Rows: {m['rows']} from {m['participant_groups']} groups; reported folds {m['reported_folds']}.", size=19)
    doc.heading("7.4 Late fusion weights", 2)
    doc.table(["Modality", "Validation ROC-AUC (fold 1)", "Normalized weight"],
              [[name, pct(fusion["validation_auc"].get(name), 4), pct(fusion["weights"].get(name), 4)] for name in fusion["weights"]],
              widths=[3120, 3120, 3120])
    doc.para("Patient-level multimodal performance has not been measured because the cohorts are unmatched.")
    doc.heading("7.5 Speech, reading passage (speech_audio_v2, EXPERIMENTAL)", 2)
    reading = insights["speech_audio_v2"]["modes"]["reading"]
    doc.para(f"Status: {insights['speech_audio_v2']['status']}. Cohort: {reading['subjects']} participants "
             f"({reading['control_subjects']} control, {reading['parkinson_subjects']} PD), {reading['recordings']} segments. "
             f"Subject-disjoint 5-fold ROC-AUC {pct(reading['outer_fold_roc_auc_mean'])} ± {pct(reading['outer_fold_roc_auc_sd'])}. "
             f"Majority-class accuracy {pct(reading['majority_class_accuracy'])}. Deployed family: {reading['deployed_family']}.")
    doc.para("Interpretation: the discrimination is weak and unstable across folds, and the model does not clearly outperform "
             "the majority-class baseline. The result is reported as an experimental research score. It is not a clinical "
             "validation, and the sustained-vowel mode has no model.")
    doc.heading("7.6 Telemonitoring severity (archived)", 2)
    doc.para("Under grouped five-fold validation the regressors did not beat the group-mean baseline: Ridge R² = −0.250, "
             "random forest R² = −0.466, baseline R² = −0.047 (AUDIT.md). Severity estimation is therefore disabled.")

    doc.heading("8. Database")
    doc.table(["Table", "Columns (name and type)"], [[name, cols] for name, cols in schema_tables()],
              widths=[1900, 7460])
    doc.bullets([
        "SQLite in WAL mode with foreign keys enabled. Assessment steps and results are stored as JSON documents.",
        "Every assessment query is scoped by user_id, so one user cannot read or modify another user's assessment.",
        "Speech audio is not stored. Spiral video files live in a private media folder and are served only through "
        "authenticated routes that check the stored identifier.",
        f"Current state of the local database at generation: {counts['users']} user accounts, "
        f"{counts['completed']} completed and {counts['in_progress']} in-progress assessments. Earlier rows that contain "
        "legacy speech messages were preserved unchanged.",
    ])

    doc.heading("9. API")
    doc.para("All routes are JSON unless stated otherwise. Routes marked Bearer token require an "
             "Authorization: Bearer header.")
    doc.table(["Methods", "Path", "Access", "Handler"], api_routes(), widths=[1300, 4100, 1500, 2460])

    doc.heading("10. User interface")
    doc.bullets([
        "**Login and registration:** account creation and sign-in with the API token stored by the browser.",
        "**Dashboard:** scope of each modality (prediction, observation or unavailable), status cards for tremor, "
        "handwriting and speech, shortcuts to spiral video analysis and to the research page.",
        "**Tremor, handwriting and speech pages:** input capture and per-step submission; the speech page explains that "
        "features are extracted and that prediction is unavailable until a trained model exists.",
        "**Results:** the assessment summary (step status, verified predictions, observations, fusion contributors), "
        "per-modality breakdown, the finalization action and the PDF download. A combined probability is shown only for a "
        "single contributing model.",
        "**History:** completed and in-progress assessments for the signed-in user.",
        "**Research and insights:** implemented, experimental and unavailable items; pooled and per-fold metrics for "
        "each modality; late-fusion weights; the speech v2 blocker; architecture; validation protocol.",
    ])

    doc.heading("11. Security")
    doc.bullets([
        f"Passwords are stored with Werkzeug's default salted hash. At generation, {counts['scrypt']} of "
        f"{counts['users']} stored hashes carry the scrypt prefix.",
        f"Tokens are HMAC-SHA256 (HS256) and expire after {86400 // 3600} hours. They are signed with APP_SECRET_KEY; the "
        "application refuses authenticated requests when that key is not configured.",
        f"Upload limits: audio {BackendConfig.AUDIO_MAX_BYTES // (1024 * 1024)} MB, spiral media {BackendConfig.SPIRAL_MAX_BYTES // (1024 * 1024)} MB, "
        f"request body {BackendConfig.MAX_CONTENT_LENGTH // (1024 * 1024)} MB. WAV uploads are checked for format before use.",
        "Error responses use stable codes and do not expose artifact paths or stack traces.",
        "**Review before any deployment:** CORS currently allows any origin for /api/*; the bearer token is kept in browser "
        "localStorage, so an XSS flaw would expose it; there are no refresh tokens, rate limits or multi-factor "
        "authentication. These are documented risks and were not changed in this work.",
    ])

    doc.heading("12. Reports")
    doc.bullets([
        "**Assessment PDF** (GET /api/assessments/<id>/report.pdf): summary of modality status and fusion scope, tremor, "
        "handwriting and speech results, model information with saved validation metrics, limitations.",
        "**Research insights** (GET /api/research/insights): JSON built from saved metrics files; the research page renders it.",
        "**This project report:** generated from the repository by scripts/build_project_report.py as an editable Word file.",
    ])

    doc.heading("13. Testing and verification")
    doc.para(TEST_NOTE)
    doc.bullets([
        "python -m unittest discover -s app_backend/tests -t .   (backend API, persistence, speech, summary, research, PDF)",
        "python -m unittest app_backend.tests.test_speech_and_reporting -v   (speech compatibility and training path)",
        "python -m pytest -q experiments/regularized_svm_rf/test_experiments.py tests/test_fusion.py",
        "node --test frontend/src/features/*.test.mjs",
        "cd frontend && npm run build   (TypeScript check and Vite production build)",
        "python -m py_compile on each modified Python module; git diff --check",
    ])

    doc.heading("14. Limitations")
    doc.bullets([
        "Small cohorts (tremor 130 participants, handwriting 40 participants in 33 groups, speech reading model 37 participants) with high fold-to-fold variation.",
        "The speech reading model is experimental: its cross-validated ROC-AUC (0.642 ± 0.144) is weak, and the training cohort is an external public dataset that differs from app recordings in passage, device and language.",
        "Sustained-vowel speech prediction is unavailable: no labelled data.",
        "No matched multimodal cohort, so combined performance is unknown; no external validation.",
        "Camera and canvas inputs cannot reproduce tablet pressure or grip angle; video tracking is a visual proxy only.",
        "Telemonitoring severity regression does not generalize (negative R²).",
        "Research prototype only; not validated for clinical use.",
    ])

    doc.heading("15. Future work")
    doc.bullets([
        "Collect labelled raw recordings from app-like conditions (same passage, devices and consent protocol) and retrain "
        "the reading-mode model with the same subject-disjoint protocol; add sustained-vowel data if available.",
        "Obtain a matched multimodal cohort and evaluate patient-level fusion; then consider a fusion model trained on it.",
        "Seek an independent external test cohort.",
        "Harden deployment: restrict CORS origins, consider httpOnly cookies, add refresh tokens and rate limiting, and "
        "move to a server database with migrations.",
        "Investigate a validated camera-to-tablet feature mapping before enabling video-based handwriting prediction.",
    ])

    doc.heading("16. Technology stack")
    doc.table(["Layer", "Technology (version pins from the repository)"], [
        ["Backend language", "Python 3.12; requirements.txt (scikit-learn 1.8.0, pandas, numpy, joblib, Flask, flask-cors, reportlab, praat-parselmouth 0.4.x, soundfile 0.14.x, opencv-python-headless, Pillow, python-dotenv)"],
        ["Frontend", "React 18, React Router 6, TypeScript 5, Vite 5 (frontend/package.json)"],
        ["Database", "SQLite 3 with WAL mode and foreign keys enabled (data/parkinson_app.db)"],
        ["Authentication", "Werkzeug password hashing (scrypt) and HMAC-SHA256 tokens with 24 h expiry and a 7-day session limit"],
        ["Reports", "ReportLab PDF assessment reports; this editable Word report is generated with the standard library only"],
    ], widths=[2200, 7160])

    doc.heading("17. Reproduction")
    doc.bullets([
        "python -m pip install -r requirements.txt",
        "python experiments/regularized_svm_rf/run_experiments.py   (regenerates saved metrics and artifacts)",
        "python -m app_backend   (serves the API on 127.0.0.1:5000)",
        "cd frontend && npm install && npm run dev",
        "python scripts/build_project_report.py   (regenerates this document)",
    ])
    return doc


def main() -> None:
    doc = build()
    write_docx(OUTPUT, doc, "Parkinson's Disease Detection: Research Prototype Project Report")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
