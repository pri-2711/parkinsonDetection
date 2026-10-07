from models.handwriting import run_handwriting_models
from models.speech import run_speech_models
from models.telemonitoring import run_telemonitoring_models
from models.tremor import run_tremor_models


def run_classification_models():
    return {
        "speech": run_speech_models(),
        "handwriting": run_handwriting_models(),
        "tremor": run_tremor_models(),
    }


def run_regression_models():
    return run_telemonitoring_models()


def build_fusion_probability_summary():
    return {
        "fusion_strategy": "not trained",
        "description": "Speech, handwriting, and tremor cohorts do not contain matched participants. A multimodal fusion model requires a matched cohort and must not be fabricated from independent datasets.",
    }


if __name__ == "__main__":
    print("Training multimodal Parkinson detection models...")
    classification_results = run_classification_models()
    regression_results = run_regression_models()
    print(f"\nClassification results: {classification_results}")
    print(f"\nRegression results: {regression_results}")
    print(f"\nFusion strategy: {build_fusion_probability_summary()}")
