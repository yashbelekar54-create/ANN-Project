"""Tests that need the TRAINED model. They are skipped (not passed) when it does not exist."""
import json
import subprocess
import sys

import pytest

from src.config import PROJECT_ROOT, load_config, resolve
from src.predict import artifact_status

pytest.importorskip("tensorflow")
CFG = load_config()
STATUS = artifact_status(CFG)
pytestmark = pytest.mark.skipif(not STATUS["ready"], reason="trained model artifacts not present - run python -m src.train")


def test_artifacts_exist_and_flagged_trained():
    assert all(STATUS["present"].values()) and STATUS["trained"]


def test_model_loads_with_expected_shapes():
    import joblib
    from tensorflow import keras
    model = keras.models.load_model(resolve(CFG, "model_file"))
    vec = joblib.load(resolve(CFG, "vectorizer_file"))
    assert model.input_shape[-1] == len(vec.vocabulary_)
    assert model.output_shape[-1] == CFG["data"]["expected_num_classes"]


def test_label_encoder_has_20_classes():
    import joblib
    assert len(joblib.load(resolve(CFG, "label_encoder_file")).classes_) == 20


def test_model_loads_and_predicts_in_fresh_process():
    code = ("from src.predict import DocumentClassifier; c=DocumentClassifier();"
            "import json; r=c.predict('The space shuttle launch was delayed by NASA engineers'); print(json.dumps(r))")
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert 0.0 <= json.loads(out.stdout.strip().splitlines()[-1])["confidence"] <= 1.0
