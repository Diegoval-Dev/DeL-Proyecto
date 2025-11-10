"""
Unit tests for classifier inference
"""

import pytest
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestClassifier:
    """Tests for trained classifier"""

    @pytest.fixture
    def model_path(self):
        """Path to best baseline model"""
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / "models" / "best_baseline"

    def test_model_files_exist(self, model_path):
        """Should have required model files"""
        assert (model_path / "encoder.pkl").exists()
        assert (model_path / "classifier.pkl").exists()

    def test_can_load_encoder(self, model_path):
        """Should be able to load encoder"""
        encoder = joblib.load(model_path / "encoder.pkl")
        assert encoder is not None

    def test_can_load_classifier(self, model_path):
        """Should be able to load classifier"""
        classifier = joblib.load(model_path / "classifier.pkl")
        assert classifier is not None

    def test_predicts_task_sentence(self, model_path):
        """Should predict TAREA for task sentence"""
        encoder = joblib.load(model_path / "encoder.pkl")
        classifier = joblib.load(model_path / "classifier.pkl")

        text = "Juan debe enviar el informe antes del viernes"
        embedding = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        prediction = classifier.predict(embedding)[0]

        # 1 = TAREA, 0 = NO_TAREA
        assert prediction == 1

    def test_predicts_non_task_sentence(self, model_path):
        """Should predict NO_TAREA for non-task sentence"""
        encoder = joblib.load(model_path / "encoder.pkl")
        classifier = joblib.load(model_path / "classifier.pkl")

        text = "Se discutieron los resultados de la reunión anterior"
        embedding = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        prediction = classifier.predict(embedding)[0]

        # 1 = TAREA, 0 = NO_TAREA
        assert prediction == 0

    def test_returns_probabilities(self, model_path):
        """Should return valid probabilities"""
        encoder = joblib.load(model_path / "encoder.pkl")
        classifier = joblib.load(model_path / "classifier.pkl")

        text = "María coordinará la presentación"
        embedding = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        probas = classifier.predict_proba(embedding)[0]

        assert len(probas) == 2
        assert 0 <= probas[0] <= 1
        assert 0 <= probas[1] <= 1
        assert abs(sum(probas) - 1.0) < 0.01  # Probabilities should sum to 1

    def test_handles_empty_string(self, model_path):
        """Should handle empty string without crashing"""
        encoder = joblib.load(model_path / "encoder.pkl")
        classifier = joblib.load(model_path / "classifier.pkl")

        text = ""
        embedding = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        prediction = classifier.predict(embedding)

        assert len(prediction) == 1

    def test_handles_long_sentence(self, model_path):
        """Should handle very long sentence"""
        encoder = joblib.load(model_path / "encoder.pkl")
        classifier = joblib.load(model_path / "classifier.pkl")

        text = "Juan " + "debe coordinar " * 50 + "la reunión"
        embedding = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        prediction = classifier.predict(embedding)

        assert len(prediction) == 1
