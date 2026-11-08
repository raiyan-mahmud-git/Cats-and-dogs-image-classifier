"""Unit tests for packaging configuration, CI/CD, and fast test fixtures."""

from pathlib import Path
import tomllib
import yaml

from catdog import __version__
from catdog.predict import CatDogPredictor


class TestPackagingConfig:
    """Verify pyproject.toml configuration adherence to standards."""

    def test_pyproject_toml_exists_and_valid(self):
        """pyproject.toml must exist and parse as valid TOML."""
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        assert pyproject_path.is_file(), "pyproject.toml file not found at repository root"

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert "project" in data
        proj = data["project"]
        assert proj["name"] == "catdog"
        assert proj["version"] == __version__
        assert "dependencies" in proj
        assert len(proj["dependencies"]) >= 4

        # Entry point CLI scripts
        assert "scripts" in proj
        assert "catdog-predict" in proj["scripts"]
        assert "catdog-export" in proj["scripts"]

        # Tool configs
        assert "tool" in data
        assert "pytest" in data["tool"]
        assert "ruff" in data["tool"]
        assert "coverage" in data["tool"]

    def test_optional_dependencies_defined(self):
        """pyproject.toml should define test, web, and dev extras."""
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        extras = data["project"].get("optional-dependencies", {})
        assert "test" in extras
        assert "web" in extras
        assert "dev" in extras


class TestGitHubActionsWorkflow:
    """Validate GitHub Actions CI workflow specification."""

    def test_ci_workflow_exists_and_valid_yaml(self):
        """CI workflow YAML must be syntactically valid and define expected jobs."""
        ci_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
        assert ci_path.is_file(), ".github/workflows/ci.yml not found"

        with open(ci_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        assert "name" in workflow
        assert "jobs" in workflow

        # Required jobs: lint and test
        assert "lint" in workflow["jobs"]
        assert "test" in workflow["jobs"]

        # Test job matrix
        test_job = workflow["jobs"]["test"]
        assert "strategy" in test_job
        assert "matrix" in test_job["strategy"]
        matrix = test_job["strategy"]["matrix"]
        assert "python-version" in matrix
        assert "3.11" in matrix["python-version"]
        assert "3.12" in matrix["python-version"]


class TestFastCIFixtures:
    """Verify tests can run in lightweight CI mode with synthetic dummy models."""

    def test_tiny_model_fixture_builds_and_predicts(self, tiny_model, tmp_images):
        """conftest.tiny_model can perform forward passes without heavy weights."""
        model, model_path = tiny_model
        assert model_path.exists()
        assert model.input_shape == (None, 150, 150, 3)
        assert model.output_shape == (None, 1)

        # Predict directly via CatDogPredictor
        predictor = CatDogPredictor(model_path=model_path)
        res = predictor.predict_image(tmp_images["rgb"])

        assert res["label"] in ("Cat", "Dog")
        assert 0.0 <= res["dog_probability"] <= 1.0
        assert 0.0 <= res["confidence_pct"] <= 100.0
        assert res["time_ms"] > 0

    def test_tiny_tflite_fixture_inference(self, tiny_tflite, tmp_images):
        """conftest.tiny_tflite can execute TFLite inference in CI."""
        assert tiny_tflite.exists()
        assert tiny_tflite.suffix == ".tflite"

        predictor = CatDogPredictor(model_path=tiny_tflite)
        assert predictor._is_tflite is True

        res = predictor.predict_image(tmp_images["rgb"])
        assert res["label"] in ("Cat", "Dog")
        assert 0.0 <= res["dog_probability"] <= 1.0
