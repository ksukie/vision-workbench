import zipfile
from pathlib import Path
from typing import Tuple

from PIL import Image

from image_classification import api
from image_classification.configuration import ImageClassificationConfig
from image_classification.domain import ClassificationModelName, ClassificationTrainingConfig
from image_classification.runner import main as runner_main


def make_image(path: Path, color: Tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def write_model_archive(path: Path, payload: bytes = b"fake weight") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.pkl", payload)


def make_classification_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        for class_name, color in (("cat", (200, 40, 40)), ("dog", (40, 90, 220))):
            make_image(root / split / class_name / "001.png", color)
            make_image(root / split / class_name / "002.png", color)
    return root


def test_supported_models_include_resnet18_and_mobilenet() -> None:
    models = api.supported_models()

    assert ClassificationModelName.RESNET18 in models
    assert ClassificationModelName.MOBILENET_V3_SMALL in models


def test_validate_classification_dataset_accepts_folder_layout(tmp_path: Path) -> None:
    dataset = make_classification_dataset(tmp_path / "dataset")

    report = api.validate_classification_dataset(dataset)

    assert report.ok
    assert report.class_names == ["cat", "dog"]
    assert report.train_image_count == 4
    assert report.val_image_count == 4
    assert report.checked_images == 8


def test_validate_classification_dataset_rejects_missing_val(tmp_path: Path) -> None:
    make_image(tmp_path / "dataset" / "train" / "cat" / "001.png", (200, 40, 40))

    report = api.validate_classification_dataset(tmp_path / "dataset")

    assert not report.ok
    assert any("Missing val" in message for message in report.messages)


def test_split_classification_dataset_creates_train_val_layout(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for class_name, color in (("cat", (200, 40, 40)), ("dog", (40, 90, 220))):
        for index in range(5):
            make_image(raw / class_name / f"{index}.png", color)
    output = tmp_path / "split"

    summary = api.split_classification_dataset(raw, output, train_ratio=0.8, seed=1)
    report = api.validate_classification_dataset(output)

    assert summary["cat"]["train"] == 4
    assert summary["cat"]["val"] == 1
    assert summary["dog"]["train"] == 4
    assert summary["dog"]["val"] == 1
    assert report.ok


def test_list_classification_models_contains_pretrained_options() -> None:
    service = api.create_image_classification_service(ImageClassificationConfig())

    model_names = [model.name for model in service.list_models()]

    assert "resnet18" in model_names
    assert "mobilenet_v3_small" in model_names


def test_classification_runner_command_preserves_training_options(tmp_path: Path) -> None:
    config = ImageClassificationConfig(
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
        pretrained_model_dir=tmp_path / "models" / "pretrained",
        dataset_dir=tmp_path / "datasets",
        runs_dir=tmp_path / "runs",
    )
    service = api.create_image_classification_service(config)
    job = ClassificationTrainingConfig(
        model_name="resnet18",
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "runs",
        run_name="quickstart",
        epochs=2,
        image_size=128,
        batch_size=4,
        device="cpu",
        learning_rate=0.002,
        workers=1,
        freeze_backbone=False,
        pretrained=False,
    )

    command = service.build_runner_command(job)

    assert command[1:3] == ["-m", "image_classification.runner"]
    assert command[command.index("--name") + 1] == "quickstart"
    assert command[command.index("--epochs") + 1] == "2"
    assert command[command.index("--lr") + 1] == "0.002"
    assert "--no-pretrained" in command
    assert "--unfreeze" in command


def test_pretrained_weight_status_and_import(tmp_path: Path) -> None:
    config = ImageClassificationConfig(
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
        pretrained_model_dir=tmp_path / "models" / "pretrained",
        dataset_dir=tmp_path / "datasets",
        runs_dir=tmp_path / "runs",
    )
    service = api.create_image_classification_service(config)
    source_weight = tmp_path / "source.pth"
    write_model_archive(source_weight, b"fake local weight")

    status_before = service.pretrained_weight_status("resnet18")[0]
    status_after = service.import_pretrained_weight("resnet18", source_weight)

    assert not status_before.exists
    assert status_after.exists
    assert status_after.local_path.read_bytes() == source_weight.read_bytes()
    assert status_after.filename == "resnet18-f37072fd.pth"


def test_pretrained_weight_status_rejects_corrupt_cache(tmp_path: Path) -> None:
    config = ImageClassificationConfig(
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
        pretrained_model_dir=tmp_path / "models" / "pretrained",
        dataset_dir=tmp_path / "datasets",
        runs_dir=tmp_path / "runs",
    )
    service = api.create_image_classification_service(config)
    bad_weight = config.pretrained_model_dir / "resnet18-f37072fd.pth"
    bad_weight.parent.mkdir(parents=True)
    bad_weight.write_bytes(b"partial")

    status = service.pretrained_weight_status("resnet18")[0]

    assert not status.exists


def test_pretrained_prediction_reuses_loaded_classifier(tmp_path: Path) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.load_pretrained_calls = []

        def load_pretrained(self, model_name: str, device: str, weight_path: Path | None):
            classifier = object()
            self.load_pretrained_calls.append((model_name, device, weight_path, classifier))
            return classifier

        def predict(self, classifier, image_path: Path, topk: int):
            return api.PredictionResult(
                image_path=Path(image_path),
                model_name="resnet18",
                model_path=None,
                inference_ms=1.0,
                predictions=[api.PredictionItem("cat", 0.9)],
            )

    config = ImageClassificationConfig(
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
        pretrained_model_dir=tmp_path / "models" / "pretrained",
        dataset_dir=tmp_path / "datasets",
        runs_dir=tmp_path / "runs",
    )
    service = api.create_image_classification_service(config)
    fake_backend = FakeBackend()
    service._backend = fake_backend
    image_path = tmp_path / "image.png"
    make_image(image_path, (200, 40, 40))

    service.predict_with_pretrained("resnet18", image_path, topk=1, device="cpu")
    service.predict_with_pretrained("resnet18", image_path, topk=1, device="cpu")

    assert len(fake_backend.load_pretrained_calls) == 1

    source_weight = tmp_path / "source.pth"
    write_model_archive(source_weight, b"new local weight")
    service.import_pretrained_weight("resnet18", source_weight)
    service.predict_with_pretrained("resnet18", image_path, topk=1, device="cpu")

    assert len(fake_backend.load_pretrained_calls) == 2


def test_pretrained_prediction_switch_evicts_previous_classifier(tmp_path: Path) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.load_pretrained_calls = []

        def load_pretrained(self, model_name: str, device: str, weight_path: Path | None):
            classifier = object()
            self.load_pretrained_calls.append((model_name, device, weight_path, classifier))
            return classifier

        def predict(self, classifier, image_path: Path, topk: int):
            model_name = next(
                model_name
                for model_name, _device, _weight_path, loaded in self.load_pretrained_calls
                if loaded is classifier
            )
            return api.PredictionResult(
                image_path=Path(image_path),
                model_name=model_name,
                model_path=None,
                inference_ms=1.0,
                predictions=[api.PredictionItem("cat", 0.9)],
            )

    config = ImageClassificationConfig(
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
        pretrained_model_dir=tmp_path / "models" / "pretrained",
        dataset_dir=tmp_path / "datasets",
        runs_dir=tmp_path / "runs",
    )
    service = api.create_image_classification_service(config)
    fake_backend = FakeBackend()
    service._backend = fake_backend
    image_path = tmp_path / "image.png"
    make_image(image_path, (200, 40, 40))

    service.predict_with_pretrained("resnet18", image_path, topk=1, device="cpu")
    service.predict_with_pretrained("mobilenet_v3_small", image_path, topk=1, device="cpu")

    assert len(fake_backend.load_pretrained_calls) == 2
    assert len(service._pretrained_cache) == 1
    assert next(iter(service._pretrained_cache))[0] == "mobilenet_v3_small"

    service.predict_with_pretrained("resnet18", image_path, topk=1, device="cpu")

    assert len(fake_backend.load_pretrained_calls) == 3
    assert len(service._pretrained_cache) == 1
    assert next(iter(service._pretrained_cache))[0] == "resnet18"


def test_runner_dry_run_validates_without_training(tmp_path: Path) -> None:
    dataset = make_classification_dataset(tmp_path / "dataset")

    exit_code = runner_main(
        [
            "--data",
            str(dataset),
            "--model",
            "mobilenet_v3_small",
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_runner_dataset_failure_prints_troubleshooting(tmp_path: Path, capsys) -> None:
    exit_code = runner_main(["--data", str(tmp_path / "missing_dataset"), "--dry-run"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "docs/troubleshooting/en/datasets-and-training.md" in captured.err
