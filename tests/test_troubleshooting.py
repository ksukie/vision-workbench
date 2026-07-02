from vision_workbench.troubleshooting import (
    DATASETS_AND_TRAINING,
    MODULE_RUNTIME_ERRORS,
    all_doc_paths,
    doc_paths,
    with_help,
)


def test_troubleshooting_doc_paths_include_bilingual_guides() -> None:
    docs = doc_paths(DATASETS_AND_TRAINING)

    assert docs.english == "docs/troubleshooting/en/datasets-and-training.md"
    assert docs.chinese == "docs/troubleshooting/zh-CN/数据集与训练.md"


def test_unknown_troubleshooting_category_falls_back_to_runtime_errors() -> None:
    docs = doc_paths("unknown")

    assert docs == doc_paths(MODULE_RUNTIME_ERRORS)


def test_with_help_appends_paths_once() -> None:
    message = with_help("Training failed.", DATASETS_AND_TRAINING)

    assert "Training failed." in message
    assert "docs/troubleshooting/en/datasets-and-training.md" in message
    assert with_help(message, DATASETS_AND_TRAINING) == message


def test_all_doc_paths_contains_expected_categories() -> None:
    docs = all_doc_paths()

    assert DATASETS_AND_TRAINING in docs
    assert MODULE_RUNTIME_ERRORS in docs
