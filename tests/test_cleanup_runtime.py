from scripts.cleanup_runtime import is_vision_workbench_process


def test_cleanup_runtime_matches_project_entry_points() -> None:
    assert is_vision_workbench_process(
        r"R:\Anaconda\envs\vision-workbench\Scripts\vision-workbench.exe"
    )
    assert is_vision_workbench_process(
        "python -m yolo26_detection.window.app"
    )


def test_cleanup_runtime_does_not_match_unrelated_python() -> None:
    assert not is_vision_workbench_process("python unrelated_script.py")
    assert not is_vision_workbench_process("python unrelated_cv_basics_tool.py")
    assert not is_vision_workbench_process(
        r"R:\Anaconda\envs\cv-demo\Scripts\yolo26-detection-workbench.exe"
    )
