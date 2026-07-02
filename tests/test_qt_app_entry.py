def test_qt_app_entry_imports_without_loading_qt_widgets():
    import vision_workbench.desktop.app as app

    assert callable(app.main)


def test_legacy_window_packages_do_not_eager_import_app_modules():
    import importlib
    import sys

    package_names = [
        "camera_diagnostics.window",
        "panorama_reconstruction.window",
        "image_classification.window",
        "cv_basics.window",
    ]

    for package_name in package_names:
        app_module_name = f"{package_name}.app"
        sys.modules.pop(app_module_name, None)
        sys.modules.pop(package_name, None)

        importlib.import_module(package_name)

        assert app_module_name not in sys.modules
