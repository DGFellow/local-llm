def test_modules_importable():
    # Basic smoke: modules import without executing heavy code
    import src.app  # noqa: F401
    import src.config  # noqa: F401
    import src.llm.engine  # noqa: F401
    import src.llm.worker  # noqa: F401
    import src.ui.main_window  # noqa: F401
    import src.utils.logging  # noqa: F401
