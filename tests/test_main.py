import pytest
from unittest import mock


def test_main_function_import():
    """Test that main functions can be imported."""
    from qpm.__main__ import main, _our_excepthook

    assert callable(main)
    assert callable(_our_excepthook)


def test_our_excepthook():
    """Test the custom exception hook."""
    from qpm.__main__ import _our_excepthook

    # Mock traceback.print_exception
    with mock.patch("qpm.__main__.traceback.print_exception") as mock_print:
        # Call the excepthook with test exception
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            _our_excepthook(type(e), e, e.__traceback__)

        # Verify print_exception was called
        mock_print.assert_called_once()


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in __import__("os").environ, reason="Skip GUI tests in CI"
)
def test_main_function_creates_app():
    """Test that main function creates QApplication and widget."""
    from qpm.__main__ import main

    # Mock QApplication and QPMWidget
    with (
        mock.patch("qpm.__main__.QApplication") as mock_app_cls,
        mock.patch("qpm.__main__.QPMWidget") as mock_widget_cls,
        mock.patch("sys.argv", ["test"]),
    ):
        mock_app = mock.Mock()
        mock_app_cls.return_value = mock_app
        mock_widget = mock.Mock()
        mock_widget_cls.return_value = mock_widget

        # Mock app.exec() to avoid hanging
        mock_app.exec.return_value = 0

        # Call main
        main()

        # Verify QApplication was created
        mock_app_cls.assert_called_once_with(["test"])

        # Verify QPMWidget was created and shown
        mock_widget_cls.assert_called_once()
        mock_widget.show.assert_called_once()

        # Verify app.exec was called
        mock_app.exec.assert_called_once()
