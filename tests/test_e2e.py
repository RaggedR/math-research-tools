"""End-to-end Playwright tests for the Knowledge Graph web app.

These tests start the FastAPI server and test the full user flow:
upload files → observe progress → interact with the graph.

Run with: pytest tests/test_e2e.py -v
Requires: pip install playwright && playwright install chromium
"""

import asyncio
import multiprocessing
import time
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Check if playwright is installed
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(
    not HAS_PLAYWRIGHT,
    reason="Playwright not installed (pip install playwright && playwright install chromium)"
)


def run_server(port):
    """Run uvicorn in a subprocess."""
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=port, log_level="error")


@pytest.fixture(scope="module")
def server():
    """Start the FastAPI server for E2E tests."""
    port = 18765
    proc = multiprocessing.Process(target=run_server, args=(port,), daemon=True)
    proc.start()
    # Wait for server to start
    time.sleep(2)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    """Launch a Playwright browser."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


class TestUploadView:
    def test_page_loads(self, server, browser):
        """The upload page loads and shows the title."""
        page = browser.new_page()
        page.goto(server)
        assert page.title() == "Knowledge Graph Builder"
        assert page.locator("h1").text_content() == "Knowledge Graph Builder"
        page.close()

    def test_drop_zone_visible(self, server, browser):
        """The drop zone is visible on page load."""
        page = browser.new_page()
        page.goto(server)
        drop_zone = page.locator("#drop-zone")
        assert drop_zone.is_visible()
        assert "Drop files here" in drop_zone.text_content()
        page.close()

    def test_build_button_disabled_initially(self, server, browser):
        """Build button starts disabled with no files selected."""
        page = browser.new_page()
        page.goto(server)
        btn = page.locator("#build-btn")
        assert btn.is_disabled()
        page.close()

    def test_file_picker_accepts_files(self, server, browser):
        """Selecting files via the file picker shows them in the list."""
        page = browser.new_page()
        page.goto(server)

        # Use the file input directly (simulates file picker)
        file_input = page.locator("#file-input")
        file_input.set_input_files(str(FIXTURES_DIR / "sample.txt"))

        # File should appear in list
        file_list = page.locator("#file-list")
        assert "sample.txt" in file_list.text_content()

        # Build button should be enabled
        btn = page.locator("#build-btn")
        assert not btn.is_disabled()
        page.close()

    def test_file_removal(self, server, browser):
        """Removing a file updates the list and disables the button."""
        page = browser.new_page()
        page.goto(server)

        file_input = page.locator("#file-input")
        file_input.set_input_files(str(FIXTURES_DIR / "sample.txt"))

        # Click remove button
        page.locator(".file-remove").click()

        # List should be empty
        file_list = page.locator("#file-list")
        assert file_list.text_content().strip() == ""

        # Button should be disabled again
        btn = page.locator("#build-btn")
        assert btn.is_disabled()
        page.close()

    def test_multiple_files(self, server, browser):
        """Selecting multiple files shows all of them."""
        page = browser.new_page()
        page.goto(server)

        file_input = page.locator("#file-input")
        file_input.set_input_files([
            str(FIXTURES_DIR / "sample.txt"),
            str(FIXTURES_DIR / "sample.md"),
        ])

        file_count = page.locator("#file-count")
        assert "2 files" in file_count.text_content()
        page.close()


class TestProgressView:
    def test_upload_switches_to_progress(self, server, browser):
        """Clicking Build switches to the progress view."""
        page = browser.new_page()
        page.goto(server)

        file_input = page.locator("#file-input")
        file_input.set_input_files(str(FIXTURES_DIR / "sample.txt"))

        # Click build
        page.locator("#build-btn").click()

        # Should switch to progress view
        page.wait_for_selector("#progress-view.active", timeout=5000)
        assert page.locator("#progress-view").is_visible()
        page.close()

    def test_stage_indicators_present(self, server, browser):
        """Progress view shows stage indicators."""
        page = browser.new_page()
        page.goto(server)

        file_input = page.locator("#file-input")
        file_input.set_input_files(str(FIXTURES_DIR / "sample.txt"))
        page.locator("#build-btn").click()

        page.wait_for_selector("#progress-view.active", timeout=5000)

        assert page.locator("#stage-ingesting").is_visible()
        assert page.locator("#stage-extracting").is_visible()
        assert page.locator("#stage-building").is_visible()
        page.close()
