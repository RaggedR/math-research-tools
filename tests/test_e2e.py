"""End-to-end Playwright tests for the Research Explorer web app.

These tests start the FastAPI server and test the UI:
explore page loads → query input works → tabs switch → graph renders.

Run with: pytest tests/test_e2e.py -v
Requires: pip install playwright && playwright install chromium
"""

import multiprocessing
import socket
import time

import pytest

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


def _find_free_port():
    """Find a random available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_server(port):
    """Run uvicorn in a subprocess."""
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=port, log_level="error")


@pytest.fixture(scope="module")
def server():
    """Start the FastAPI server for E2E tests on a random free port."""
    port = _find_free_port()
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


class TestExploreView:
    """Tests for the Research Explorer page served at /."""

    def test_page_loads(self, server, browser):
        """The explore page loads and shows the title."""
        page = browser.new_page()
        page.goto(server)
        assert page.title() == "Research Explorer"
        assert page.locator("h1").text_content() == "Research Explorer"
        page.close()

    def test_query_input_visible(self, server, browser):
        """The query input and search button are visible."""
        page = browser.new_page()
        page.goto(server)
        assert page.locator("#query-input").is_visible()
        assert page.locator("#query-btn").is_visible()
        page.close()

    def test_query_input_placeholder(self, server, browser):
        """The query input has a helpful placeholder."""
        page = browser.new_page()
        page.goto(server)
        placeholder = page.locator("#query-input").get_attribute("placeholder")
        assert "key findings" in placeholder.lower()
        page.close()

    def test_tabs_present(self, server, browser):
        """Both tabs are present: Ask a Question and Concepts."""
        page = browser.new_page()
        page.goto(server)
        tabs = page.locator(".tab")
        assert tabs.count() == 2
        assert "Ask a Question" in tabs.nth(0).text_content()
        assert "Concepts" in tabs.nth(1).text_content()
        page.close()

    def test_query_tab_active_by_default(self, server, browser):
        """The query tab is active by default."""
        page = browser.new_page()
        page.goto(server)
        query_tab = page.locator("#tab-query")
        assert "active" in query_tab.get_attribute("class")
        page.close()

    def test_graph_svg_present(self, server, browser):
        """The graph SVG element exists."""
        page = browser.new_page()
        page.goto(server)
        assert page.locator("#graph-svg").is_visible()
        page.close()

    def test_sidebar_layout(self, server, browser):
        """The sidebar and graph container form the main layout."""
        page = browser.new_page()
        page.goto(server)
        assert page.locator(".sidebar").is_visible()
        assert page.locator(".graph-container").is_visible()
        page.close()

    def test_empty_query_no_action(self, server, browser):
        """Clicking search with empty input does not crash or navigate."""
        page = browser.new_page()
        page.goto(server)
        page.locator("#query-btn").click()
        # Should still be on the same page, no error
        assert page.title() == "Research Explorer"
        page.close()
