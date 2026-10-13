"""
Shared HTML snippets for the Aequitas UI.

Every tab builds its section headers, loading indicators and empty states
from here, so markup and CSS classes (see style.css) stay consistent.
"""
from html import escape


def section_header(title: str, spaced: bool = False) -> str:
    """Pill-shaped header placed at the top of a card or card section."""
    classes = "aq-header aq-header-spaced" if spaced else "aq-header"
    return (
        f'<div class="{classes}">'
        f'<span class="aq-header-dot"></span>{escape(title)}'
        f"</div>"
    )


def loading_overlay(text: str) -> str:
    """Blurred overlay with a spinner, covering the parent `.aq-loading-host` card."""
    return f"""
    <div class="aq-loading-overlay">
        <div class="aq-loading-box">
            <div class="aq-spinner"></div>
            <div class="aq-loading-text">{escape(text)}</div>
        </div>
    </div>
    """


def loading_inline(text: str) -> str:
    """Spinner pill shown inline, in the document flow."""
    return f"""
    <div class="aq-loading-box">
        <div class="aq-spinner"></div>
        <div class="aq-loading-text">{escape(text)}</div>
    </div>
    """


def loading_badge(text: str) -> str:
    """Small spinner badge pinned to the top-right corner of a card."""
    return f"""
    <div class="aq-loading-badge">
        <div class="aq-spinner aq-spinner-sm"></div>
        <span>{escape(text)}</span>
    </div>
    """


def empty_state(message: str, icon: bool = False, raw_html: bool = False) -> str:
    """Placeholder shown inside a table card when there is nothing to display."""
    body = message if raw_html else escape(message)
    svg = """
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.4" stroke-linecap="round">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <path d="M3 9h18M9 21V9"/>
      </svg>
    """ if icon else ""
    return f"""
    <div class="aq-table-wrap">
        <div class="aq-empty">{svg}<span>{body}</span></div>
    </div>
    """


def error_alert(title: str, items: list[str]) -> str:
    """Red alert box with a bullet list of error messages."""
    lis = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"""
    <div class="aq-alert-row">
        <div class="aq-alert-error">
            <b>{escape(title)}</b>
            <ul>{lis}</ul>
        </div>
    </div>
    """
