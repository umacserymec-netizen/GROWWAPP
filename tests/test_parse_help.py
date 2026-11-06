"""Help-article parsing: keep answer prose and steps, drop site chrome."""

from __future__ import annotations

from ingest.parse import parse_html

HTML = """
<html>
  <body>
    <h1>Customer Support</h1>
    <div class="layout-main">
      <h1 class="qap761Title">How to change password on Groww?</h1>
      <div class="answerWrapper">
        <div><p>To change your password:</p></div>
        <div>
          <ol>
            <li>Go to the 'Profile' section in the app.</li>
            <li>Click on the 'Settings' icon.</li>
            <li>Select 'Change Password'.</li>
          </ol>
        </div>
        <p>Click the button 'CHANGE PASSWORD' below to go directly to the page.</p>
        <p>Was the answer helpful?</p>
      </div>
    </div>
    <footer>Footer chrome</footer>
  </body>
</html>
"""


def _doc():
    return parse_html(
        HTML,
        scheme="How to change password on Groww?",
        source_url="https://groww.in/help/my-account/ma-account-details/how-to-change-password-on-groww",
        source_title="How to change password on Groww?",
        page_type="help",
        fetched_at="2026-11-06",
    )


def test_help_title_skips_customer_support_banner() -> None:
    doc = _doc()
    assert len(doc.sections) == 1
    assert doc.sections[0].title == "How to change password on Groww?"


def test_help_body_keeps_numbered_steps_and_drops_cta() -> None:
    body = _doc().sections[0].body
    assert "1. Go to the 'Profile' section in the app." in body
    assert "2. Click on the 'Settings' icon." in body
    assert "3. Select 'Change Password'." in body
    assert "button" not in body.lower()
    assert "helpful" not in body.lower()
    assert "Footer chrome" not in body
