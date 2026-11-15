"""
Renders every page of the Streamlit app headlessly and fails on any exception.
Run: python tests/smoke_app.py   (also runs in CI)
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["home", "single_prediction", "bulk_prediction", "overview", "analytics", "model_performance", "responsible_use"]


def main() -> int:
    failed = 0
    for page in PAGES:
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        at.switch_page(f"app_pages/{page}.py").run()
        problems = [e.value for e in at.exception] + [e.value for e in at.error]
        print(f"{page:<20} {'OK' if not problems else problems}")
        failed += bool(problems)

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.switch_page("app_pages/single_prediction.py").run()
    at.button[0].click().run()
    ok = not at.exception and any("Estimated no-show probability" in m.value for m in at.markdown)
    print(f"{'single submit':<20} {'OK' if ok else 'FAILED'}")
    failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
