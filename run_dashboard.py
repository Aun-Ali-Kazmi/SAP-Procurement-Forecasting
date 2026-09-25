"""Start the forecasting dashboard at http://127.0.0.1:8010"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from forecasting.app import app  # noqa: E402

if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:8010  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=8010, debug=False)
