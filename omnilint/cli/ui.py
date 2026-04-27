"""UI launcher for Streamlit."""

import subprocess
import sys
from pathlib import Path


def run():
    """Launch the Streamlit UI."""
    app_path = Path("omnilint/app/streamlit_app.py")
    
    if not app_path.exists():
        print("Error: omnilint/app/streamlit_app.py not found")
        sys.exit(1)
    
    subprocess.run(["streamlit", "run", str(app_path)])


if __name__ == "__main__":
    run()