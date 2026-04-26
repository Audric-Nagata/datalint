import subprocess
import os
import inspect
import app

def run():
    # find installed path of your app package
    app_path = os.path.dirname(inspect.getfile(app))
    target = os.path.join(app_path, "streamlit_app.py")

    subprocess.run([
        "streamlit", "run", target
    ])