import os
import sys
from main import app

def start_server():
    os.chdir(os.path.dirname(__file__))
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
