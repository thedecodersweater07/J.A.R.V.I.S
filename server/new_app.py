from flask import Flask, send_from_directory
import os

# The 'web' directory is in the same directory as this script.
web_dir = os.path.join(os.path.dirname(__file__), 'web')

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory(web_dir, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(web_dir, path)
