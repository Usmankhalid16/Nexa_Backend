# //call flask api in this code
from fetch_data import funct
# Path: venv\Lib\main.py
# Compare this snippet from venv\Lib\api.py:
from flask import Flask,request,jsonify

app = Flask(__name__)


@app.get("/data")
def getdata():
    data = funct()
    return jsonify(data)
