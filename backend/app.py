from flask import Flask, jsonify
import json
from data import *

app = Flask(__name__)

noaaURL = 'https://api.water.noaa.gov/nwps/v1/gauges/NMDM7/stageflow/forecast'

@app.route('/observed/<string:stationID>/<string:parameterID>')
def getStationData(stationID, parameterID, days=28):
    dates, sequence = getSingleGauge(stationID, parameterID, days)

    response = {
        'dates': dates,
        'sequence': sequence
    }

    return jsonify(response)

@app.route('/model/<int:horizon>/<int:offset>')
def getPredictions(horizon, offset):
    pass

@app.route('/floodhub/gauge')
def getFloodHubGauge():
    pass

@app.route('/floodhub/discharge')
def getFloodHubDischarge():
    pass

@app.route('/noaa/gauge')
def getNOAAGauge():
    response = requestNOAA(noaaURL, 'primary')

    return jsonify(response)

@app.route('/noaa/discharge')
def getNOAADischarge():
    response = requestNOAA(noaaURL, 'secondary')

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)