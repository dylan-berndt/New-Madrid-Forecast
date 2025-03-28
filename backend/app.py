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

@app.route('/model/<int:horizon>')
def getPredictions(horizon):
    pass

@app.route('/floodhub/gauge')
def getFloodHubGauge():
    pass

@app.route('/floodhub/discharge')
def getFloodHubDischarge():
    pass

@app.route('/noaa/gauge')
def getNOAAGauge():
    dates, sequence = requestNOAA(noaaURL, 'primary')

    response = {
        'dates': dates,
        'sequence': sequence
    }

    return jsonify(response)

@app.route('/noaa/discharge')
def getNOAADischarge():
    dates, sequence = requestNOAA(noaaURL, 'secondary')

    response = {
        'dates': dates,
        'sequence': sequence
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

