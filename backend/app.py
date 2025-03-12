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
    data = requests.get(noaaURL).json()
    dates = []
    sequence = []
    for datum in data['data']:
        date = datetime.strptime(datum['validTime'], "%Y-%m-%dT%H:%M:%SZ")
        dates.append(date.timestamp())
        sequence.append(datum['primary'])
    
    response = {
        'dates': dates,
        'sequence': sequence
    }

    return jsonify(response)

@app.route('/noaa/discharge')
def getNOAADischarge():
    data = requests.get(noaaURL).json()
    dates = []
    sequence = []
    for datum in data['data']:
        date = datetime.strptime(datum['validTime'], "%Y-%m-%dT%H:%M:%SZ")
        dates.append(date.timestamp())
        sequence.append(datum['secondary'])
    
    response = {
        'dates': dates,
        'sequence': sequence
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)