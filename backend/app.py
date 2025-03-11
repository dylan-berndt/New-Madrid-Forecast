from flask import Flask, jsonify
import json
import requests

app = Flask(__name__)

noaaURL = 'https://api.water.noaa.gov/nwps/v1/gauges/NMDM7/stageflow/forecast'

@app.route('/testData')
def testData():
    return jsonify(list(range(10)))

@app.route('/observed/<int:stationID>')
def getStationData(stationID):
    pass

@app.route('/modelPredictions/<int:horizon>')
def getPredictions(horizon):
    pass

@app.route('/noaa/gauge')
def getNOAAGauge():
    data = requests.get(noaaURL).json()
    sequence = []
    for datum in data['data']:
        sequence.append(datum['primary'])
    
    response = {
        'dates': [],
        'sequence': sequence
    }

    return jsonify(response)

@app.route('/noaa/discharge')
def getNOAADischarge():
    data = requests.get(noaaURL).json()
    sequence = []
    for datum in data['data']:
        sequence.append(datum['secondary'])
    
    response = {
        'dates': [],
        'sequence': sequence
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)