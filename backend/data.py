from datetime import datetime, timedelta
import requests

mostRecent = None

model = None


def dateInt(dateString):
    dateString = dateString[:dateString.index(".")][:-3]

    date = datetime.strptime(dateString, "%Y-%m-%dT%H:%M")

    return date.timestamp()


def getSingleGauge(stationID, parameterID, days):
    startDate = datetime.now() - timedelta(days=days)
    endDate = datetime.now()
    dates = [datetime.strftime(startDate, "%Y-%m-%d"), datetime.strftime(endDate, "%Y-%m-%d")]

    url = "https://waterservices.usgs.gov/nwis/iv/?"
    url += f"sites={stationID}"
    url += f"&startDT={dates[0]}&endDT={dates[1]}" 
    url += f"&parameterCd={parameterID}"
    url += f"&format=json"

    response = requests.get(url)
    # TODO: Actual error handling
    if not response.ok:
        print("ISSUE")

    data = response.json()

    timeSeries = data['value']['timeSeries']
    parameters = [obj['variable']['variableCode'][0]['value'] for obj in timeSeries]
    values = [obj['values'][0]['value'] for obj in timeSeries]

    sortedValues = [values[parameters.index(parameterCode)] for parameterCode in [parameterID]]

    data = sortedValues
    
    for p in range(len(data)):
        dates = [dateInt(node["dateTime"]) for node in data[p] if float(node["value"]) > -1e+5]
        values = [float(node["value"]) for node in data[p] if float(node["value"]) > -1e+5]

        paired = sorted(zip(dates, values))
        dates, values = zip(*paired)

    return dates, values


def refreshData():
    if (datetime.now() - mostRecent).minutes > 15:
        getData()

    mostRecent = datetime.now()


def getData():
    if mostRecent is None:
        return

    startDate = mostRecent
    endDate = datetime.now()





