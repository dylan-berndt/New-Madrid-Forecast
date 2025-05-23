from datetime import datetime, timedelta
import requests
import numpy as np
import yaml
from types import SimpleNamespace
import os
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
import h5py

load_dotenv()

with open("backend/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    config = SimpleNamespace(**config)


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


def getAG2(stationID, parameterID, days):
    startDate = datetime.now() - timedelta(days=days)
    endDate = datetime.now()

    parameters = {
        "Account": os.getenv("AG2ACCOUNT"),
        "profile": os.getenv("AG2PROFILE"),
        "password": os.getenv("AG2PASSWORD"),
        "HistoricalProductID": "HISTORICAL_HOURLY_OBSERVED",
        "DataTypes[]": [parameterID],
        "TempUnits": "F",
        "StartDate": startDate.strftime("%m/%d/%Y"),
        "EndDate": endDate.strftime("%m/%d/%Y"),
        "CityIds[]": [stationID]
    }

    url = "https://www.wsitrader.com/Services/CSVDownloadService.svc/GetHistoricalObservations?"
    for key in parameters:
        if type(parameters[key]) == list:
            for parameter in parameters[key]:
                url += f"{key}={parameter}&"
        else:
            url += f"{key}={parameters[key]}&"

    # TODO: Actual error handling
    response = requests.get(url)
    if not response.ok:
        print("ISSUE")

    data = response.text.replace("\r\n", "\n")

    frames = data.split(" - ")[1:]
    for frame in frames:
        frame = '\n'.join(frame.split('\n')[:-1])
        csvIO = StringIO(frame)
        df = pd.read_csv(csvIO, sep=",", header=1)
        df.columns = ["Date", "Hour", parameterID]

        df["Hour"] = df["Hour"].astype('str')

        df["Hour"] = df["Hour"].transform(lambda x: x if len(x) > 1 else "0" + x)

        df["Date"] = df[["Date", "Hour"]].agg(':'.join, axis=1)

        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y:%H")
        df["Date"] = df["Date"].transform(lambda x: x.timestamp())

        return df["Date"].to_list(), df[parameterID].to_list()
    

def ag2Forecast(stationID):
    parameters = {
        "Account": os.getenv("AG2ACCOUNT"),
        "profile": os.getenv("AG2PROFILE"),
        "password": os.getenv("AG2PASSWORD"),
        "HistoricalProductID": "HISTORICAL_HOURLY_OBSERVED",
        "TempUnits": "F",
        "Region": "NA",
        "SiteId": [stationID]
    }

    url = "https://www.wsitrader.com/Services/CSVDownloadService.svc/GetHourlyForecast?"
    for key in parameters:
        if type(parameters[key]) == list:
            for parameter in parameters[key]:
                url += f"{key}={parameter}&"
        else:
            url += f"{key}={parameters[key]}&"

    print(f"Fetching from: {url}")

    # TODO: Actual error handling
    response = requests.get(url)
    if not response.ok:
        print("ISSUE")

    data = response.text.replace("\r\n", "\n")

    # TODO: Proper parameter handling, ordering
    frames = data.split(" - ")[1:]
    for frame in frames:
        frame = '\n'.join(frame.split('\n')[:-1])
        csvIO = StringIO(frame)
        df = pd.read_csv(csvIO, sep=",", header=1)
        df.columns[:2] = ["Date", "Hour"]

        df["Hour"] = df["Hour"].astype('str')

        df["Hour"] = df["Hour"].transform(lambda x: x if len(x) > 1 else "0" + x)

        df["Date"] = df[["Date", "Hour"]].agg(':'.join, axis=1)

        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y:%H")
        df["Date"] = df["Date"].transform(lambda x: x.timestamp())

        return df["Date"].to_numpy(), df[parameterID].to_numpy()


def requestNOAA(url, target):
    data = requests.get(url).json()
    dates = []
    sequence = []
    for datum in data['data']:
        date = datetime.strptime(datum['validTime'], "%Y-%m-%dT%H:%M:%SZ")
        dates.append(date.timestamp())
        sequence.append(datum[target])
    
    return dates, sequence


# TODO: Request regularly to keep reasonable log
def archiveNOAA(url, target):
    dates, sequence = requestNOAA(url, target)

    file = h5py.File(f"NOAA/{target} {datetime.now().strftime("%Y-%m-%d %H:%M")}.hdf5", "w")
    file.create_dataset("Dates", data=dates)
    file.create_dataset("Values", data=sequence)


