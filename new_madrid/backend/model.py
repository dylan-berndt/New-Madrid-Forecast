
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import intel_extension_for_pytorch as ipex

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from sklearn.preprocessing import MinMaxScaler

import requests
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib

from functools import reduce

import h5py
import pandas as pd
from io import StringIO

import os

import json

import math

import itertools

device = "cpu" if not torch.xpu.is_available() else "xpu"
torch.set_default_device(device)
# print(f"Running on: {device}")
# print(f"Total XPU devices: {torch.xpu.device_count()}")
# print(f"Properties: {torch.xpu.get_device_properties()}")
torch.set_default_dtype(torch.float32)
# print(f"Memory: {' '.join([key + value for key, value in torch.xpu.memory_stats().items()])}")


def dateInt(dateString):
    dateString = dateString[:dateString.index(".")][:-3]

    date = datetime.strptime(dateString, "%Y-%m-%dT%H:%M")

    return date.timestamp()


def loadHistoricalNewMadrid(folder="Data"):
    df = pd.read_csv(f"Data/New Madrid Gauge Height.csv")
    df.columns = ["Date", "Height"]

    df["Date"] = df["Date"].transform(lambda x: x[:-7])
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
    df["Date"] = df["Date"].transform(lambda x: x.timestamp())

    df = df[df["Height"].notnull()]

    file = h5py.File(f"{folder}/07024175.hdf5", "w")
    file.create_dataset("00065 values", data=df["Height"])
    file.create_dataset("00065 dates", data=df["Date"])
    file.close()


def downloadStationData(stationID, parameterCodes, startDate=None, endDate=datetime.now(), period=None, log=True):
    if startDate is not None:
        dates = [datetime.strftime(startDate, "%Y-%m-%d"), datetime.strftime(endDate, "%Y-%m-%d")]

    url = "https://waterservices.usgs.gov/nwis/iv/?"
    url += f"sites={stationID}"
    url += f"&startDT={dates[0]}&endDT={dates[1]}" if period is None else f"&period={period}"
    url += f"&parameterCd={','.join(parameterCodes)}"
    url += f"&format=json"

    if log: print("Fetching from:", url)

    response = requests.get(url)
    if not response.ok:
        print("Uh Oh.")
        quit()

    data = response.json()

    if log: print("Fetched")

    timeSeries = data['value']['timeSeries']
    parameters = [obj['variable']['variableCode'][0]['value'] for obj in timeSeries]
    values = [obj['values'][0]['value'] for obj in timeSeries]

    sortedValues = [values[parameters.index(parameterCode)] for parameterCode in parameterCodes]

    return sortedValues


def downloadStationDataset(stationParameters, startDate, endDate=datetime.now(), folder="Data"):
    for station in stationParameters:
        data = downloadStationData(station["id"], parameterCodes=station["parameters"], startDate=startDate, endDate=endDate)

        file = h5py.File(f"{folder}/{station['id']}.hdf5", "w")
        
        for p in range(len(data)):
            dates = [dateInt(node["dateTime"]) for node in data[p] if float(node["value"]) > -1e+5]
            values = [float(node["value"]) for node in data[p] if float(node["value"]) > -1e+5]

            paired = sorted(zip(dates, values))
            dates, values = zip(*paired)

            file.create_dataset(f"{station['parameters'][p]} values", data=values)
            file.create_dataset(f"{station['parameters'][p]} dates", data=dates)

        file.close()


def downloadHistoricalAG2(parameters, save=True, folder="Data"):
    url = "https://www.wsitrader.com/Services/CSVDownloadService.svc/GetHistoricalObservations?"
    for key in parameters:
        if type(parameters[key]) == list:
            for parameter in parameters[key]:
                url += f"{key}={parameter}&"
        else:
            url += f"{key}={parameters[key]}&"

    print(f"Fetching from: {url}")

    data = requests.get(url).text.replace("\r\n", "\n")

    columnNames = parameters["DataTypes[]"]

    aggregated = []

    frames = data.split(" - ")[1:]
    for frame in frames:
        frame = '\n'.join(frame.split('\n')[:-1])
        csvIO = StringIO(frame)
        df = pd.read_csv(csvIO, sep=",", header=1)
        df.columns = ["Date", "Hour"] + columnNames
        
        name = frame[:4]
        file = h5py.File(f"{folder}/{name}.hdf5", "w")

        df["Hour"] = df["Hour"].astype('str')

        # df["Date"] = df["Date"].transform(lambda x: f"{x[6:]}-{x[3:5]}-{x[:2]}")

        df["Hour"] = df["Hour"].transform(lambda x: x if len(x) > 1 else "0" + x)

        df["Date"] = df[["Date", "Hour"]].agg(':'.join, axis=1)

        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y:%H")
        df["Date"] = df["Date"].transform(lambda x: x.timestamp())

        # if not save:
        #     aggregated.append([df["Date"].to_numpy(), df["Temperature"].to_numpy()])
        #     aggregated.append([df["Date"].to_numpy(), df["Precipitation"].to_numpy()])
        #     continue

        # Create dataset per parameter, date values and measurements
        for columnName in columnNames:
            file.create_dataset(f"{columnName} dates", data=df["Date"])
            file.create_dataset(f"{columnName} values", data=df[columnName])

        file.close()

    if not save:
        return aggregated


class StationData(Dataset):
    def __init__(self, predictorIndex=None, contextLength=32, sliceNum=1, timesteps=1, folder="Data", dumpScales=True, display=False, test=False,
                 thresholds = [-3, -6], desampleDepth=3, useScales=False, convolved=False, resampleDepth=0, resampleStrength=0.05, stationary=False,
                 invertThreshold=False):
        super().__init__()

        self.contextLength = contextLength
        self.predictorIndex = predictorIndex
        self.timesteps = timesteps
        self.sliceNum = sliceNum
        self.stationary = stationary

        self.resampleDepth = resampleDepth
        self.resampleStrength = resampleStrength

        self.x = []
        self.y = []

        self.scales = []
        self.yIndex = -1

        self.scaleDict = {}

        linspace = None

        self.names = []

        previousScales = json.load(open("scales.json"))

        for fileName in sorted(os.listdir(f"{folder}")):
            if not fileName.endswith(".hdf5"):
                continue

            file = h5py.File(f"{folder}/{fileName}", "r")
            
            parameterNames = [name[:name.index(" ")] for name in file if "values" in name]

            stationID = fileName[:fileName.index(".")]

            self.scaleDict[stationID] = {}

            for name in parameterNames:
                dates = file[name + " dates"]
                values = file[name + " values"]

                self.names.append(stationID + " " + name)

                if linspace is None:
                    length = (np.max(dates) - np.min(dates)) // (60 * 15)
                    linspace = np.linspace(np.min(dates), np.max(dates), int(length))
                # print(np.min(dates))
                # print(datetime.fromtimestamp(np.min(dates)), fileName, name)

                uniqueDates = np.unique(dates, return_index=True)[1]

                dates = dates[uniqueDates]
                values = values[uniqueDates]

                if name in ["precipitation", "dewpoint"]:
                    values[values < 0] = 0

                spline = CubicSpline(dates, values, bc_type='natural')
                valid = (np.min(values), np.max(values))
                interpolated = spline(linspace)
                interpolated = np.clip(interpolated, valid[0], valid[1])

                if stationary:
                    interpolated = interpolated[sliceNum:] - interpolated[:-sliceNum]

                # interpolated = interpolated[::sliceNum]

                # if name == "Precipitation":
                #     scaler = MinMaxScaler()
                #     # interpolated[interpolated < 0] = 0
                #     # print(min(interpolated), min(values), max(interpolated), max(values))
                #     normalized = scaler.fit_transform(interpolated.reshape(-1, 1))
                #     normalized = normalized.squeeze()

                # else:
                # interpolated = np.log(interpolated)
                mean, std = np.mean(interpolated), np.std(interpolated)
                if useScales:
                    mean, std = previousScales[stationID][name]
                normalized = (interpolated - mean) / std

                if convolved:
                    kernel = np.zeros([sliceNum * 2])
                    kernel[sliceNum:] = 1
                    kernel = kernel / np.sum(kernel)
                    
                    normalized = np.convolve(normalized, kernel, mode='same')

                self.scales.append([mean, std])
                self.scaleDict[stationID][name] = [mean, std]

                if predictorIndex is not None:
                    if predictorIndex[0] in fileName and name == predictorIndex[1]:
                        self.y = normalized
                        self.yIndex = len(self.x)
                        self.thresholds = [(threshold - mean) / std for threshold in thresholds]

                self.x.append(normalized)

            file.close()

        secondsPerDay = 60 * 60 * 24
        day = linspace / secondsPerDay
        day += 90
        seasonal = np.cos(2 * np.pi * (day / 365))

        if stationary:
            seasonal = seasonal[sliceNum:] - seasonal[:-sliceNum]

        self.x.append(seasonal)

        self.names.append("Season")

        minimumDate = datetime(2008, 1, 1).timestamp()
        dataStart = np.min(linspace) - minimumDate
        # TODO: Double check bc weird
        dataOffset = dataStart / (60 * 15)

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.t = np.arange(len(self.y)) + dataOffset
        self.t = (self.t / sliceNum).astype(int)

        self.parameters = len(self.x)

        if display:
            plt.figure(figsize=(20, 12))

            for i in range(len(self.x)):
                plt.subplot(14, 4, i+1)
                plt.title(self.names[i])
                plt.plot(np.linspace(0, 1, len(self.x[i])), self.x[i], label=self.names[i])
            
            plt.legend()
            plt.show()

        # under = np.sum(self.y < self.thresholds[0])
        # total = np.sum(np.ones(self.y.shape))
        # originalSplit = total / under

        totalBins = 2 ** (desampleDepth)
        xBins = []
        yBins = []
        tBins = []

        length = self.x.shape[1]

        if desampleDepth == 0:
            xBins = [self.x.T]
            yBins = [self.y]
            tBins = [self.t]

        else:
            for b in range(0, length, length // totalBins):
                xBin = self.x[:, b: b + length // totalBins]
                yBin = self.y[b: b + length // totalBins]
                tBin = self.t[b: b + length // totalBins]

                mask = yBin < self.thresholds[0]
                if invertThreshold:
                    mask = yBin > self.thresholds[0]
                if np.any(mask):
                    if xBins:
                        if xBin.T.shape[0] != xBins[0].shape[0]:
                            continue
                    xBins.append(xBin.T)
                    yBins.append(yBin)
                    tBins.append(tBin)

        # for i in range(1, resampleDepth + 1):
        #     noiseLevel = resampleStrength * (i / resampleDepth)
        #     for b in range(len(xBins)):
        #         noise1 = np.random.normal(loc=0, scale=noiseLevel, size=xBins[0].shape)
        #         noise2 = np.random.normal(loc=0, scale=noiseLevel, size=yBins[0].shape)
        #         xBins.append(xBins[b] + noise1)
        #         # yBins.append(yBins[b] + noise2)
        #         yBins.append(yBins[b])
        #         tBins.append(tBins[b])

        # yAdjust = np.concatenate([yBins], axis=0)
        # under = np.sum(yAdjust < self.thresholds[0])
        # total = np.sum(np.ones(yAdjust.shape))
        # self.positiveSplit = total / under

        # print(originalSplit, total / under)

        # print(len(xBins))

        self.xBins = torch.tensor(np.array(xBins), dtype=torch.float32)
        self.yBins = torch.tensor(np.array(yBins), dtype=torch.float32)
        self.tBins = torch.tensor(np.array(tBins), dtype=torch.long)

        self.x = self.x[:, ::sliceNum]
        self.y = self.y[::sliceNum]
        self.t = self.t[::sliceNum]
        self.x = torch.tensor(self.x.T, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.t = torch.tensor(self.t, dtype=torch.long)
        
        if test:
            self.y = torch.concatenate([self.y, 5000 + torch.zeros([self.timesteps])])
            self.t = torch.concatenate([self.t, torch.arange(self.timesteps) + torch.max(self.t)])
        self.test = test

        if dumpScales:
            file = open("scales.json", "w")
            json.dump(self.scaleDict, file)
            file.close()

    def getStationData(self, stationID, parameterIndex):
        name = stationID + " " + parameterIndex
        index = self.names.index(name)
        stationData = self.x.T[index]
        return self.unscale(stationData, index)

    def unscale(self, data, index=None):
        index = self.yIndex if index is None else index
        return (data * self.scales[index][1]) + self.scales[index][0]
    
    def scale(self, data, index=None):
        index = self.yIndex if index is None else index
        return (data - self.scales[index][0]) / self.scales[index][1]

    def __len__(self):
        if self.test:
            return (self.x.shape[0] - self.contextLength)
        
        return len(self.xBins) * (self.xBins[0].shape[0] - ((self.contextLength + self.timesteps) * self.sliceNum))
    
    def __getitem__(self, i):
        if self.test:
            x = self.x[i: i + self.contextLength]

            if self.predictorIndex is None:
                y = self.x[i + self.timesteps: i + self.contextLength + self.timesteps]

            else:
                y = self.y[i + self.timesteps: i + self.contextLength + self.timesteps]

            return x, y
    
        binSize = self.xBins[0].shape[0] - ((self.contextLength + self.timesteps) * self.sliceNum)
        bin = i // binSize
        j = i % binSize

        x = self.xBins[bin][j: j + self.contextLength * self.sliceNum: self.sliceNum]
        y = self.yBins[bin][j + self.timesteps * self.sliceNum: j + ((self.contextLength + self.timesteps) * self.sliceNum): self.sliceNum]

        return x, y
        
    

class TransformerData(StationData, Dataset):
    def __len__(self):
        if self.test:
            return (self.x.shape[0] - self.contextLength)

        return len(self.xBins) * int((self.xBins[0].shape[0] - ((self.contextLength + self.timesteps) * self.sliceNum)))
    
    def __getitem__(self, i):
        if self.test:
            x = self.x[i: i + self.contextLength]

            if self.predictorIndex is None:
                y = self.x[i + self.timesteps: i + self.contextLength + self.timesteps]

            else:
                y = self.y[i + self.timesteps: i + self.contextLength + self.timesteps]

            return x, y
    
        binSize = int((self.xBins[0].shape[0] - ((self.contextLength + self.timesteps) * self.sliceNum)))
        bin = i // binSize
        j = i % binSize

        x = self.xBins[bin][j: j + self.contextLength * self.sliceNum: self.sliceNum]
        y = self.yBins[bin][j + self.timesteps * self.sliceNum: j + ((self.contextLength + self.timesteps) * self.sliceNum): self.sliceNum]

        if self.predictorIndex is None:
            y = self.xBins[bin][j + self.timesteps * self.sliceNum: j + (self.contextLength + self.timesteps) * self.sliceNum: self.sliceNum]

        return x, y
    

class PatchData(StationData, Dataset):
    def __len__(self):
        pass
    
    def __getitem__(self, i):
        pass
    

class StackData(StationData, Dataset):
    def __len__(self):
        if self.test:
            return (self.x.shape[0] - self.contextLength)

        return len(self.xBins) * (self.resampleDepth + 1) * int((self.xBins[0].shape[0] - ((self.contextLength + self.timesteps) * self.sliceNum)))
    
    def __getitem__(self, i):
        if self.test:
            x = self.x[i: i + self.contextLength]

            if self.predictorIndex is None:
                y = self.x[i + self.timesteps: i + self.contextLength + self.timesteps]

            else:
                y = self.y[i + self.timesteps: i + self.contextLength + self.timesteps]

            t = self.t[i + self.timesteps: i + self.contextLength + self.timesteps]

            return (x, t), y
        
        noiseLevel = (i / len(self)) // (self.resampleDepth + 1)
    
        binSize = int((self.xBins[0].shape[0] - ((self.contextLength + self.timesteps) * self.sliceNum)))
        i = i // (self.resampleDepth + 1)
        bin = i // binSize
        j = i % binSize

        x = self.xBins[bin][j: j + self.contextLength * self.sliceNum: self.sliceNum]
        y = self.yBins[bin][j + self.timesteps * self.sliceNum: j + ((self.contextLength + self.timesteps) * self.sliceNum): self.sliceNum]

        if self.predictorIndex is None:
            y = self.xBins[bin][j + self.timesteps * self.sliceNum: j + (self.contextLength + self.timesteps) * self.sliceNum: self.sliceNum]

        t = self.tBins[bin][j + self.timesteps * self.sliceNum: j + ((self.contextLength + self.timesteps) * self.sliceNum): self.sliceNum]

        noise = torch.tensor(np.random.normal(loc=0, scale=noiseLevel, size=y.shape), dtype=y.dtype)
        y += noise

        return (x, t), y
    

class RelativeEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1e4, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(int(max_len), 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if not self.batch_first:
            x = x + self.pe[:x.size(0)]
        else:
            x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)
    

class IndexedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, batch_first=False):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model)).view(1, 1, -1)

    def forward(self, x, t):
        if self.batch_first:
            position = t.unsqueeze(-1).float()
            pe = torch.zeros(t.size(0), t.size(1), self.d_model)
            pe[..., 0::2] = torch.sin(position * self.div_term)
            pe[..., 1::2] = torch.cos(position * self.div_term)

            x = x + pe
            return self.dropout(x)

        else:
            raise NotImplementedError
    

class CMAL(nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(inFeatures, inFeatures * 2),
            nn.ReLU(),
            nn.Linear(inFeatures * 2, outFeatures * 4)
        )
        # self.fc1 = nn.Linear(inFeatures, outFeatures * 4)

        self.softplus = nn.Softplus(2)

        self.eps = 1e-5

    def forward(self, x):
        h = self.ff(x)

        m, b, t, p = h.chunk(4, dim=-1)

        b = self.softplus(b) + self.eps
        t = (1 - self.eps) * torch.sigmoid(t) + self.eps
        p = (1 - self.eps) * torch.softmax(p, dim=-1) + self.eps

        return m, b, t, p
    

class GaussianHead(nn.Module):
    def __init__(self, inFeatures, outFeatures, eps=1e-5):
        super().__init__()

        self.fc1 = nn.Linear(inFeatures, inFeatures * 4)
        # self.silu = nn.SiLU()
        self.fc2 = nn.Linear(inFeatures * 4, outFeatures * 2)

        self.eps = eps

    def forward(self, x):
        x = self.fc1(x)
        h = self.fc2(x)

        m, v = h.chunk(2, dim=-1)

        v = torch.exp(v) + self.eps

        return m, v
    

class BatchNorm(nn.Module):
    def __init__(self, hiddenDim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hiddenDim)

    def forward(self, x):
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
    

class StackedLSTM(nn.Module):
    def __init__(self, timesteps, parameters, hiddenDim, components=1, numLayers=3, dropout=0.2, head=GaussianHead):
        super().__init__()
        self.timesteps = timesteps

        self.fc1 = nn.Sequential(
            nn.Linear(parameters, hiddenDim * 2),
            nn.ReLU(),
            nn.Linear(hiddenDim * 2, hiddenDim)
        )
        self.bn1 = BatchNorm(hiddenDim)

        self.encoder = nn.LSTM(hiddenDim, hiddenDim, num_layers=numLayers, dropout=dropout, batch_first=True)

        self.skipProjection = nn.Linear(hiddenDim, hiddenDim)

        self.encodings = IndexedPositionalEncoding(hiddenDim, batch_first=True)
        self.bn2 = BatchNorm(hiddenDim)
        # self.encodings = LearnedEncoding(hiddenDim, max_len=timesteps, batch_first=True)
        self.decoder = nn.LSTM(hiddenDim, hiddenDim, num_layers=numLayers, dropout=dropout, batch_first=True)

        self.bn3 = BatchNorm(hiddenDim)
        # self.mergeProjection = nn.Linear(hiddenDim * 2, hiddenDim)

        self.head = head(hiddenDim, components)


    def forward(self, x):
        x, t = x

        x = self.fc1(x)
        x = self.bn1(x)

        x, (hidden, cell) = self.encoder(x)

        y = x[:, -self.timesteps:, :]
        z = self.skipProjection(y)

        y = self.bn2(y)

        t = t[:, -self.timesteps:]
        y = self.encodings(y, t)

        x, _ = self.decoder(y, (hidden, cell))
        x = self.bn3(x)

        # x = torch.cat([x, z], dim=-1)
        # x = self.mergeProjection(x)
        x = x + z
        x = self.head(x)

        return x
    
    def save(self, saveLocation):
        state = self.state_dict()

        torch.save(
                {'state_dict': state}, 
                saveLocation)
        

    def load(self, saveLocation):
        loaded = torch.load(saveLocation, weights_only=True)['state_dict']

        self.load_state_dict(loaded)


class StackTST(nn.Module):
    def __init__(self, timesteps, features, mixtures, dModel, heads, feed, layers, head=nn.Linear):
        super().__init__()

        self.timesteps = timesteps

        self.patchEmbedding = nn.Linear(features, dModel)

        self.positionalEncoding = RelativeEncoding(dModel, max_len=1000, batch_first=True)
        self.temporalEncoding = IndexedPositionalEncoding(dModel, batch_first=True)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dModel, heads, feed, batch_first=True),
            num_layers=layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dModel, heads, feed, batch_first=True),
            num_layers=layers
        )

        self.enforcer = nn.LSTM(dModel, dModel, 1, batch_first=True)

        self.output = head(dModel, mixtures)

    def forward(self, x):
        x, t = x

        x = self.patchEmbedding(x)
        x = self.positionalEncoding(x)
        x = self.encoder(x)

        t = self.temporalEncoding(torch.zeros_like(x), t)[:, :self.timesteps]
        x = self.positionalEncoding(x)
        x = self.decoder(t, x)
        x, _ = self.enforcer(x)

        x = self.output(x)

        return x


    def save(self, saveLocation):
        state = self.state_dict()

        torch.save(
                {'state_dict': state}, 
                saveLocation)
        

    def load(self, saveLocation):
        loaded = torch.load(saveLocation, weights_only=True)['state_dict']

        self.load_state_dict(loaded)
    

def sampleCMAL(mu, beta, tau, pi, numSamples):
    batchSize, timesteps, components = mu.shape

    mu = torch.repeat_interleave(mu, numSamples, dim=0)
    beta = torch.repeat_interleave(beta, numSamples, dim=0)
    tau = torch.repeat_interleave(tau, numSamples, dim=0)
    pi = torch.repeat_interleave(pi, numSamples, dim=0)

    samples = torch.zeros(batchSize * numSamples, timesteps)
    
    for t in range(timesteps):
        choices = torch.multinomial(pi[:, t, :], num_samples=1)

        tChosen = tau[:, t, :].gather(1, choices)
        mChosen = mu[:, t, :].gather(1, choices)
        bChosen = beta[:, t, :].gather(1, choices)

        u = torch.rand_like(mChosen)

        # thing = torch.where(
        #     u < tChosen,
        #     mChosen + ((bChosen * torch.log(u / tChosen)) / (1 - tChosen)),
        #     mChosen - ((bChosen * torch.log((1 - u) / (1 - tChosen))) / tChosen)
        # ).flatten()

        samples[:, t] = (mChosen + bChosen * (
            torch.where(
                u < tChosen,
                torch.log(u / tChosen) / (1 - tChosen),
                -torch.log((1 - u) / (1 - tChosen)) / tChosen
            )
        )).flatten()

    samples = samples.reshape(batchSize, numSamples, timesteps).transpose(1, 2)

    return samples


def sampleGaussian(mean, variance, numSamples):
    std = torch.sqrt(variance)
    noise = torch.randn(*mean.shape, numSamples)
    
    return mean.unsqueeze(-1) + std.unsqueeze(-1) * noise


