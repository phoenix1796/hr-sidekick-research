import json
import csv
from pprint import pprint

def _getJSON(path):
    with open("./datasets/%s" % path) as JsonFile:
        data = json.load(JsonFile)
    return data

def _getCSV(path):
    data = []
    with open("./datasets/%s" % path) as csvFile:
        dataReader = csv.reader(csvFile)
        for row in dataReader:
            data.append(row)
    return data

def getDataset(path):
    obj = {}
    #get file extension
    extension = path.split(".")[-1].lower()
    if extension == "csv":
        data = _getCSV(path)
    elif extension == "json":
        data = _getJSON(path)
    else:
        return "file extension not supported"
    # separate column headers and actual data
    obj["headers"]=data[0]
    obj["data"]=data[1:]
    return obj

def main():
    json = getDataset("allTogether/AmbientAirQuality-Delhi/2004.json")
    csv = getDataset("weather/delhiWeatherData - kaggle.csv")
    print(csv["headers"])
    print(json["headers"])

if __name__ == '__main__':
    main()