from datetime import datetime

mostRecent = None

model = None


def refreshData():
    if (datetime.now() - mostRecent).minutes > 15:
        getData()

    mostRecent = datetime.now()


def getData():
    if mostRecent is None:
        return

    startDate = mostRecent
    endDate = datetime.now()





