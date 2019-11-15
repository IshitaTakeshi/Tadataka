class TimeStamp(object):
    def __init__(self):
        self.timestamp = 0

    def increment(self):
        self.timestamp += 1

    def get(self):
        return self.timestamp
