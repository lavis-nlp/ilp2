class MissingGenericException(Exception):
    def __init__(self, nachricht):
        self.nachricht = nachricht

class MissingTemplatesException(Exception):
    def __init__(self, nachricht):
        self.nachricht = nachricht