from rich.console import Console

debug = False
version = "0.1.0"

# rich console is quiet by default
console = Console(quiet=True)


class MissingGenericException(Exception):
    def __init__(self, nachricht):
        self.nachricht = nachricht


class MissingTemplatesException(Exception):
    def __init__(self, nachricht):
        self.nachricht = nachricht
