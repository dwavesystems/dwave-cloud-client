class ConfigFileError(Exception):
    """Base exception for all config file processing errors."""

class ConfigFileReadError(ConfigFileError):
    """Non-existing or unreadable config file specified or implied."""

class ConfigFileParseError(ConfigFileError):
    """Invalid format of config file."""


class SolverError(Exception):
    """Generic base class for all solver-related errors."""

class SolverFailureError(SolverError):
    """An exception raised when there is a remote failure calling a solver."""

class SolverAuthenticationError(SolverError):
    """An exception raised when there is an authentication error."""

    def __init__(self):
        super(SolverAuthenticationError, self).__init__("Token not accepted for that action.")

class UnsupportedSolverError(SolverError):
    """The solver we received from the API is not supported by the client."""


class Timeout(Exception):
    """General timeout error."""

class RequestTimeout(Timeout):
    """REST API request timed out."""

class PollingTimeout(Timeout):
    """Problem polling timed out."""


class CanceledFutureError(Exception):
    """An exception raised when code tries to read from a canceled future."""

    def __init__(self):
        super(CanceledFutureError, self).__init__("An error occurred reading results from a canceled request")


class InvalidAPIResponseError(Exception):
    """Raised when an invalid/unexpected response from D-Wave Solver API is received."""
