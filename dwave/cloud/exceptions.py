class SolverFailureError(Exception):
    """An exception raised when there is a remote failure calling a solver."""


class SolverAuthenticationError(Exception):
    """An exception raised when there is an authentication error."""

    def __init__(self):
        super(SolverAuthenticationError, self).__init__("Token not accepted for that action.")


class CanceledFutureError(Exception):
    """An exception raised when code tries to read from a canceled future."""

    def __init__(self):
        super(CanceledFutureError, self).__init__("An error occurred reading results from a canceled request")


class InvalidAPIResponseError(Exception):
    """Raised when an invalid/unexpected response from D-Wave Solver API is received."""


class UnsupportedSolverError(Exception):
    """The solver we received from the API is not supported by the client."""
