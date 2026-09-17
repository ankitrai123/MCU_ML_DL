"""Exception hierarchy for EdgeForge.

The CLI catches EdgeForgeError at the top level and prints ``str(err)`` without
a traceback, so every message raised here must stand on its own as something a
user can act on.
"""


class EdgeForgeError(Exception):
    """Base class for all expected EdgeForge failures."""


class BoardError(EdgeForgeError):
    """A board YAML file is missing, malformed, or fails schema validation."""


class UnsupportedModelError(EdgeForgeError):
    """The input file is not a model type/format EdgeForge can ingest."""


class UnsupportedOpError(EdgeForgeError):
    """The model uses an operator/layer outside the supported op set."""


class TierMismatchError(EdgeForgeError):
    """The model's tier (deep) exceeds what the target board accepts (classical-only)."""


class FootprintExceededError(EdgeForgeError):
    """The model's estimated flash/RAM footprint exceeds the board's declared budget."""


class ToolchainNotFoundError(EdgeForgeError):
    """The board's declared compiler toolchain is not installed on this machine."""


class BuildError(EdgeForgeError):
    """A toolchain invocation (compile/link/objcopy) returned a nonzero exit code."""


class ValidationError(EdgeForgeError):
    """Golden-vector validation failed: generated C output diverged from the reference model."""
