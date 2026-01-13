"""Protocol subsystem for Delibera.

Provides declarative protocol specifications and interpretation.
"""

from delibera.protocol.defaults import (
    DEFAULT_PROTOCOL,
    create_simple_protocol,
    create_tree_protocol_v1,
)
from delibera.protocol.interpreter import (
    ExpandAction,
    FinalizeAction,
    ProtocolInterpreter,
    PruneAction,
    ReduceAction,
    ValidateAction,
    WorkAction,
)
from delibera.protocol.loader import (
    ProtocolLoadError,
    load_protocol_from_yaml,
    protocol_from_dict,
)
from delibera.protocol.spec import (
    ConvergenceSpec,
    ExpandSpec,
    ProtocolSpec,
    PruneSpec,
    ReduceSpec,
    StepSpec,
    validate_protocol,
)

__all__ = [
    # Spec
    "ProtocolSpec",
    "StepSpec",
    "ExpandSpec",
    "PruneSpec",
    "ReduceSpec",
    "ConvergenceSpec",
    "validate_protocol",
    # Defaults
    "DEFAULT_PROTOCOL",
    "create_simple_protocol",
    "create_tree_protocol_v1",
    # Loader
    "load_protocol_from_yaml",
    "protocol_from_dict",
    "ProtocolLoadError",
    # Interpreter
    "ProtocolInterpreter",
    "ExpandAction",
    "WorkAction",
    "ValidateAction",
    "PruneAction",
    "ReduceAction",
    "FinalizeAction",
]
