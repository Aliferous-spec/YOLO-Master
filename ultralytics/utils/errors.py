# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Exception hierarchy for YOLO-Master.

Base hierarchy:
    YOLOMasterError
    |-- HUBModelError          - model not found on Ultralytics HUB
    |-- MoERouterError          - MoE router input/config violation
    |-- PEFTPlannerError        - PEFT planner decision failure
    |   `-- PEFTRefusalError    - planner refused (valid decision, not a crash)
    `-- ShapeMismatchError      - tensor shape contract violation
"""

from ultralytics.utils import emojis


class YOLOMasterError(Exception):
    """Base exception for all YOLO-Master custom errors.

    All YOLO-Master-specific exceptions inherit from this class so callers
    can catch the entire family with a single ``except YOLOMasterError``.
    """

    def __init__(self, message: str = ""):
        super().__init__(emojis(message))


class HUBModelError(YOLOMasterError):
    """Exception raised when a model cannot be found or retrieved from Ultralytics HUB.

    This custom exception is used specifically for handling errors related to model fetching in Ultralytics YOLO. The
    error message is processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Methods:
        __init__: Initialize the HUBModelError with a custom message.

    Examples:
        >>> try:
        ...     # Code that might fail to find a model
        ...     raise HUBModelError("Custom model not found message")
        ... except HUBModelError as e:
        ...     print(e)  # Displays the emoji-enhanced error message
    """

    def __init__(self, message: str = "Model not found. Please check model URL and try again."):
        """Initialize a HUBModelError exception.

        This exception is raised when a requested model is not found or cannot be retrieved from Ultralytics HUB. The
        message is processed to include emojis for better user experience.

        Args:
            message (str, optional): The error message to display when the exception is raised.
        """
        super().__init__(message)


class MoERouterError(YOLOMasterError):
    """Raised when a MoE router receives invalid input or configuration.

    Common causes:
        - Input tensor is not 4-D (NCHW expected).
        - ``in_channels`` does not match the router's first conv layer.
        - ``top_k`` exceeds ``num_experts`` at call time.
        - Numerical breakdown (NaN/Inf) in routing logits.
    """


class PEFTPlannerError(YOLOMasterError):
    """Base error for PEFT Planner operational failures."""


class PEFTRefusalError(PEFTPlannerError):
    """Raised when the PEFT Planner refuses a configuration.

    This is a **valid planning decision**, not a crash. The caller should
    catch this and fall back to full fine-tuning (Full-SFT).

    Attributes:
        reason: Human-readable refusal explanation.
        predicted_delta: Predicted ΔmAP that triggered the refusal.
    """

    def __init__(self, reason: str = "", predicted_delta: float = 0.0):
        self.reason = reason
        self.predicted_delta = predicted_delta
        msg = f"PEFT Planner refused: {reason}" if reason else "PEFT Planner refused."
        if predicted_delta:
            msg += f" (predicted ΔmAP={predicted_delta:.3f})"
        super().__init__(msg)


class ShapeMismatchError(YOLOMasterError):
    """Raised when a tensor shape contract is violated.

    Attributes:
        expected: Expected shape (or shape pattern string).
        actual: Actual tensor shape.
        context: Optional description of where the mismatch occurred.
    """

    def __init__(self, expected, actual, context: str = ""):
        self.expected = expected
        self.actual = actual
        self.context = context
        msg = f"Shape mismatch: expected {expected}, got {actual}"
        if context:
            msg += f" [{context}]"
        super().__init__(msg)
