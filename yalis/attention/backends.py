# These imports trigger @register_attention decorators
from . import sdpa_and_flex  # noqa: F401
from . import flash  # noqa: F401
from . import thresh  # noqa: F401
from . import thresh_nowmp  # noqa: F401
from . import double_sparse  # noqa: F401

from enum import Enum


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASH = "flash"
    FLEX = "flex"
    THRESH = "thresh"
    THRESH_ATTN_NOWMP = "thresh_attn_nowmp"
    DOUBLE_SPARSE = "double_sparse"
