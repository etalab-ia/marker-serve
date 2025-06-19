from enum import Enum
from typing import Any

from pydantic import BaseModel

class OutputFormat(str, Enum):
    markdown = "markdown"
    json = "json"
    html = "html"


class ParseResponse(BaseModel):
    format: OutputFormat
    output: str
    images: dict[str, str]
    metadata: dict[str, Any]
    success: bool
