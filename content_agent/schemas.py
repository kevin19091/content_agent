from typing import Literal, Optional

from pydantic import BaseModel


class ComplianceResult(BaseModel):
    passed: bool
    issues: list[str] = []
    severity: Literal["none", "minor", "blocking"]


class WhatsAppContent(BaseModel):
    template_name: str
    header: Optional[str] = None
    body: str
    footer: Optional[str] = None
    cta_button_text: Optional[str] = None


class PushContent(BaseModel):
    title: str
    body: str
    cta_button_text: Optional[str] = None
