from typing import Literal, Optional

from typing_extensions import TypedDict


class ContentRequest(TypedDict):
    client_id: str
    channel: Literal["whatsapp", "push"]
    campaign_topic: str


class AgentState(TypedDict):
    request: ContentRequest
    brand_guidelines: Optional[dict]
    brief: Optional[dict]
    angle: Optional[str]
    draft_content: Optional[dict]
    compliance_result: Optional[dict]
    stage: Optional[Literal["ideation", "creation", "compliance"]]
    human_decision: Optional[Literal["approve", "edit", "reject"]]
    human_edit_notes: Optional[str]
    final_content: Optional[dict]
