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
    angle_options: Optional[list[str]]
    draft_content: Optional[dict]
    compliance_result: Optional[dict]
    stage: Optional[Literal["ideation", "creation", "compliance"]]
    human_message: Optional[str]
    human_decision: Optional[Literal["approve", "edit", "reject"]]
    human_edit_notes: Optional[str]
    target_stage: Optional[Literal["ideation", "creation"]]
    final_content: Optional[dict]
    node_error: Optional[str]
    node_error_source: Optional[str]
