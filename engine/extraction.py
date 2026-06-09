from pydantic import BaseModel, HttpUrl, field_validator


# Anti-hallucination contract: every subagent result MUST validate against this schema.
# Results that fail validation are rejected before synthesis reaches them.
class Finding(BaseModel):
    claim: str
    evidence_span: str
    citation_url: HttpUrl

    @field_validator("claim", "evidence_span")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class FindingList(BaseModel):
    """Wrapper used for structured LLM output in the subagent extraction step."""
    findings: list[Finding]
