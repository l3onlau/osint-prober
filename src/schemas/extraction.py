"""
Pydantic schemas for structured LLM extraction output.

Used by the Gatherer agent's `extract_and_save` tool via
`ChatOllama.with_structured_output(ExtractionResult)`.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A named entity extracted from source text."""

    id: str = Field(
        ...,
        description="A unique slugified identifier (e.g., 'jeffrey-epstein').",
    )
    name: str = Field(
        ...,
        description="The full, precise name of the entity.",
    )
    type: str = Field(
        ...,
        description="One of: 'Person', 'Organization', 'Location', 'Event'.",
    )
    summary: str = Field(
        ...,
        description="A 1–2 sentence description based on the source text.",
    )


class Relationship(BaseModel):
    """A directed, evidence-backed connection between two entities."""

    source_entity_id: str = Field(..., description="ID of the source entity.")
    target_entity_id: str = Field(..., description="ID of the target entity.")
    description: str = Field(
        ...,
        description="How these entities are connected (e.g., 'founded by').",
    )
    date: Optional[str] = Field(
        None,
        description="Date or timeframe if mentioned (e.g., '2019', 'July 2008').",
    )
    justifying_quote: str = Field(
        ...,
        description=(
            "An exact, character-for-character substring from the source text "
            "that proves this relationship."
        ),
    )


class ExtractionResult(BaseModel):
    """Complete extraction output from a single text chunk."""

    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
