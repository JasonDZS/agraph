from pydantic import BaseModel, Field

from ..entities import Entity
from ..relations import Relation


class RetrievalEntity(BaseModel):
    entity: Entity = Field(..., description="Entity object representing the retrieved entity")
    score: float = Field(default=0.0, description="Relevance score of the entity in the retrieval context")


class RetrievalRelation(BaseModel):
    relation: Relation = Field(..., description="Relation object representing the retrieved relation")
    score: float = Field(default=0.0, description="Relevance score of the relation in the retrieval context")


class RetrievalResult(BaseModel):
    entities: list[RetrievalEntity] = Field(default_factory=list, description="List of retrieved entities with scores")
    relations: list[RetrievalRelation] = Field(
        default_factory=list, description="List of retrieved relations with scores"
    )

    query_time_ms: int = Field(default=0, description="Time taken to execute the query in milliseconds")
    success: bool = Field(default=True, description="Indicates if the retrieval was successful")
    msg: str = Field(default="", description="Message providing additional context about the retrieval result")
