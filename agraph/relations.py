"""
关系相关的数据结构和操作
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .entities import Entity
from .types import RelationType


@dataclass
class Relation:
    """关系类"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    head_entity: Optional[Entity] = None
    tail_entity: Optional[Entity] = None
    relation_type: Any = field(default_factory=lambda: RelationType.RELATED_TO)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Relation):
            return self.id == other.id
        return False

    def add_property(self, key: str, value: Any) -> None:
        """添加属性"""
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """获取属性"""
        return self.properties.get(key, default)

    def is_valid(self) -> bool:
        """验证关系有效性"""
        return self.head_entity is not None and self.tail_entity is not None and self.head_entity != self.tail_entity

    def reverse(self) -> "Relation":
        """创建反向关系"""
        return Relation(
            head_entity=self.tail_entity,
            tail_entity=self.head_entity,
            relation_type=self._get_reverse_relation_type(),
            properties=self.properties.copy(),
            confidence=self.confidence,
            source=self.source,
        )

    def _get_reverse_relation_type(self) -> Any:
        """获取反向关系类型"""
        reverse_map = {
            getattr(RelationType, "CONTAINS", None): getattr(RelationType, "BELONGS_TO", None),
            getattr(RelationType, "BELONGS_TO", None): getattr(RelationType, "CONTAINS", None),
            getattr(RelationType, "REFERENCES", None): getattr(RelationType, "REFERENCES", None),
            getattr(RelationType, "SIMILAR_TO", None): getattr(RelationType, "SIMILAR_TO", None),
            getattr(RelationType, "SYNONYMS", None): getattr(RelationType, "SYNONYMS", None),
        }
        return reverse_map.get(self.relation_type, self.relation_type)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "head_entity_id": self.head_entity.id if self.head_entity else None,
            "tail_entity_id": self.tail_entity.id if self.tail_entity else None,
            "relation_type": getattr(self.relation_type, "value", str(self.relation_type)),
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], entities_map: Dict[str, Entity]) -> "Relation":
        """从字典创建关系"""
        head_entity_id = data.get("head_entity_id")
        tail_entity_id = data.get("tail_entity_id")
        head_entity = entities_map.get(head_entity_id) if head_entity_id else None
        tail_entity = entities_map.get(tail_entity_id) if tail_entity_id else None

        relation_type_value = data.get("relation_type", "RELATED_TO")
        try:
            relation_type = RelationType(relation_type_value)
        except (ValueError, AttributeError):
            relation_type = getattr(RelationType, "RELATED_TO", None) or RelationType._member_map_.get("RELATED_TO")

        relation = cls(
            id=data.get("id", str(uuid.uuid4())),
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=relation_type,
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
        )
        if "created_at" in data:
            relation.created_at = datetime.fromisoformat(data["created_at"])
        return relation
