"""
实体相关的数据结构和操作
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from .types import EntityType


@dataclass
class Entity:
    """实体类"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: Any = field(default_factory=lambda: EntityType.UNKNOWN)
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """返回实体的哈希值"""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id

    def add_alias(self, alias: str) -> None:
        """添加别名"""
        if alias and alias not in self.aliases:
            self.aliases.append(alias)

    def add_property(self, key: str, value: Any) -> None:
        """添加属性"""
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """获取属性"""
        return self.properties.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": getattr(self.entity_type, "value", str(self.entity_type)),
            "description": self.description,
            "properties": self.properties,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """从字典创建实体"""
        entity_type_value = data.get("entity_type", "UNKNOWN")
        try:
            entity_type = EntityType(entity_type_value)
        except (ValueError, AttributeError):
            entity_type = getattr(EntityType, "UNKNOWN", None) or EntityType._member_map_.get("UNKNOWN")

        entity = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            entity_type=entity_type,
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            aliases=data.get("aliases", []),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
        )
        if "created_at" in data:
            entity.created_at = datetime.fromisoformat(data["created_at"])
        return entity
