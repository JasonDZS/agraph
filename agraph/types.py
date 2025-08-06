"""
图谱类型定义和枚举
"""

from enum import Enum
from typing import Any, Union

from .config import settings


def create_entity_type_enum() -> Any:
    """从配置动态创建实体类型枚举"""
    entity_types = {
        entity_type.upper().replace("-", "_").replace(" ", "_"): entity_type for entity_type in settings.ENTITY_TYPES
    }
    return Enum("EntityType", entity_types)


def create_relation_type_enum() -> Any:
    """从配置动态创建关系类型枚举"""
    relation_types = {
        relation_type.upper().replace("-", "_").replace(" ", "_"): relation_type
        for relation_type in settings.RELATION_TYPES
    }
    return Enum("RelationType", relation_types)


# 创建动态枚举类型
EntityType = create_entity_type_enum()
RelationType = create_relation_type_enum()

# 类型别名，用于类型注解
EntityTypeType = Union[Any, Enum]
RelationTypeType = Union[Any, Enum]
