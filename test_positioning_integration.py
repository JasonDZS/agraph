#!/usr/bin/env python3
"""
Test script to verify entity positioning integration in AGraph.

This script tests the complete entity positioning workflow:
1. Text chunking with position tracking
2. Entity extraction with positioning
3. Position-aware entity queries
"""

import asyncio
from pathlib import Path

from agraph.base.models import Entity
from agraph.base.models.positioning import AlignmentStatus, CharInterval, Position
from agraph import KnowledgeGraphBuilder
from agraph.chunker import TokenChunker
from agraph.config import BuilderConfig


async def test_positioning_integration():
    """Test the complete positioning integration workflow."""
    print("=== AGraph 实体定位集成测试 ===\n")

    # Test text with clear entities
    test_text = "患者张三，男性，65岁，因急性心肌梗死入院治疗。医生李四负责主治。"

    print(f"测试文本: {test_text}")
    print(f"文本长度: {len(test_text)} 字符\n")

    # 1. Test text chunking with positions
    print("1. 测试文本分块和位置追踪:")
    chunker = TokenChunker(chunk_size=50, chunk_overlap=0)
    chunks_with_positions = chunker.split_text_with_positions(test_text)

    for i, (chunk_text, start_pos, end_pos) in enumerate(chunks_with_positions):
        print(f"   分块 {i}: '{chunk_text}' 位置 [{start_pos}-{end_pos}]")
    print()

    # 2. Test knowledge graph builder with positioning
    print("2. 测试知识图谱构建和实体定位:")

    config = BuilderConfig(
        chunk_size=50,
        chunk_overlap=0,
        entity_confidence_threshold=0.3,
        relation_confidence_threshold=0.3,
        cache_dir="./test_cache"
    )

    builder = KnowledgeGraphBuilder(config=config, enable_knowledge_graph=True)

    try:
        # Build knowledge graph from the test text
        kg = await builder.build_from_text(
            texts=[test_text],
            graph_name="定位测试图谱",
            graph_description="用于测试实体定位功能",
            use_cache=False
        )

        print(f"   构建完成: {len(kg.entities)} 个实体, {len(kg.relations)} 个关系")
        print(f"   文本块数量: {len(kg.text_chunks)}")
        print()

        # 3. Check entity positioning information
        print("3. 检查实体定位信息:")
        positioned_entities = []
        for entity_id, entity in kg.entities.items():
            if entity.has_position():
                positioned_entities.append(entity)
                char_pos = entity.get_char_position()
                alignment_status = entity.get_alignment_status()
                confidence = entity.get_position_confidence()

                print(f"   实体: '{entity.name}' ({entity.entity_type})")
                print(f"      位置: {char_pos}")
                print(f"      对齐状态: {alignment_status.value}")
                print(f"      置信度: {confidence:.2f}")

                # Verify position accuracy
                if char_pos:
                    start_pos, end_pos = char_pos
                    actual_text = test_text[start_pos:end_pos]
                    print(f"      实际文本: '{actual_text}'")
                    print(f"      匹配准确: {actual_text.lower() == entity.name.lower()}")
                print()

        print(f"总计 {len(positioned_entities)} 个实体有定位信息")

        # 4. Test position-based entity operations
        print("\n4. 测试基于位置的实体操作:")

        if len(positioned_entities) >= 2:
            entity1, entity2 = positioned_entities[0], positioned_entities[1]
            overlap = entity1.overlaps_with(entity2)
            print(f"   实体 '{entity1.name}' 与 '{entity2.name}' 重叠: {overlap}")

            # Test position serialization
            entity1_dict = entity1.to_dict()
            if "position" in entity1_dict:
                print(f"   实体序列化包含位置信息: ✓")

                # Test deserialization
                restored_entity = Entity.from_dict(entity1_dict)
                if restored_entity.has_position():
                    print(f"   位置信息反序列化成功: ✓")
                else:
                    print(f"   位置信息反序列化失败: ✗")
            else:
                print(f"   实体序列化缺少位置信息: ✗")

        # 5. Test position filtering and queries
        print("\n5. 测试位置过滤和查询:")

        # Find entities in specific character range
        target_range = (0, 10)  # First 10 characters
        entities_in_range = []

        for entity in positioned_entities:
            char_pos = entity.get_char_position()
            if char_pos:
                start_pos, end_pos = char_pos
                if start_pos >= target_range[0] and end_pos <= target_range[1]:
                    entities_in_range.append(entity)

        print(f"   字符范围 {target_range} 内的实体:")
        for entity in entities_in_range:
            print(f"      - {entity.name} {entity.get_char_position()}")

        if len(entities_in_range) == 0:
            print(f"      (无实体在指定范围内)")

        print("\n✅ 定位集成测试完成!")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await builder.aclose()


async def test_position_models():
    """Test the positioning models in isolation."""
    print("=== 定位模型单元测试 ===\n")

    # Test CharInterval
    print("1. CharInterval 测试:")
    char1 = CharInterval(start_pos=0, end_pos=10)
    char2 = CharInterval(start_pos=5, end_pos=15)
    char3 = CharInterval(start_pos=2, end_pos=8)

    print(f"   char1: {char1}")
    print(f"   char2: {char2}")
    print(f"   char3: {char3}")
    print(f"   char1 与 char2 重叠: {char1.overlaps(char2)}")
    print(f"   char3 嵌套在 char1 中: {char3.is_nested_in(char1)}")
    print()

    # Test Position with alignment status
    print("2. Position 对齐状态测试:")
    positions = [
        Position(
            char_interval=CharInterval(start_pos=0, end_pos=5),
            alignment_status=AlignmentStatus.MATCH_EXACT,
            confidence=1.0
        ),
        Position(
            char_interval=CharInterval(start_pos=10, end_pos=20),
            alignment_status=AlignmentStatus.MATCH_FUZZY,
            confidence=0.8
        ),
        Position(
            char_interval=CharInterval(start_pos=25, end_pos=30),
            alignment_status=AlignmentStatus.MATCH_LESSER,
            confidence=0.6
        )
    ]

    for i, pos in enumerate(positions):
        print(f"   位置 {i+1}: {pos.get_char_range()}")
        print(f"      对齐状态: {pos.alignment_status.value}")
        print(f"      精确对齐: {pos.is_precisely_aligned}")
        print(f"      置信度: {pos.confidence}")
    print()

    # Test position overlaps
    print("3. Position 重叠检测:")
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            overlap = positions[i].overlaps_with(positions[j])
            print(f"   位置 {i+1} 与位置 {j+1} 重叠: {overlap}")

    print("\n✅ 定位模型测试完成!")


async def main():
    """Main test function."""
    print("开始 AGraph 实体定位功能测试...\n")

    # Test positioning models
    await test_position_models()
    print()

    # Test positioning integration
    success = await test_positioning_integration()

    if success:
        print("\n🎉 所有测试通过! AGraph 已成功集成实体定位功能")
    else:
        print("\n❌ 部分测试失败，请检查实现")


if __name__ == "__main__":
    asyncio.run(main())
