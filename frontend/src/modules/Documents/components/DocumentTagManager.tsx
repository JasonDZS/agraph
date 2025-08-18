import React, { useState, useEffect } from 'react';
import {
  Modal,
  Input,
  Tag,
  Space,
  Button,
  List,
  Typography,
  Divider,
  message,
  Popconfirm,
  Card,
  Row,
  Col,
  Statistic,
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  TagsOutlined,
  CheckOutlined,
  CloseOutlined,
} from '@ant-design/icons';
import { documentService } from '../../../services';

const { Text } = Typography;

interface DocumentTagManagerProps {
  open: boolean;
  onClose: () => void;
  projectName?: string;
  onTagsChange?: (tags: string[]) => void;
}

interface TagInfo {
  name: string;
  count: number;
}

const DocumentTagManager: React.FC<DocumentTagManagerProps> = ({
  open,
  onClose,
  projectName,
  onTagsChange,
}) => {
  const [tags, setTags] = useState<TagInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [newTagName, setNewTagName] = useState('');
  const [editingTag, setEditingTag] = useState<string | null>(null);
  const [editTagName, setEditTagName] = useState('');

  // Load tags
  const loadTags = async () => {
    setLoading(true);
    try {
      const response = await documentService.getStats(projectName);
      const tagData = response.data.tags || {};

      const tagList: TagInfo[] = Object.entries(tagData).map(
        ([name, count]) => ({
          name,
          count: count as number,
        })
      );

      tagList.sort((a, b) => b.count - a.count); // Sort by usage count
      setTags(tagList);
    } catch (error) {
      console.error('Failed to load tags:', error);
      message.error('加载标签失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      loadTags();
    }
  }, [open, projectName]);

  // Add new tag
  const handleAddTag = () => {
    if (!newTagName.trim()) {
      message.warning('请输入标签名称');
      return;
    }

    if (tags.some(tag => tag.name === newTagName.trim())) {
      message.warning('标签已存在');
      return;
    }

    const newTag: TagInfo = {
      name: newTagName.trim(),
      count: 0,
    };

    setTags(prev => [...prev, newTag]);
    setNewTagName('');
    message.success('标签添加成功');

    onTagsChange?.(tags.map(t => t.name).concat(newTag.name));
  };

  // Start editing tag
  const handleEditTag = (tagName: string) => {
    setEditingTag(tagName);
    setEditTagName(tagName);
  };

  // Save edited tag
  const handleSaveEdit = () => {
    if (!editTagName.trim()) {
      message.warning('请输入标签名称');
      return;
    }

    if (
      editTagName.trim() !== editingTag &&
      tags.some(tag => tag.name === editTagName.trim())
    ) {
      message.warning('标签已存在');
      return;
    }

    setTags(prev =>
      prev.map(tag =>
        tag.name === editingTag ? { ...tag, name: editTagName.trim() } : tag
      )
    );

    setEditingTag(null);
    setEditTagName('');
    message.success('标签编辑成功');

    onTagsChange?.(
      tags.map(t => (t.name === editingTag ? editTagName.trim() : t.name))
    );
  };

  // Cancel editing
  const handleCancelEdit = () => {
    setEditingTag(null);
    setEditTagName('');
  };

  // Delete tag
  const handleDeleteTag = (tagName: string) => {
    const tag = tags.find(t => t.name === tagName);
    if (tag && tag.count > 0) {
      message.warning(
        `标签 "${tagName}" 正在被 ${tag.count} 个文档使用，无法删除`
      );
      return;
    }

    setTags(prev => prev.filter(tag => tag.name !== tagName));
    message.success('标签删除成功');

    onTagsChange?.(tags.filter(t => t.name !== tagName).map(t => t.name));
  };

  const getTagColor = (count: number) => {
    if (count === 0) return 'default';
    if (count <= 2) return 'blue';
    if (count <= 5) return 'green';
    if (count <= 10) return 'orange';
    return 'red';
  };

  const totalTags = tags.length;
  const usedTags = tags.filter(tag => tag.count > 0).length;
  const totalUsage = tags.reduce((sum, tag) => sum + tag.count, 0);

  return (
    <Modal
      title={
        <Space>
          <TagsOutlined />
          <span>标签管理</span>
        </Space>
      }
      open={open}
      onCancel={onClose}
      footer={null}
      width={700}
    >
      {/* Statistics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Statistic
            title="总标签数"
            value={totalTags}
            prefix={<TagsOutlined />}
          />
        </Col>
        <Col span={8}>
          <Statistic title="使用中标签" value={usedTags} />
        </Col>
        <Col span={8}>
          <Statistic title="标签使用次数" value={totalUsage} />
        </Col>
      </Row>

      <Divider />

      {/* Add New Tag */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space.Compact style={{ width: '100%' }}>
          <Input
            placeholder="输入新标签名称"
            value={newTagName}
            onChange={e => setNewTagName(e.target.value)}
            onPressEnter={handleAddTag}
          />
          <Button type="primary" icon={<PlusOutlined />} onClick={handleAddTag}>
            添加标签
          </Button>
        </Space.Compact>
      </Card>

      {/* Tag List */}
      <List
        loading={loading}
        dataSource={tags}
        renderItem={tag => (
          <List.Item
            actions={[
              <Button
                key="edit"
                type="text"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleEditTag(tag.name)}
                disabled={editingTag === tag.name}
              />,
              <Popconfirm
                key="delete"
                title={
                  tag.count > 0
                    ? `标签 "${tag.name}" 正在被 ${tag.count} 个文档使用，确定要删除吗？`
                    : `确定要删除标签 "${tag.name}" 吗？`
                }
                onConfirm={() => handleDeleteTag(tag.name)}
                okText="删除"
                cancelText="取消"
                disabled={editingTag === tag.name}
              >
                <Button
                  type="text"
                  size="small"
                  icon={<DeleteOutlined />}
                  danger
                  disabled={editingTag === tag.name}
                />
              </Popconfirm>,
            ]}
          >
            <List.Item.Meta
              title={
                editingTag === tag.name ? (
                  <Space.Compact>
                    <Input
                      size="small"
                      value={editTagName}
                      onChange={e => setEditTagName(e.target.value)}
                      onPressEnter={handleSaveEdit}
                      autoFocus
                    />
                    <Button
                      size="small"
                      type="primary"
                      icon={<CheckOutlined />}
                      onClick={handleSaveEdit}
                    />
                    <Button
                      size="small"
                      icon={<CloseOutlined />}
                      onClick={handleCancelEdit}
                    />
                  </Space.Compact>
                ) : (
                  <Space>
                    <Tag color={getTagColor(tag.count)}>{tag.name}</Tag>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {tag.count} 个文档使用
                    </Text>
                  </Space>
                )
              }
            />
          </List.Item>
        )}
        locale={{ emptyText: '暂无标签' }}
      />
    </Modal>
  );
};

export default DocumentTagManager;
