import React from 'react';
import {
  Tabs,
  Table,
  Tag,
  Typography,
  Space,
  Card,
  Empty,
  Tooltip,
} from 'antd';
import {
  NodeIndexOutlined,
  ShareAltOutlined,
  FileTextOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { ContextPanelProps } from '../types';

const { Title, Text, Paragraph } = Typography;

const ContextPanel: React.FC<ContextPanelProps> = ({
  context,
  visible,
  onClose,
}) => {
  if (!context) {
    return (
      <Empty
        description="暂无上下文信息"
        image={Empty.PRESENTED_IMAGE_SIMPLE}
      />
    );
  }

  const {
    entities = [],
    relations = [],
    text_chunks = [],
    reasoning,
  } = context;

  const entityColumns = [
    {
      title: '实体名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: any) => (
        <Space>
          <NodeIndexOutlined style={{ color: '#1890ff' }} />
          <Text strong>{name}</Text>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'entity_type',
      key: 'entity_type',
      render: (type: string) => <Tag color="blue">{type}</Tag>,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      render: (description: string) => (
        <Text type="secondary">{description || '无描述'}</Text>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Tag
          color={
            confidence > 0.8 ? 'green' : confidence > 0.6 ? 'orange' : 'red'
          }
        >
          {(confidence * 100).toFixed(1)}%
        </Tag>
      ),
    },
  ];

  const relationColumns = [
    {
      title: '关系类型',
      dataIndex: 'relation_type',
      key: 'relation_type',
      render: (type: string) => (
        <Space>
          <ShareAltOutlined style={{ color: '#52c41a' }} />
          <Tag color="green">{type}</Tag>
        </Space>
      ),
    },
    {
      title: '源实体',
      dataIndex: 'head_entity_id',
      key: 'head_entity_id',
      render: (entityId: string) => {
        const entity = entities.find(e => e.id === entityId);
        return entity ? entity.name : entityId;
      },
    },
    {
      title: '目标实体',
      dataIndex: 'tail_entity_id',
      key: 'tail_entity_id',
      render: (entityId: string) => {
        const entity = entities.find(e => e.id === entityId);
        return entity ? entity.name : entityId;
      },
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      render: (description: string) => (
        <Text type="secondary">{description || '无描述'}</Text>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Tag
          color={
            confidence > 0.8 ? 'green' : confidence > 0.6 ? 'orange' : 'red'
          }
        >
          {(confidence * 100).toFixed(1)}%
        </Tag>
      ),
    },
  ];

  const textChunkColumns = [
    {
      title: '文本内容',
      dataIndex: 'content',
      key: 'content',
      render: (content: string) => (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Space>
            <FileTextOutlined style={{ color: '#722ed1' }} />
            <Text strong>文本片段</Text>
          </Space>
          <Paragraph
            ellipsis={{ rows: 3, expandable: true, symbol: '展开' }}
            style={{ margin: 0 }}
          >
            {content}
          </Paragraph>
        </Space>
      ),
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      width: 200,
      render: (source: string) => (
        <Tooltip title={source}>
          <Text type="secondary" ellipsis>
            {source || '未知来源'}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: '关联实体',
      dataIndex: 'entities',
      key: 'entities',
      width: 150,
      render: (entityIds: string[]) => (
        <Space wrap>
          {entityIds?.slice(0, 3).map(entityId => {
            const entity = entities.find(e => e.id === entityId);
            return (
              <Tag key={entityId} size="small" color="blue">
                {entity?.name || entityId}
              </Tag>
            );
          })}
          {entityIds?.length > 3 && (
            <Tag size="small">+{entityIds.length - 3}</Tag>
          )}
        </Space>
      ),
    },
  ];

  const tabItems = [
    {
      key: 'entities',
      label: (
        <Space>
          <NodeIndexOutlined />
          实体信息
          <Tag>{entities.length}</Tag>
        </Space>
      ),
      children: (
        <Table
          columns={entityColumns}
          dataSource={entities.map(entity => ({ ...entity, key: entity.id }))}
          pagination={{ pageSize: 10, showSizeChanger: false }}
          size="small"
          scroll={{ y: 400 }}
        />
      ),
    },
    {
      key: 'relations',
      label: (
        <Space>
          <ShareAltOutlined />
          关系信息
          <Tag>{relations.length}</Tag>
        </Space>
      ),
      children: (
        <Table
          columns={relationColumns}
          dataSource={relations.map(relation => ({
            ...relation,
            key: relation.id,
          }))}
          pagination={{ pageSize: 10, showSizeChanger: false }}
          size="small"
          scroll={{ y: 400 }}
        />
      ),
    },
    {
      key: 'text_chunks',
      label: (
        <Space>
          <FileTextOutlined />
          文本片段
          <Tag>{text_chunks.length}</Tag>
        </Space>
      ),
      children: (
        <Table
          columns={textChunkColumns}
          dataSource={text_chunks.map(chunk => ({ ...chunk, key: chunk.id }))}
          pagination={{ pageSize: 5, showSizeChanger: false }}
          size="small"
          scroll={{ y: 400 }}
        />
      ),
    },
  ];

  // Add reasoning tab if available
  if (reasoning) {
    tabItems.push({
      key: 'reasoning',
      label: (
        <Space>
          <InfoCircleOutlined />
          推理过程
        </Space>
      ),
      children: (
        <Card size="small" style={{ height: 400, overflow: 'auto' }}>
          <Paragraph style={{ whiteSpace: 'pre-wrap' }}>{reasoning}</Paragraph>
        </Card>
      ),
    });
  }

  return (
    <div style={{ height: '500px' }}>
      <div style={{ marginBottom: 16 }}>
        <Space>
          <InfoCircleOutlined style={{ color: '#1890ff' }} />
          <Title level={5} style={{ margin: 0 }}>
            该回答基于以下知识图谱信息生成
          </Title>
        </Space>
      </div>

      <Tabs
        defaultActiveKey="entities"
        items={tabItems}
        size="small"
        style={{ height: '100%' }}
        tabBarStyle={{ marginBottom: 16 }}
      />
    </div>
  );
};

export default ContextPanel;
