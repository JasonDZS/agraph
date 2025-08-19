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
      width: 120,
      render: (entityId: string, record: any) => {
        // Try to get entity name from multiple possible sources
        let entityName = entityId;

        // First try: direct head_entity object
        if (record.head_entity?.name) {
          entityName = record.head_entity.name;
        }
        // Second try: find in entities list by ID
        else if (entityId) {
          const entity = entities.find(e => e.id === entityId);
          if (entity?.name) {
            entityName = entity.name;
          }
        }

        return (
          <Tag color="blue" style={{ maxWidth: '100px' }}>
            <Text ellipsis title={entityName}>
              {entityName}
            </Text>
          </Tag>
        );
      },
    },
    {
      title: '目标实体',
      dataIndex: 'tail_entity_id',
      key: 'tail_entity_id',
      width: 120,
      render: (entityId: string, record: any) => {
        // Try to get entity name from multiple possible sources
        let entityName = entityId;

        // First try: direct tail_entity object
        if (record.tail_entity?.name) {
          entityName = record.tail_entity.name;
        }
        // Second try: find in entities list by ID
        else if (entityId) {
          const entity = entities.find(e => e.id === entityId);
          if (entity?.name) {
            entityName = entity.name;
          }
        }

        return (
          <Tag color="purple" style={{ maxWidth: '100px' }}>
            <Text ellipsis title={entityName}>
              {entityName}
            </Text>
          </Tag>
        );
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
      title: '文本片段',
      dataIndex: 'content',
      key: 'content',
      width: '60%',
      render: (content: string, record: any) => (
        <div style={{ width: '100%' }}>
          <Card
            size="small"
            style={{
              margin: '4px 0',
              border: '1px solid #e8e8e8',
              borderRadius: '8px',
              backgroundColor: '#fafafa',
            }}
            bodyStyle={{ padding: '12px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size={8}>
              <Space>
                <FileTextOutlined style={{ color: '#722ed1' }} />
                <Text strong style={{ color: '#722ed1' }}>
                  {record.title || '文本片段'}
                </Text>
                {record.confidence && (
                  <Tag color="blue" size="small">
                    {(record.confidence * 100).toFixed(0)}%
                  </Tag>
                )}
              </Space>

              <Paragraph
                style={{
                  margin: 0,
                  lineHeight: '1.6',
                  fontSize: '14px',
                  color: '#333',
                }}
                ellipsis={{
                  rows: 4,
                  expandable: true,
                  symbol: (
                    <Text style={{ color: '#1890ff', cursor: 'pointer' }}>
                      展开全文 ↓
                    </Text>
                  ),
                }}
              >
                {content}
              </Paragraph>

              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: '8px',
                  paddingTop: '8px',
                  borderTop: '1px solid #f0f0f0',
                }}
              >
                <Space>
                  {record.source && (
                    <Tooltip title={`来源: ${record.source}`}>
                      <Tag color="geekblue" size="small">
                        📄{' '}
                        {record.source.length > 15
                          ? record.source.substring(0, 15) + '...'
                          : record.source}
                      </Tag>
                    </Tooltip>
                  )}

                  {record.chunk_type && (
                    <Tag color="cyan" size="small">
                      {record.chunk_type}
                    </Tag>
                  )}
                </Space>

                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {record.start_index !== undefined &&
                  record.end_index !== undefined
                    ? `位置: ${record.start_index}-${record.end_index}`
                    : `长度: ${content.length} 字符`}
                </Text>
              </div>
            </Space>
          </Card>
        </div>
      ),
    },
    {
      title: '关联信息',
      dataIndex: 'entities',
      key: 'entities',
      width: '40%',
      render: (entityIds: string[] | undefined, record: any) => (
        <Space direction="vertical" style={{ width: '100%' }} size={8}>
          {/* 关联实体 */}
          <div>
            <Text strong style={{ fontSize: '12px', color: '#666' }}>
              🏷️ 关联实体
            </Text>
            <div style={{ marginTop: '4px' }}>
              {entityIds && entityIds.length > 0 ? (
                <Space wrap size={4}>
                  {entityIds.slice(0, 4).map(entityId => {
                    const entity = entities.find(e => e.id === entityId);
                    return (
                      <Tag
                        key={entityId}
                        size="small"
                        color="blue"
                        style={{ margin: '1px' }}
                      >
                        {entity?.name || entityId}
                      </Tag>
                    );
                  })}
                  {entityIds.length > 4 && (
                    <Tag size="small" color="default">
                      +{entityIds.length - 4}
                    </Tag>
                  )}
                </Space>
              ) : (
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  暂无关联实体
                </Text>
              )}
            </div>
          </div>

          {/* 关联关系 */}
          <div>
            <Text strong style={{ fontSize: '12px', color: '#666' }}>
              🔗 关联关系
            </Text>
            <div style={{ marginTop: '4px' }}>
              {record.relations && record.relations.length > 0 ? (
                <Space wrap size={4}>
                  {record.relations
                    .slice(0, 3)
                    .map((relationId: string, index: number) => {
                      const relation = relations.find(r => r.id === relationId);
                      return (
                        <Tag
                          key={relationId}
                          size="small"
                          color="green"
                          style={{ margin: '1px' }}
                        >
                          {relation?.relation_type || `关系${index + 1}`}
                        </Tag>
                      );
                    })}
                  {record.relations.length > 3 && (
                    <Tag size="small" color="default">
                      +{record.relations.length - 3}
                    </Tag>
                  )}
                </Space>
              ) : (
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  暂无关联关系
                </Text>
              )}
            </div>
          </div>

          {/* 统计信息 */}
          {record.metadata && Object.keys(record.metadata).length > 0 && (
            <div>
              <Text strong style={{ fontSize: '12px', color: '#666' }}>
                📊 元数据
              </Text>
              <div style={{ marginTop: '4px' }}>
                <Space wrap size={4}>
                  {Object.entries(record.metadata)
                    .slice(0, 2)
                    .map(([key, value]: [string, any]) => (
                      <Tag
                        key={key}
                        size="small"
                        color="orange"
                        style={{ margin: '1px' }}
                      >
                        {key}:{' '}
                        {String(value).length > 10
                          ? String(value).substring(0, 10) + '...'
                          : String(value)}
                      </Tag>
                    ))}
                </Space>
              </div>
            </div>
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
          <NodeIndexOutlined style={{ color: '#1890ff' }} />
          <Text strong>实体信息</Text>
          <Tag color="blue" size="small">
            {entities.length}
          </Tag>
        </Space>
      ),
      children: (
        <Table
          columns={entityColumns}
          dataSource={entities.map(entity => ({ ...entity, key: entity.id }))}
          pagination={{
            pageSize: 8,
            showSizeChanger: false,
            showQuickJumper: false,
            showTotal: (total, range) => `${range[0]}-${range[1]} / ${total}`,
            size: 'small',
          }}
          size="small"
          scroll={{ y: 350 }}
          style={{ backgroundColor: '#fff' }}
          rowClassName={(record, index) =>
            index % 2 === 0 ? '' : 'ant-table-row-striped'
          }
        />
      ),
    },
    {
      key: 'relations',
      label: (
        <Space>
          <ShareAltOutlined style={{ color: '#52c41a' }} />
          <Text strong>关系信息</Text>
          <Tag color="green" size="small">
            {relations.length}
          </Tag>
        </Space>
      ),
      children: (
        <Table
          columns={relationColumns}
          dataSource={relations.map(relation => ({
            ...relation,
            key: relation.id,
          }))}
          pagination={{
            pageSize: 8,
            showSizeChanger: false,
            showQuickJumper: false,
            showTotal: (total, range) => `${range[0]}-${range[1]} / ${total}`,
            size: 'small',
          }}
          size="small"
          scroll={{ y: 350 }}
          style={{ backgroundColor: '#fff' }}
          rowClassName={(record, index) =>
            index % 2 === 0 ? '' : 'ant-table-row-striped'
          }
        />
      ),
    },
    {
      key: 'text_chunks',
      label: (
        <Space>
          <FileTextOutlined style={{ color: '#722ed1' }} />
          <Text strong>文本片段</Text>
          <Tag color="purple" size="small">
            {text_chunks.length}
          </Tag>
        </Space>
      ),
      children: (
        <div
          style={{
            backgroundColor: '#f8f9fa',
            padding: '8px',
            borderRadius: '6px',
          }}
        >
          <Table
            columns={textChunkColumns}
            dataSource={text_chunks.map(chunk => ({ ...chunk, key: chunk.id }))}
            pagination={{
              pageSize: 3,
              showSizeChanger: false,
              showQuickJumper: false,
              showTotal: (total, range) =>
                `${range[0]}-${range[1]} / ${total} 条文本片段`,
              size: 'small',
            }}
            size="small"
            scroll={{ y: 450 }}
            style={{ backgroundColor: 'transparent' }}
            showHeader={false}
          />
        </div>
      ),
    },
  ];

  // Add reasoning tab if available
  if (reasoning) {
    tabItems.push({
      key: 'reasoning',
      label: (
        <Space>
          <InfoCircleOutlined style={{ color: '#fa8c16' }} />
          <Text strong>推理过程</Text>
        </Space>
      ),
      children: (
        <Card
          size="small"
          style={{
            height: 400,
            overflow: 'auto',
            backgroundColor: '#f6f8fa',
            border: '1px solid #e1e4e8',
          }}
          bodyStyle={{ padding: '16px' }}
        >
          <Paragraph
            style={{
              whiteSpace: 'pre-wrap',
              lineHeight: '1.6',
              fontSize: '14px',
              color: '#24292e',
              margin: 0,
            }}
          >
            {reasoning}
          </Paragraph>
        </Card>
      ),
    });
  }

  return (
    <>
      <div
        style={{
          height: '500px',
          backgroundColor: '#fff',
          borderRadius: '8px',
        }}
      >
        <div
          style={{
            marginBottom: 16,
            padding: '16px 16px 0 16px',
            borderBottom: '1px solid #f0f0f0',
          }}
        >
          <Space>
            <InfoCircleOutlined
              style={{ color: '#1890ff', fontSize: '16px' }}
            />
            <Title level={5} style={{ margin: 0, color: '#1890ff' }}>
              📊 该回答基于以下知识图谱信息生成
            </Title>
          </Space>
          <Text
            type="secondary"
            style={{ fontSize: '12px', display: 'block', marginTop: '4px' }}
          >
            点击查看 AI 回答依据的实体、关系和文本信息
          </Text>
        </div>

        <div
          style={{ padding: '0 16px 16px 16px', height: 'calc(100% - 80px)' }}
        >
          <Tabs
            defaultActiveKey="entities"
            items={tabItems}
            size="small"
            style={{ height: '100%' }}
            tabBarStyle={{
              marginBottom: 12,
              paddingLeft: 0,
            }}
            tabBarGutter={24}
          />
        </div>
      </div>

      <style jsx global>{`
        .ant-table-row-striped {
          background-color: #fafafa !important;
        }
        .ant-table-row-striped:hover {
          background-color: #f0f0f0 !important;
        }
        .ant-table-tbody > tr:hover > td {
          background-color: #e6f7ff !important;
        }
      `}</style>
    </>
  );
};

export default ContextPanel;
