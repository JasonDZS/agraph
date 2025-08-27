import React, { useState } from 'react';
import {
  Modal,
  Form,
  Input,
  Switch,
  Button,
  Space,
  Typography,
  Divider,
  Alert,
  Checkbox,
  message,
} from 'antd';
import {
  RocketOutlined,
  InfoCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import type {
  EnhancedProject,
  KnowledgeGraphBuildRequest,
} from '../types/project';

const { Text, Title } = Typography;

interface ProjectBuildModalProps {
  visible: boolean;
  onCancel: () => void;
  onBuild: (request: KnowledgeGraphBuildRequest) => Promise<void>;
  project: EnhancedProject | null;
  loading?: boolean;
}

export const ProjectBuildModal: React.FC<ProjectBuildModalProps> = ({
  visible,
  onCancel,
  onBuild,
  project,
  loading = false,
}) => {
  const [form] = Form.useForm();
  const [useAllDocuments, setUseAllDocuments] = useState(true);
  const [specificDocumentIds, setSpecificDocumentIds] = useState<string>('');

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();

      const buildRequest: KnowledgeGraphBuildRequest = {
        graph_name: values.graph_name,
        graph_description: values.graph_description,
        use_cache: values.use_cache,
        save_to_vector_store: values.save_to_vector_store,
        enable_graph: values.enable_graph,
      };

      // Handle document selection
      if (!useAllDocuments && specificDocumentIds.trim()) {
        buildRequest.document_ids = specificDocumentIds
          .split(',')
          .map(id => id.trim())
          .filter(id => id.length > 0);
      }
      // If useAllDocuments is true, we don't set document_ids, which means use all documents

      await onBuild(buildRequest);
      form.resetFields();
      setUseAllDocuments(true);
      setSpecificDocumentIds('');
    } catch (error) {
      console.error('Form validation failed:', error);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setUseAllDocuments(true);
    setSpecificDocumentIds('');
    onCancel();
  };

  const initialValues = {
    graph_name: project ? `${project.name} Knowledge Graph` : 'Knowledge Graph',
    graph_description: project
      ? `Knowledge graph built from ${project.name} project`
      : 'Built by AGraph',
    use_cache: true,
    save_to_vector_store: true,
    enable_graph: true,
  };

  return (
    <Modal
      title={
        <Space>
          <RocketOutlined />
          <span>Build Knowledge Graph</span>
        </Space>
      }
      open={visible}
      onCancel={handleCancel}
      footer={[
        <Button key="cancel" onClick={handleCancel}>
          Cancel
        </Button>,
        <Button
          key="build"
          type="primary"
          icon={<RocketOutlined />}
          loading={loading}
          onClick={handleSubmit}
        >
          Build Knowledge Graph
        </Button>,
      ]}
      width={600}
      destroyOnClose
    >
      <div style={{ marginBottom: 16 }}>
        <Alert
          message="Build Knowledge Graph"
          description={`This will create a knowledge graph from the documents in project "${project?.name}". The process may take some time depending on the amount of data.`}
          type="info"
          icon={<InfoCircleOutlined />}
          showIcon
        />
      </div>

      <Form
        form={form}
        layout="vertical"
        initialValues={initialValues}
        preserve={false}
      >
        <Title level={5}>
          <SettingOutlined /> Basic Configuration
        </Title>

        <Form.Item
          name="graph_name"
          label="Graph Name"
          rules={[{ required: true, message: 'Please enter a graph name' }]}
        >
          <Input placeholder="Enter knowledge graph name" />
        </Form.Item>

        <Form.Item
          name="graph_description"
          label="Graph Description"
          rules={[
            { required: true, message: 'Please enter a graph description' },
          ]}
        >
          <Input.TextArea
            rows={3}
            placeholder="Enter description for the knowledge graph"
          />
        </Form.Item>

        <Divider />

        <Title level={5}>Document Selection</Title>

        <div style={{ marginBottom: 16 }}>
          <Checkbox
            checked={useAllDocuments}
            onChange={e => setUseAllDocuments(e.target.checked)}
          >
            Use all documents in the project
          </Checkbox>
          <Text
            type="secondary"
            style={{ display: 'block', marginTop: 4, fontSize: '12px' }}
          >
            {project?.statistics?.document_count
              ? `This will include all ${project.statistics.document_count} documents in the project`
              : 'This will include all documents in the project'}
          </Text>
        </div>

        {!useAllDocuments && (
          <div style={{ marginBottom: 16 }}>
            <Text strong style={{ fontSize: '14px' }}>
              Specific Document IDs
            </Text>
            <Input.TextArea
              rows={3}
              placeholder="Enter document IDs separated by commas (e.g., doc1, doc2, doc3)"
              value={specificDocumentIds}
              onChange={e => setSpecificDocumentIds(e.target.value)}
              style={{ marginTop: 8 }}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Leave empty to use all documents
            </Text>
          </div>
        )}

        <Divider />

        <Title level={5}>Processing Options</Title>

        <Form.Item
          name="use_cache"
          valuePropName="checked"
          style={{ marginBottom: 12 }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <Text strong>Use Cache</Text>
              <Text
                type="secondary"
                style={{ display: 'block', fontSize: '12px' }}
              >
                Reuse previously processed data to speed up the build
              </Text>
            </div>
            <Switch />
          </div>
        </Form.Item>

        <Form.Item
          name="save_to_vector_store"
          valuePropName="checked"
          style={{ marginBottom: 12 }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <Text strong>Save to Vector Store</Text>
              <Text
                type="secondary"
                style={{ display: 'block', fontSize: '12px' }}
              >
                Store embeddings for semantic search capabilities
              </Text>
            </div>
            <Switch />
          </div>
        </Form.Item>

        <Form.Item
          name="enable_graph"
          valuePropName="checked"
          style={{ marginBottom: 0 }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <Text strong>Enable Knowledge Graph</Text>
              <Text
                type="secondary"
                style={{ display: 'block', fontSize: '12px' }}
              >
                Build entity-relation knowledge graph structure
              </Text>
            </div>
            <Switch />
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default ProjectBuildModal;
