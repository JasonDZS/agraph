import React, { useState, useEffect } from 'react';
import {
  Modal,
  Form,
  Input,
  InputNumber,
  Button,
  Tabs,
  Space,
  Typography,
  Card,
  Tag,
  Select,
  Switch,
  message,
  Spin,
  Alert,
  Divider,
  Tooltip,
  Row,
  Col,
} from 'antd';
import {
  SaveOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  SettingOutlined,
  DatabaseOutlined,
  BranchesOutlined,
  ExperimentOutlined,
  CloudUploadOutlined,
  FileTextOutlined,
  QuestionCircleOutlined,
  SafetyOutlined,
  FolderOutlined,
} from '@ant-design/icons';
import { configService } from '@/services/configService';
import type { ProjectConfig, ConfigUpdateRequest } from '@/types/config';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

interface ProjectConfigModalProps {
  visible: boolean;
  projectName: string;
  onCancel: () => void;
  onSuccess?: () => void;
}

const ProjectConfigModal: React.FC<ProjectConfigModalProps> = ({
  visible,
  projectName,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [activeTab, setActiveTab] = useState('basic');

  const loadConfig = async () => {
    if (!projectName) return;

    setLoading(true);
    try {
      const response = await configService.getProjectConfig(projectName);

      if (response.success && response.data) {
        const configData = response.data as unknown as ProjectConfig;
        setConfig(configData);

        // Set form values
        if (configData && configData.settings) {
          const settings = configData.settings;
          form.setFieldsValue({
            // Basic settings
            description: configData.project_info?.description || '',
            workdir: settings.workdir || '',
            current_project: settings.current_project || projectName,
            max_current: settings.max_current || 5,

            // OpenAI settings
            openai_api_key: settings.openai?.api_key || '',
            openai_api_base: settings.openai?.api_base || '',

            // LLM settings
            llm_model: settings.llm?.model || '',
            llm_temperature: settings.llm?.temperature || 0,
            llm_max_tokens: settings.llm?.max_tokens || 4096,
            llm_provider: settings.llm?.provider || 'openai',

            // Embedding settings
            embedding_model: settings.embedding?.model || '',
            embedding_provider: settings.embedding?.provider || 'openai',
            embedding_dimension: settings.embedding?.dimension || 1024,
            embedding_max_token_size: settings.embedding?.max_token_size || 8192,
            embedding_batch_size: settings.embedding?.batch_size || 32,

            // Graph settings
            entity_types: settings.graph?.entity_types || [],
            relation_types: settings.graph?.relation_types || [],

            // Text processing settings
            max_chunk_size: settings.text?.max_chunk_size || 512,
            chunk_overlap: settings.text?.chunk_overlap || 100,

            // RAG settings
            system_prompt: settings.rag?.system_prompt || '',

            // Builder settings
            builder_enable_cache: settings.builder?.enable_cache ?? true,
            builder_cache_dir: settings.builder?.cache_dir || 'cache',
            builder_cache_ttl: settings.builder?.cache_ttl || 86400,
            builder_auto_cleanup: settings.builder?.auto_cleanup ?? false,
            builder_chunk_size: settings.builder?.chunk_size || 512,
            builder_chunk_overlap: settings.builder?.chunk_overlap || 100,
            builder_entity_confidence_threshold: settings.builder?.entity_confidence_threshold || 0.5,
            builder_relation_confidence_threshold: settings.builder?.relation_confidence_threshold || 0.5,
            builder_cluster_algorithm: settings.builder?.cluster_algorithm || 'leiden',
            builder_min_cluster_size: settings.builder?.min_cluster_size || 5,
            builder_enable_user_interaction: settings.builder?.enable_user_interaction ?? true,
            builder_auto_save_edits: settings.builder?.auto_save_edits ?? false,
          });
        }
      }
    } catch (error) {
      console.error('Failed to load config:', error);
      message.error('Failed to load project configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const values = await form.validateFields();

      const updates: ConfigUpdateRequest = {
        // Basic settings
        workdir: values.workdir,
        current_project: values.current_project,
        max_current: values.max_current,

        // OpenAI settings
        openai_api_key: values.openai_api_key,
        openai_api_base: values.openai_api_base,

        // LLM settings
        llm_model: values.llm_model,
        llm_temperature: values.llm_temperature,
        llm_max_tokens: values.llm_max_tokens,
        llm_provider: values.llm_provider,

        // Embedding settings
        embedding_model: values.embedding_model,
        embedding_provider: values.embedding_provider,
        embedding_dimension: values.embedding_dimension,
        embedding_max_token_size: values.embedding_max_token_size,
        embedding_batch_size: values.embedding_batch_size,

        // Graph settings
        entity_types: values.entity_types,
        relation_types: values.relation_types,

        // Text processing settings
        max_chunk_size: values.max_chunk_size,
        chunk_overlap: values.chunk_overlap,

        // RAG settings
        system_prompt: values.system_prompt,

        // Builder settings
        builder_enable_cache: values.builder_enable_cache,
        builder_cache_dir: values.builder_cache_dir,
        builder_cache_ttl: values.builder_cache_ttl,
        builder_auto_cleanup: values.builder_auto_cleanup,
        builder_chunk_size: values.builder_chunk_size,
        builder_chunk_overlap: values.builder_chunk_overlap,
        builder_entity_confidence_threshold: values.builder_entity_confidence_threshold,
        builder_relation_confidence_threshold: values.builder_relation_confidence_threshold,
        builder_cluster_algorithm: values.builder_cluster_algorithm,
        builder_min_cluster_size: values.builder_min_cluster_size,
        builder_enable_user_interaction: values.builder_enable_user_interaction,
        builder_auto_save_edits: values.builder_auto_save_edits,
      };

      // Validate configuration before sending
      const validation = configService.validateConfigUpdate(updates);
      if (!validation.valid) {
        message.error(`Configuration validation failed: ${validation.errors.join(', ')}`);
        return;
      }

      const response = await configService.updateProjectConfig(
        projectName,
        updates
      );

      if (response.success) {
        message.success('Project configuration saved successfully');
        await loadConfig(); // Reload to show updated values
        onSuccess?.();
      } else {
        message.error('Failed to save configuration');
      }
    } catch (error) {
      console.error('Failed to save config:', error);
      message.error('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleRecoverFromBackup = async () => {
    try {
      setSaving(true);
      const response = await configService.loadProjectConfigFromBackup(projectName);

      if (response.success) {
        message.success('Configuration recovered from backup successfully');
        await loadConfig(); // Reload to show recovered values
      } else {
        message.error('Failed to recover configuration from backup');
      }
    } catch (error) {
      console.error('Failed to recover from backup:', error);
      message.error('Failed to recover configuration from backup');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    try {
      setSaving(true);
      const response = await configService.resetProjectConfig(projectName);

      if (response.success) {
        message.success('Project configuration reset to defaults');
        await loadConfig(); // Reload to show reset values
      } else {
        message.error('Failed to reset configuration');
      }
    } catch (error) {
      console.error('Failed to reset config:', error);
      message.error('Failed to reset configuration');
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    if (visible && projectName) {
      loadConfig();
    }
  }, [visible, projectName]);

  const basicInfoTab = (
    <div>
      <Title level={4}>
        <InfoCircleOutlined style={{ marginRight: 8 }} />
        Project Information
      </Title>

      {config?.project_info && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Project Name: </Text>
              <Text>{config.project_info.project_name}</Text>
            </div>
            <div>
              <Text strong>Created: </Text>
              <Text>
                {new Date(config.project_info.created_at).toLocaleString()}
              </Text>
            </div>
            <div>
              <Text strong>Version: </Text>
              <Tag>{config.project_info.version}</Tag>
            </div>
            {config.project_info.paths && (
              <div>
                <Text strong>Project Directory: </Text>
                <Text code>{config.project_info.paths.project_dir}</Text>
              </div>
            )}
          </Space>
        </Card>
      )}

      <Row gutter={16}>
        <Col span={12}>
          <Form.Item
            label="Description"
            name="description"
            rules={[
              { max: 500, message: 'Description cannot exceed 500 characters' },
            ]}
          >
            <TextArea rows={3} placeholder="Project description..." />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item
            label="Working Directory"
            name="workdir"
            rules={[{ required: true, message: 'Please enter working directory' }]}
          >
            <Input placeholder="workdir" prefix={<FolderOutlined />} />
          </Form.Item>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={12}>
          <Form.Item
            label="Current Project"
            name="current_project"
            tooltip="The name of the currently active project"
          >
            <Input placeholder="Project name" disabled />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item
            label="Max Current Items"
            name="max_current"
            tooltip="Maximum number of current items to keep in memory"
            rules={[
              { type: 'number', min: 1, max: 100, message: 'Must be between 1 and 100' },
            ]}
          >
            <InputNumber min={1} max={100} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
      </Row>

      {config?.runtime_info && (
        <Card title="Runtime Information" size="small">
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Collection: </Text>
              <Text code>{config.runtime_info.collection_name}</Text>
            </div>
            <div>
              <Text strong>Vector Store: </Text>
              <Text>{config.runtime_info.vector_store_type}</Text>
            </div>
            <div>
              <Text strong>Knowledge Graph: </Text>
              <Tag
                color={
                  config.runtime_info.enable_knowledge_graph
                    ? 'green'
                    : 'orange'
                }
              >
                {config.runtime_info.enable_knowledge_graph
                  ? 'Enabled'
                  : 'Disabled'}
              </Tag>
            </div>
            <div>
              <Text strong>Initialized: </Text>
              <Tag color={config.runtime_info.is_initialized ? 'green' : 'red'}>
                {config.runtime_info.is_initialized ? 'Yes' : 'No'}
              </Tag>
            </div>
          </Space>
        </Card>
      )}
    </div>
  );

  const openaiTab = (
    <div>
      <Title level={4}>OpenAI Configuration</Title>

      <Form.Item
        label="API Key"
        name="openai_api_key"
        rules={[{ required: true, message: 'Please enter OpenAI API key' }]}
      >
        <Input.Password placeholder="sk-..." />
      </Form.Item>

      <Form.Item
        label="API Base URL"
        name="openai_api_base"
        rules={[
          { required: true, message: 'Please enter API base URL' },
          { type: 'url', message: 'Please enter a valid URL' },
        ]}
      >
        <Input placeholder="https://api.openai.com/v1" />
      </Form.Item>
    </div>
  );

  const llmTab = (
    <div>
      <Title level={4}>Language Model Configuration</Title>

      <Form.Item
        label="Model"
        name="llm_model"
        rules={[{ required: true, message: 'Please enter LLM model' }]}
      >
        <Input placeholder="gpt-4" />
      </Form.Item>

      <Form.Item
        label="Provider"
        name="llm_provider"
        rules={[{ required: true, message: 'Please select provider' }]}
      >
        <Select placeholder="Select provider">
          <Option value="openai">OpenAI</Option>
          <Option value="huggingface">HuggingFace</Option>
          <Option value="ollama">Ollama</Option>
        </Select>
      </Form.Item>

      <Form.Item
        label="Temperature"
        name="llm_temperature"
        rules={[
          { required: true, message: 'Please enter temperature' },
          {
            type: 'number',
            min: 0,
            max: 2,
            message: 'Temperature must be between 0 and 2',
          },
        ]}
      >
        <InputNumber
          min={0}
          max={2}
          step={0.1}
          style={{ width: '100%' }}
          placeholder="0.7"
        />
      </Form.Item>

      <Form.Item
        label="Max Tokens"
        name="llm_max_tokens"
        rules={[
          { required: true, message: 'Please enter max tokens' },
          { type: 'number', min: 1, message: 'Max tokens must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={32768}
          style={{ width: '100%' }}
          placeholder="4096"
        />
      </Form.Item>
    </div>
  );

  const embeddingTab = (
    <div>
      <Title level={4}>Embedding Configuration</Title>

      <Form.Item
        label="Model"
        name="embedding_model"
        rules={[{ required: true, message: 'Please enter embedding model' }]}
      >
        <Input placeholder="text-embedding-ada-002" />
      </Form.Item>

      <Form.Item
        label="Provider"
        name="embedding_provider"
        rules={[{ required: true, message: 'Please select provider' }]}
      >
        <Select placeholder="Select provider">
          <Option value="openai">OpenAI</Option>
          <Option value="huggingface">HuggingFace</Option>
        </Select>
      </Form.Item>

      <Form.Item
        label="Dimension"
        name="embedding_dimension"
        rules={[
          { required: true, message: 'Please enter dimension' },
          { type: 'number', min: 1, message: 'Dimension must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={4096}
          style={{ width: '100%' }}
          placeholder="1536"
        />
      </Form.Item>

      <Form.Item
        label="Max Token Size"
        name="embedding_max_token_size"
        rules={[
          { required: true, message: 'Please enter max token size' },
          {
            type: 'number',
            min: 1,
            message: 'Max token size must be positive',
          },
        ]}
      >
        <InputNumber
          min={1}
          max={32768}
          style={{ width: '100%' }}
          placeholder="8192"
        />
      </Form.Item>

      <Form.Item
        label="Batch Size"
        name="embedding_batch_size"
        rules={[
          { required: true, message: 'Please enter batch size' },
          { type: 'number', min: 1, message: 'Batch size must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={100}
          style={{ width: '100%' }}
          placeholder="32"
        />
      </Form.Item>
    </div>
  );

  const textTab = (
    <div>
      <Title level={4}>Text Processing Configuration</Title>

      <Form.Item
        label="Max Chunk Size"
        name="max_chunk_size"
        rules={[
          { required: true, message: 'Please enter max chunk size' },
          { type: 'number', min: 1, message: 'Chunk size must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={4096}
          style={{ width: '100%' }}
          placeholder="512"
        />
      </Form.Item>

      <Form.Item
        label="Chunk Overlap"
        name="chunk_overlap"
        rules={[
          { required: true, message: 'Please enter chunk overlap' },
          {
            type: 'number',
            min: 0,
            message: 'Chunk overlap must be non-negative',
          },
        ]}
      >
        <InputNumber
          min={0}
          max={500}
          style={{ width: '100%' }}
          placeholder="100"
        />
      </Form.Item>
    </div>
  );

  const graphTab = (
    <div>
      <Title level={4}>
        <BranchesOutlined style={{ marginRight: 8 }} />
        Knowledge Graph Configuration
      </Title>

      <Alert
        message="Graph Configuration"
        description="Configure entity types and relation types for knowledge graph construction."
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      <Form.Item
        label="Entity Types"
        name="entity_types"
        tooltip="Define the types of entities to extract from documents"
        rules={[{ type: 'array', min: 1, message: 'Please add at least one entity type' }]}
      >
        <Select
          mode="tags"
          style={{ width: '100%' }}
          placeholder="Add entity types (e.g., person, organization, location)"
          tokenSeparators={[',']}
        >
          <Option value="person">Person</Option>
          <Option value="organization">Organization</Option>
          <Option value="location">Location</Option>
          <Option value="concept">Concept</Option>
          <Option value="event">Event</Option>
          <Option value="product">Product</Option>
          <Option value="software">Software</Option>
          <Option value="document">Document</Option>
          <Option value="keyword">Keyword</Option>
          <Option value="database">Database</Option>
          <Option value="table">Table</Option>
          <Option value="column">Column</Option>
        </Select>
      </Form.Item>

      <Form.Item
        label="Relation Types"
        name="relation_types"
        tooltip="Define the types of relationships between entities"
        rules={[{ type: 'array', min: 1, message: 'Please add at least one relation type' }]}
      >
        <Select
          mode="tags"
          style={{ width: '100%' }}
          placeholder="Add relation types (e.g., contains, belongs_to, located_in)"
          tokenSeparators={[',']}
        >
          <Option value="contains">Contains</Option>
          <Option value="belongs_to">Belongs To</Option>
          <Option value="located_in">Located In</Option>
          <Option value="works_for">Works For</Option>
          <Option value="causes">Causes</Option>
          <Option value="part_of">Part Of</Option>
          <Option value="is_a">Is A</Option>
          <Option value="references">References</Option>
          <Option value="similar_to">Similar To</Option>
          <Option value="related_to">Related To</Option>
          <Option value="depends_on">Depends On</Option>
          <Option value="foreign_key">Foreign Key</Option>
          <Option value="mentions">Mentions</Option>
          <Option value="describes">Describes</Option>
          <Option value="creates">Creates</Option>
          <Option value="develops">Develops</Option>
        </Select>
      </Form.Item>
    </div>
  );

  const builderTab = (
    <div>
      <Title level={4}>
        <ExperimentOutlined style={{ marginRight: 8 }} />
        Knowledge Graph Builder Configuration
      </Title>

      <Alert
        message="Builder Settings"
        description="Advanced settings for knowledge graph construction and processing."
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      {/* Cache Settings */}
      <Card title="Cache Settings" size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item
              label="Enable Cache"
              name="builder_enable_cache"
              valuePropName="checked"
              tooltip="Enable caching for improved performance"
            >
              <Switch />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              label="Auto Cleanup"
              name="builder_auto_cleanup"
              valuePropName="checked"
              tooltip="Automatically clean up old cache entries"
            >
              <Switch />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              label="Cache TTL (seconds)"
              name="builder_cache_ttl"
              tooltip="Time to live for cache entries"
              rules={[
                { type: 'number', min: 60, message: 'TTL must be at least 60 seconds' },
              ]}
            >
              <InputNumber min={60} max={604800} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          label="Cache Directory"
          name="builder_cache_dir"
          rules={[{ required: true, message: 'Please enter cache directory' }]}
          tooltip="Directory for storing cache files"
        >
          <Input placeholder="cache" prefix={<FolderOutlined />} />
        </Form.Item>
      </Card>

      {/* Processing Settings */}
      <Card title="Processing Settings" size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="Builder Chunk Size"
              name="builder_chunk_size"
              tooltip="Text chunk size for builder processing"
              rules={[
                { type: 'number', min: 1, message: 'Chunk size must be positive' },
              ]}
            >
              <InputNumber min={1} max={4096} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Builder Chunk Overlap"
              name="builder_chunk_overlap"
              tooltip="Overlap between text chunks"
              rules={[
                { type: 'number', min: 0, message: 'Overlap must be non-negative' },
              ]}
            >
              <InputNumber min={0} max={500} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
      </Card>

      {/* Confidence Thresholds */}
      <Card title="Confidence Thresholds" size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="Entity Confidence"
              name="builder_entity_confidence_threshold"
              tooltip="Minimum confidence for entity extraction"
              rules={[
                { type: 'number', min: 0, max: 1, message: 'Must be between 0 and 1' },
              ]}
            >
              <InputNumber
                min={0}
                max={1}
                step={0.1}
                style={{ width: '100%' }}
                placeholder="0.5"
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Relation Confidence"
              name="builder_relation_confidence_threshold"
              tooltip="Minimum confidence for relation extraction"
              rules={[
                { type: 'number', min: 0, max: 1, message: 'Must be between 0 and 1' },
              ]}
            >
              <InputNumber
                min={0}
                max={1}
                step={0.1}
                style={{ width: '100%' }}
                placeholder="0.5"
              />
            </Form.Item>
          </Col>
        </Row>
      </Card>

      {/* Clustering Settings */}
      <Card title="Clustering Settings" size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="Cluster Algorithm"
              name="builder_cluster_algorithm"
              tooltip="Algorithm for entity clustering"
              rules={[{ required: true, message: 'Please select cluster algorithm' }]}
            >
              <Select placeholder="Select algorithm">
                <Option value="leiden">Leiden</Option>
                <Option value="louvain">Louvain</Option>
                <Option value="kmeans">K-Means</Option>
                <Option value="hierarchical">Hierarchical</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Min Cluster Size"
              name="builder_min_cluster_size"
              tooltip="Minimum size for entity clusters"
              rules={[
                { type: 'number', min: 1, message: 'Min cluster size must be positive' },
              ]}
            >
              <InputNumber min={1} max={100} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
      </Card>

      {/* Interaction Settings */}
      <Card title="User Interaction" size="small">
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="Enable User Interaction"
              name="builder_enable_user_interaction"
              valuePropName="checked"
              tooltip="Allow user to interact during building process"
            >
              <Switch />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Auto Save Edits"
              name="builder_auto_save_edits"
              valuePropName="checked"
              tooltip="Automatically save user edits"
            >
              <Switch />
            </Form.Item>
          </Col>
        </Row>
      </Card>
    </div>
  );

  const ragTab = (
    <div>
      <Title level={4}>
        <FileTextOutlined style={{ marginRight: 8 }} />
        RAG System Prompt Configuration
      </Title>

      <Alert
        message="System Prompt"
        description="Configure the system prompt that guides the AI's behavior during retrieval-augmented generation."
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      <Form.Item
        label="System Prompt"
        name="system_prompt"
        rules={[{ required: true, message: 'Please enter system prompt' }]}
        tooltip="The system prompt defines how the AI should behave when answering questions"
      >
        <TextArea
          rows={12}
          placeholder="Enter the system prompt for RAG..."
          showCount
          maxLength={5000}
          style={{ fontFamily: 'monospace' }}
        />
      </Form.Item>

      <Space>
        <Text type="secondary">Tip:</Text>
        <Text type="secondary">
          Use clear instructions about the AI's role, response format, and how it should use the knowledge graph context.
        </Text>
      </Space>
    </div>
  );

  const items = [
    {
      key: 'basic',
      label: (
        <span>
          <InfoCircleOutlined />
          Basic Info
        </span>
      ),
      children: basicInfoTab,
    },
    {
      key: 'openai',
      label: (
        <span>
          <CloudUploadOutlined />
          OpenAI
        </span>
      ),
      children: openaiTab,
    },
    {
      key: 'llm',
      label: (
        <span>
          <ExperimentOutlined />
          Language Model
        </span>
      ),
      children: llmTab,
    },
    {
      key: 'embedding',
      label: (
        <span>
          <DatabaseOutlined />
          Embedding
        </span>
      ),
      children: embeddingTab,
    },
    {
      key: 'graph',
      label: (
        <span>
          <BranchesOutlined />
          Knowledge Graph
        </span>
      ),
      children: graphTab,
    },
    {
      key: 'builder',
      label: (
        <span>
          <ExperimentOutlined />
          Builder
        </span>
      ),
      children: builderTab,
    },
    {
      key: 'text',
      label: (
        <span>
          <FileTextOutlined />
          Text Processing
        </span>
      ),
      children: textTab,
    },
    {
      key: 'rag',
      label: (
        <span>
          <SafetyOutlined />
          RAG Prompt
        </span>
      ),
      children: ragTab,
    },
  ];

  return (
    <Modal
      title={
        <Space>
          <SettingOutlined />
          <span>Project Configuration - {projectName}</span>
        </Space>
      }
      open={visible}
      onCancel={onCancel}
      width={800}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          Cancel
        </Button>,
        <Tooltip key="recover-tooltip" title="Recover settings from backup file">
          <Button
            key="recover"
            onClick={handleRecoverFromBackup}
            loading={saving}
            icon={<SafetyOutlined />}
          >
            Recover from Backup
          </Button>
        </Tooltip>,
        <Button
          key="reset"
          onClick={handleReset}
          loading={saving}
          icon={<ReloadOutlined />}
        >
          Reset to Defaults
        </Button>,
        <Button
          key="save"
          type="primary"
          onClick={handleSave}
          loading={saving}
          icon={<SaveOutlined />}
        >
          Save Configuration
        </Button>,
      ]}
    >
      <Spin spinning={loading}>
        {config ? (
          <Form form={form} layout="vertical" autoComplete="off">
            <Tabs activeKey={activeTab} onChange={setActiveTab} items={items} />
          </Form>
        ) : (
          <Alert
            message="Loading Configuration"
            description="Please wait while we load the project configuration..."
            type="info"
            showIcon
          />
        )}
      </Spin>
    </Modal>
  );
};

export default ProjectConfigModal;
