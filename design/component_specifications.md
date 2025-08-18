# AGraph 前端组件规格说明

## 核心组件详细设计

### 1. 项目管理组件

#### ProjectCard 组件

```typescript
// modules/Projects/components/ProjectCard.tsx

interface ProjectCardProps {
  project: Project;
  selected?: boolean;
  onSelect?: (project: Project) => void;
  onEdit?: (project: Project) => void;
  onDelete?: (project: Project) => void;
  actions?: React.ReactNode;
}

interface Project {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt?: string;
  documentCount: number;
  hasKnowledgeGraph: boolean;
  statistics: {
    sizeInMB: number;
    entityCount: number;
    relationCount: number;
    textChunkCount: number;
  };
  status: 'active' | 'building' | 'error';
}

const ProjectCard: React.FC<ProjectCardProps> = ({
  project,
  selected = false,
  onSelect,
  onEdit,
  onDelete,
  actions,
}) => {
  const [loading, setLoading] = useState(false);

  const handleSelect = () => {
    if (onSelect && !loading) {
      onSelect(project);
    }
  };

  const renderStatusBadge = () => {
    const statusConfig = {
      active: { color: 'green', text: '正常' },
      building: { color: 'orange', text: '构建中' },
      error: { color: 'red', text: '错误' },
    };

    const config = statusConfig[project.status];
    return <Badge color={config.color} text={config.text} />;
  };

  return (
    <Card
      className={`project-card ${selected ? 'selected' : ''}`}
      hoverable
      onClick={handleSelect}
      title={
        <div className="project-header">
          <Space>
            <Typography.Title level={4} className="project-name">
              {project.name}
            </Typography.Title>
            {renderStatusBadge()}
          </Space>
          <Dropdown
            menu={{
              items: [
                {
                  key: 'edit',
                  label: '编辑项目',
                  icon: <EditOutlined />,
                  onClick: () => onEdit?.(project),
                },
                {
                  key: 'delete',
                  label: '删除项目',
                  icon: <DeleteOutlined />,
                  danger: true,
                  onClick: () => onDelete?.(project),
                },
              ],
            }}
            trigger={['click']}
            placement="bottomRight"
          >
            <Button type="text" icon={<MoreOutlined />} />
          </Dropdown>
        </div>
      }
      extra={actions}
      loading={loading}
    >
      <div className="project-content">
        {project.description && (
          <Typography.Paragraph
            ellipsis={{ rows: 2, expandable: true }}
            className="project-description"
          >
            {project.description}
          </Typography.Paragraph>
        )}

        <Row gutter={[16, 16]} className="project-stats">
          <Col span={12}>
            <Statistic
              title="文档数量"
              value={project.documentCount}
              prefix={<FileTextOutlined />}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="存储大小"
              value={project.statistics.sizeInMB}
              suffix="MB"
              prefix={<DatabaseOutlined />}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="实体数量"
              value={project.statistics.entityCount}
              prefix={<NodeIndexOutlined />}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="关系数量"
              value={project.statistics.relationCount}
              prefix={<BranchesOutlined />}
            />
          </Col>
        </Row>

        <div className="project-footer">
          <Space>
            <Typography.Text type="secondary" className="project-date">
              创建于 {dayjs(project.createdAt).format('YYYY-MM-DD')}
            </Typography.Text>
            {project.hasKnowledgeGraph ? (
              <Tag color="green">已构建知识图谱</Tag>
            ) : (
              <Tag color="orange">未构建知识图谱</Tag>
            )}
          </Space>
        </div>
      </div>
    </Card>
  );
};
```

#### ProjectCreateModal 组件

```typescript
// modules/Projects/components/ProjectCreateModal.tsx

interface ProjectCreateModalProps {
  visible: boolean;
  onCancel: () => void;
  onSuccess: (project: Project) => void;
}

interface CreateProjectFormData {
  name: string;
  description?: string;
  template?: string;
  copyFrom?: string;
}

const ProjectCreateModal: React.FC<ProjectCreateModalProps> = ({
  visible,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm<CreateProjectFormData>();
  const [loading, setLoading] = useState(false);
  const { projects } = useProjectStore();

  const handleSubmit = async (values: CreateProjectFormData) => {
    setLoading(true);
    try {
      const project = await projectService.create({
        name: values.name,
        description: values.description,
      });

      // 如果选择了复制源项目
      if (values.copyFrom) {
        await configService.copyProjectConfig(values.copyFrom, values.name);
      }

      onSuccess(project.data);
      form.resetFields();
    } catch (error) {
      console.error('Failed to create project:', error);
    } finally {
      setLoading(false);
    }
  };

  const validateProjectName = async (rule: any, value: string) => {
    if (!value) {
      throw new Error('请输入项目名称');
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(value)) {
      throw new Error('项目名称只能包含字母、数字、下划线和连字符');
    }

    if (projects.some(p => p.name === value)) {
      throw new Error('项目名称已存在');
    }
  };

  return (
    <Modal
      title="创建新项目"
      open={visible}
      onCancel={onCancel}
      confirmLoading={loading}
      onOk={() => form.submit()}
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
      >
        <Form.Item
          name="name"
          label="项目名称"
          rules={[
            { required: true, message: '请输入项目名称' },
            { validator: validateProjectName },
            { max: 50, message: '项目名称不能超过50个字符' },
          ]}
        >
          <Input
            placeholder="请输入项目名称"
            showCount
            maxLength={50}
          />
        </Form.Item>

        <Form.Item
          name="description"
          label="项目描述"
          rules={[
            { max: 500, message: '项目描述不能超过500个字符' },
          ]}
        >
          <Input.TextArea
            placeholder="请输入项目描述（可选）"
            rows={3}
            showCount
            maxLength={500}
          />
        </Form.Item>

        <Form.Item
          name="template"
          label="项目模板"
        >
          <Select placeholder="选择项目模板（可选）" allowClear>
            <Select.Option value="research">学术研究</Select.Option>
            <Select.Option value="business">商业分析</Select.Option>
            <Select.Option value="legal">法律文档</Select.Option>
            <Select.Option value="technical">技术文档</Select.Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="copyFrom"
          label="复制配置"
        >
          <Select placeholder="从现有项目复制配置（可选）" allowClear>
            {projects.map(project => (
              <Select.Option key={project.id} value={project.name}>
                {project.name}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>
      </Form>
    </Modal>
  );
};
```

### 2. 文档管理组件

#### DocumentUploader 组件

```typescript
// modules/Documents/components/DocumentUploader.tsx

interface DocumentUploaderProps {
  onUploadComplete?: (documents: UploadedDocument[]) => void;
  onUploadError?: (error: Error) => void;
  acceptedTypes?: string[];
  maxFileSize?: number;
  maxFiles?: number;
  multiple?: boolean;
}

interface UploadedDocument {
  id: string;
  filename: string;
  size: number;
  contentType: string;
  uploadProgress: number;
  status: 'uploading' | 'success' | 'error';
  error?: string;
}

const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  onUploadComplete,
  onUploadError,
  acceptedTypes = ['.pdf', '.docx', '.txt', '.md'],
  maxFileSize = 50 * 1024 * 1024, // 50MB
  maxFiles = 10,
  multiple = true,
}) => {
  const [files, setFiles] = useState<UploadedDocument[]>([]);
  const [uploading, setUploading] = useState(false);

  const uploadFile = async (file: File): Promise<UploadedDocument> => {
    const formData = new FormData();
    formData.append('files', file);

    const uploadDoc: UploadedDocument = {
      id: generateId(),
      filename: file.name,
      size: file.size,
      contentType: file.type,
      uploadProgress: 0,
      status: 'uploading',
    };

    setFiles(prev => [...prev, uploadDoc]);

    try {
      const response = await documentService.upload(formData, {
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );

          setFiles(prev => prev.map(f =>
            f.id === uploadDoc.id
              ? { ...f, uploadProgress: progress }
              : f
          ));
        },
      });

      const successDoc: UploadedDocument = {
        ...uploadDoc,
        id: response.data.uploaded_documents[0].id,
        status: 'success',
        uploadProgress: 100,
      };

      setFiles(prev => prev.map(f =>
        f.id === uploadDoc.id ? successDoc : f
      ));

      return successDoc;
    } catch (error) {
      const errorDoc: UploadedDocument = {
        ...uploadDoc,
        status: 'error',
        error: error.message,
      };

      setFiles(prev => prev.map(f =>
        f.id === uploadDoc.id ? errorDoc : f
      ));

      throw error;
    }
  };

  const handleFilesSelected = async (fileList: FileList) => {
    if (files.length + fileList.length > maxFiles) {
      message.error(`最多只能上传 ${maxFiles} 个文件`);
      return;
    }

    setUploading(true);

    const uploadPromises = Array.from(fileList).map(file => {
      // 文件类型验证
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!acceptedTypes.includes(fileExtension)) {
        message.error(`不支持的文件类型: ${file.name}`);
        return Promise.reject(new Error(`Unsupported file type: ${fileExtension}`));
      }

      // 文件大小验证
      if (file.size > maxFileSize) {
        const sizeMB = Math.round(maxFileSize / (1024 * 1024));
        message.error(`文件 ${file.name} 超过大小限制 (${sizeMB}MB)`);
        return Promise.reject(new Error(`File too large: ${file.name}`));
      }

      return uploadFile(file);
    });

    try {
      const results = await Promise.allSettled(uploadPromises);
      const successful = results
        .filter((result): result is PromiseFulfilledResult<UploadedDocument> =>
          result.status === 'fulfilled'
        )
        .map(result => result.value);

      if (successful.length > 0) {
        onUploadComplete?.(successful);
        message.success(`成功上传 ${successful.length} 个文件`);
      }

      const failed = results.filter(result => result.status === 'rejected');
      if (failed.length > 0) {
        message.error(`${failed.length} 个文件上传失败`);
      }
    } catch (error) {
      onUploadError?.(error as Error);
    } finally {
      setUploading(false);
    }
  };

  const removeFile = (fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const retryUpload = async (fileId: string) => {
    const file = files.find(f => f.id === fileId);
    if (!file) return;

    // 重新创建File对象需要额外的处理，这里简化为重置状态
    setFiles(prev => prev.map(f =>
      f.id === fileId
        ? { ...f, status: 'uploading', uploadProgress: 0, error: undefined }
        : f
    ));

    // 实际应用中需要保存原始File对象引用
  };

  return (
    <div className="document-uploader">
      <Dragger
        name="files"
        multiple={multiple}
        showUploadList={false}
        beforeUpload={() => false} // 阻止默认上传
        onChange={(info) => {
          if (info.fileList.length > 0) {
            const fileList = new DataTransfer();
            info.fileList.forEach(file => {
              if (file.originFileObj) {
                fileList.items.add(file.originFileObj);
              }
            });
            handleFilesSelected(fileList.files);
          }
        }}
        disabled={uploading}
        accept={acceptedTypes.join(',')}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          点击或拖拽文件到此区域上传
        </p>
        <p className="ant-upload-hint">
          支持 {acceptedTypes.join(', ')} 格式，单个文件不超过 {Math.round(maxFileSize / (1024 * 1024))}MB
        </p>
      </Dragger>

      {files.length > 0 && (
        <div className="upload-list">
          <Divider>上传列表</Divider>
          <List
            dataSource={files}
            renderItem={(file) => (
              <List.Item
                actions={[
                  file.status === 'error' && (
                    <Button
                      type="link"
                      icon={<RedoOutlined />}
                      onClick={() => retryUpload(file.id)}
                    >
                      重试
                    </Button>
                  ),
                  <Button
                    type="link"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => removeFile(file.id)}
                  >
                    删除
                  </Button>,
                ].filter(Boolean)}
              >
                <List.Item.Meta
                  avatar={
                    <Avatar
                      icon={
                        file.status === 'success' ? (
                          <CheckOutlined style={{ color: '#52c41a' }} />
                        ) : file.status === 'error' ? (
                          <CloseOutlined style={{ color: '#ff4d4f' }} />
                        ) : (
                          <FileOutlined />
                        )
                      }
                    />
                  }
                  title={file.filename}
                  description={
                    <Space>
                      <span>{formatFileSize(file.size)}</span>
                      {file.status === 'uploading' && (
                        <Progress
                          percent={file.uploadProgress}
                          size="small"
                          status="active"
                        />
                      )}
                      {file.status === 'error' && (
                        <Typography.Text type="danger">
                          {file.error}
                        </Typography.Text>
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </div>
      )}
    </div>
  );
};
```

#### DocumentList 组件

```typescript
// modules/Documents/components/DocumentList.tsx

interface DocumentListProps {
  onDocumentSelect?: (document: Document) => void;
  onDocumentDelete?: (documentIds: string[]) => void;
  selectable?: boolean;
  showActions?: boolean;
}

interface DocumentFilters {
  search: string;
  tags: string[];
  dateRange: [string, string] | null;
  fileType: string[];
}

const DocumentList: React.FC<DocumentListProps> = ({
  onDocumentSelect,
  onDocumentDelete,
  selectable = false,
  showActions = true,
}) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 20,
    total: 0,
  });
  const [filters, setFilters] = useState<DocumentFilters>({
    search: '',
    tags: [],
    dateRange: null,
    fileType: [],
  });

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    try {
      const response = await documentService.list({
        page: pagination.current,
        page_size: pagination.pageSize,
        search_query: filters.search || undefined,
        tag_filter: filters.tags.length > 0 ? filters.tags : undefined,
      });

      setDocuments(response.data.documents);
      setPagination(prev => ({
        ...prev,
        total: response.data.pagination.total_count,
      }));
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    } finally {
      setLoading(false);
    }
  }, [pagination.current, pagination.pageSize, filters]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleSearch = useDebouncedCallback((value: string) => {
    setFilters(prev => ({ ...prev, search: value }));
    setPagination(prev => ({ ...prev, current: 1 }));
  }, 500);

  const handleDelete = async (documentIds: string[]) => {
    try {
      await documentService.delete({ document_ids: documentIds });
      await fetchDocuments();
      setSelectedRowKeys([]);
      onDocumentDelete?.(documentIds);
      message.success(`成功删除 ${documentIds.length} 个文档`);
    } catch (error) {
      console.error('Failed to delete documents:', error);
      message.error('删除文档失败');
    }
  };

  const columns: ColumnsType<Document> = [
    {
      title: '文档名称',
      dataIndex: 'filename',
      key: 'filename',
      render: (filename: string, record: Document) => (
        <Space>
          <FileIcon type={getFileType(filename)} />
          <Button
            type="link"
            onClick={() => onDocumentSelect?.(record)}
            style={{ padding: 0 }}
          >
            {filename}
          </Button>
        </Space>
      ),
      sorter: true,
    },
    {
      title: '文件大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatFileSize(size),
      sorter: true,
      width: 120,
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <Space wrap>
          {tags.map(tag => (
            <Tag key={tag} color="blue">
              {tag}
            </Tag>
          ))}
        </Space>
      ),
      filterDropdown: ({ setSelectedKeys, selectedKeys, confirm, clearFilters }) => (
        <TagFilter
          value={selectedKeys as string[]}
          onChange={(tags) => setSelectedKeys(tags)}
          onConfirm={confirm}
          onClear={clearFilters}
        />
      ),
    },
    {
      title: '上传时间',
      dataIndex: 'uploaded_at',
      key: 'uploaded_at',
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm'),
      sorter: true,
      width: 180,
    },
    ...(showActions ? [{
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record: Document) => (
        <Space>
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => onDocumentSelect?.(record)}
            title="查看"
          />
          <Button
            type="text"
            icon={<DownloadOutlined />}
            onClick={() => downloadDocument(record.id)}
            title="下载"
          />
          <Popconfirm
            title="确定要删除这个文档吗？"
            onConfirm={() => handleDelete([record.id])}
            okText="确定"
            cancelText="取消"
          >
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              title="删除"
            />
          </Popconfirm>
        </Space>
      ),
    }] : []),
  ];

  const rowSelection = selectable ? {
    selectedRowKeys,
    onChange: setSelectedRowKeys,
    onSelectAll: (selected: boolean, selectedRows: Document[], changeRows: Document[]) => {
      console.log('Select all:', selected, selectedRows, changeRows);
    },
  } : undefined;

  return (
    <div className="document-list">
      <Card>
        <div className="document-list-header">
          <Row gutter={[16, 16]} align="middle">
            <Col flex="auto">
              <Input.Search
                placeholder="搜索文档..."
                allowClear
                onSearch={handleSearch}
                style={{ maxWidth: 400 }}
              />
            </Col>
            <Col>
              <Space>
                <FilterDropdown
                  filters={filters}
                  onChange={setFilters}
                />
                <Button
                  icon={<ReloadOutlined />}
                  onClick={fetchDocuments}
                >
                  刷新
                </Button>
                {selectedRowKeys.length > 0 && (
                  <Popconfirm
                    title={`确定要删除选中的 ${selectedRowKeys.length} 个文档吗？`}
                    onConfirm={() => handleDelete(selectedRowKeys)}
                    okText="确定"
                    cancelText="取消"
                  >
                    <Button danger>
                      批量删除 ({selectedRowKeys.length})
                    </Button>
                  </Popconfirm>
                )}
              </Space>
            </Col>
          </Row>
        </div>

        <Table
          columns={columns}
          dataSource={documents}
          rowKey="id"
          rowSelection={rowSelection}
          loading={loading}
          pagination={{
            current: pagination.current,
            pageSize: pagination.pageSize,
            total: pagination.total,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
            onChange: (page, pageSize) => {
              setPagination(prev => ({
                ...prev,
                current: page,
                pageSize: pageSize || prev.pageSize,
              }));
            },
          }}
        />
      </Card>
    </div>
  );
};
```

### 3. 知识图谱可视化组件

#### GraphVisualizer 组件

```typescript
// modules/KnowledgeGraph/components/GraphVisualizer.tsx

interface GraphVisualizerProps {
  data?: GraphData;
  options?: GraphVisualizationOptions;
  eventHandlers?: GraphEventHandlers;
  height?: number;
  className?: string;
  style?: React.CSSProperties;
  showDataTable?: boolean;
}

interface EChartsGraphData {
  nodes: Array<{
    id: string;
    name: string;
    category: number;
    symbolSize: number;
    value: number;
    entityType?: string;
    confidence?: number;
    properties?: any;
  }>;
  links: Array<{
    source: string;
    target: string;
    relationshipType?: string;
    confidence?: number;
    properties?: any;
  }>;
  categories: Array<{
    name: string;
    itemStyle?: {
      color?: string;
    };
  }>;
}

const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  data,
  options = {},
  eventHandlers = {},
  height = 600,
  className,
  style,
  showDataTable = true,
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const [graphData, setGraphData] = useState<EChartsGraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [selectedEdge, setSelectedEdge] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<string>('nodes');

  // Store hooks
  const {
    entities,
    relations,
    currentLayout,
    showNodeLabels,
    edgeWidth,
    getFilteredEntities,
    getFilteredRelations,
  } = useKnowledgeGraphStore();

  const { theme } = useAppStore();

  // Helper function to get node name by ID
  const getNodeNameById = useCallback((nodeId: string) => {
    const node = graphData?.nodes?.find(n => n.id === nodeId);
    return node?.name || nodeId;
  }, [graphData]);

  // Initialize ECharts
  const initializeChart = useCallback(() => {
    if (!chartRef.current || chartInstance.current) return;

    try {
      chartInstance.current = echarts.init(chartRef.current);

      // Setup event handlers
      chartInstance.current.on('click', function (params: any) {
        if (params.dataType === 'node') {
          const node = params.data;
          setSelectedNode(node);
          setSelectedEdge(null);
          setActiveTab('nodes');
          eventHandlers.onNodeSelect?.(node);
        } else if (params.dataType === 'edge') {
          const edge = params.data;
          setSelectedEdge(edge);
          setSelectedNode(null);
          setActiveTab('edges');
          eventHandlers.onEdgeSelect?.(edge);
        }
      });

      // Resize chart when window resizes
      const handleResize = () => {
        if (chartInstance.current) {
          chartInstance.current.resize();
        }
      };
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
      };
    } catch (err) {
      console.error('Failed to initialize ECharts:', err);
    }
  }, [eventHandlers]);

  // Convert data to ECharts format
  const convertToEChartsData = useCallback(() => {
    const entitiesToUse = data ? data.nodes.map(node => ({
      id: node.id,
      name: node.label,
      entity_type: node.entityType || 'concept',
      confidence: node.confidence || 1.0,
      properties: node.properties || {},
      description: '',
      aliases: [],
    })) : getFilteredEntities();

    const relationsToUse = data ? data.edges.map(edge => ({
      id: edge.id,
      head_entity_id: edge.source,
      tail_entity_id: edge.target,
      relation_type: edge.relationType || 'related',
      confidence: edge.confidence || 1.0,
      properties: edge.properties || {},
      description: '',
    })) : getFilteredRelations();

    if (entitiesToUse.length === 0) {
      return null;
    }

    // Get unique entity types for categories
    const entityTypes = [...new Set(entitiesToUse.map(e => e.entity_type))];
    const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc'];

    const categories = entityTypes.map((type, index) => ({
      name: type,
      itemStyle: { color: colors[index % colors.length] }
    }));

    // Create nodes
    const nodes = entitiesToUse.map(entity => {
      const categoryIndex = entityTypes.indexOf(entity.entity_type);
      return {
        id: entity.id,
        name: entity.name,
        category: categoryIndex,
        symbolSize: Math.min(Math.max((entity.confidence || 0.5) * 30, 15), 50),
        value: entity.confidence || 0.5,
        entityType: entity.entity_type,
        confidence: entity.confidence,
        properties: entity.properties,
      };
    });

    // Create links
    const links = relationsToUse.map(relation => ({
      source: relation.head_entity_id,
      target: relation.tail_entity_id,
      relationshipType: relation.relation_type,
      confidence: relation.confidence,
      properties: relation.properties,
    }));

    return { nodes, links, categories };
  }, [data, getFilteredEntities, getFilteredRelations]);

  // Render graph
  const renderGraph = useCallback(() => {
    if (!chartInstance.current || !graphData) return;

    const option = {
      title: {
        text: '知识图谱',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
          color: theme === 'dark' ? '#ffffff' : '#1e293b',
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params: any) {
          if (params.dataType === 'node') {
            return `<strong>${params.data.name}</strong><br/>
                    类别: ${graphData.categories[params.data.category]?.name || 'Unknown'}<br/>
                    置信度: ${(params.data.confidence || 0).toFixed(2)}`;
          } else if (params.dataType === 'edge') {
            return `${getNodeNameById(params.data.source)} → ${getNodeNameById(params.data.target)}<br/>
                    关系类型: ${params.data.relationshipType || '关系'}<br/>
                    置信度: ${(params.data.confidence || 0).toFixed(2)}`;
          }
          return params.name;
        },
      },
      legend: {
        data: graphData.categories.map(cat => cat.name),
        top: 'bottom',
        left: 'center',
        textStyle: {
          color: theme === 'dark' ? '#ffffff' : '#374151',
        },
      },
      series: [
        {
          name: '知识图谱',
          type: 'graph',
          layout: 'force',
          data: graphData.nodes,
          links: graphData.links,
          categories: graphData.categories,
          roam: true,
          focusNodeAdjacency: true,
          force: {
            repulsion: 200,
            gravity: 0.05,
            edgeLength: 80,
            layoutAnimation: true,
          },
          label: {
            show: showNodeLabels !== false,
            position: 'right',
            formatter: '{b}',
            color: theme === 'dark' ? '#ffffff' : '#374151',
          },
          labelLayout: {
            hideOverlap: true,
          },
          lineStyle: {
            color: 'source',
            curveness: 0.3,
            opacity: 0.7,
            width: edgeWidth || 2,
          },
          emphasis: {
            focus: 'adjacency',
            lineStyle: {
              width: 4,
            },
          },
        },
      ],
      backgroundColor: theme === 'dark' ? '#1f1f1f' : '#ffffff',
      animation: true,
    };

    try {
      chartInstance.current.clear();
      chartInstance.current.setOption(option, true);

      setTimeout(() => {
        if (chartInstance.current) {
          chartInstance.current.resize();
        }
      }, 200);
    } catch (error) {
      console.error('Error setting ECharts option:', error);
    }
  }, [graphData, theme, showNodeLabels, edgeWidth, getNodeNameById]);

  // Effects
  useEffect(() => {
    const timer = setTimeout(() => {
      if (chartRef.current && !chartInstance.current) {
        const cleanup = initializeChart();
        return cleanup;
      }
    }, 100);

    return () => {
      clearTimeout(timer);
      if (chartInstance.current) {
        try {
          chartInstance.current.dispose();
        } catch (err) {
          console.warn('Error during ECharts cleanup:', err);
        } finally {
          chartInstance.current = null;
        }
      }
    };
  }, [initializeChart]);

  // Update graph when data changes
  useEffect(() => {
    const newGraphData = convertToEChartsData();
    setGraphData(newGraphData);
  }, [entities, relations, data, convertToEChartsData]);

  // Render graph when chart is ready and data is available
  useEffect(() => {
    if (chartInstance.current && graphData) {
      setTimeout(() => {
        renderGraph();
      }, 100);
    }
  }, [graphData, renderGraph]);

  // Data table columns
  const nodeColumns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
      width: '45%',
    },
    {
      title: '类别',
      dataIndex: 'entityType',
      key: 'entityType',
      width: '35%',
      render: (type: string) => (
        <Tag color="blue" size="small">
          {type}
        </Tag>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: '20%',
      align: 'center' as const,
      render: (confidence: number) => (
        <span style={{ fontSize: '12px' }}>
          {(confidence || 0).toFixed(2)}
        </span>
      ),
    },
  ];

  const edgeColumns = [
    {
      title: '源节点',
      dataIndex: 'source',
      key: 'source',
      ellipsis: true,
      width: '35%',
      render: (sourceId: string) => getNodeNameById(sourceId),
    },
    {
      title: '目标节点',
      dataIndex: 'target',
      key: 'target',
      ellipsis: true,
      width: '35%',
      render: (targetId: string) => getNodeNameById(targetId),
    },
    {
      title: '关系',
      dataIndex: 'relationshipType',
      key: 'relationshipType',
      width: '30%',
      ellipsis: true,
      render: (type: string) => (
        <Tag size="small" color="blue">
          {type || '关系'}
        </Tag>
      ),
    },
  ];

  if (!showDataTable) {
    // Render only the graph
    return (
      <Card
        className={className}
        style={style}
        styles={{ body: { padding: 0 } }}
      >
        <div
          ref={chartRef}
          style={{
            width: '100%',
            height,
            backgroundColor: theme === 'dark' ? '#1f1f1f' : '#ffffff',
          }}
        />
      </Card>
    );
  }

  // 右侧数据面板的标签项
  const dataTabItems = [
    {
      key: 'nodes',
      label: `节点 (${graphData?.nodes?.length || 0})`,
      children: (
        <div style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <Table
            columns={nodeColumns}
            dataSource={graphData?.nodes || []}
            rowKey="id"
            pagination={false}
            size="small"
            scroll={{
              y: height - 180,
              scrollToFirstRowOnChange: false
            }}
            onRow={(record) => ({
              onClick: () => {
                setSelectedNode(record);
                setSelectedEdge(null);
              },
              style: {
                backgroundColor: selectedNode?.id === record.id ? '#e6f7ff' : undefined,
                cursor: 'pointer',
              },
            })}
          />
        </div>
      ),
    },
    {
      key: 'edges',
      label: `关系 (${graphData?.links?.length || 0})`,
      children: (
        <div style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <Table
            columns={edgeColumns}
            dataSource={graphData?.links || []}
            rowKey={(record, index) => `${record.source}-${record.target}-${index}`}
            pagination={false}
            size="small"
            scroll={{
              y: height - 180,
              scrollToFirstRowOnChange: false
            }}
            onRow={(record) => ({
              onClick: () => {
                setSelectedEdge(record);
                setSelectedNode(null);
              },
              style: {
                backgroundColor:
                  selectedEdge?.source === record.source &&
                  selectedEdge?.target === record.target
                    ? '#e6f7ff' : undefined,
                cursor: 'pointer',
              },
            })}
          />
        </div>
      ),
    },
  ];

  // 主布局：左侧图谱 + 右侧数据面板
  return (
    <Card
      className={className}
      style={style}
      styles={{ body: { padding: 0 } }}
    >
      <div style={{ display: 'flex', height }}>
        {/* 左侧知识图谱区域 - 2/3宽度 */}
        <div style={{ flex: 2, position: 'relative', borderRight: '1px solid #f0f0f0' }}>
          <div style={{ padding: '16px 16px 0 16px' }}>
            <div style={{ marginBottom: 16, textAlign: 'center' }}>
              <Text strong style={{ fontSize: 16 }}>知识图谱</Text>
            </div>
            {graphData && (
              <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'center', gap: 24 }}>
                <span>节点: <Text strong>{graphData.nodes?.length || 0}</Text></span>
                <span>关系: <Text strong>{graphData.links?.length || 0}</Text></span>
                <span>类别: <Text strong>{graphData.categories?.length || 0}</Text></span>
              </div>
            )}
          </div>
          <div
            ref={chartRef}
            style={{
              width: '100%',
              height: 'calc(100% - 80px)',
              minHeight: '400px',
              backgroundColor: theme === 'dark' ? '#1f1f1f' : '#ffffff',
            }}
          />

          {/* 选中项显示 */}
          {(selectedNode || selectedEdge) && (
            <div style={{
              position: 'absolute',
              bottom: 16,
              left: 16,
              right: 16,
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              padding: 12,
              borderRadius: 6,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              backdropFilter: 'blur(4px)'
            }}>
              {selectedNode && (
                <div>
                  <Text strong>选中节点: </Text>
                  <Text>{selectedNode.name}</Text>
                  <Text type="secondary" style={{ marginLeft: 12 }}>
                    类别: {selectedNode.entityType} | 置信度: {(selectedNode.confidence || 0).toFixed(2)}
                  </Text>
                </div>
              )}
              {selectedEdge && (
                <div>
                  <Text strong>选中关系: </Text>
                  <Text>{getNodeNameById(selectedEdge.source)} → {getNodeNameById(selectedEdge.target)}</Text>
                  <Text type="secondary" style={{ marginLeft: 12 }}>
                    类型: {selectedEdge.relationshipType || '关系'} | 置信度: {(selectedEdge.confidence || 0).toFixed(2)}
                  </Text>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 右侧数据面板 - 1/3宽度 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div style={{
            padding: '16px 16px 12px 16px',
            textAlign: 'center',
            borderBottom: '1px solid #f0f0f0',
            flexShrink: 0
          }}>
            <Text strong style={{ fontSize: 16 }}>数据视图</Text>
          </div>
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <Tabs
              activeKey={activeTab}
              onChange={setActiveTab}
              items={dataTabItems}
              style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
              tabPosition="top"
              size="small"
              tabBarStyle={{
                padding: '0 16px',
                margin: 0,
                flexShrink: 0
              }}
            />
          </div>
        </div>
      </div>
    </Card>
  );
};
```

### 4. 对话系统组件

#### ChatInterface 组件

```typescript
// modules/Chat/components/ChatInterface.tsx

interface ChatInterfaceProps {
  conversationId?: string;
  onConversationChange?: (conversationId: string) => void;
  height?: number;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  context?: ChatContext;
  streaming?: boolean;
}

interface ChatContext {
  retrievedEntities: Entity[];
  retrievedRelations: Relation[];
  retrievedTextChunks: TextChunk[];
  searchQueries: string[];
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  conversationId,
  onConversationChange,
  height = 600,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [currentContext, setCurrentContext] = useState<ChatContext | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<Input.TextArea>(null);

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 发送消息
  const handleSendMessage = async () => {
    if (!inputValue.trim() || streaming) return;

    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setStreaming(true);

    try {
      // 创建助手消息占位符
      const assistantMessageId = generateId();
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        streaming: true,
      };

      setMessages(prev => [...prev, assistantMessage]);

      // 流式响应
      const response = await chatService.streamChat({
        question: userMessage.content,
        conversation_history: messages.map(m => ({
          role: m.role,
          content: m.content,
        })),
        stream: true,
      });

      let partialContent = '';

      for await (const chunk of response) {
        if (chunk.chunk) {
          partialContent += chunk.chunk;

          setMessages(prev => prev.map(msg =>
            msg.id === assistantMessageId
              ? { ...msg, content: partialContent }
              : msg
          ));
        }

        if (chunk.context) {
          setCurrentContext(chunk.context);
        }

        if (chunk.finished) {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantMessageId
              ? {
                  ...msg,
                  content: chunk.answer || partialContent,
                  streaming: false,
                  context: chunk.context,
                }
              : msg
          ));
          break;
        }
      }
    } catch (error) {
      console.error('Chat error:', error);

      const errorMessage: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content: '抱歉，我现在无法回答您的问题。请稍后再试。',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setStreaming(false);
    }
  };

  // 处理键盘事件
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 清空对话
  const handleClearConversation = () => {
    Modal.confirm({
      title: '确定要清空对话吗？',
      content: '此操作不可撤销',
      onOk: () => {
        setMessages([]);
        setCurrentContext(null);
      },
    });
  };

  // 重新生成回答
  const handleRegenerateResponse = async (messageIndex: number) => {
    const userMessage = messages[messageIndex - 1];
    if (!userMessage || userMessage.role !== 'user') return;

    // 移除当前助手回答
    const newMessages = messages.slice(0, messageIndex);
    setMessages(newMessages);

    // 重新发送请求
    setInputValue(userMessage.content);
    await handleSendMessage();
  };

  return (
    <div className="chat-interface" style={{ height }}>
      <div className="chat-header">
        <Row justify="space-between" align="middle">
          <Col>
            <Typography.Title level={4} style={{ margin: 0 }}>
              知识问答
            </Typography.Title>
          </Col>
          <Col>
            <Space>
              <Button
                icon={<ClearOutlined />}
                onClick={handleClearConversation}
                disabled={messages.length === 0}
              >
                清空对话
              </Button>
              <Button
                icon={<SettingOutlined />}
                onClick={() => {/* 打开设置 */}}
              >
                设置
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      <div className="chat-messages" style={{ height: height - 120, overflow: 'auto' }}>
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="开始对话吧！我可以帮您查询知识库中的信息。"
            />
          </div>
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              onRegenerate={
                message.role === 'assistant' && !message.streaming
                  ? () => handleRegenerateResponse(index)
                  : undefined
              }
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <Row gutter={8} align="bottom">
          <Col flex="auto">
            <Input.TextArea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onPressEnter={handleKeyPress}
              placeholder="输入您的问题... (Shift+Enter 换行，Enter 发送)"
              autoSize={{ minRows: 1, maxRows: 4 }}
              disabled={streaming}
            />
          </Col>
          <Col>
            <Button
              type="primary"
              icon={streaming ? <LoadingOutlined /> : <SendOutlined />}
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || streaming}
              loading={streaming}
            >
              发送
            </Button>
          </Col>
        </Row>
      </div>

      {currentContext && (
        <ContextPanel
          context={currentContext}
          style={{ position: 'absolute', right: 0, top: 0, width: 300 }}
        />
      )}
    </div>
  );
};
```

## 通用工具组件

### LoadingSpinner 组件

```typescript
// components/Common/LoadingSpinner.tsx

interface LoadingSpinnerProps {
  size?: 'small' | 'default' | 'large';
  tip?: string;
  spinning?: boolean;
  children?: React.ReactNode;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'default',
  tip = '加载中...',
  spinning = true,
  children,
}) => {
  if (children) {
    return (
      <Spin size={size} tip={tip} spinning={spinning}>
        {children}
      </Spin>
    );
  }

  return (
    <div className="loading-spinner">
      <Spin size={size} tip={tip} />
    </div>
  );
};
```

### ErrorBoundary 组件

```typescript
// components/Common/ErrorBoundary.tsx

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends React.Component<
  React.PropsWithChildren<{}>,
  ErrorBoundaryState
> {
  constructor(props: React.PropsWithChildren<{}>) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    });

    // 记录错误到日志服务
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Result
          status="error"
          title="页面出现错误"
          subTitle="抱歉，页面遇到了一些问题。您可以尝试刷新页面或返回上一页。"
          extra={[
            <Button type="primary" key="reload" onClick={this.handleReload}>
              刷新页面
            </Button>,
            <Button key="reset" onClick={this.handleReset}>
              重试
            </Button>,
          ]}
        >
          {process.env.NODE_ENV === 'development' && (
            <details style={{ whiteSpace: 'pre-wrap', marginTop: 16 }}>
              <summary>错误详情 (开发模式)</summary>
              {this.state.error && this.state.error.toString()}
              <br />
              {this.state.errorInfo.componentStack}
            </details>
          )}
        </Result>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

## 组件样式规范

### CSS Modules 规范

```scss
// components/ProjectCard/ProjectCard.module.scss

.projectCard {
  .selected {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
  }

  .projectHeader {
    display: flex;
    justify-content: space-between;
    align-items: center;

    .projectName {
      margin: 0;
      color: var(--text-color);
    }
  }

  .projectContent {
    .projectDescription {
      color: var(--text-color-secondary);
      margin-bottom: 16px;
    }

    .projectStats {
      margin-bottom: 16px;
    }

    .projectFooter {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 12px;
      border-top: 1px solid var(--border-color);

      .projectDate {
        font-size: 12px;
      }
    }
  }
}
```

### 主题变量

```scss
// styles/variables.scss

:root {
  // 颜色系统
  --primary-color: #1890ff;
  --success-color: #52c41a;
  --warning-color: #faad14;
  --error-color: #ff4d4f;

  // 文本颜色
  --text-color: rgba(0, 0, 0, 0.85);
  --text-color-secondary: rgba(0, 0, 0, 0.65);
  --text-color-disabled: rgba(0, 0, 0, 0.25);

  // 背景颜色
  --background-color: #fff;
  --background-color-light: #fafafa;
  --background-color-dark: #f5f5f5;

  // 边框颜色
  --border-color: #d9d9d9;
  --border-color-light: #f0f0f0;

  // 阴影
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.03);
  --shadow: 0 1px 8px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 4px 16px rgba(0, 0, 0, 0.12);

  // 圆角
  --border-radius: 6px;
  --border-radius-sm: 4px;
  --border-radius-lg: 8px;

  // 间距
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
}

// 暗色主题
[data-theme='dark'] {
  --text-color: rgba(255, 255, 255, 0.85);
  --text-color-secondary: rgba(255, 255, 255, 0.65);
  --text-color-disabled: rgba(255, 255, 255, 0.25);

  --background-color: #141414;
  --background-color-light: #1f1f1f;
  --background-color-dark: #0f0f0f;

  --border-color: #434343;
  --border-color-light: #303030;
}
```

## 性能优化最佳实践

### 组件懒加载

```typescript
// utils/componentLoader.ts

import { lazy, ComponentType } from 'react';
import { LoadingSpinner } from '../components/Common';

export function lazyLoadComponent<T extends ComponentType<any>>(
  importFunc: () => Promise<{ default: T }>,
  fallback: ComponentType = LoadingSpinner
) {
  const LazyComponent = lazy(importFunc);

  return (props: React.ComponentProps<T>) => (
    <Suspense fallback={<fallback />}>
      <LazyComponent {...props} />
    </Suspense>
  );
}

// 使用示例
export const LazyProjectList = lazyLoadComponent(
  () => import('../modules/Projects/components/ProjectList')
);

export const LazyGraphVisualizer = lazyLoadComponent(
  () => import('../modules/KnowledgeGraph/components/GraphVisualizer')
);
```

### React.memo 优化

```typescript
// 使用React.memo优化组件重渲染
const ProjectCard = React.memo<ProjectCardProps>(({ project, onSelect }) => {
  // 组件实现
}, (prevProps, nextProps) => {
  // 自定义比较逻辑
  return (
    prevProps.project.id === nextProps.project.id &&
    prevProps.project.name === nextProps.project.name &&
    prevProps.project.documentCount === nextProps.project.documentCount &&
    prevProps.selected === nextProps.selected
  );
});
```

### useMemo 和 useCallback 优化

```typescript
const DocumentList: React.FC<DocumentListProps> = ({ onDocumentSelect }) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [filters, setFilters] = useState<DocumentFilters>({});

  // 缓存过滤后的文档列表
  const filteredDocuments = useMemo(() => {
    return documents.filter(doc => {
      if (filters.search && !doc.filename.includes(filters.search)) {
        return false;
      }
      if (filters.tags.length > 0 && !filters.tags.some(tag => doc.tags.includes(tag))) {
        return false;
      }
      return true;
    });
  }, [documents, filters.search, filters.tags]);

  // 缓存事件处理函数
  const handleDocumentClick = useCallback((document: Document) => {
    onDocumentSelect?.(document);
  }, [onDocumentSelect]);

  return (
    <div>
      {filteredDocuments.map(doc => (
        <DocumentCard
          key={doc.id}
          document={doc}
          onClick={handleDocumentClick}
        />
      ))}
    </div>
  );
};
```

这份组件规格说明文档详细描述了AGraph前端的核心组件设计，包括组件接口、实现细节、样式规范和性能优化策略，为前端开发提供了完整的技术指南。
