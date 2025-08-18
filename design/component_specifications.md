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
  entities: Entity[];
  relations: Relation[];
  width?: number;
  height?: number;
  onNodeClick?: (entity: Entity) => void;
  onEdgeClick?: (relation: Relation) => void;
  onSelectionChange?: (selection: GraphSelection) => void;
  layout?: LayoutOptions;
  theme?: GraphTheme;
}

interface GraphSelection {
  nodes: string[];
  edges: string[];
}

interface LayoutOptions {
  name: 'cola' | 'cose' | 'grid' | 'circle' | 'concentric';
  animate?: boolean;
  fit?: boolean;
  padding?: number;
}

interface GraphTheme {
  nodeColors: Record<string, string>;
  edgeColors: Record<string, string>;
  backgroundColor: string;
  textColor: string;
}

const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  entities,
  relations,
  width = 800,
  height = 600,
  onNodeClick,
  onEdgeClick,
  onSelectionChange,
  layout = { name: 'cose', animate: true, fit: true },
  theme = defaultGraphTheme,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core>();
  const [selectedElements, setSelectedElements] = useState<GraphSelection>({
    nodes: [],
    edges: [],
  });
  
  // 初始化Cytoscape实例
  useEffect(() => {
    if (!containerRef.current) return;
    
    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) => {
              const entityType = ele.data('entity_type');
              return theme.nodeColors[entityType] || '#666';
            },
            'label': 'data(name)',
            'text-valign': 'center',
            'text-halign': 'center',
            'color': theme.textColor,
            'font-size': '12px',
            'width': (ele) => Math.max(30, ele.data('name').length * 8),
            'height': (ele) => Math.max(30, ele.data('name').length * 8),
            'border-width': 2,
            'border-color': '#fff',
            'text-wrap': 'wrap',
            'text-max-width': '100px',
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#007bff',
            'border-width': 3,
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': (ele) => {
              const relationType = ele.data('relation_type');
              return theme.edgeColors[relationType] || '#ccc';
            },
            'target-arrow-color': (ele) => {
              const relationType = ele.data('relation_type');
              return theme.edgeColors[relationType] || '#ccc';
            },
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(relation_type)',
            'font-size': '10px',
            'color': theme.textColor,
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
          },
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#007bff',
            'target-arrow-color': '#007bff',
            'width': 3,
          },
        },
      ],
      layout,
      minZoom: 0.1,
      maxZoom: 3,
      wheelSensitivity: 0.2,
    });
    
    cyRef.current = cy;
    
    // 添加事件监听器
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const entityId = node.data('id');
      const entity = entities.find(e => e.id === entityId);
      if (entity && onNodeClick) {
        onNodeClick(entity);
      }
    });
    
    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target;
      const relationId = edge.data('id');
      const relation = relations.find(r => r.id === relationId);
      if (relation && onEdgeClick) {
        onEdgeClick(relation);
      }
    });
    
    cy.on('select unselect', () => {
      const selectedNodes = cy.$('node:selected').map(node => node.data('id'));
      const selectedEdges = cy.$('edge:selected').map(edge => edge.data('id'));
      
      const selection = {
        nodes: selectedNodes,
        edges: selectedEdges,
      };
      
      setSelectedElements(selection);
      onSelectionChange?.(selection);
    });
    
    return () => {
      cy.destroy();
    };
  }, []);
  
  // 更新图数据
  useEffect(() => {
    if (!cyRef.current) return;
    
    const cy = cyRef.current;
    
    // 转换实体为节点
    const nodes = entities.map(entity => ({
      data: {
        id: entity.id,
        name: entity.name,
        entity_type: entity.entity_type,
        description: entity.description,
        confidence: entity.confidence,
      },
    }));
    
    // 转换关系为边
    const edges = relations.map(relation => ({
      data: {
        id: relation.id,
        source: relation.head_entity_id,
        target: relation.tail_entity_id,
        relation_type: relation.relation_type,
        description: relation.description,
        confidence: relation.confidence,
      },
    }));
    
    // 更新图数据
    cy.json({ elements: { nodes, edges } });
    
    // 重新应用布局
    cy.layout(layout).run();
  }, [entities, relations, layout]);
  
  // 工具栏功能
  const handleZoomIn = () => {
    cyRef.current?.zoom(cyRef.current.zoom() * 1.25);
  };
  
  const handleZoomOut = () => {
    cyRef.current?.zoom(cyRef.current.zoom() * 0.8);
  };
  
  const handleFit = () => {
    cyRef.current?.fit();
  };
  
  const handleCenter = () => {
    cyRef.current?.center();
  };
  
  const handleExportPNG = () => {
    if (!cyRef.current) return;
    
    const png = cyRef.current.png({
      output: 'blob',
      bg: theme.backgroundColor,
      full: true,
    });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(png);
    link.download = 'knowledge-graph.png';
    link.click();
  };
  
  const handleLayoutChange = (layoutName: string) => {
    if (!cyRef.current) return;
    
    const newLayout = { ...layout, name: layoutName as any };
    cyRef.current.layout(newLayout).run();
  };
  
  return (
    <div className="graph-visualizer">
      <div className="graph-toolbar">
        <Space>
          <Button.Group>
            <Button icon={<ZoomInOutlined />} onClick={handleZoomIn} title="放大" />
            <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut} title="缩小" />
            <Button icon={<BorderOutlined />} onClick={handleFit} title="适应窗口" />
            <Button icon={<AimOutlined />} onClick={handleCenter} title="居中" />
          </Button.Group>
          
          <Select
            defaultValue={layout.name}
            style={{ width: 120 }}
            onChange={handleLayoutChange}
          >
            <Select.Option value="cose">力导向</Select.Option>
            <Select.Option value="cola">约束布局</Select.Option>
            <Select.Option value="grid">网格布局</Select.Option>
            <Select.Option value="circle">环形布局</Select.Option>
            <Select.Option value="concentric">同心圆</Select.Option>
          </Select>
          
          <Button icon={<DownloadOutlined />} onClick={handleExportPNG}>
            导出图片
          </Button>
        </Space>
      </div>
      
      <div
        ref={containerRef}
        className="cytoscape-container"
        style={{
          width,
          height,
          backgroundColor: theme.backgroundColor,
          border: '1px solid #d9d9d9',
          borderRadius: '6px',
        }}
      />
      
      {selectedElements.nodes.length > 0 || selectedElements.edges.length > 0 ? (
        <div className="selection-info">
          <Typography.Text>
            已选择: {selectedElements.nodes.length} 个节点, {selectedElements.edges.length} 条边
          </Typography.Text>
        </div>
      ) : null}
    </div>
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