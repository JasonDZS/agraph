import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  Input,
  Button,
  Space,
  Pagination,
  Select,
  Tag,
  Typography,
  Row,
  Col,
  Checkbox,
  Dropdown,
  Modal,
  message,
  Empty,
  Spin,
  Statistic,
} from 'antd';
import {
  SearchOutlined,
  FilterOutlined,
  DeleteOutlined,
  DownloadOutlined,
  ReloadOutlined,
  EyeOutlined,
  SettingOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import DocumentCard from './DocumentCard';
import DocumentViewer from './DocumentViewer';
import { useDocumentStore } from '../../../store/documentStore';
import { documentService } from '../../../services';
import type { Document, DocumentFilterState } from '../types';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;

interface DocumentListProps {
  projectName?: string;
  height?: number;
}

const DocumentList: React.FC<DocumentListProps> = ({
  projectName,
  height = 600,
}) => {
  const {
    documents,
    documentsLoading: loading,
    selectedDocuments,
    currentPage,
    pageSize,
    totalDocuments,
    setDocuments,
    setDocumentsLoading: setLoading,
    setCurrentPage,
    setPageSize,
    setTotalDocuments,
    toggleDocumentSelection,
    selectAllDocuments,
    clearSelection: clearSelectedDocuments,
  } = useDocumentStore();

  const [filterState, setFilterState] = useState<DocumentFilterState>({
    searchQuery: '',
    selectedTags: [],
    selectedDocuments: [],
    sortBy: 'stored_at',
    sortOrder: 'desc',
  });

  const [viewerDocument, setViewerDocument] = useState<Document | null>(null);
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [stats, setStats] = useState<any>(null);

  // Load documents
  const loadDocuments = async (page: number = 1) => {
    setLoading(true);
    try {
      const response = await documentService.listDocuments({
        page,
        page_size: pageSize,
        search_query: filterState.searchQuery || undefined,
        tag_filter:
          filterState.selectedTags.length > 0
            ? filterState.selectedTags
            : undefined,
        project_name: projectName,
      });

      setDocuments(response.data.documents);
      setCurrentPage(response.data.pagination.page);
      setPageSize(response.data.pagination.page_size);
      setTotalDocuments(response.data.pagination.total_count);
    } catch (error) {
      console.error('Failed to load documents:', error);
      message.error('加载文档列表失败');
    } finally {
      setLoading(false);
    }
  };

  // Load stats
  const loadStats = async () => {
    try {
      const response = await documentService.getStats(projectName);
      setStats(response.data);

      // Extract available tags
      const tags = Object.keys(response.data.tags || {});
      setAvailableTags(tags);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  useEffect(() => {
    loadDocuments(1); // Reset to first page when project changes
    loadStats();
  }, [projectName, filterState.searchQuery, filterState.selectedTags]);

  // Sorted documents
  const sortedDocuments = useMemo(() => {
    const sorted = [...documents].sort((a, b) => {
      let aValue: any, bValue: any;

      switch (filterState.sortBy) {
        case 'stored_at':
          aValue = a.created_at || a.updated_at || '';
          bValue = b.created_at || b.updated_at || '';
          break;
        case 'filename':
          aValue = a.title || a.source || '';
          bValue = b.title || b.source || '';
          break;
        case 'content_length':
          aValue = a.file_size || a.content?.length || 0;
          bValue = b.file_size || b.content?.length || 0;
          break;
        default:
          return 0;
      }

      if (filterState.sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
    return sorted;
  }, [documents, filterState.sortBy, filterState.sortOrder]);

  // Handle search
  const handleSearch = (value: string) => {
    setFilterState(prev => ({ ...prev, searchQuery: value }));
  };

  // Handle tag filter
  const handleTagFilter = (tags: string[]) => {
    setFilterState(prev => ({ ...prev, selectedTags: tags }));
  };

  // Handle sort
  const handleSort = (
    sortBy: DocumentFilterState['sortBy'],
    sortOrder: DocumentFilterState['sortOrder']
  ) => {
    setFilterState(prev => ({ ...prev, sortBy, sortOrder }));
  };

  // Handle pagination
  const handlePageChange = (page: number, newPageSize?: number) => {
    if (newPageSize && newPageSize !== pageSize) {
      setPageSize(newPageSize);
    }
    loadDocuments(page);
  };

  // Handle document selection
  const handleDocumentSelect = (documentId: string, selected: boolean) => {
    toggleDocumentSelection(documentId);
  };

  // Handle select all
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      selectAllDocuments();
    } else {
      clearSelectedDocuments();
    }
  };

  // Handle bulk delete
  const handleBulkDelete = () => {
    if (selectedDocuments.length === 0) {
      message.warning('请先选择要删除的文档');
      return;
    }

    Modal.confirm({
      title: '确认删除',
      content: `确定要删除 ${selectedDocuments.length} 个文档吗？此操作不可撤销。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await documentService.deleteDocuments(selectedDocuments, projectName);
          message.success(`成功删除 ${selectedDocuments.length} 个文档`);
          clearSelectedDocuments();
          loadDocuments(currentPage);
          loadStats();
        } catch (error) {
          console.error('Failed to delete documents:', error);
          message.error('删除文档失败');
        }
      },
    });
  };

  // Handle document actions
  const handleDocumentView = (document: Document) => {
    setViewerDocument(document);
  };

  const handleDocumentDelete = (document: Document) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除文档 "${document.filename}" 吗？`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await documentService.deleteDocuments([document.id], projectName);
          message.success('文档删除成功');
          loadDocuments(currentPage);
          loadStats();
        } catch (error) {
          console.error('Failed to delete document:', error);
          message.error('删除文档失败');
        }
      },
    });
  };

  const handleDocumentDownload = (document: Document) => {
    // Create blob and download
    const blob = new Blob([document.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = window.document.createElement('a');
    a.href = url;
    a.download =
      document.filename || document.title || document.source || '未命名文档';
    window.document.body.appendChild(a);
    a.click();
    window.document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const sortMenuItems: MenuProps['items'] = [
    {
      key: 'stored_at_desc',
      label: '按时间排序 (最新)',
      onClick: () => handleSort('stored_at', 'desc'),
    },
    {
      key: 'stored_at_asc',
      label: '按时间排序 (最旧)',
      onClick: () => handleSort('stored_at', 'asc'),
    },
    {
      key: 'filename_asc',
      label: '按文件名排序 (A-Z)',
      onClick: () => handleSort('filename', 'asc'),
    },
    {
      key: 'filename_desc',
      label: '按文件名排序 (Z-A)',
      onClick: () => handleSort('filename', 'desc'),
    },
    {
      key: 'content_length_desc',
      label: '按大小排序 (大到小)',
      onClick: () => handleSort('content_length', 'desc'),
    },
    {
      key: 'content_length_asc',
      label: '按大小排序 (小到大)',
      onClick: () => handleSort('content_length', 'asc'),
    },
  ];

  const isAllSelected =
    sortedDocuments.length > 0 &&
    sortedDocuments.every(doc => selectedDocuments.includes(doc.id));
  const isIndeterminate = selectedDocuments.length > 0 && !isAllSelected;

  return (
    <div className="document-list" style={{ height }}>
      <Card
        title={
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Title level={4} style={{ margin: 0 }}>
              文档管理
            </Title>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                loadDocuments(currentPage);
                loadStats();
              }}
              loading={loading}
            >
              刷新
            </Button>
          </div>
        }
        style={{ height: '100%' }}
        bodyStyle={{ padding: 0 }}
      >
        {/* Stats Row */}
        {stats && (
          <div style={{ padding: 16, borderBottom: '1px solid #f0f0f0' }}>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="总文档数"
                  value={stats.total_documents}
                  prefix={<FileTextOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="总大小"
                  value={stats.total_content_size}
                  formatter={value => {
                    const size = Number(value);
                    if (size < 1024) return `${size} B`;
                    if (size < 1024 * 1024)
                      return `${(size / 1024).toFixed(1)} KB`;
                    return `${(size / 1024 / 1024).toFixed(1)} MB`;
                  }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="标签数"
                  value={Object.keys(stats.tags || {}).length}
                />
              </Col>
              <Col span={6}>
                <Statistic title="已选择" value={selectedDocuments.length} />
              </Col>
            </Row>
          </div>
        )}

        {/* Filters */}
        <div style={{ padding: 16, borderBottom: '1px solid #f0f0f0' }}>
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Search
                placeholder="搜索文档..."
                allowClear
                onSearch={handleSearch}
                style={{ width: '100%' }}
              />
            </Col>
            <Col span={8}>
              <Select
                mode="multiple"
                placeholder="筛选标签"
                style={{ width: '100%' }}
                value={filterState.selectedTags}
                onChange={handleTagFilter}
                allowClear
              >
                {availableTags.map(tag => (
                  <Option key={tag} value={tag}>
                    <Tag size="small">{tag}</Tag>
                  </Option>
                ))}
              </Select>
            </Col>
            <Col span={4}>
              <Dropdown menu={{ items: sortMenuItems }} trigger={['click']}>
                <Button icon={<FilterOutlined />} style={{ width: '100%' }}>
                  排序
                </Button>
              </Dropdown>
            </Col>
          </Row>
        </div>

        {/* Bulk Actions */}
        {selectedDocuments.length > 0 && (
          <div
            style={{
              padding: 16,
              backgroundColor: '#f5f5f5',
              borderBottom: '1px solid #f0f0f0',
            }}
          >
            <Space>
              <Checkbox
                indeterminate={isIndeterminate}
                checked={isAllSelected}
                onChange={e => handleSelectAll(e.target.checked)}
              >
                全选 ({selectedDocuments.length} 已选择)
              </Checkbox>
              <Button
                type="primary"
                danger
                icon={<DeleteOutlined />}
                onClick={handleBulkDelete}
                size="small"
              >
                批量删除
              </Button>
              <Button
                icon={<DownloadOutlined />}
                onClick={() => message.info('批量下载功能开发中')}
                size="small"
              >
                批量下载
              </Button>
            </Space>
          </div>
        )}

        {/* Document List */}
        <div
          style={{
            padding: 16,
            height: 'calc(100% - 200px)',
            overflowY: 'auto',
          }}
        >
          <Spin spinning={loading}>
            {sortedDocuments.length > 0 ? (
              <Row gutter={[16, 16]}>
                {sortedDocuments.map(document => {
                  // Convert API Document to DocumentCard expected format
                  const cardDocument = {
                    ...document,
                    filename:
                      document.filename ||
                      document.title ||
                      document.source ||
                      '未命名文档',
                    content_length:
                      document.content_length ||
                      document.file_size ||
                      document.content?.length ||
                      0,
                    stored_at:
                      document.stored_at ||
                      document.created_at ||
                      document.updated_at ||
                      new Date().toISOString(),
                    tags: document.tags || [],
                  };

                  return (
                    <Col span={24} key={document.id}>
                      <DocumentCard
                        document={cardDocument}
                        selected={selectedDocuments.includes(document.id)}
                        onSelect={handleDocumentSelect}
                        onView={handleDocumentView}
                        onDelete={handleDocumentDelete}
                        onDownload={handleDocumentDownload}
                        showCheckbox={true}
                      />
                    </Col>
                  );
                })}
              </Row>
            ) : (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="暂无文档"
              />
            )}
          </Spin>
        </div>

        {/* Pagination */}
        {sortedDocuments.length > 0 && (
          <div
            style={{
              padding: 16,
              borderTop: '1px solid #f0f0f0',
              textAlign: 'center',
            }}
          >
            <Pagination
              current={currentPage}
              pageSize={pageSize}
              total={totalDocuments}
              showSizeChanger
              showQuickJumper
              showTotal={(total, range) =>
                `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
              }
              onChange={handlePageChange}
              onShowSizeChange={handlePageChange}
            />
          </div>
        )}
      </Card>

      {/* Document Viewer Modal */}
      <DocumentViewer
        document={viewerDocument}
        open={!!viewerDocument}
        onClose={() => setViewerDocument(null)}
      />
    </div>
  );
};

export default DocumentList;
