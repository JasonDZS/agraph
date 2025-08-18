import React, { useState } from 'react';
import { Card, Tabs, Space, Button, Typography, Row, Col } from 'antd';
import {
  UploadOutlined,
  FileTextOutlined,
  TagsOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import DocumentUploader from './DocumentUploader';
import DocumentList from './DocumentList';
import DocumentTagManager from './DocumentTagManager';
import DocumentBatchActions from './DocumentBatchActions';
import { useDocumentStore } from '../../../store/documentStore';
import { useAppStore } from '../../../store/appStore';

const { Title } = Typography;
const { TabPane } = Tabs;

const DocumentManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState('list');
  const [tagManagerVisible, setTagManagerVisible] = useState(false);

  const { selectedDocuments, documents, clearSelection } = useDocumentStore();
  const { currentProject } = useAppStore();

  // Get selected document objects
  const selectedDocumentObjects = documents.filter(doc =>
    selectedDocuments.includes(doc.id)
  );

  const handleUploadComplete = () => {
    // Switch to list tab after upload
    setActiveTab('list');
  };

  const handleBatchActionComplete = () => {
    clearSelection();
  };

  const tabBarExtraContent = (
    <Space>
      <Button
        icon={<TagsOutlined />}
        onClick={() => setTagManagerVisible(true)}
      >
        标签管理
      </Button>
      {selectedDocuments.length > 0 && (
        <Button type="primary" onClick={() => setActiveTab('batch')}>
          批量操作 ({selectedDocuments.length})
        </Button>
      )}
    </Space>
  );

  return (
    <div className="document-management" style={{ height: '100%' }}>
      <Card
        title={
          <Space>
            <FileTextOutlined />
            <Title level={3} style={{ margin: 0 }}>
              文档管理
            </Title>
            {currentProject && (
              <span style={{ color: '#666', fontSize: '14px' }}>
                - {currentProject}
              </span>
            )}
          </Space>
        }
        style={{ height: '100%' }}
        bodyStyle={{ padding: 0, height: 'calc(100% - 57px)' }}
        tabBarExtraContent={tabBarExtraContent}
      >
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          style={{ height: '100%' }}
          tabBarStyle={{ paddingLeft: 24, paddingRight: 24, marginBottom: 0 }}
        >
          <TabPane
            tab={
              <Space>
                <FileTextOutlined />
                <span>文档列表</span>
              </Space>
            }
            key="list"
          >
            <div style={{ height: 'calc(100vh - 200px)' }}>
              {selectedDocumentObjects.length > 0 && (
                <div style={{ padding: 16, borderBottom: '1px solid #f0f0f0' }}>
                  <DocumentBatchActions
                    selectedDocuments={selectedDocumentObjects}
                    onComplete={handleBatchActionComplete}
                    projectName={currentProject}
                  />
                </div>
              )}
              <DocumentList
                projectName={currentProject}
                height={
                  selectedDocumentObjects.length > 0 ? undefined : undefined
                }
              />
            </div>
          </TabPane>

          <TabPane
            tab={
              <Space>
                <UploadOutlined />
                <span>上传文档</span>
              </Space>
            }
            key="upload"
          >
            <div style={{ padding: 24 }}>
              <DocumentUploader
                projectName={currentProject}
                onUploadComplete={handleUploadComplete}
              />
            </div>
          </TabPane>

          {selectedDocumentObjects.length > 0 && (
            <TabPane
              tab={
                <Space>
                  <SettingOutlined />
                  <span>批量操作</span>
                </Space>
              }
              key="batch"
            >
              <div style={{ padding: 24 }}>
                <DocumentBatchActions
                  selectedDocuments={selectedDocumentObjects}
                  onComplete={handleBatchActionComplete}
                  projectName={currentProject}
                />

                <Card title="选中的文档" style={{ marginTop: 16 }}>
                  <Row gutter={[16, 16]}>
                    {selectedDocumentObjects.map(doc => (
                      <Col span={12} key={doc.id}>
                        <Card size="small">
                          <Card.Meta
                            title={
                              doc.filename ||
                              doc.title ||
                              doc.source ||
                              '未命名文档'
                            }
                            description={
                              <Space direction="vertical" size="small">
                                <span>
                                  {doc.file_size || doc.content?.length || 0}{' '}
                                  bytes
                                </span>
                                <Space wrap>
                                  {(doc.tags || []).map(tag => (
                                    <span
                                      key={tag}
                                      style={{
                                        background: '#f0f0f0',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        fontSize: '12px',
                                      }}
                                    >
                                      {tag}
                                    </span>
                                  ))}
                                </Space>
                              </Space>
                            }
                          />
                        </Card>
                      </Col>
                    ))}
                  </Row>
                </Card>
              </div>
            </TabPane>
          )}
        </Tabs>
      </Card>

      {/* Tag Manager Modal */}
      <DocumentTagManager
        open={tagManagerVisible}
        onClose={() => setTagManagerVisible(false)}
        projectName={currentProject}
      />
    </div>
  );
};

export default DocumentManagement;
