import { Card, Row, Col, Statistic } from 'antd';
import {
  FileTextOutlined,
  NodeIndexOutlined,
  MessageOutlined,
  FolderOutlined,
} from '@ant-design/icons';

const Dashboard = () => {
  return (
    <div style={{ padding: '24px' }}>
      <h1>Dashboard</h1>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic title="Projects" value={5} prefix={<FolderOutlined />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Documents"
              value={128}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Knowledge Entities"
              value={1024}
              prefix={<NodeIndexOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Chat Sessions"
              value={42}
              prefix={<MessageOutlined />}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
