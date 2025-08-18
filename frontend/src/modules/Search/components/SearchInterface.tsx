import { Card } from 'antd';
import { SearchOutlined } from '@ant-design/icons';

const SearchInterface = () => {
  return (
    <div>
      <h1>Search</h1>
      <Card>
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <SearchOutlined style={{ fontSize: '48px', color: '#ccc' }} />
          <p>Search interface will be implemented in TODO 9</p>
        </div>
      </Card>
    </div>
  );
};

export default SearchInterface;
