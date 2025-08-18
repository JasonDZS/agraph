import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Alert, Button, Card } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReload = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Card>
          <Alert
            message="Something went wrong"
            description={
              <div>
                <p>An unexpected error occurred in this component.</p>
                {this.state.error && (
                  <details style={{ marginTop: 8 }}>
                    <summary>Error details</summary>
                    <pre
                      style={{
                        fontSize: 12,
                        overflow: 'auto',
                        maxHeight: 200,
                        background: '#f5f5f5',
                        padding: 8,
                        borderRadius: 4,
                        marginTop: 8,
                      }}
                    >
                      {this.state.error.toString()}
                      {this.state.errorInfo?.componentStack}
                    </pre>
                  </details>
                )}
              </div>
            }
            type="error"
            showIcon
            action={
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={this.handleReload}
              >
                Try Again
              </Button>
            }
          />
        </Card>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
