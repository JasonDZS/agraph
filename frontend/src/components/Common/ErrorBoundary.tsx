import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { Result, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    this.setState({
      error,
      errorInfo,
    });

    // Here you can log the error to an error reporting service
    // logErrorToService(error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Result
          status="error"
          title="Something went wrong"
          subTitle="An unexpected error occurred. Please try refreshing the page."
          extra={[
            <Button key="retry" onClick={this.handleReset}>
              Try Again
            </Button>,
            <Button
              key="reload"
              icon={<ReloadOutlined />}
              onClick={this.handleReload}
            >
              Reload Page
            </Button>,
          ]}
        >
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <div
              style={{
                textAlign: 'left',
                marginTop: '20px',
                padding: '16px',
                background: '#f5f5f5',
                borderRadius: '6px',
                fontSize: '12px',
                fontFamily: 'monospace',
              }}
            >
              <details>
                <summary>Error Details (Development Only)</summary>
                <pre style={{ whiteSpace: 'pre-wrap', margin: '10px 0' }}>
                  {this.state.error.toString()}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            </div>
          )}
        </Result>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
