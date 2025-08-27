import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { Result, Button, Typography, Card, Space } from 'antd';
import { ReloadOutlined, BugOutlined, HomeOutlined } from '@ant-design/icons';

const { Text, Paragraph } = Typography;

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showReportButton?: boolean;
  level?: 'page' | 'component';
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

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log to error reporting service in production
    if (process.env.NODE_ENV === 'production') {
      // logErrorToService(error, errorInfo);
    }
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  handleReportError = () => {
    if (this.state.error) {
      const errorReport = {
        message: this.state.error.message,
        stack: this.state.error.stack,
        componentStack: this.state.errorInfo?.componentStack,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        url: window.location.href,
      };

      console.log('Error Report:', errorReport);

      // Here you would typically send to an error reporting service
      // reportError(errorReport);

      // For now, copy to clipboard
      navigator.clipboard.writeText(JSON.stringify(errorReport, null, 2))
        .then(() => alert('Error report copied to clipboard'))
        .catch(() => console.log('Failed to copy error report'));
    }
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    const { level = 'page', showReportButton = true } = this.props;

    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const isComponentLevel = level === 'component';

      if (isComponentLevel) {
        return (
          <Card
            style={{
              margin: '16px 0',
              border: '1px solid #ff4d4f',
              borderRadius: '6px'
            }}
          >
            <Result
              status="error"
              title={<Text type="danger">Component Error</Text>}
              subTitle="This component encountered an error and cannot be displayed."
              extra={
                <Space>
                  <Button size="small" onClick={this.handleReset}>
                    Retry
                  </Button>
                  {showReportButton && (
                    <Button
                      size="small"
                      icon={<BugOutlined />}
                      onClick={this.handleReportError}
                    >
                      Report
                    </Button>
                  )}
                </Space>
              }
            />
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <Card
                size="small"
                title="Error Details (Development Only)"
                style={{ marginTop: '16px' }}
              >
                <Paragraph>
                  <Text code>{this.state.error.message}</Text>
                </Paragraph>
                <details>
                  <summary>Stack Trace</summary>
                  <pre style={{
                    whiteSpace: 'pre-wrap',
                    fontSize: '11px',
                    background: '#f5f5f5',
                    padding: '8px',
                    borderRadius: '4px',
                    marginTop: '8px'
                  }}>
                    {this.state.error.stack}
                    {this.state.errorInfo?.componentStack}
                  </pre>
                </details>
              </Card>
            )}
          </Card>
        );
      }

      return (
        <Result
          status="error"
          title="Something went wrong"
          subTitle="An unexpected error occurred. Please try one of the options below."
          extra={
            <Space wrap>
              <Button type="primary" onClick={this.handleReset}>
                Try Again
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={this.handleReload}
              >
                Reload Page
              </Button>
              <Button
                icon={<HomeOutlined />}
                onClick={this.handleGoHome}
              >
                Go Home
              </Button>
              {showReportButton && (
                <Button
                  icon={<BugOutlined />}
                  onClick={this.handleReportError}
                >
                  Report Error
                </Button>
              )}
            </Space>
          }
        >
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <Card
              title="Error Details (Development Only)"
              style={{
                textAlign: 'left',
                marginTop: '20px',
                maxWidth: '800px',
                margin: '20px auto 0',
              }}
            >
              <Paragraph>
                <Text strong>Error Message:</Text>
                <br />
                <Text code>{this.state.error.message}</Text>
              </Paragraph>
              <details>
                <summary>Full Stack Trace</summary>
                <pre style={{
                  whiteSpace: 'pre-wrap',
                  fontSize: '11px',
                  background: '#f5f5f5',
                  padding: '12px',
                  borderRadius: '4px',
                  marginTop: '8px',
                  overflow: 'auto'
                }}>
                  {this.state.error.stack}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            </Card>
          )}
        </Result>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
