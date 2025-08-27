import React from 'react';
import { Spin } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

interface LoadingSpinnerProps {
  size?: 'small' | 'default' | 'large';
  tip?: string;
  spinning?: boolean;
  children?: React.ReactNode;
  delay?: number;
  indicator?: React.ReactNode;
  style?: React.CSSProperties;
  className?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'default',
  tip,
  spinning = true,
  children,
  delay = 0,
  indicator,
  style,
  className,
}) => {
  const defaultIndicator = <LoadingOutlined style={{ fontSize: 24 }} spin />;

  if (children) {
    return (
      <Spin
        size={size}
        tip={tip}
        spinning={spinning}
        delay={delay}
        indicator={indicator || defaultIndicator}
        style={style}
        className={className}
      >
        {children}
      </Spin>
    );
  }

  return (
    <div
      className={className}
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100px',
        ...style,
      }}
    >
      <Spin
        size={size}
        tip={tip}
        spinning={spinning}
        delay={delay}
        indicator={indicator || defaultIndicator}
      />
    </div>
  );
};

export default LoadingSpinner;
