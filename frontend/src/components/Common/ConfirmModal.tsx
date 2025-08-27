import React, { useState } from 'react';
import { Modal, Button, Space, Typography, Input, Alert } from 'antd';
import {
  ExclamationCircleOutlined,
  DeleteOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';

const { Text, Paragraph } = Typography;

export interface ConfirmModalProps {
  open: boolean;
  title?: string;
  content?: React.ReactNode;
  type?: 'info' | 'success' | 'warning' | 'error' | 'danger';
  confirmText?: string;
  cancelText?: string;
  onConfirm?: () => void | Promise<void>;
  onCancel?: () => void;
  loading?: boolean;
  danger?: boolean;
  requiresConfirmation?: boolean;
  confirmationText?: string;
  confirmationPlaceholder?: string;
  width?: number;
  centered?: boolean;
  maskClosable?: boolean;
  keyboard?: boolean;
  okButtonProps?: any;
  cancelButtonProps?: any;
  icon?: React.ReactNode;
}

const ConfirmModal: React.FC<ConfirmModalProps> = ({
  open,
  title = 'Confirm Action',
  content,
  type = 'warning',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  onConfirm,
  onCancel,
  loading = false,
  danger = false,
  requiresConfirmation = false,
  confirmationText,
  confirmationPlaceholder = 'Type to confirm',
  width = 416,
  centered = true,
  maskClosable = false,
  keyboard = true,
  okButtonProps,
  cancelButtonProps,
  icon,
}) => {
  const [confirmInput, setConfirmInput] = useState('');

  const getIcon = () => {
    if (icon) return icon;

    switch (type) {
      case 'success':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'info':
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'error':
      case 'danger':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
    }
  };

  const getAlertType = () => {
    switch (type) {
      case 'success':
        return 'success';
      case 'info':
        return 'info';
      case 'warning':
        return 'warning';
      case 'error':
      case 'danger':
        return 'error';
      default:
        return 'warning';
    }
  };

  const handleConfirm = async () => {
    if (requiresConfirmation && confirmationText && confirmInput !== confirmationText) {
      return;
    }

    if (onConfirm) {
      await onConfirm();
    }
  };

  const handleCancel = () => {
    setConfirmInput('');
    if (onCancel) {
      onCancel();
    }
  };

  const isConfirmDisabled = requiresConfirmation &&
    confirmationText &&
    confirmInput !== confirmationText;

  const isDangerAction = danger || type === 'error' || type === 'danger';

  return (
    <Modal
      open={open}
      title={
        <Space>
          {getIcon()}
          <span>{title}</span>
        </Space>
      }
      onCancel={handleCancel}
      onOk={handleConfirm}
      width={width}
      centered={centered}
      maskClosable={maskClosable}
      keyboard={keyboard}
      confirmLoading={loading}
      okText={confirmText}
      cancelText={cancelText}
      okButtonProps={{
        danger: isDangerAction,
        disabled: isConfirmDisabled,
        ...okButtonProps,
      }}
      cancelButtonProps={cancelButtonProps}
      footer={[
        <Button
          key="cancel"
          onClick={handleCancel}
          {...cancelButtonProps}
        >
          {cancelText}
        </Button>,
        <Button
          key="confirm"
          type="primary"
          danger={isDangerAction}
          loading={loading}
          disabled={isConfirmDisabled}
          onClick={handleConfirm}
          {...okButtonProps}
        >
          {confirmText}
        </Button>,
      ]}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        {content && (
          <div style={{ marginBottom: '16px' }}>
            {typeof content === 'string' ? (
              <Paragraph>{content}</Paragraph>
            ) : (
              content
            )}
          </div>
        )}

        {isDangerAction && (
          <Alert
            message="This action cannot be undone"
            type="error"
            showIcon
            style={{ marginBottom: '16px' }}
          />
        )}

        {requiresConfirmation && confirmationText && (
          <div>
            <Text type="secondary" style={{ marginBottom: '8px', display: 'block' }}>
              Please type <Text code strong>{confirmationText}</Text> to confirm:
            </Text>
            <Input
              value={confirmInput}
              onChange={(e) => setConfirmInput(e.target.value)}
              placeholder={confirmationPlaceholder}
              autoFocus
            />
          </div>
        )}
      </Space>
    </Modal>
  );
};

export default ConfirmModal;

// Utility function for quick usage
export const confirm = (options: Omit<ConfirmModalProps, 'open'>) => {
  return new Promise<boolean>((resolve) => {
    let modal: any;

    const onConfirm = async () => {
      try {
        if (options.onConfirm) {
          await options.onConfirm();
        }
        modal.destroy();
        resolve(true);
      } catch (error) {
        console.error('Confirmation error:', error);
        resolve(false);
      }
    };

    const onCancel = () => {
      if (options.onCancel) {
        options.onCancel();
      }
      modal.destroy();
      resolve(false);
    };

    modal = Modal.confirm({
      title: options.title,
      content: options.content,
      icon: options.icon || <ExclamationCircleOutlined />,
      okText: options.confirmText || 'Confirm',
      cancelText: options.cancelText || 'Cancel',
      onOk: onConfirm,
      onCancel,
      width: options.width,
      centered: options.centered,
      maskClosable: options.maskClosable,
      keyboard: options.keyboard,
      okButtonProps: {
        danger: options.danger || options.type === 'error' || options.type === 'danger',
        ...options.okButtonProps,
      },
      cancelButtonProps: options.cancelButtonProps,
    });
  });
};
