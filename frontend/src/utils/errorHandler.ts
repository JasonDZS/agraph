import { notification } from 'antd';
import { env } from './env';

export interface AppError extends Error {
  code?: string;
  status?: number;
  details?: any;
}

class ErrorHandler {
  private static instance: ErrorHandler;

  private constructor() {
    // Set up global error handlers
    this.setupGlobalHandlers();
  }

  static getInstance(): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler();
    }
    return ErrorHandler.instance;
  }

  private setupGlobalHandlers() {
    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', event => {
      console.error('Unhandled promise rejection:', event.reason);
      this.handleError(event.reason);
      event.preventDefault();
    });

    // Handle uncaught errors
    window.addEventListener('error', event => {
      console.error('Uncaught error:', event.error);
      this.handleError(event.error);
    });
  }

  handleError(error: Error | AppError | string, context?: string) {
    let processedError: AppError;

    if (typeof error === 'string') {
      processedError = new Error(error) as AppError;
    } else {
      processedError = error as AppError;
    }

    // Log error details
    this.logError(processedError, context);

    // Show user notification based on error type
    this.showErrorNotification(processedError);

    // Send to error tracking service in production
    if (env.isProduction()) {
      this.reportError(processedError, context);
    }
  }

  private logError(error: AppError, context?: string) {
    const logLevel = env.get('LOG_LEVEL');

    if (logLevel === 'debug' || env.isDevelopment()) {
      console.group(`🚨 Error ${context ? `in ${context}` : ''}`);
      console.error('Message:', error.message);
      console.error('Code:', error.code);
      console.error('Status:', error.status);
      console.error('Stack:', error.stack);
      console.error('Details:', error.details);
      console.groupEnd();
    } else {
      console.error(`Error${context ? ` in ${context}` : ''}:`, error.message);
    }
  }

  private showErrorNotification(error: AppError) {
    let message = 'An error occurred';
    let description = error.message;

    // Customize messages based on error type
    if (error.status) {
      switch (error.status) {
        case 400:
          message = 'Invalid Request';
          break;
        case 401:
          message = 'Unauthorized';
          description = 'Please log in to continue';
          break;
        case 403:
          message = 'Access Denied';
          description = 'You do not have permission to perform this action';
          break;
        case 404:
          message = 'Not Found';
          description = 'The requested resource was not found';
          break;
        case 500:
          message = 'Server Error';
          description =
            'Something went wrong on our end. Please try again later';
          break;
        default:
          message = 'Network Error';
      }
    }

    notification.error({
      message,
      description,
      duration: 5,
      placement: 'topRight',
    });
  }

  private reportError(error: AppError, context?: string) {
    // Here you would integrate with error tracking services like:
    // - Sentry
    // - Bugsnag
    // - LogRocket
    // - Custom error tracking API

    console.log('Error reported to tracking service:', {
      error: error.message,
      code: error.code,
      status: error.status,
      context,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
    });
  }

  // Helper methods for specific error types
  handleNetworkError(error: any, url?: string) {
    const networkError: AppError = new Error(
      'Network request failed'
    ) as AppError;
    networkError.code = 'NETWORK_ERROR';
    networkError.details = { originalError: error, url };
    this.handleError(networkError, 'Network');
  }

  handleValidationError(errors: Record<string, string[]>) {
    const validationError: AppError = new Error(
      'Validation failed'
    ) as AppError;
    validationError.code = 'VALIDATION_ERROR';
    validationError.details = errors;
    this.handleError(validationError, 'Validation');
  }

  handlePermissionError(action?: string) {
    const permissionError: AppError = new Error(
      `Permission denied${action ? ` for ${action}` : ''}`
    ) as AppError;
    permissionError.code = 'PERMISSION_ERROR';
    permissionError.status = 403;
    this.handleError(permissionError, 'Authorization');
  }
}

// Export singleton instance
export const errorHandler = ErrorHandler.getInstance();

// Export utility functions
export const handleError = (error: Error | string, context?: string) => {
  errorHandler.handleError(error, context);
};

export const handleNetworkError = (error: any, url?: string) => {
  errorHandler.handleNetworkError(error, url);
};

export const handleValidationError = (errors: Record<string, string[]>) => {
  errorHandler.handleValidationError(errors);
};

export const handlePermissionError = (action?: string) => {
  errorHandler.handlePermissionError(action);
};
