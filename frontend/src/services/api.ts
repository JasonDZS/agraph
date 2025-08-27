import { env } from '@/utils/env';
import { handleNetworkError } from '@/utils/errorHandler';

export interface ApiResponse<T = any> {
  data?: T;
  message?: string;
  success: boolean;
  status: 'success' | 'error' | 'processing';
  timestamp?: string;
  error?: ApiError;
}

export interface ApiError {
  message: string;
  code?: string;
  status?: number;
  details?: any;
}

export interface RequestConfig extends Omit<RequestInit, 'cache'> {
  retry?: number;
  timeout?: number;
  cache?: boolean;
  params?: Record<string, string | number | boolean | null | undefined>;
}

export interface RequestInterceptor {
  (config: RequestConfig): RequestConfig | Promise<RequestConfig>;
}

export interface ResponseInterceptor {
  onFulfilled?: (response: Response) => Response | Promise<Response>;
  onRejected?: (error: any) => any;
}

class ApiClient {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;
  private requestInterceptors: RequestInterceptor[] = [];
  private responseInterceptors: ResponseInterceptor[] = [];
  private cache = new Map<
    string,
    { data: any; timestamp: number; ttl: number }
  >();
  private readonly DEFAULT_TIMEOUT = 30000;
  private readonly DEFAULT_RETRY_COUNT = 3;
  private readonly CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  constructor() {
    this.baseURL = env.get('API_BASE_URL');
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  addRequestInterceptor(interceptor: RequestInterceptor): void {
    this.requestInterceptors.push(interceptor);
  }

  addResponseInterceptor(interceptor: ResponseInterceptor): void {
    this.responseInterceptors.push(interceptor);
  }

  private async applyRequestInterceptors(
    config: RequestConfig
  ): Promise<RequestConfig> {
    let result = config;
    for (const interceptor of this.requestInterceptors) {
      result = await interceptor(result);
    }
    return result;
  }

  private async applyResponseInterceptors(
    response: Response
  ): Promise<Response> {
    let result = response;
    for (const interceptor of this.responseInterceptors) {
      if (interceptor.onFulfilled) {
        try {
          result = await interceptor.onFulfilled(result);
        } catch (error) {
          if (interceptor.onRejected) {
            await interceptor.onRejected(error);
          }
          throw error;
        }
      }
    }
    return result;
  }

  private getCacheKey(url: string, options: RequestConfig): string {
    const method = options.method || 'GET';
    const body = options.body ? JSON.stringify(options.body) : '';
    return `${method}:${url}:${body}`;
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    const now = Date.now();
    if (now - cached.timestamp > cached.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.data;
  }

  private setCache(key: string, data: any, ttl = this.CACHE_TTL): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  private createTimeoutController(timeout: number): {
    signal: AbortSignal;
    cleanup: () => void;
  } {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, timeout);

    return {
      signal: controller.signal,
      cleanup: () => clearTimeout(timeoutId),
    };
  }

  private async requestWithRetry<T>(
    url: string,
    config: RequestConfig,
    retryCount = 0
  ): Promise<ApiResponse<T>> {
    const maxRetries = config.retry ?? this.DEFAULT_RETRY_COUNT;
    const timeout = config.timeout ?? this.DEFAULT_TIMEOUT;

    const { signal, cleanup } = this.createTimeoutController(timeout);
    const requestConfig = { ...config, signal };

    try {
      // Check cache for GET requests
      if (config.cache && (!config.method || config.method === 'GET')) {
        const cacheKey = this.getCacheKey(url, config);
        const cachedData = this.getFromCache(cacheKey);
        if (cachedData) {
          cleanup();
          return cachedData;
        }
      }

      const {
        cache: _cache,
        retry: _retry,
        timeout: _timeout,
        params: _params,
        ...fetchConfig
      } = requestConfig;
      const response = await fetch(url, fetchConfig);
      cleanup();

      // Apply response interceptors
      const processedResponse = await this.applyResponseInterceptors(
        response.clone()
      );

      if (!processedResponse.ok) {
        const errorData = await processedResponse.json().catch(() => ({}));
        const error = {
          message: errorData.message || `HTTP ${processedResponse.status}`,
          status: processedResponse.status,
          details: errorData,
        } as ApiError;

        // Retry for specific status codes
        if (
          retryCount < maxRetries &&
          [500, 502, 503, 504].includes(processedResponse.status)
        ) {
          await this.delay(Math.pow(2, retryCount) * 1000); // Exponential backoff
          return this.requestWithRetry(url, config, retryCount + 1);
        }

        throw error;
      }

      const data = await processedResponse.json();
      const result: ApiResponse<T> = {
        data: data.data || data,
        success: data.status === 'success' || true,
        status: data.status || 'success',
        message: data.message,
        timestamp: data.timestamp,
      };

      // Cache successful GET requests
      if (config.cache && (!config.method || config.method === 'GET')) {
        const cacheKey = this.getCacheKey(url, config);
        this.setCache(cacheKey, result);
      }

      return result;
    } catch (error: any) {
      cleanup();

      // Retry for network errors
      if (
        retryCount < maxRetries &&
        (error.name === 'AbortError' || error.name === 'TypeError')
      ) {
        await this.delay(Math.pow(2, retryCount) * 1000);
        return this.requestWithRetry(url, config, retryCount + 1);
      }

      handleNetworkError(error, url);
      throw error;
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private async request<T = any>(
    endpoint: string,
    options: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    let url = `${this.baseURL}${endpoint}`;

    // Add query parameters if provided
    if (options.params) {
      const searchParams = new URLSearchParams();
      Object.entries(options.params).forEach(([key, value]) => {
        if (value !== null && value !== undefined) {
          searchParams.append(key, String(value));
        }
      });
      if (searchParams.toString()) {
        url += `?${searchParams.toString()}`;
      }
    }

    let config: RequestConfig = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    };

    // Remove Content-Type for FormData uploads to let browser set it correctly
    if (config.body instanceof FormData && config.headers) {
      const headers = config.headers as Record<string, string>;
      if (headers['Content-Type'] === 'application/json') {
        delete headers['Content-Type'];
      }
    }

    // Apply request interceptors
    config = await this.applyRequestInterceptors(config);

    return this.requestWithRetry<T>(url, config);
  }

  async get<T = any>(
    endpoint: string,
    options: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'GET',
      cache: options.cache ?? true,
    });
  }

  async post<T = any>(
    endpoint: string,
    data?: any,
    options: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T = any>(
    endpoint: string,
    data?: any,
    options: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T = any>(
    endpoint: string,
    options: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }

  setAuthToken(token: string): void {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  removeAuthToken(): void {
    delete this.defaultHeaders['Authorization'];
  }

  clearCache(): void {
    this.cache.clear();
  }

  setCacheTimeout(_ttl: number): void {
    // Update cache TTL for future requests
  }
}

export const apiClient = new ApiClient();
