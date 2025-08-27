import { apiClient, ApiResponse } from './api';
import { useProjectStore } from '@/store/projectStore';
import { projectSync } from '@/utils/projectSync';
import type {
  ChatRequest,
  ChatResponse,
  ChatMessage,
  StreamChatChunk,
  Entity,
  Relation,
  TextChunk,
  BaseApiResponse,
} from '@/types/api';

export interface ChatHistoryResponse extends BaseApiResponse {
  data?: {
    conversations: Array<{
      id: string;
      title: string;
      created_at: string;
      updated_at: string;
      message_count: number;
    }>;
  };
}

export interface ConversationResponse extends BaseApiResponse {
  data?: {
    conversation_id: string;
    messages: ChatMessage[];
  };
}

export interface StreamEventData {
  type: 'chunk' | 'context' | 'complete' | 'error';
  data: any;
}

class ChatService {
  private readonly baseEndpoint = '/chat';

  private async getCurrentProjectName(): Promise<string | null> {
    // First try to get from store
    let projectName = useProjectStore.getState().currentProject?.name;

    // If not found, try to sync with backend
    if (!projectName) {
      projectName = await projectSync.ensureProjectSync();
    }

    return projectName;
  }

  private async addProjectParam(url: string): Promise<string> {
    const projectName = await this.getCurrentProjectName();
    if (!projectName) return url;

    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}project_name=${encodeURIComponent(projectName)}`;
  }

  async sendMessage(request: ChatRequest): Promise<ApiResponse<ChatResponse>> {
    return apiClient.post<ChatResponse>(
      await this.addProjectParam(this.baseEndpoint),
      request,
      {
        timeout: 120000, // 2 minutes for chat response
      }
    );
  }

  async sendStreamMessage(
    request: ChatRequest,
    onChunk: (chunk: StreamChatChunk) => void,
    onError?: (error: any) => void,
    onComplete?: () => void
  ): Promise<void> {
    const endpoint = await this.addProjectParam(`${this.baseEndpoint}/stream`);
    const url = `${(apiClient as any).baseURL}${endpoint}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          ...(apiClient as any).defaultHeaders,
        },
        body: JSON.stringify({ ...request, stream: true }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');

          // Keep the last incomplete line in the buffer
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);

              if (data === '[DONE]') {
                onComplete?.();
                return;
              }

              try {
                const chunk: StreamChatChunk = JSON.parse(data);
                onChunk(chunk);
              } catch (parseError) {
                console.warn('Failed to parse SSE data:', data, parseError);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

      onComplete?.();
    } catch (error) {
      onError?.(error);
      throw error;
    }
  }

  async sendStreamMessageWithEventSource(
    request: ChatRequest,
    onChunk: (chunk: StreamChatChunk) => void,
    onError?: (error: any) => void,
    onComplete?: () => void
  ): Promise<() => void> {
    const endpoint = await this.addProjectParam(`${this.baseEndpoint}/stream`);
    const url = `${(apiClient as any).baseURL}${endpoint}`;

    // Create a POST request first to initiate the stream
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiClient as any).defaultHeaders,
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    // Get the stream ID from response
    const { stream_id } = await response.json();

    // Connect to the event stream
    const eventSource = new EventSource(`${url}/${stream_id}`);

    eventSource.onmessage = event => {
      if (event.data === '[DONE]') {
        eventSource.close();
        onComplete?.();
        return;
      }

      try {
        const chunk: StreamChatChunk = JSON.parse(event.data);
        onChunk(chunk);
      } catch (parseError) {
        console.warn('Failed to parse SSE data:', event.data, parseError);
      }
    };

    eventSource.onerror = error => {
      eventSource.close();
      onError?.(error);
    };

    // Return cleanup function
    return () => {
      eventSource.close();
    };
  }

  async getChatHistory(): Promise<ApiResponse<ChatHistoryResponse>> {
    return apiClient.get<ChatHistoryResponse>(
      await this.addProjectParam(`${this.baseEndpoint}/history`),
      {
        cache: true,
      }
    );
  }

  async getConversation(
    conversationId: string
  ): Promise<ApiResponse<ConversationResponse>> {
    return apiClient.get<ConversationResponse>(
      `${this.baseEndpoint}/conversations/${encodeURIComponent(conversationId)}`,
      {
        cache: true,
      }
    );
  }

  async createConversation(
    title?: string
  ): Promise<ApiResponse<{ conversation_id: string }>> {
    return apiClient.post(`${this.baseEndpoint}/conversations`, { title });
  }

  async deleteConversation(
    conversationId: string
  ): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.delete(
      `${this.baseEndpoint}/conversations/${encodeURIComponent(conversationId)}`
    );
  }

  async updateConversationTitle(
    conversationId: string,
    title: string
  ): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.put(
      `${this.baseEndpoint}/conversations/${encodeURIComponent(conversationId)}/title`,
      {
        title,
      }
    );
  }

  async clearChatHistory(): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.post(`${this.baseEndpoint}/clear-history`);
  }

  async regenerateResponse(
    conversationId: string,
    messageId: string
  ): Promise<ApiResponse<ChatResponse>> {
    return apiClient.post<ChatResponse>(`${this.baseEndpoint}/regenerate`, {
      conversation_id: conversationId,
      message_id: messageId,
    });
  }

  async getSuggestions(
    query: string
  ): Promise<ApiResponse<{ suggestions: string[] }>> {
    return apiClient.post(`${this.baseEndpoint}/suggestions`, { query });
  }

  async getRelatedEntities(
    query: string
  ): Promise<ApiResponse<{ entities: Entity[] }>> {
    return apiClient.post(`${this.baseEndpoint}/related-entities`, { query });
  }

  async getContextPreview(
    query: string,
    options?: {
      entity_top_k?: number;
      relation_top_k?: number;
      text_chunk_top_k?: number;
    }
  ): Promise<
    ApiResponse<{
      entities: Entity[];
      relations: Relation[];
      text_chunks: TextChunk[];
    }>
  > {
    return apiClient.post(`${this.baseEndpoint}/context-preview`, {
      query,
      ...options,
    });
  }

  formatConversationHistory(
    messages: ChatMessage[]
  ): Array<{ role: string; content: string }> {
    return messages.map(msg => ({
      role: msg.role,
      content: msg.content,
    }));
  }

  generateConversationTitle(firstMessage: string): string {
    // Extract a meaningful title from the first message
    const cleaned = firstMessage.trim().slice(0, 50);
    return cleaned.length < firstMessage.length ? `${cleaned}...` : cleaned;
  }

  validateMessage(content: string): { valid: boolean; error?: string } {
    if (!content || content.trim().length === 0) {
      return { valid: false, error: '消息不能为空' };
    }

    if (content.length > 4000) {
      return { valid: false, error: '消息长度不能超过 4000 个字符' };
    }

    return { valid: true };
  }

  escapeMarkdown(text: string): string {
    return text.replace(/[*_`~]/g, '\\$&');
  }

  parseMarkdown(text: string): string {
    // Simple markdown parsing - in a real app, use a proper markdown library
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }

  clearCache(): void {
    apiClient.clearCache();
  }
}

export const chatService = new ChatService();
