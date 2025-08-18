import { useCallback, useEffect, useRef } from 'react';
import { useChatStore } from '@/store/chatStore';
import { SendMessageOptions } from '../types';

export interface UseStreamChatOptions {
  onMessageStart?: () => void;
  onMessageComplete?: () => void;
  onError?: (error: string) => void;
  autoCreateConversation?: boolean;
}

export const useStreamChat = (options: UseStreamChatOptions = {}) => {
  const {
    onMessageStart,
    onMessageComplete,
    onError,
    autoCreateConversation = true,
  } = options;

  const {
    conversations,
    currentConversationId,
    messages,
    isLoading,
    isStreaming,
    error,
    currentContext,
    showContext,

    // Actions
    createNewConversation,
    loadConversation,
    deleteConversation,
    clearConversationHistory,
    sendMessage,
    regenerateLastResponse,
    setShowContext,
    reset,
  } = useChatStore();

  // Refs for stable callbacks
  const onMessageStartRef = useRef(onMessageStart);
  const onMessageCompleteRef = useRef(onMessageComplete);
  const onErrorRef = useRef(onError);

  // Update refs when props change
  useEffect(() => {
    onMessageStartRef.current = onMessageStart;
    onMessageCompleteRef.current = onMessageComplete;
    onErrorRef.current = onError;
  }, [onMessageStart, onMessageComplete, onError]);

  // Auto-create conversation if needed
  const ensureConversation = useCallback(async () => {
    if (!currentConversationId && autoCreateConversation) {
      const newConversationId = await createNewConversation();
      return newConversationId;
    }
    return currentConversationId;
  }, [currentConversationId, createNewConversation, autoCreateConversation]);

  // Send message with auto-conversation creation
  const sendStreamMessage = useCallback(
    async (content: string, messageOptions?: SendMessageOptions) => {
      try {
        onMessageStartRef.current?.();

        await ensureConversation();
        await sendMessage(content, messageOptions);

        onMessageCompleteRef.current?.();
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : '发送消息失败';
        onErrorRef.current?.(errorMessage);
      }
    },
    [ensureConversation, sendMessage]
  );

  // Create new conversation
  const startNewConversation = useCallback(
    async (title?: string) => {
      try {
        const conversationId = await createNewConversation(title);
        return conversationId;
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : '创建对话失败';
        onErrorRef.current?.(errorMessage);
        throw err;
      }
    },
    [createNewConversation]
  );

  // Load conversation
  const switchToConversation = useCallback(
    async (conversationId: string) => {
      try {
        await loadConversation(conversationId);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : '加载对话失败';
        onErrorRef.current?.(errorMessage);
        throw err;
      }
    },
    [loadConversation]
  );

  // Delete conversation with confirmation
  const removeConversation = useCallback(
    async (conversationId: string) => {
      try {
        await deleteConversation(conversationId);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : '删除对话失败';
        onErrorRef.current?.(errorMessage);
        throw err;
      }
    },
    [deleteConversation]
  );

  // Clear all conversations with confirmation
  const clearAllConversations = useCallback(async () => {
    try {
      await clearConversationHistory();
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : '清空对话历史失败';
      onErrorRef.current?.(errorMessage);
      throw err;
    }
  }, [clearConversationHistory]);

  // Regenerate last response
  const regenerateResponse = useCallback(async () => {
    try {
      onMessageStartRef.current?.();

      await regenerateLastResponse();

      onMessageCompleteRef.current?.();
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : '重新生成回答失败';
      onErrorRef.current?.(errorMessage);
    }
  }, [regenerateLastResponse]);

  // Get current conversation
  const currentConversation = conversations.find(
    c => c.id === currentConversationId
  );

  // Helper functions
  const hasMessages = messages.length > 0;
  const hasConversations = conversations.length > 0;
  const canRegenerate =
    hasMessages &&
    messages[messages.length - 1]?.role === 'assistant' &&
    !isStreaming;

  // Context management
  const toggleContext = useCallback(() => {
    setShowContext(!showContext);
  }, [showContext, setShowContext]);

  const closeContext = useCallback(() => {
    setShowContext(false);
  }, [setShowContext]);

  // Reset chat state
  const resetChat = useCallback(() => {
    reset();
  }, [reset]);

  return {
    // State
    conversations,
    currentConversation,
    currentConversationId,
    messages,
    isLoading,
    isStreaming,
    error,
    currentContext,
    showContext,

    // Computed state
    hasMessages,
    hasConversations,
    canRegenerate,

    // Actions
    sendStreamMessage,
    startNewConversation,
    switchToConversation,
    removeConversation,
    clearAllConversations,
    regenerateResponse,

    // Context management
    toggleContext,
    closeContext,

    // Utils
    resetChat,
  };
};

export default useStreamChat;
