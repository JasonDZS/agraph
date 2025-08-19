import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  ChatMessage,
  ChatConversation,
  ChatContext,
} from '@/modules/Chat/types';
import { chatService } from '@/services/chatService';
import { generateId } from '@/utils/helpers';

interface ChatState {
  // Conversations
  conversations: ChatConversation[];
  currentConversationId: string | null;

  // Messages
  messages: ChatMessage[];

  // UI State
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;

  // Context
  currentContext: ChatContext | null;
  showContext: boolean;

  // Actions
  setConversations: (conversations: ChatConversation[]) => void;
  setCurrentConversation: (conversationId: string | null) => void;
  addMessage: (message: ChatMessage) => void;
  updateLastMessage: (content: string) => void;
  setMessages: (messages: ChatMessage[]) => void;
  clearMessages: () => void;

  // Loading states
  setLoading: (loading: boolean) => void;
  setStreaming: (streaming: boolean) => void;
  setError: (error: string | null) => void;

  // Context
  setCurrentContext: (context: ChatContext | null) => void;
  setShowContext: (show: boolean) => void;

  // Conversation management
  createNewConversation: (title?: string) => Promise<string>;
  loadConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  clearConversationHistory: () => Promise<void>;

  // Message operations
  sendMessage: (
    content: string,
    options?: {
      entity_top_k?: number;
      relation_top_k?: number;
      text_chunk_top_k?: number;
      response_type?: string;
    }
  ) => Promise<void>;

  regenerateLastResponse: () => Promise<void>;

  // Utilities
  reset: () => void;
}

const initialState = {
  conversations: [],
  currentConversationId: null,
  messages: [],
  isLoading: false,
  isStreaming: false,
  error: null,
  currentContext: null,
  showContext: true,
};

export const useChatStore = create<ChatState>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Basic setters
        setConversations: conversations =>
          set({ conversations }, false, 'setConversations'),

        setCurrentConversation: conversationId =>
          set(
            { currentConversationId: conversationId },
            false,
            'setCurrentConversation'
          ),

        addMessage: message =>
          set(
            state => ({
              messages: [...state.messages, message],
            }),
            false,
            'addMessage'
          ),

        updateLastMessage: content =>
          set(
            state => {
              const messages = [...state.messages];
              if (messages.length > 0) {
                messages[messages.length - 1] = {
                  ...messages[messages.length - 1],
                  content,
                };
              }
              return { messages };
            },
            false,
            'updateLastMessage'
          ),

        setMessages: messages => set({ messages }, false, 'setMessages'),

        clearMessages: () => set({ messages: [] }, false, 'clearMessages'),

        // Loading states
        setLoading: isLoading => set({ isLoading }, false, 'setLoading'),

        setStreaming: isStreaming =>
          set({ isStreaming }, false, 'setStreaming'),

        setError: error => set({ error }, false, 'setError'),

        // Context
        setCurrentContext: currentContext =>
          set({ currentContext }, false, 'setCurrentContext'),

        setShowContext: showContext =>
          set({ showContext }, false, 'setShowContext'),

        // Conversation management
        createNewConversation: async title => {
          const conversationId = generateId();
          const newConversation: ChatConversation = {
            id: conversationId,
            title: title || 'New Chat',
            messages: [],
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          };

          set(
            state => ({
              conversations: [newConversation, ...state.conversations],
              currentConversationId: conversationId,
              messages: [],
              currentContext: null,
              error: null,
            }),
            false,
            'createNewConversation'
          );

          return conversationId;
        },

        loadConversation: async conversationId => {
          const state = get();
          const conversation = state.conversations.find(
            c => c.id === conversationId
          );

          if (conversation) {
            set(
              {
                currentConversationId: conversationId,
                messages: conversation.messages,
                currentContext: null,
                error: null,
              },
              false,
              'loadConversation'
            );
          }
        },

        deleteConversation: async conversationId => {
          set(
            state => {
              const conversations = state.conversations.filter(
                c => c.id !== conversationId
              );
              const currentConversationId =
                state.currentConversationId === conversationId
                  ? null
                  : state.currentConversationId;

              return {
                conversations,
                currentConversationId,
                messages: currentConversationId === null ? [] : state.messages,
              };
            },
            false,
            'deleteConversation'
          );
        },

        clearConversationHistory: async () => {
          set(
            {
              conversations: [],
              currentConversationId: null,
              messages: [],
              currentContext: null,
              error: null,
            },
            false,
            'clearConversationHistory'
          );
        },

        // Message operations
        sendMessage: async (content, options = {}) => {
          const state = get();

          try {
            set({ isStreaming: true, error: null }, false, 'sendMessage:start');

            // Add user message
            const userMessage: ChatMessage = {
              role: 'user',
              content,
              timestamp: new Date().toISOString(),
            };

            // Add assistant message placeholder
            const assistantMessage: ChatMessage = {
              role: 'assistant',
              content: '',
              timestamp: new Date().toISOString(),
            };

            set(
              state => ({
                messages: [...state.messages, userMessage, assistantMessage],
              }),
              false,
              'sendMessage:addMessages'
            );

            // Prepare conversation history
            const conversationHistory = chatService.formatConversationHistory([
              ...state.messages,
              userMessage,
            ]);

            // Send streaming request
            await chatService.sendStreamMessage(
              {
                question: content,
                conversation_history: conversationHistory,
                entity_top_k: options.entity_top_k || 5,
                relation_top_k: options.relation_top_k || 5,
                text_chunk_top_k: options.text_chunk_top_k || 3,
                response_type: options.response_type || 'conversational',
                stream: true,
              },
              // onChunk
              chunk => {
                const currentState = get();

                if (chunk.chunk) {
                  // Update the last message with new content
                  set(
                    state => {
                      const messages = [...state.messages];
                      if (messages.length > 0) {
                        messages[messages.length - 1] = {
                          ...messages[messages.length - 1],
                          content: chunk.partial_answer || chunk.chunk,
                        };
                      }
                      return { messages };
                    },
                    false,
                    'sendMessage:updateChunk'
                  );
                }

                if (chunk.context) {
                  // Transform context data structure from backend format to frontend format
                  const transformedContext = {
                    entities:
                      chunk.context.entities?.map(
                        (item: any) => item.entity || item
                      ) || [],
                    relations:
                      chunk.context.relations?.map(
                        (item: any) => item.relation || item
                      ) || [],
                    text_chunks:
                      chunk.context.text_chunks?.map(
                        (item: any) => item.text_chunk || item
                      ) || [],
                    reasoning: chunk.context.reasoning,
                  };

                  set(
                    {
                      currentContext: transformedContext,
                      showContext: true, // Auto-show context when available
                    },
                    false,
                    'sendMessage:updateContext'
                  );
                }

                if (chunk.finished) {
                  set(
                    state => {
                      const messages = [...state.messages];
                      if (messages.length > 0) {
                        messages[messages.length - 1] = {
                          ...messages[messages.length - 1],
                          content: chunk.answer || chunk.partial_answer || '',
                        };
                      }
                      return {
                        messages,
                        isStreaming: false,
                      };
                    },
                    false,
                    'sendMessage:finish'
                  );

                  // Update conversation
                  const finalState = get();
                  if (finalState.currentConversationId) {
                    set(
                      state => ({
                        conversations: state.conversations.map(conv =>
                          conv.id === finalState.currentConversationId
                            ? {
                                ...conv,
                                messages: finalState.messages,
                                updated_at: new Date().toISOString(),
                                title:
                                  conv.title === 'New Chat'
                                    ? chatService.generateConversationTitle(
                                        content
                                      )
                                    : conv.title,
                              }
                            : conv
                        ),
                      }),
                      false,
                      'sendMessage:updateConversation'
                    );
                  }
                }
              },
              // onError
              error => {
                console.error('Streaming chat error:', error);
                set(
                  {
                    error: `Chat error: ${error.message || error}`,
                    isStreaming: false,
                  },
                  false,
                  'sendMessage:error'
                );
              },
              // onComplete
              () => {
                set({ isStreaming: false }, false, 'sendMessage:complete');
              }
            );
          } catch (error) {
            console.error('Send message error:', error);
            set(
              {
                error: `Failed to send message: ${error.message || error}`,
                isStreaming: false,
              },
              false,
              'sendMessage:catchError'
            );
          }
        },

        regenerateLastResponse: async () => {
          const state = get();
          const messages = [...state.messages];

          if (
            messages.length < 2 ||
            messages[messages.length - 1].role !== 'assistant'
          ) {
            return;
          }

          // Remove last assistant message
          messages.pop();
          const userMessage = messages[messages.length - 1];

          if (userMessage && userMessage.role === 'user') {
            set(
              { messages },
              false,
              'regenerateLastResponse:removeLastMessage'
            );
            await get().sendMessage(userMessage.content);
          }
        },

        reset: () => set(initialState, false, 'reset'),
      }),
      {
        name: 'agraph-chat-storage',
        partialize: state => ({
          conversations: state.conversations,
          currentConversationId: state.currentConversationId,
          showContext: state.showContext,
        }),
      }
    ),
    { name: 'ChatStore' }
  )
);

// Add to window for debugging
if (typeof window !== 'undefined') {
  (window as any).__AGRAPH_CHAT_STORE__ = useChatStore;
}
