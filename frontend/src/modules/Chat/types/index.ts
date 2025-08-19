import {
  ChatMessage,
  ChatResponse,
  StreamChatChunk,
  Entity,
  Relation,
  TextChunk,
} from '@/types/api';

export interface ChatConversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

export interface ChatContext {
  entities: Entity[];
  relations: Relation[];
  text_chunks: TextChunk[];
  reasoning?: string;
}

export interface ChatState {
  conversations: ChatConversation[];
  currentConversationId: string | null;
  messages: ChatMessage[];
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
  currentContext: ChatContext | null;
}

export interface SendMessageOptions {
  entity_top_k?: number;
  relation_top_k?: number;
  text_chunk_top_k?: number;
  response_type?: string;
}

export interface MessageBubbleProps {
  message: ChatMessage;
  index: number;
  isStreaming?: boolean;
  context?: ChatContext;
  onRegenerate?: () => void;
  onCopy?: (content: string) => void;
  onShowContext?: (context: ChatContext) => void;
}

export interface ContextPanelProps {
  context: ChatContext | null;
  visible: boolean;
  onClose: () => void;
}

export interface ChatInterfaceProps {
  height?: number;
  showContext?: boolean;
  defaultSettings?: SendMessageOptions;
}

export {
  ChatMessage,
  ChatResponse,
  StreamChatChunk,
  Entity,
  Relation,
  TextChunk,
};
