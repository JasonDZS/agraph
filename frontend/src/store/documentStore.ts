import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { Document } from '../types/api';

export interface UploadProgress {
  id: string;
  filename: string;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  startTime: number;
  endTime?: number;
}

export interface DocumentState {
  // Document List
  documents: Document[];
  documentsLoading: boolean;
  documentsError: string | null;

  // Pagination
  currentPage: number;
  pageSize: number;
  totalDocuments: number;
  hasNextPage: boolean;

  // Filters and Search
  searchQuery: string;
  tagFilter: string[];
  sortBy: 'title' | 'created_at' | 'updated_at' | 'file_size';
  sortOrder: 'asc' | 'desc';

  // Selected Documents
  selectedDocuments: string[];
  selectAll: boolean;

  // Upload State
  uploadProgress: UploadProgress[];
  isUploading: boolean;

  // Document Operations
  isDeleting: boolean;
  isProcessing: boolean;

  // UI State
  showUploadModal: boolean;
  showDeleteConfirmModal: boolean;
  documentsToDelete: string[];

  // Document Viewer
  viewingDocument: Document | null;
  showDocumentViewer: boolean;
}

export interface DocumentActions {
  // Document List Actions
  setDocuments: (documents: Document[]) => void;
  addDocument: (document: Document) => void;
  updateDocument: (documentId: string, updates: Partial<Document>) => void;
  removeDocuments: (documentIds: string[]) => void;
  setDocumentsLoading: (loading: boolean) => void;
  setDocumentsError: (error: string | null) => void;

  // Pagination Actions
  setCurrentPage: (page: number) => void;
  setPageSize: (size: number) => void;
  setTotalDocuments: (total: number) => void;
  setHasNextPage: (hasNext: boolean) => void;

  // Filter and Search Actions
  setSearchQuery: (query: string) => void;
  setTagFilter: (tags: string[]) => void;
  setSortBy: (sortBy: DocumentState['sortBy']) => void;
  setSortOrder: (order: DocumentState['sortOrder']) => void;

  // Selection Actions
  setSelectedDocuments: (ids: string[]) => void;
  toggleDocumentSelection: (id: string) => void;
  selectAllDocuments: () => void;
  clearSelection: () => void;
  setSelectAll: (selectAll: boolean) => void;

  // Upload Actions
  addUploadProgress: (progress: UploadProgress) => void;
  updateUploadProgress: (id: string, updates: Partial<UploadProgress>) => void;
  removeUploadProgress: (id: string) => void;
  clearUploadProgress: () => void;
  setIsUploading: (uploading: boolean) => void;

  // Document Operations
  setIsDeleting: (deleting: boolean) => void;
  setIsProcessing: (processing: boolean) => void;

  // UI Actions
  setShowUploadModal: (show: boolean) => void;
  setShowDeleteConfirmModal: (show: boolean) => void;
  setDocumentsToDelete: (ids: string[]) => void;
  setViewingDocument: (document: Document | null) => void;
  setShowDocumentViewer: (show: boolean) => void;

  // Computed Values
  getFilteredDocuments: () => Document[];
  getSelectedDocumentsData: () => Document[];
  getUploadStats: () => {
    total: number;
    completed: number;
    failed: number;
    inProgress: number;
  };
  getAllTags: () => string[];

  // Reset Actions
  reset: () => void;
  resetOperationStates: () => void;
}

export type DocumentStore = DocumentState & DocumentActions;

const initialState: DocumentState = {
  documents: [],
  documentsLoading: false,
  documentsError: null,
  currentPage: 1,
  pageSize: 20,
  totalDocuments: 0,
  hasNextPage: false,
  searchQuery: '',
  tagFilter: [],
  sortBy: 'updated_at',
  sortOrder: 'desc',
  selectedDocuments: [],
  selectAll: false,
  uploadProgress: [],
  isUploading: false,
  isDeleting: false,
  isProcessing: false,
  showUploadModal: false,
  showDeleteConfirmModal: false,
  documentsToDelete: [],
  viewingDocument: null,
  showDocumentViewer: false,
};

export const useDocumentStore = create<DocumentStore>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Document List Actions
        setDocuments: documents => {
          set({ documents, documentsError: null });
        },

        addDocument: document => {
          set(state => ({
            documents: [document, ...state.documents],
            totalDocuments: state.totalDocuments + 1,
          }));
        },

        updateDocument: (documentId, updates) => {
          set(state => ({
            documents: state.documents.map(doc =>
              doc.id === documentId ? { ...doc, ...updates } : doc
            ),
            viewingDocument:
              state.viewingDocument?.id === documentId
                ? { ...state.viewingDocument, ...updates }
                : state.viewingDocument,
          }));
        },

        removeDocuments: documentIds => {
          set(state => ({
            documents: state.documents.filter(
              doc => !documentIds.includes(doc.id)
            ),
            selectedDocuments: state.selectedDocuments.filter(
              id => !documentIds.includes(id)
            ),
            totalDocuments: Math.max(
              0,
              state.totalDocuments - documentIds.length
            ),
            viewingDocument: documentIds.includes(
              state.viewingDocument?.id || ''
            )
              ? null
              : state.viewingDocument,
          }));
        },

        setDocumentsLoading: documentsLoading => {
          set({ documentsLoading });
        },

        setDocumentsError: documentsError => {
          set({ documentsError, documentsLoading: false });
        },

        // Pagination Actions
        setCurrentPage: currentPage => {
          set({ currentPage });
        },

        setPageSize: pageSize => {
          set({ pageSize, currentPage: 1 });
        },

        setTotalDocuments: totalDocuments => {
          set({ totalDocuments });
        },

        setHasNextPage: hasNextPage => {
          set({ hasNextPage });
        },

        // Filter and Search Actions
        setSearchQuery: searchQuery => {
          set({ searchQuery, currentPage: 1 });
        },

        setTagFilter: tagFilter => {
          set({ tagFilter, currentPage: 1 });
        },

        setSortBy: sortBy => {
          set({ sortBy });
        },

        setSortOrder: sortOrder => {
          set({ sortOrder });
        },

        // Selection Actions
        setSelectedDocuments: selectedDocuments => {
          set({ selectedDocuments, selectAll: false });
        },

        toggleDocumentSelection: id => {
          set(state => {
            const isSelected = state.selectedDocuments.includes(id);
            const selectedDocuments = isSelected
              ? state.selectedDocuments.filter(docId => docId !== id)
              : [...state.selectedDocuments, id];

            return {
              selectedDocuments,
              selectAll: selectedDocuments.length === state.documents.length,
            };
          });
        },

        selectAllDocuments: () => {
          set(state => ({
            selectedDocuments: state.documents.map(doc => doc.id),
            selectAll: true,
          }));
        },

        clearSelection: () => {
          set({ selectedDocuments: [], selectAll: false });
        },

        setSelectAll: selectAll => {
          set(state => ({
            selectAll,
            selectedDocuments: selectAll
              ? state.documents.map(doc => doc.id)
              : [],
          }));
        },

        // Upload Actions
        addUploadProgress: progress => {
          set(state => ({
            uploadProgress: [...state.uploadProgress, progress],
          }));
        },

        updateUploadProgress: (id, updates) => {
          set(state => ({
            uploadProgress: state.uploadProgress.map(progress =>
              progress.id === id ? { ...progress, ...updates } : progress
            ),
          }));
        },

        removeUploadProgress: id => {
          set(state => ({
            uploadProgress: state.uploadProgress.filter(
              progress => progress.id !== id
            ),
          }));
        },

        clearUploadProgress: () => {
          set({ uploadProgress: [] });
        },

        setIsUploading: isUploading => {
          set({ isUploading });
        },

        // Document Operations
        setIsDeleting: isDeleting => {
          set({ isDeleting });
        },

        setIsProcessing: isProcessing => {
          set({ isProcessing });
        },

        // UI Actions
        setShowUploadModal: showUploadModal => {
          set({ showUploadModal });
        },

        setShowDeleteConfirmModal: showDeleteConfirmModal => {
          set({ showDeleteConfirmModal });
        },

        setDocumentsToDelete: documentsToDelete => {
          set({ documentsToDelete });
        },

        setViewingDocument: viewingDocument => {
          set({ viewingDocument });
        },

        setShowDocumentViewer: showDocumentViewer => {
          set({ showDocumentViewer });
        },

        // Computed Values
        getFilteredDocuments: () => {
          const { documents, searchQuery, tagFilter, sortBy, sortOrder } =
            get();

          let filtered = documents;

          // Apply search filter
          if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(
              doc =>
                doc.title?.toLowerCase().includes(query) ||
                doc.content.toLowerCase().includes(query) ||
                doc.source?.toLowerCase().includes(query)
            );
          }

          // Apply tag filter
          if (tagFilter.length > 0) {
            filtered = filtered.filter(doc =>
              doc.tags?.some(tag => tagFilter.includes(tag))
            );
          }

          // Apply sorting
          filtered.sort((a, b) => {
            let aValue: any, bValue: any;

            switch (sortBy) {
              case 'title':
                aValue = (a.title || '').toLowerCase();
                bValue = (b.title || '').toLowerCase();
                break;
              case 'created_at':
                aValue = new Date(a.created_at || 0).getTime();
                bValue = new Date(b.created_at || 0).getTime();
                break;
              case 'updated_at':
                aValue = new Date(a.updated_at || 0).getTime();
                bValue = new Date(b.updated_at || 0).getTime();
                break;
              case 'file_size':
                aValue = a.file_size || 0;
                bValue = b.file_size || 0;
                break;
              default:
                return 0;
            }

            if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
            return 0;
          });

          return filtered;
        },

        getSelectedDocumentsData: () => {
          const { documents, selectedDocuments } = get();
          return documents.filter(doc => selectedDocuments.includes(doc.id));
        },

        getUploadStats: () => {
          const { uploadProgress } = get();
          return uploadProgress.reduce(
            (stats, progress) => {
              stats.total++;
              switch (progress.status) {
                case 'completed':
                  stats.completed++;
                  break;
                case 'error':
                  stats.failed++;
                  break;
                case 'uploading':
                case 'processing':
                  stats.inProgress++;
                  break;
              }
              return stats;
            },
            { total: 0, completed: 0, failed: 0, inProgress: 0 }
          );
        },

        getAllTags: () => {
          const { documents } = get();
          const tagSet = new Set<string>();
          documents.forEach(doc => {
            doc.tags?.forEach(tag => tagSet.add(tag));
          });
          return Array.from(tagSet).sort();
        },

        // Reset Actions
        reset: () => {
          set(initialState);
        },

        resetOperationStates: () => {
          set({
            documentsLoading: false,
            isUploading: false,
            isDeleting: false,
            isProcessing: false,
            documentsError: null,
          });
        },
      }),
      {
        name: 'agraph-document-store',
        partialize: state => ({
          currentPage: state.currentPage,
          pageSize: state.pageSize,
          searchQuery: state.searchQuery,
          tagFilter: state.tagFilter,
          sortBy: state.sortBy,
          sortOrder: state.sortOrder,
        }),
      }
    ),
    {
      name: 'DocumentStore',
    }
  )
);

// Document state change listeners
let documentListeners: ((documents: Document[]) => void)[] = [];
let selectionListeners: ((selectedIds: string[]) => void)[] = [];
let uploadListeners: ((progress: UploadProgress[]) => void)[] = [];

export const subscribeToDocuments = (
  callback: (documents: Document[]) => void
) => {
  documentListeners.push(callback);
  return () => {
    documentListeners = documentListeners.filter(
      listener => listener !== callback
    );
  };
};

export const subscribeToSelection = (
  callback: (selectedIds: string[]) => void
) => {
  selectionListeners.push(callback);
  return () => {
    selectionListeners = selectionListeners.filter(
      listener => listener !== callback
    );
  };
};

export const subscribeToUploadProgress = (
  callback: (progress: UploadProgress[]) => void
) => {
  uploadListeners.push(callback);
  return () => {
    uploadListeners = uploadListeners.filter(listener => listener !== callback);
  };
};

// Subscribe to store changes to notify listeners
useDocumentStore.subscribe((state, prevState) => {
  if (state.documents !== prevState.documents) {
    documentListeners.forEach(listener => listener(state.documents));
  }

  if (state.selectedDocuments !== prevState.selectedDocuments) {
    selectionListeners.forEach(listener => listener(state.selectedDocuments));
  }

  if (state.uploadProgress !== prevState.uploadProgress) {
    uploadListeners.forEach(listener => listener(state.uploadProgress));
  }
});
