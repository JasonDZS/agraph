import { useState, useCallback } from 'react';
import { message } from 'antd';
import { documentService } from '../../../services';
import { useDocumentStore } from '../../../store/documentStore';
import type {
  DocumentUploadProgress,
  UploadOptions,
  DocumentUploadItem,
} from '../types';

export const useDocumentUpload = () => {
  const [uploadProgress, setUploadProgress] = useState<
    DocumentUploadProgress[]
  >([]);
  const [uploading, setUploading] = useState(false);

  const { addDocument } = useDocumentStore();

  // Add files to upload queue
  const addFiles = useCallback((files: File[]) => {
    const newProgress: DocumentUploadProgress[] = files.map(file => ({
      file,
      progress: 0,
      status: 'pending',
    }));

    setUploadProgress(prev => [...prev, ...newProgress]);
    return newProgress;
  }, []);

  // Remove file from upload queue
  const removeFile = useCallback((filename: string) => {
    setUploadProgress(prev => prev.filter(item => item.file.name !== filename));
  }, []);

  // Clear completed uploads
  const clearCompleted = useCallback(() => {
    setUploadProgress(prev => prev.filter(item => item.status !== 'success'));
  }, []);

  // Clear all uploads
  const clearAll = useCallback(() => {
    setUploadProgress([]);
  }, []);

  // Update progress for specific file
  const updateProgress = useCallback(
    (filename: string, updates: Partial<DocumentUploadProgress>) => {
      setUploadProgress(prev =>
        prev.map(item =>
          item.file.name === filename ? { ...item, ...updates } : item
        )
      );
    },
    []
  );

  // Upload single file
  const uploadSingleFile = useCallback(
    async (
      file: File,
      options: UploadOptions = {}
    ): Promise<DocumentUploadItem | null> => {
      try {
        // Update status to uploading
        updateProgress(file.name, { status: 'uploading', progress: 0 });

        // Create FormData
        const formData = new FormData();
        formData.append('files', file);

        if (options.metadata) {
          formData.append('metadata', JSON.stringify(options.metadata));
        }

        if (options.tags) {
          formData.append('tags', JSON.stringify(options.tags));
        }

        // Simulate progress updates (since we can't get real progress from fetch)
        const progressInterval = setInterval(() => {
          updateProgress(file.name, (prev: any) => ({
            progress: Math.min(prev.progress + Math.random() * 20, 90),
          }));
        }, 200);

        try {
          const response = await documentService.uploadFiles([file], options);

          clearInterval(progressInterval);

          if (response.data.uploaded_documents.length > 0) {
            const uploadedDoc = response.data.uploaded_documents[0];

            // Update progress to complete
            updateProgress(file.name, {
              status: 'success',
              progress: 100,
              result: uploadedDoc,
            });

            // Add to store
            addDocument({
              id: uploadedDoc.id,
              filename: uploadedDoc.filename,
              content: '', // Content not returned in upload response
              content_length: uploadedDoc.content_length,
              content_type: uploadedDoc.content_type,
              stored_at: new Date().toISOString(),
              project_name: options.project_name,
              metadata: options.metadata || {},
              tags: options.tags || [],
              file_size: uploadedDoc.size,
              extracted_metadata: uploadedDoc.extracted_metadata,
            });

            return uploadedDoc;
          } else {
            throw new Error('No documents returned from upload');
          }
        } catch (error) {
          clearInterval(progressInterval);
          throw error;
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Upload failed';

        updateProgress(file.name, {
          status: 'error',
          progress: 0,
          error: errorMessage,
        });

        console.error(`Failed to upload ${file.name}:`, error);
        throw error;
      }
    },
    [updateProgress, addDocument]
  );

  // Upload multiple files
  const uploadFiles = useCallback(
    async (
      files: File[],
      options: UploadOptions = {}
    ): Promise<DocumentUploadItem[]> => {
      if (files.length === 0) {
        message.warning('没有选择文件');
        return [];
      }

      setUploading(true);

      try {
        // Add files to queue
        addFiles(files);

        const results: DocumentUploadItem[] = [];
        const errors: string[] = [];

        // Upload files sequentially to avoid overwhelming the server
        for (const file of files) {
          try {
            const result = await uploadSingleFile(file, options);
            if (result) {
              results.push(result);
            }
          } catch (error) {
            const errorMessage =
              error instanceof Error ? error.message : 'Unknown error';
            errors.push(`${file.name}: ${errorMessage}`);
          }
        }

        // Show summary message
        if (results.length > 0) {
          message.success(`成功上传 ${results.length} 个文件`);
        }

        if (errors.length > 0) {
          message.error(`${errors.length} 个文件上传失败`);
          console.error('Upload errors:', errors);
        }

        return results;
      } catch (error) {
        console.error('Batch upload failed:', error);
        message.error('批量上传失败');
        return [];
      } finally {
        setUploading(false);
      }
    },
    [addFiles, uploadSingleFile]
  );

  // Upload text content directly
  const uploadText = useCallback(
    async (
      content: string,
      filename: string,
      options: UploadOptions = {}
    ): Promise<DocumentUploadItem | null> => {
      try {
        setUploading(true);

        const response = await documentService.uploadTexts([content], options);

        if (response.data.uploaded_documents.length > 0) {
          const uploadedDoc = response.data.uploaded_documents[0];

          // Add to store
          addDocument({
            id: uploadedDoc.id,
            filename: uploadedDoc.filename,
            content: content,
            content_length: uploadedDoc.content_length,
            content_type: 'text/plain',
            stored_at: new Date().toISOString(),
            project_name: options.project_name,
            metadata: options.metadata || {},
            tags: options.tags || [],
          });

          message.success('文本内容上传成功');
          return uploadedDoc;
        } else {
          throw new Error('No documents returned from text upload');
        }
      } catch (error) {
        console.error('Failed to upload text:', error);
        message.error('文本上传失败');
        return null;
      } finally {
        setUploading(false);
      }
    },
    [addDocument]
  );

  // Get upload statistics
  const getUploadStats = useCallback(() => {
    const total = uploadProgress.length;
    const completed = uploadProgress.filter(p => p.status === 'success').length;
    const failed = uploadProgress.filter(p => p.status === 'error').length;
    const pending = uploadProgress.filter(p => p.status === 'pending').length;
    const uploading_count = uploadProgress.filter(
      p => p.status === 'uploading'
    ).length;

    const avgProgress =
      total > 0
        ? Math.round(
            uploadProgress.reduce((sum, p) => sum + p.progress, 0) / total
          )
        : 0;

    return {
      total,
      completed,
      failed,
      pending,
      uploading: uploading_count,
      avgProgress,
      isComplete: total > 0 && completed + failed === total,
    };
  }, [uploadProgress]);

  return {
    uploadProgress,
    uploading,
    uploadFiles,
    uploadSingleFile,
    uploadText,
    addFiles,
    removeFile,
    clearCompleted,
    clearAll,
    getUploadStats,
  };
};
