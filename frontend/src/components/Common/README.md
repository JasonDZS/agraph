# Common Components Library

A collection of reusable, high-performance common components for the AGraph frontend application.

## Components Overview

### 🔄 LoadingSpinner
A flexible loading spinner component with various configurations.

**Features:**
- Multiple sizes (small, default, large)
- Custom indicators support
- Wrapper and standalone modes
- Customizable delay and styling

**Basic Usage:**
```typescript
import { LoadingSpinner } from '@/components/Common';

// Standalone spinner
<LoadingSpinner size="large" tip="Loading data..." />

// Wrapper mode
<LoadingSpinner spinning={loading}>
  <YourComponent />
</LoadingSpinner>
```

### 🛡️ ErrorBoundary
Enhanced error boundary component with multiple display modes and error reporting.

**Features:**
- Page-level and component-level error handling
- Error reporting functionality
- Development-mode detailed error display
- Multiple recovery options
- Custom error handlers

**Basic Usage:**
```typescript
import { ErrorBoundary } from '@/components/Common';

// Page-level error boundary
<ErrorBoundary
  level="page"
  onError={(error, errorInfo) => console.log('Page error:', error)}
>
  <App />
</ErrorBoundary>

// Component-level error boundary
<ErrorBoundary
  level="component"
  showReportButton={true}
>
  <RiskyComponent />
</ErrorBoundary>
```

### ✅ ConfirmModal
Advanced confirmation dialog with typed confirmations and various alert levels.

**Features:**
- Multiple confirmation types (info, success, warning, error, danger)
- Typed confirmation requirements
- Custom actions and buttons
- Promise-based API
- Danger mode with additional warnings

**Basic Usage:**
```typescript
import { ConfirmModal, confirm } from '@/components/Common';

// Component usage
<ConfirmModal
  open={showConfirm}
  title="Delete Project"
  content="Are you sure you want to delete this project? This action cannot be undone."
  type="danger"
  requiresConfirmation={true}
  confirmationText="DELETE"
  onConfirm={handleDelete}
  onCancel={() => setShowConfirm(false)}
/>

// Function usage
const result = await confirm({
  title: 'Confirm Action',
  content: 'This will permanently delete the item.',
  type: 'warning',
  danger: true,
});
```

### 🔔 NotificationCenter
Comprehensive notification system with persistent storage and management.

**Features:**
- Multiple notification types
- Persistent notification history
- Unread count badges
- Custom actions per notification
- Drawer-based notification center
- System notification integration

**Basic Usage:**
```typescript
import { NotificationCenter, notify } from '@/components/Common';

// Add to your layout
<NotificationCenter
  showBadge={true}
  maxItems={50}
  onNotificationClick={(notification) => {
    console.log('Clicked:', notification);
  }}
/>

// Send notifications
notify.success('Success!', 'Operation completed successfully');
notify.error('Error', 'Something went wrong');
notify.warning('Warning', 'Please check your input');
notify.info('Info', 'New update available');

// Custom notification with actions
notify.info('Update Available', 'A new version is ready', {
  persistent: true,
  actions: [
    {
      label: 'Update Now',
      onClick: () => handleUpdate(),
      type: 'primary'
    },
    {
      label: 'Later',
      onClick: () => handleLater(),
    }
  ]
});
```

### 📜 VirtualList
High-performance virtualized list component for large datasets.

**Features:**
- Fixed and dynamic item heights
- Smooth scrolling with overscan
- Scroll-to-index functionality
- End-reached callbacks for infinite scroll
- Loading states and empty states
- Memory efficient for large lists

**Basic Usage:**
```typescript
import { VirtualList, FixedHeightVirtualList } from '@/components/Common';

// Fixed height items
<FixedHeightVirtualList
  items={largeDataset}
  itemHeight={50}
  height={400}
  renderItem={(item, index) => (
    <div key={index} style={{ padding: '8px 16px' }}>
      {item.name}
    </div>
  )}
  onEndReached={() => loadMoreData()}
  onEndReachedThreshold={0.8}
  loading={loadingMore}
/>

// Dynamic height items
<VirtualList
  items={variableHeightItems}
  itemHeight={(item, index) => item.height || 60}
  height={500}
  renderItem={(item, index) => (
    <CustomItemComponent item={item} />
  )}
  scrollToIndex={selectedIndex}
  scrollToAlignment="center"
  getItemKey={(item) => item.id}
/>
```

## Advanced Examples

### Error Boundary with Custom Fallback
```typescript
const CustomErrorFallback = ({ error, retry }) => (
  <div style={{ padding: '20px', textAlign: 'center' }}>
    <h3>Oops! Something went wrong</h3>
    <p>{error.message}</p>
    <Button onClick={retry}>Try Again</Button>
  </div>
);

<ErrorBoundary fallback={<CustomErrorFallback />}>
  <ComplexComponent />
</ErrorBoundary>
```

### Notification with Rich Content
```typescript
const richNotification = {
  title: 'Processing Complete',
  message: 'Your data has been processed successfully',
  type: 'success' as const,
  actions: [
    {
      label: 'View Results',
      onClick: () => navigateToResults(),
      type: 'primary'
    },
    {
      label: 'Download',
      onClick: () => downloadResults(),
    }
  ],
  data: { resultId: 'abc123' } // Custom data
};

notificationManager.add(richNotification);
```

### Virtual List with Complex Items
```typescript
interface ListItem {
  id: string;
  title: string;
  content: string;
  type: 'text' | 'image' | 'video';
  height?: number;
}

const ComplexListItem = ({ item, index }: { item: ListItem; index: number }) => (
  <Card
    style={{ margin: '8px', height: item.height || 100 }}
    hoverable
  >
    <Card.Meta
      title={item.title}
      description={item.content}
    />
    {item.type === 'image' && <img src={item.content} alt={item.title} />}
  </Card>
);

<VirtualList
  items={complexItems}
  itemHeight={(item) => item.height || 100}
  height={600}
  renderItem={(item, index) => (
    <ComplexListItem key={item.id} item={item} index={index} />
  )}
  overscan={3}
  onScroll={(scrollTop, scrollHeight, clientHeight) => {
    // Handle scroll events
    const scrollPercent = scrollTop / (scrollHeight - clientHeight);
    updateScrollProgress(scrollPercent);
  }}
/>
```

## Performance Considerations

### LoadingSpinner
- Use appropriate `delay` prop to prevent flashing for quick operations
- Prefer wrapper mode over multiple spinners for better UX

### ErrorBoundary
- Place at appropriate component tree levels
- Use component-level boundaries for non-critical components
- Implement proper error reporting in production

### NotificationCenter
- Limit `maxItems` to prevent memory issues
- Use `persistent: false` for temporary notifications
- Implement notification cleanup for long-running applications

### VirtualList
- Choose appropriate `overscan` values (3-10 items)
- Use `getItemKey` for better performance with dynamic data
- Implement proper `onEndReached` throttling for infinite scroll

## Integration with Existing Components

### With Ant Design
All components are designed to work seamlessly with Ant Design:

```typescript
// Loading states in forms
<Form>
  <LoadingSpinner spinning={submitting}>
    <Form.Item>
      <Input placeholder="Username" />
    </Form.Item>
    <Form.Item>
      <Button type="primary" htmlType="submit">
        Submit
      </Button>
    </Form.Item>
  </LoadingSpinner>
</Form>

// Error boundaries in modals
<Modal open={open} title="Complex Operation">
  <ErrorBoundary level="component">
    <ComplexModalContent />
  </ErrorBoundary>
</Modal>
```

### With Zustand Store
```typescript
// Using notifications in store actions
const useProjectStore = create((set) => ({
  deleteProject: async (id: string) => {
    try {
      await projectService.delete(id);
      notify.success('Project Deleted', 'Project has been successfully removed');
      set(state => ({
        projects: state.projects.filter(p => p.id !== id)
      }));
    } catch (error) {
      notify.error('Delete Failed', error.message);
    }
  }
}));
```

## TypeScript Support

All components come with comprehensive TypeScript definitions:

```typescript
import type {
  ConfirmModalProps,
  NotificationItem,
  NotificationType
} from '@/components/Common';

// Type-safe notification handling
const handleNotification = (notification: NotificationItem) => {
  if (notification.type === 'error') {
    // Handle error notifications
    console.error('Error notification:', notification.title);
  }
};

// Type-safe confirm modal props
const confirmProps: ConfirmModalProps = {
  open: true,
  title: 'Confirm',
  type: 'warning',
  onConfirm: async () => {
    // Type-safe async confirm handler
    await someAsyncOperation();
  }
};
```

## Best Practices

1. **Error Boundaries**: Place them at strategic points in your component tree
2. **Loading States**: Always provide loading feedback for async operations
3. **Notifications**: Use appropriate types and don't overwhelm users
4. **Virtual Lists**: Profile your list performance and adjust overscan accordingly
5. **Confirmations**: Use typed confirmations for dangerous operations

## Accessibility

All components follow WCAG 2.1 AA guidelines:
- Proper ARIA labels and roles
- Keyboard navigation support
- Screen reader compatibility
- Focus management
- Color contrast compliance
