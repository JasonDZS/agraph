import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Empty, Spin } from 'antd';
import LoadingSpinner from './LoadingSpinner';

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number | ((item: T, index: number) => number);
  height: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  overscan?: number;
  onScroll?: (scrollTop: number, scrollHeight: number, clientHeight: number) => void;
  onEndReached?: () => void;
  onEndReachedThreshold?: number;
  loading?: boolean;
  empty?: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  scrollToIndex?: number;
  scrollToAlignment?: 'auto' | 'start' | 'center' | 'end';
  getItemKey?: (item: T, index: number) => string | number;
}

interface ItemStyle {
  position: 'absolute';
  top: number;
  left: number;
  right: number;
  height: number;
}

function VirtualList<T>({
  items,
  itemHeight,
  height,
  renderItem,
  overscan = 5,
  onScroll,
  onEndReached,
  onEndReachedThreshold = 0.8,
  loading = false,
  empty,
  className,
  style,
  scrollToIndex,
  scrollToAlignment = 'auto',
  getItemKey,
}: VirtualListProps<T>) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollElementRef = useRef<HTMLDivElement>(null);
  const lastScrollTop = useRef(0);
  const isScrollingRef = useRef(false);

  const getItemHeight = useCallback((item: T, index: number): number => {
    return typeof itemHeight === 'function' ? itemHeight(item, index) : itemHeight;
  }, [itemHeight]);

  const itemPositions = useMemo(() => {
    const positions: number[] = [0];
    let totalHeight = 0;

    for (let i = 0; i < items.length; i++) {
      totalHeight += getItemHeight(items[i], i);
      positions.push(totalHeight);
    }

    return positions;
  }, [items, getItemHeight]);

  const totalHeight = itemPositions[itemPositions.length - 1] || 0;

  const visibleRange = useMemo(() => {
    if (items.length === 0) {
      return { startIndex: 0, endIndex: 0, visibleItems: [] };
    }

    let startIndex = 0;
    let endIndex = items.length - 1;

    // Binary search for start index
    let left = 0;
    let right = itemPositions.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (itemPositions[mid] <= scrollTop) {
        startIndex = mid;
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    // Binary search for end index
    const bottom = scrollTop + height;
    left = 0;
    right = itemPositions.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (itemPositions[mid] <= bottom) {
        endIndex = mid;
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    // Apply overscan
    startIndex = Math.max(0, startIndex - overscan);
    endIndex = Math.min(items.length - 1, endIndex + overscan);

    const visibleItems = items.slice(startIndex, endIndex + 1).map((item, i) => ({
      item,
      index: startIndex + i,
      key: getItemKey ? getItemKey(item, startIndex + i) : startIndex + i,
      style: {
        position: 'absolute' as const,
        top: itemPositions[startIndex + i],
        left: 0,
        right: 0,
        height: getItemHeight(item, startIndex + i),
      },
    }));

    return { startIndex, endIndex, visibleItems };
  }, [items, scrollTop, height, itemPositions, overscan, getItemHeight, getItemKey]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop: newScrollTop, scrollHeight, clientHeight } = e.currentTarget;

    setScrollTop(newScrollTop);
    lastScrollTop.current = newScrollTop;
    isScrollingRef.current = true;

    if (onScroll) {
      onScroll(newScrollTop, scrollHeight, clientHeight);
    }

    // Handle end reached
    if (onEndReached) {
      const threshold = scrollHeight * onEndReachedThreshold;
      if (newScrollTop + clientHeight >= threshold) {
        onEndReached();
      }
    }

    // Reset scrolling flag after some time
    setTimeout(() => {
      isScrollingRef.current = false;
    }, 150);
  }, [onScroll, onEndReached, onEndReachedThreshold]);

  // Handle scroll to index
  useEffect(() => {
    if (scrollToIndex !== undefined && scrollToIndex >= 0 && scrollToIndex < items.length) {
      const targetPosition = itemPositions[scrollToIndex];
      const itemHeightValue = getItemHeight(items[scrollToIndex], scrollToIndex);

      let scrollPosition = targetPosition;

      switch (scrollToAlignment) {
        case 'start':
          scrollPosition = targetPosition;
          break;
        case 'center':
          scrollPosition = targetPosition - (height - itemHeightValue) / 2;
          break;
        case 'end':
          scrollPosition = targetPosition - height + itemHeightValue;
          break;
        case 'auto':
        default:
          if (targetPosition < scrollTop) {
            scrollPosition = targetPosition;
          } else if (targetPosition + itemHeightValue > scrollTop + height) {
            scrollPosition = targetPosition - height + itemHeightValue;
          } else {
            return; // Already in view
          }
          break;
      }

      scrollPosition = Math.max(0, Math.min(scrollPosition, totalHeight - height));

      if (scrollElementRef.current) {
        scrollElementRef.current.scrollTop = scrollPosition;
      }
    }
  }, [scrollToIndex, scrollToAlignment, itemPositions, getItemHeight, items, height, totalHeight, scrollTop]);

  if (items.length === 0 && !loading) {
    return (
      <div style={{ height, ...style }} className={className}>
        {empty || <Empty description="No data" />}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        height,
        overflow: 'hidden',
        position: 'relative',
        ...style,
      }}
    >
      <div
        ref={scrollElementRef}
        style={{
          height: '100%',
          overflow: 'auto',
          position: 'relative',
        }}
        onScroll={handleScroll}
      >
        <div
          style={{
            height: totalHeight,
            position: 'relative',
          }}
        >
          {visibleRange.visibleItems.map(({ item, index, key, style: itemStyle }) => (
            <div key={key} style={itemStyle}>
              {renderItem(item, index)}
            </div>
          ))}
        </div>
      </div>

      {loading && (
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            padding: '16px',
            textAlign: 'center',
            background: 'rgba(255, 255, 255, 0.9)',
            borderTop: '1px solid #f0f0f0',
          }}
        >
          <LoadingSpinner size="small" tip="Loading more..." />
        </div>
      )}
    </div>
  );
}

export default VirtualList;

// Helper component for fixed height items
export const FixedHeightVirtualList = <T,>(props: Omit<VirtualListProps<T>, 'itemHeight'> & { itemHeight: number }) => (
  <VirtualList {...props} />
);

// Helper component for dynamic height items
export const DynamicHeightVirtualList = <T,>(props: Omit<VirtualListProps<T>, 'itemHeight'> & {
  itemHeight: (item: T, index: number) => number
}) => (
  <VirtualList {...props} />
);
