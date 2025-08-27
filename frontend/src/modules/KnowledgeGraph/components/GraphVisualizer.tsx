import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from 'react';
import { Card, Spin, Alert, Typography, Space, Tabs, Table, Tag, InputNumber, Button, Form, Row, Col } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import * as echarts from 'echarts/core';
import {
  TooltipComponent,
  LegendComponent,
  TitleComponent,
} from 'echarts/components';
import { GraphChart } from 'echarts/charts';
import { LabelLayout } from 'echarts/features';
import { CanvasRenderer } from 'echarts/renderers';

import { useKnowledgeGraphStore } from '@/store/knowledgeGraphStore';
import { knowledgeGraphService } from '@/services/knowledgeGraphService';
import { useAppStore } from '@/store/appStore';
import ErrorBoundary from '@/components/ErrorBoundary';

import type {
  GraphData,
  GraphVisualizationOptions,
  GraphEventHandlers,
} from '../types/graph';

// 注册 ECharts 组件
echarts.use([
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  GraphChart,
  CanvasRenderer,
  LabelLayout,
]);

// Re-export the main component
export { default as KnowledgeGraphMain } from './KnowledgeGraphMain';

const { Title, Text } = Typography;

interface EChartsGraphData {
  nodes: Array<{
    id: string;
    name: string;
    category: number;
    symbolSize: number;
    value: number;
    entityType?: string;
    confidence?: number;
    properties?: any;
  }>;
  links: Array<{
    source: string;
    target: string;
    relationshipType?: string;
    confidence?: number;
    properties?: any;
  }>;
  categories: Array<{
    name: string;
    itemStyle?: {
      color?: string;
    };
  }>;
}

export interface GraphVisualizerProps {
  data?: GraphData;
  options?: GraphVisualizationOptions;
  eventHandlers?: GraphEventHandlers;
  height?: number;
  className?: string;
  style?: React.CSSProperties;
  showDataTable?: boolean;
  maxEntities?: number;
  maxRelations?: number;
}

const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  data,
  options = {},
  eventHandlers = {},
  height = 600,
  className,
  style,
  showDataTable = true,
  maxEntities = 500,
  maxRelations = 750,
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const isMountedRef = useRef(true);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [graphData, setGraphData] = useState<EChartsGraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [selectedEdge, setSelectedEdge] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<string>('nodes');
  const [inputMaxEntities, setInputMaxEntities] = useState<number>(maxEntities);
  const [inputMaxRelations, setInputMaxRelations] = useState<number>(maxRelations);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Store hooks
  const {
    currentGraph,
    entities,
    relations,
    textChunks,
    visualState,
    currentLayout,
    showNodeLabels,
    showEdgeLabels,
    nodeSize,
    edgeWidth,
    confidenceThreshold,
    entityTypeFilter,
    relationTypeFilter,
    setVisualState,
    setSelectedEntity,
    setSelectedRelation,
    getFilteredEntities,
    getFilteredRelations,
  } = useKnowledgeGraphStore();

  const { theme, currentProject } = useAppStore();

  // Get category colors based on entity types
  const getCategoryColors = useCallback(() => {
    const entityTypes = [...new Set(entities.map(e => e.entity_type))];
    const colors = [
      '#5470c6',
      '#91cc75',
      '#fac858',
      '#ee6666',
      '#73c0de',
      '#3ba272',
      '#fc8452',
      '#9a60b4',
      '#ea7ccc',
    ];

    return entityTypes.map((type, index) => ({
      name: type,
      itemStyle: { color: colors[index % colors.length] },
    }));
  }, [entities]);

  // Convert data to ECharts format
  const convertToEChartsData = useCallback(
    (graphData?: GraphData): EChartsGraphData | null => {
      let entitiesToUse, relationsToUse;

      if (!graphData) {
        // Use store data if no data prop provided
        entitiesToUse = getFilteredEntities();
        relationsToUse = getFilteredRelations();
      } else {
        // Convert GraphData to entity/relation format
        entitiesToUse = graphData.nodes.map(node => ({
          id: node.id,
          name: node.label,
          entity_type: node.entityType || 'concept',
          confidence: node.confidence || 1.0,
          properties: node.properties || {},
          description: '',
          aliases: [],
        }));

        relationsToUse = graphData.edges.map(edge => ({
          id: edge.id,
          head_entity_id: edge.source,
          tail_entity_id: edge.target,
          relation_type: edge.relationType || 'related',
          confidence: edge.confidence || 1.0,
          properties: edge.properties || {},
          description: '',
        }));
      }

      if (entitiesToUse.length === 0) {
        return null;
      }

      // Get unique entity types for categories
      const entityTypes = [...new Set(entitiesToUse.map(e => e.entity_type))];
      const categories = getCategoryColors();

      console.log('Converting entities to nodes:', {
        entitiesCount: entitiesToUse.length,
        entityTypes,
        categoriesCount: categories.length,
      });

      // Create nodes
      const nodes = entitiesToUse.map(entity => {
        const categoryIndex = entityTypes.indexOf(entity.entity_type);
        return {
          id: entity.id,
          name: entity.name,
          category: categoryIndex,
          symbolSize: Math.min(
            Math.max((entity.confidence || 0.5) * 30, 15),
            50
          ),
          value: entity.confidence || 0.5,
          entityType: entity.entity_type,
          confidence: entity.confidence,
          properties: entity.properties,
        };
      });

      // Create links
      const links = relationsToUse.map(relation => ({
        source: relation.head_entity_id,
        target: relation.tail_entity_id,
        relationshipType: relation.relation_type,
        confidence: relation.confidence,
        properties: relation.properties,
      }));

      console.log('Created nodes and links:', {
        nodesCount: nodes.length,
        linksCount: links.length,
        sampleNode: nodes[0],
        sampleLink: links[0],
      });

      const result = {
        nodes,
        links,
        categories,
      };

      console.log('convertToEChartsData result:', result);
      return result;
    },
    [getFilteredEntities, getFilteredRelations, getCategoryColors]
  );

  // Initialize ECharts
  const initializeChart = useCallback(() => {
    if (!chartRef.current) {
      return;
    }

    if (chartInstance.current) {
      return;
    }

    try {
      chartInstance.current = echarts.init(chartRef.current);

      // Resize chart when window resizes
      const handleResize = () => {
        if (chartInstance.current) {
          chartInstance.current.resize();
        }
      };
      window.addEventListener('resize', handleResize);

      setupEventHandlers();
      setIsInitialized(true);
      setError(null);


      return () => {
        window.removeEventListener('resize', handleResize);
      };
    } catch (err) {
      console.error('Failed to initialize ECharts:', err);
      setError(
        `Failed to initialize graph visualization: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }, []);

  // Setup event handlers
  const setupEventHandlers = useCallback(() => {
    if (!chartInstance.current) return;

    // Add click event listeners
    chartInstance.current.off('click');
    chartInstance.current.on('click', function (params: any) {
      if (params.dataType === 'node') {
        const node = params.data;
        setSelectedNode(node);
        setSelectedEdge(null);
        setActiveTab('nodes');

        // Find and set the selected entity
        const entity = entities.find(e => e.id === node.id);
        if (entity) {
          setSelectedEntity(entity);
        }

        eventHandlers.onNodeSelect?.(node);
        eventHandlers.onNodeClick?.(node);
      } else if (params.dataType === 'edge') {
        const edge = params.data;
        setSelectedEdge(edge);
        setSelectedNode(null);
        setActiveTab('edges');

        // Find and set the selected relation
        const relation = relations.find(
          r =>
            r.head_entity_id === edge.source && r.tail_entity_id === edge.target
        );
        if (relation) {
          setSelectedRelation(relation);
        }

        eventHandlers.onEdgeSelect?.(edge);
        eventHandlers.onEdgeClick?.(edge);
      } else {
        // Background click
        setSelectedNode(null);
        setSelectedEdge(null);
        setSelectedEntity(null);
        setSelectedRelation(null);
        eventHandlers.onBackgroundClick?.();
      }
    });
  }, [
    entities,
    relations,
    setSelectedEntity,
    setSelectedRelation,
    eventHandlers,
  ]);

  // Render graph
  const renderGraph = useCallback(() => {
    if (!chartInstance.current || !graphData) {
      console.log('renderGraph: Missing chart instance or graph data', {
        hasChartInstance: !!chartInstance.current,
        hasGraphData: !!graphData,
      });
      return;
    }

    console.log('renderGraph: Rendering graph with data', {
      nodes: graphData.nodes?.length,
      links: graphData.links?.length,
      categories: graphData.categories?.length,
      sampleNode: graphData.nodes?.[0],
      sampleLink: graphData.links?.[0],
    });

    if (!graphData.nodes?.length) {
      console.warn('No nodes found in graph data');
      return;
    }

    const option = {
      title: {
        text: '知识图谱',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
          color: theme === 'dark' ? '#ffffff' : '#1e293b',
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params: any) {
          console.log('Tooltip params:', params);
          if (params.dataType === 'node') {
            return `<strong>${params.data.name}</strong><br/>
                    类别: ${graphData.categories[params.data.category]?.name || 'Unknown'}<br/>
                    置信度: ${(params.data.confidence || 0).toFixed(2)}<br/>
                    节点大小: ${params.data.symbolSize || 'N/A'}`;
          } else if (params.dataType === 'edge') {
            return `${params.data.source} → ${params.data.target}<br/>
                    关系类型: ${params.data.relationshipType || '关系'}<br/>
                    置信度: ${(params.data.confidence || 0).toFixed(2)}`;
          }
          return params.name;
        },
      },
      legend: {
        data: graphData.categories.map(cat => cat.name),
        top: 'bottom',
        left: 'center',
        textStyle: {
          color: theme === 'dark' ? '#ffffff' : '#374151',
        },
      },
      series: [
        {
          name: '知识图谱',
          type: 'graph',
          layout: 'force',
          data: graphData.nodes,
          links: graphData.links,
          categories: graphData.categories,
          roam: true,
          focusNodeAdjacency: true,
          force: {
            repulsion: 200,
            gravity: 0.05,
            edgeLength: 80,
            layoutAnimation: true,
          },
          label: {
            show: showNodeLabels !== false,
            position: 'right',
            formatter: '{b}',
            color: theme === 'dark' ? '#ffffff' : '#374151',
          },
          labelLayout: {
            hideOverlap: true,
          },
          scaleLimit: {
            min: 0.1,
            max: 10,
          },
          lineStyle: {
            color: 'source',
            curveness: 0.3,
            opacity: 0.7,
            width: edgeWidth || 2,
          },
          emphasis: {
            focus: 'adjacency',
            lineStyle: {
              width: 4,
            },
          },
          itemStyle: {
            borderColor: '#fff',
            borderWidth: 1,
          },
        },
      ],
      backgroundColor: theme === 'dark' ? '#1f1f1f' : '#ffffff',
      animation: true,
    };

    try {
      // Clear existing chart before setting new option
      chartInstance.current.clear();

      console.log('Setting ECharts option:', option);

      // Set new option with notMerge=true to completely replace
      chartInstance.current.setOption(option, true);
      console.log('ECharts setOption completed successfully');

      // Force chart resize to ensure proper display
      setTimeout(() => {
        if (chartInstance.current) {
          console.log('Resizing chart');
          chartInstance.current.resize();
        }
      }, 200);
    } catch (error) {
      console.error('Error setting ECharts option:', error);
      setError('Failed to render knowledge graph: ' + (error as Error).message);
    }
  }, [graphData, theme, showNodeLabels, edgeWidth]);

  // Update graph data
  const updateGraph = useCallback(() => {
    console.log('updateGraph called with:', {
      hasData: !!data,
      entitiesLength: entities.length,
      relationsLength: relations.length,
    });

    const newGraphData = convertToEChartsData(data);
    console.log('Converted graph data:', newGraphData);

    setGraphData(newGraphData);
    console.log('Graph data state updated');
  }, [data, convertToEChartsData, entities.length, relations.length]);

  // Load graph data from API if needed
  const loadGraphData = useCallback(async (customMaxEntities?: number, customMaxRelations?: number, forceReload = false) => {
    console.log('loadGraphData called with:', {
      customMaxEntities,
      customMaxRelations,
      forceReload,
      hasData: !!data,
      hasCurrentProject: !!currentProject
    });

    if (!currentProject) {
      console.log('loadGraphData early return: no current project');
      return;
    }

    if (!forceReload && data) {
      console.log('loadGraphData early return: has data and not force reload');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await knowledgeGraphService.getVisualizationData({
        include_clusters: false,
        include_text_chunks: true,
        max_entities: customMaxEntities || maxEntities,
        max_relations: customMaxRelations || maxRelations,
        min_confidence: confidenceThreshold || 0.5,
        entity_types:
          entityTypeFilter.length > 0 ? entityTypeFilter : undefined,
        relation_types:
          relationTypeFilter.length > 0 ? relationTypeFilter : undefined,
        use_cache: !forceReload,
      });

      if (response.success && response.data) {
        const visualData = response.data;

        const entitiesFromNodes = visualData.nodes
          .filter((node: any) => node.type === 'entity')
          .map((node: any) => ({
            id: node.id,
            name: node.label,
            entity_type: node.entityType || 'concept',
            description: '',
            confidence: node.confidence || 1.0,
            properties: node.properties || {},
            aliases: [],
          }));

        const relationsFromEdges = visualData.edges.map((edge: any) => ({
          id: edge.id,
          head_entity_id: edge.source,
          tail_entity_id: edge.target,
          relation_type: edge.relationType,
          description: '',
          confidence: edge.confidence,
          properties: edge.properties || {},
        }));

        console.log('Setting new entities and relations:', {
          entitiesCount: entitiesFromNodes.length,
          relationsCount: relationsFromEdges.length,
          textChunksCount: visualData.text_chunks?.length || 0
        });

        useKnowledgeGraphStore.getState().setEntities(entitiesFromNodes);
        useKnowledgeGraphStore.getState().setRelations(relationsFromEdges);

        // Handle text chunks if available
        if (visualData.text_chunks) {
          useKnowledgeGraphStore.getState().setTextChunks(visualData.text_chunks);
        }

        console.log('Data loaded successfully, entities and relations set in store');
      } else {
        setError(response.message || 'Failed to load graph data');
      }
    } catch (err) {
      console.error('Failed to load graph data:', err);
      setError(
        `Failed to load graph data: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setIsLoading(false);
    }
  }, [
    data,
    currentProject,
    maxEntities,
    maxRelations,
    confidenceThreshold,
    entityTypeFilter,
    relationTypeFilter,
  ]);

  // Chart control methods
  const applyLayout = useCallback(
    (layoutName?: string) => {
      // ECharts handles layout automatically with force simulation
      if (chartInstance.current && graphData) {
        renderGraph();
      }
    },
    [renderGraph, graphData]
  );

  const fitToContainer = useCallback(() => {
    if (chartInstance.current) {
      chartInstance.current.resize();
    }
  }, []);

  const centerGraph = useCallback(() => {
    // ECharts handles centering automatically
    if (chartInstance.current) {
      renderGraph();
    }
  }, [renderGraph]);

  const resetView = useCallback(() => {
    if (chartInstance.current) {
      renderGraph();
    }
  }, [renderGraph]);

  // Expose methods via ref (if needed by parent components)
  React.useImperativeHandle(
    eventHandlers.ref,
    () => ({
      applyLayout,
      fitToContainer,
      centerGraph,
      resetView,
      getChart: () => chartInstance.current,
    }),
    [applyLayout, fitToContainer, centerGraph, resetView]
  );

  // Effects
  useEffect(() => {
    isMountedRef.current = true;

    // Initialize chart when component mounts and container is available
    const timer = setTimeout(() => {
      if (chartRef.current && !chartInstance.current) {
        console.log('Delayed chart initialization');
        const cleanup = initializeChart();
        return cleanup;
      }
    }, 100);

    return () => {
      isMountedRef.current = false;
      clearTimeout(timer);
      if (chartInstance.current) {
        try {
          chartInstance.current.dispose();
        } catch (err) {
          console.warn('Error during ECharts cleanup:', err);
        } finally {
          chartInstance.current = null;
          setIsInitialized(false);
        }
      }
    };
  }, [initializeChart]);

  // Update graph when data changes
  useEffect(() => {
    console.log('Update graph effect triggered', {
      entitiesLength: entities.length,
      relationsLength: relations.length,
      hasData: !!data,
    });
    updateGraph();
  }, [entities, relations, data, updateGraph]);

  // Render graph when chart is ready and data is available
  useEffect(() => {
    if (isInitialized && chartInstance.current && graphData) {
      console.log('Rendering graph with data:', graphData);
      setTimeout(() => {
        renderGraph();
      }, 100);
    }
  }, [isInitialized, graphData, renderGraph]);

  // Refresh graph with new parameters
  const handleRefresh = useCallback(async () => {
    console.log('handleRefresh called with:', { inputMaxEntities, inputMaxRelations });
    setIsRefreshing(true);
    try {
      // Clear cache first to ensure fresh data
      knowledgeGraphService.clearCache();
      await loadGraphData(inputMaxEntities, inputMaxRelations, true);
    } finally {
      setIsRefreshing(false);
    }
  }, [inputMaxEntities, inputMaxRelations, loadGraphData]);

  // Load initial data
  useEffect(() => {
    loadGraphData();
  }, [loadGraphData]);

  // Render node table columns - 优化为紧凑布局
  const nodeColumns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
      width: '45%',
    },
    {
      title: '类别',
      dataIndex: 'entityType',
      key: 'entityType',
      width: '35%',
      render: (type: string) => {
        const color = getCategoryColors().find(c => c.name === type)?.itemStyle
          ?.color;
        return (
          <Tag color={color} size="small">
            {type}
          </Tag>
        );
      },
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: '20%',
      align: 'center' as const,
      render: (confidence: number) => (
        <span style={{ fontSize: '12px' }}>{(confidence || 0).toFixed(2)}</span>
      ),
    },
  ];

  // Helper function to get node name by ID
  const getNodeNameById = useCallback(
    (nodeId: string) => {
      const node = graphData?.nodes?.find(n => n.id === nodeId);
      return node?.name || nodeId;
    },
    [graphData]
  );

  // Render edge table columns - 优化为紧凑布局
  const edgeColumns = [
    {
      title: '源节点',
      dataIndex: 'source',
      key: 'source',
      ellipsis: true,
      width: '35%',
      render: (sourceId: string) => getNodeNameById(sourceId),
    },
    {
      title: '目标节点',
      dataIndex: 'target',
      key: 'target',
      ellipsis: true,
      width: '35%',
      render: (targetId: string) => getNodeNameById(targetId),
    },
    {
      title: '关系',
      dataIndex: 'relationshipType',
      key: 'relationshipType',
      width: '30%',
      ellipsis: true,
      render: (type: string) => (
        <Tag size="small" color="blue">
          {type || '关系'}
        </Tag>
      ),
    },
  ];

  // State for expanded text chunks
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());

  // Toggle expanded state for a text chunk
  const toggleChunkExpansion = (chunkId: string) => {
    const newExpanded = new Set(expandedChunks);
    if (newExpanded.has(chunkId)) {
      newExpanded.delete(chunkId);
    } else {
      newExpanded.add(chunkId);
    }
    setExpandedChunks(newExpanded);
  };

  // Render text chunks table columns - simplified to show only content with expand functionality
  const textChunkColumns = [
    {
      title: '文档内容',
      dataIndex: 'content',
      key: 'content',
      width: '100%',
      render: (content: string, record: any) => {
        const isExpanded = expandedChunks.has(record.id);
        const shouldTruncate = content.length > 200;
        const displayContent = isExpanded || !shouldTruncate
          ? content
          : `${content.substring(0, 200)}...`;

        return (
          <div style={{
            padding: '8px 0',
            borderBottom: '1px solid #f0f0f0',
            cursor: shouldTruncate ? 'pointer' : 'default'
          }}>
            <div
              style={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                lineHeight: '1.6',
                fontSize: '14px',
                color: '#333',
                marginBottom: shouldTruncate ? 12 : 0,
                padding: '8px 12px',
                backgroundColor: '#fafafa',
                borderRadius: '4px',
                border: '1px solid #e8e8e8'
              }}
              onClick={() => shouldTruncate && toggleChunkExpansion(record.id)}
            >
              {displayContent}
            </div>
            {shouldTruncate && (
              <div style={{ textAlign: 'center', marginTop: '8px' }}>
                <a
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleChunkExpansion(record.id);
                  }}
                  style={{
                    fontSize: '12px',
                    color: '#1890ff',
                    textDecoration: 'none',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    backgroundColor: '#f0f9ff',
                    border: '1px solid #d1e9ff'
                  }}
                  onMouseEnter={(e) => {
                    (e.target as HTMLElement).style.backgroundColor = '#e6f4ff';
                  }}
                  onMouseLeave={(e) => {
                    (e.target as HTMLElement).style.backgroundColor = '#f0f9ff';
                  }}
                >
                  {isExpanded ? '收起 ↑' : '展开查看更多 ↓'}
                </a>
              </div>
            )}
          </div>
        );
      },
    },
  ];

  // Render loading state
  if (isLoading) {
    return (
      <Card className={className} style={style}>
        <div
          style={{
            height,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Space direction="vertical" align="center">
            <Spin
              indicator={<LoadingOutlined style={{ fontSize: 48 }} spin />}
              size="large"
            />
            <Text>Loading knowledge graph...</Text>
          </Space>
        </div>
      </Card>
    );
  }

  // Render error state
  if (error) {
    return (
      <Card className={className} style={style}>
        <Alert
          message="Graph Visualization Error"
          description={error}
          type="error"
          showIcon
          style={{ margin: 20 }}
        />
      </Card>
    );
  }

  // Render empty state
  if (!data && entities.length === 0) {
    return (
      <Card className={className} style={style}>
        <div
          style={{
            height,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            textAlign: 'center',
          }}
        >
          <Title level={4} type="secondary">
            No Knowledge Graph Data
          </Title>
          <Text type="secondary">
            Build a knowledge graph from your documents to visualize entities
            and relationships.
          </Text>
        </div>
      </Card>
    );
  }

  if (!showDataTable) {
    // Render only the graph
    return (
      <ErrorBoundary>
        <Card
          className={className}
          style={style}
          styles={{ body: { padding: 0 } }}
        >
          <div
            ref={chartRef}
            style={{
              width: '100%',
              height,
              backgroundColor: theme === 'dark' ? '#1f1f1f' : '#ffffff',
            }}
          />
        </Card>
      </ErrorBoundary>
    );
  }

  // 右侧数据面板的标签项
  const dataTabItems = [
    {
      key: 'nodes',
      label: `节点 (${graphData?.nodes?.length || 0})`,
      children: (
        <div
          style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <Table
            columns={nodeColumns}
            dataSource={graphData?.nodes || []}
            rowKey="id"
            pagination={false}
            size="small"
            scroll={{
              y: height - 120, // 优化高度计算
              scrollToFirstRowOnChange: false,
            }}
            onRow={record => ({
              onClick: () => {
                setSelectedNode(record);
                setSelectedEdge(null);
              },
              style: {
                backgroundColor:
                  selectedNode?.id === record.id ? '#e6f7ff' : undefined,
                cursor: 'pointer',
              },
            })}
          />
        </div>
      ),
    },
    {
      key: 'edges',
      label: `关系 (${graphData?.links?.length || 0})`,
      children: (
        <div
          style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <Table
            columns={edgeColumns}
            dataSource={graphData?.links || []}
            rowKey={(record, index) =>
              `${record.source}-${record.target}-${index}`
            }
            pagination={false}
            size="small"
            scroll={{
              y: height - 120, // 优化高度计算
              scrollToFirstRowOnChange: false,
            }}
            onRow={record => ({
              onClick: () => {
                setSelectedEdge(record);
                setSelectedNode(null);
              },
              style: {
                backgroundColor:
                  selectedEdge?.source === record.source &&
                  selectedEdge?.target === record.target
                    ? '#e6f7ff'
                    : undefined,
                cursor: 'pointer',
              },
            })}
          />
        </div>
      ),
    },
    {
      key: 'textChunks',
      label: `文本块 (${textChunks?.length || 0})`,
      children: (
        <div
          style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <style>
            {`
              .text-chunk-row {
                border: none !important;
              }
              .text-chunk-row td {
                border: none !important;
                padding: 4px 8px !important;
              }
              .text-chunk-row:hover {
                background-color: transparent !important;
              }
            `}
          </style>
          <Table
            columns={textChunkColumns}
            dataSource={textChunks || []}
            rowKey="id"
            pagination={false}
            size="small"
            showHeader={false}
            scroll={{
              y: height - 120, // 优化高度计算
              scrollToFirstRowOnChange: false,
            }}
            onRow={record => ({
              onClick: (e) => {
                // Prevent row click if user clicked on expand/collapse button
                if ((e.target as HTMLElement).tagName === 'A') {
                  e.stopPropagation();
                }
              },
              style: {
                cursor: 'default',
                border: 'none',
              },
            })}
            rowClassName={() => 'text-chunk-row'}
            style={{
              backgroundColor: 'transparent',
            }}
          />
        </div>
      ),
    },
  ];

  // 主布局：左侧图谱 + 右侧数据面板
  return (
    <ErrorBoundary>
      <Card
        className={className}
        style={{ ...style, height: '100%' }}
        styles={{ body: { padding: 0, height: '100%' } }}
      >
        <div style={{ display: 'flex', height: '100%', minHeight: height || '600px' }}>
          {/* 左侧知识图谱区域 - 呠2/3宽度 */}
          <div
            style={{
              flex: 2,
              position: 'relative',
              borderRight: '1px solid #f0f0f0',
              display: 'flex',
              flexDirection: 'column',
              height: '100%',
            }}
          >
            <div style={{ padding: '12px 16px', flexShrink: 0 }}>
              <div style={{ marginBottom: 12, textAlign: 'center' }}>
                <Text strong style={{ fontSize: 16 }}>
                  知识图谱
                </Text>
              </div>

              {/* 配置控件 */}
              <div style={{ marginBottom: 12 }}>
                <Row gutter={16} align="middle">
                  <Col span={6}>
                    <Space direction="vertical" size={2} style={{ width: '100%' }}>
                      <Text style={{ fontSize: '12px' }}>最大节点数</Text>
                      <InputNumber
                        size="small"
                        min={1}
                        max={10000}
                        value={inputMaxEntities}
                        onChange={(value) => setInputMaxEntities(value || 500)}
                        style={{ width: '100%' }}
                        placeholder="500"
                      />
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space direction="vertical" size={2} style={{ width: '100%' }}>
                      <Text style={{ fontSize: '12px' }}>最大关系数</Text>
                      <InputNumber
                        size="small"
                        min={1}
                        max={10000}
                        value={inputMaxRelations}
                        onChange={(value) => setInputMaxRelations(value || 750)}
                        style={{ width: '100%' }}
                        placeholder="750"
                      />
                    </Space>
                  </Col>
                  <Col span={4}>
                    <Button
                      type="primary"
                      size="small"
                      loading={isRefreshing}
                      onClick={handleRefresh}
                      style={{ marginTop: '16px' }}
                    >
                      刷新
                    </Button>
                  </Col>
                  <Col span={8}>
                    {graphData && (
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'flex-end',
                          gap: 12,
                          fontSize: '11px',
                          marginTop: '8px',
                        }}
                      >
                        <span>
                          节点: <Text strong>{graphData.nodes?.length || 0}</Text>
                        </span>
                        <span>
                          关系: <Text strong>{graphData.links?.length || 0}</Text>
                        </span>
                        <span>
                          类别: <Text strong>{graphData.categories?.length || 0}</Text>
                        </span>
                      </div>
                    )}
                  </Col>
                </Row>
              </div>
            </div>
            <div
              ref={chartRef}
              style={{
                width: '100%',
                flex: 1,
                minHeight: 0,
                backgroundColor: theme === 'dark' ? '#1f1f1f' : '#ffffff',
              }}
            />

            {/* 选中项显示 */}
            {(selectedNode || selectedEdge) && (
              <div
                style={{
                  position: 'absolute',
                  bottom: 16,
                  left: 16,
                  right: 16,
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  padding: 12,
                  borderRadius: 6,
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                  backdropFilter: 'blur(4px)',
                }}
              >
                {selectedNode && (
                  <div>
                    <Text strong>选中节点: </Text>
                    <Text>{selectedNode.name}</Text>
                    <Text type="secondary" style={{ marginLeft: 12 }}>
                      类别: {selectedNode.entityType} | 置信度:{' '}
                      {(selectedNode.confidence || 0).toFixed(2)}
                    </Text>
                  </div>
                )}
                {selectedEdge && (
                  <div>
                    <Text strong>选中关系: </Text>
                    <Text>
                      {getNodeNameById(selectedEdge.source)} →{' '}
                      {getNodeNameById(selectedEdge.target)}
                    </Text>
                    <Text type="secondary" style={{ marginLeft: 12 }}>
                      类型: {selectedEdge.relationshipType || '关系'} | 置信度:{' '}
                      {(selectedEdge.confidence || 0).toFixed(2)}
                    </Text>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 右侧数据面板 - 呠1/3宽度 */}
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              height: '100%',
            }}
          >
            <div
              style={{
                padding: '12px 16px 8px 16px',
                textAlign: 'center',
                borderBottom: '1px solid #f0f0f0',
                flexShrink: 0,
              }}
            >
              <Text strong style={{ fontSize: 16 }}>
                数据视图
              </Text>
            </div>
            <div
              style={{
                flex: 1,
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                minHeight: 0,
              }}
            >
              <Tabs
                activeKey={activeTab}
                onChange={setActiveTab}
                items={dataTabItems}
                style={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                }}
                tabPosition="top"
                size="small"
                tabBarStyle={{
                  padding: '0 12px',
                  margin: 0,
                  flexShrink: 0,
                  minHeight: '40px',
                }}
              />
            </div>
          </div>
        </div>
      </Card>
    </ErrorBoundary>
  );
};

export default GraphVisualizer;
