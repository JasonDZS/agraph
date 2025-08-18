import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from 'react';
import { Card, Spin, Alert, Typography, Space, Tabs, Table, Tag } from 'antd';
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
}

const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  data,
  options = {},
  eventHandlers = {},
  height = 600,
  className,
  style,
  showDataTable = true,
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

  // Store hooks
  const {
    currentGraph,
    entities,
    relations,
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
      console.log('Chart ref not available');
      return;
    }

    if (chartInstance.current) {
      console.log('Chart instance already exists');
      return;
    }

    try {
      console.log('Initializing ECharts instance', chartRef.current);
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

      console.log('ECharts initialized successfully');

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
  }, [data, convertToEChartsData, entities.length, relations.length]);

  // Load graph data from API if needed
  const loadGraphData = useCallback(async () => {
    if (data || !currentProject) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await knowledgeGraphService.getVisualizationData({
        include_clusters: false,
        max_entities: options.maxNodes || 500,
        max_relations: 750,
        min_confidence: confidenceThreshold || 0.5,
        entity_types:
          entityTypeFilter.length > 0 ? entityTypeFilter : undefined,
        relation_types:
          relationTypeFilter.length > 0 ? relationTypeFilter : undefined,
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

        useKnowledgeGraphStore.getState().setEntities(entitiesFromNodes);
        useKnowledgeGraphStore.getState().setRelations(relationsFromEdges);
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
    options.maxNodes,
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
              y: height - 180, // 动态计算高度，减去标题和标签栏的高度
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
              y: height - 180, // 动态计算高度，减去标题和标签栏的高度
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
  ];

  // 主布局：左侧图谱 + 右侧数据面板
  return (
    <ErrorBoundary>
      <Card
        className={className}
        style={style}
        styles={{ body: { padding: 0 } }}
      >
        <div style={{ display: 'flex', height }}>
          {/* 左侧知识图谱区域 - 呠2/3宽度 */}
          <div
            style={{
              flex: 2,
              position: 'relative',
              borderRight: '1px solid #f0f0f0',
            }}
          >
            <div style={{ padding: '16px 16px 0 16px' }}>
              <div style={{ marginBottom: 16, textAlign: 'center' }}>
                <Text strong style={{ fontSize: 16 }}>
                  知识图谱
                </Text>
              </div>
              {graphData && (
                <div
                  style={{
                    marginBottom: 12,
                    display: 'flex',
                    justifyContent: 'center',
                    gap: 24,
                  }}
                >
                  <span>
                    节点: <Text strong>{graphData.nodes?.length || 0}</Text>
                  </span>
                  <span>
                    关系: <Text strong>{graphData.links?.length || 0}</Text>
                  </span>
                  <span>
                    类别:{' '}
                    <Text strong>{graphData.categories?.length || 0}</Text>
                  </span>
                </div>
              )}
            </div>
            <div
              ref={chartRef}
              style={{
                width: '100%',
                height: 'calc(100% - 80px)',
                minHeight: '400px',
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
                padding: '16px 16px 12px 16px',
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
                  padding: '0 16px',
                  margin: 0,
                  flexShrink: 0,
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
