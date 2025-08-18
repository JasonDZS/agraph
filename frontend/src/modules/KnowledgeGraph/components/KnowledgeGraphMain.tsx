import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Card, message, Spin } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

import { useKnowledgeGraphStore } from '@/store/knowledgeGraphStore';
import { useAppStore } from '@/store/appStore';
import { knowledgeGraphService } from '@/services/knowledgeGraphService';

import GraphVisualizer from './GraphVisualizer';
import type { GraphVisualizerMethods } from '../types/graph';

// ECharts 方法类型
interface EChartsVisualizerMethods {
  applyLayout: (layoutName?: string) => void;
  fitToContainer: () => void;
  centerGraph: () => void;
  resetView: () => void;
  getChart: () => any;
}

export interface KnowledgeGraphMainProps {
  height?: number;
}

const KnowledgeGraphMain: React.FC<KnowledgeGraphMainProps> = ({
  height = 700,
}) => {
  const graphRef = useRef<EChartsVisualizerMethods>(null);
  const [isInitialLoading, setIsInitialLoading] = useState(true);

  const {
    currentGraph,
    entities,
    relations,
    graphLoading,
    graphError,
    selectedEntity,
    selectedRelation,
    setCurrentGraph,
    setEntities,
    setRelations,
    setGraphLoading,
    setGraphError,
    setSelectedEntity,
    setSelectedRelation,
    clearSelection,
  } = useKnowledgeGraphStore();

  const { currentProject, showNotification } = useAppStore();

  // Load initial graph data
  useEffect(() => {
    loadGraphData();
  }, [currentProject]);

  const loadGraphData = useCallback(async () => {
    if (!currentProject) {
      setIsInitialLoading(false);
      return;
    }

    try {
      setIsInitialLoading(true);
      setGraphLoading(true);
      setGraphError(null);

      // First check if there's a knowledge graph
      const statusResponse = await knowledgeGraphService.getGraphStatus();

      if (!statusResponse.success || !statusResponse.data?.exists) {
        setGraphError(
          'No knowledge graph found. Please build a knowledge graph first.'
        );
        setIsInitialLoading(false);
        setGraphLoading(false);
        return;
      }

      // Load the full graph data
      const graphResponse = await knowledgeGraphService.getGraph();

      if (graphResponse.success && graphResponse.data?.data) {
        const graph = graphResponse.data.data;
        setCurrentGraph(graph);
        setEntities(graph.entities || []);
        setRelations(graph.relations || []);

        showNotification?.({
          type: 'success',
          message: 'Knowledge graph loaded successfully',
          description: `Loaded ${graph.entities?.length || 0} entities and ${graph.relations?.length || 0} relations`,
        });
      } else {
        setGraphError(
          graphResponse.message || 'Failed to load knowledge graph'
        );
      }
    } catch (error) {
      console.error('Failed to load graph data:', error);
      setGraphError('Failed to load knowledge graph data');
    } finally {
      setIsInitialLoading(false);
      setGraphLoading(false);
    }
  }, [
    currentProject,
    setCurrentGraph,
    setEntities,
    setRelations,
    setGraphLoading,
    setGraphError,
    showNotification,
  ]);

  // Graph event handlers
  const graphEventHandlers = {
    onNodeSelect: (node: any) => {
      const entity = entities.find(e => e.id === node.id);
      if (entity) {
        setSelectedEntity(entity);
      }
    },
    onNodeDeselect: () => {
      setSelectedEntity(null);
    },
    onEdgeSelect: (edge: any) => {
      const relation = relations.find(
        r =>
          r.head_entity_id === edge.source && r.tail_entity_id === edge.target
      );
      if (relation) {
        setSelectedRelation(relation);
      }
    },
    onEdgeDeselect: () => {
      setSelectedRelation(null);
    },
    onBackgroundClick: () => {
      clearSelection();
    },
    ref: graphRef,
  };

  if (isInitialLoading) {
    return (
      <Card
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <Spin
            indicator={<LoadingOutlined style={{ fontSize: 48 }} spin />}
            size="large"
          />
          <div style={{ marginTop: 16, fontSize: 16, fontWeight: 500 }}>
            Loading Knowledge Graph
          </div>
          <div style={{ fontSize: 14, color: '#666', marginTop: 8 }}>
            Initializing visualization components...
          </div>
        </div>
      </Card>
    );
  }

  const mainContent = (
    <GraphVisualizer
      eventHandlers={graphEventHandlers}
      height={height}
      showDataTable={true}
      options={{
        fitToContent: false,
        animate: true,
      }}
    />
  );

  return <div style={{ height }}>{mainContent}</div>;
};

export default KnowledgeGraphMain;
