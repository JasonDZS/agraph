import type {
  ElementDefinition,
  Core,
  NodeSingular,
  EdgeSingular,
} from 'cytoscape';

export interface GraphNode {
  id: string;
  label: string;
  type: 'entity' | 'cluster';
  entityType?: string;
  confidence?: number;
  properties?: Record<string, any>;
  position?: { x: number; y: number };
  selected?: boolean;
  highlighted?: boolean;
  dimmed?: boolean;
  hidden?: boolean;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  relationType: string;
  confidence: number;
  properties?: Record<string, any>;
  selected?: boolean;
  highlighted?: boolean;
  dimmed?: boolean;
  hidden?: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphVisualizationOptions {
  layout?: string;
  fitToContent?: boolean;
  animate?: boolean;
  includeEdgeLabels?: boolean;
  includeNodeLabels?: boolean;
  minConfidence?: number;
  maxNodes?: number;
  entityTypeFilter?: string[];
  relationTypeFilter?: string[];
}

export interface GraphInteractionState {
  selectedNodes: Set<string>;
  selectedEdges: Set<string>;
  highlightedNodes: Set<string>;
  highlightedEdges: Set<string>;
  hoveredNode?: string;
  hoveredEdge?: string;
  isSelecting: boolean;
  isDragging: boolean;
  isPanning: boolean;
  isZooming: boolean;
}

export interface GraphExportOptions {
  format: 'png' | 'svg' | 'json' | 'graphml';
  quality?: number;
  scale?: number;
  includeBounds?: boolean;
  includeStyles?: boolean;
  backgroundColor?: string;
}

export interface GraphSearchResult {
  type: 'node' | 'edge';
  id: string;
  score: number;
  matches: {
    field: string;
    value: string;
    highlight: string;
  }[];
}

export interface GraphMetrics {
  nodeCount: number;
  edgeCount: number;
  averageDegree: number;
  density: number;
  clusteringCoefficient: number;
  connectedComponents: number;
  centralityScores: Record<string, number>;
}

export interface GraphViewport {
  zoom: number;
  pan: { x: number; y: number };
  extent: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    w: number;
    h: number;
  };
}

export interface GraphVisualizerMethods {
  applyLayout: (layoutName?: string) => void;
  stopLayout: () => void;
  fitToContainer: (padding?: number) => void;
  centerGraph: () => void;
  resetView: () => void;
  getCytoscape: () => Core | null;
}

export interface GraphEventHandlers {
  onNodeSelect?: (node: NodeSingular) => void;
  onNodeDeselect?: (node: NodeSingular) => void;
  onEdgeSelect?: (edge: EdgeSingular) => void;
  onEdgeDeselect?: (edge: EdgeSingular) => void;
  onNodeHover?: (node: NodeSingular) => void;
  onNodeUnhover?: (node: NodeSingular) => void;
  onEdgeHover?: (edge: EdgeSingular) => void;
  onEdgeUnhover?: (edge: EdgeSingular) => void;
  onNodeClick?: (node: NodeSingular) => void;
  onEdgeClick?: (edge: EdgeSingular) => void;
  onNodeDoubleClick?: (node: NodeSingular) => void;
  onBackgroundClick?: () => void;
  onLayoutStop?: () => void;
  onZoom?: (viewport: GraphViewport) => void;
  onPan?: (viewport: GraphViewport) => void;
  ref?: React.RefObject<GraphVisualizerMethods>;
}

export interface CytoscapeElementData {
  id: string;
  label: string;
  type?: 'entity' | 'cluster';
  entityType?: string;
  relationType?: string;
  confidence?: number;
  color?: string;
  position?: { x: number; y: number };
  source?: string;
  target?: string;
  properties?: Record<string, any>;
}

export interface CytoscapeElement extends ElementDefinition {
  data: CytoscapeElementData;
  classes?: string;
  selected?: boolean;
  selectable?: boolean;
  locked?: boolean;
  grabbable?: boolean;
  position?: { x: number; y: number };
}

export interface GraphContextMenuAction {
  id: string;
  label: string;
  icon?: string;
  disabled?: boolean;
  separator?: boolean;
  action: (target: NodeSingular | EdgeSingular | Core) => void;
}

export interface GraphLegend {
  nodeTypes: Array<{
    type: string;
    label: string;
    color: string;
    shape: string;
    count: number;
  }>;
  edgeTypes: Array<{
    type: string;
    label: string;
    color: string;
    count: number;
  }>;
}

export interface GraphFilterState {
  entityTypes: Set<string>;
  relationTypes: Set<string>;
  confidenceRange: [number, number];
  searchQuery: string;
  hideIsolatedNodes: boolean;
  showClusters: boolean;
}

export interface GraphLayoutAnimationOptions {
  duration: number;
  easing: string;
  complete?: () => void;
}

export interface GraphPerformanceMetrics {
  renderTime: number;
  layoutTime: number;
  nodeCount: number;
  edgeCount: number;
  fps: number;
  memoryUsage?: number;
}
