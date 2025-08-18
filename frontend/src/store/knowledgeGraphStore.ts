import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type {
  Entity,
  Relation,
  KnowledgeGraph,
  TextChunk,
  Cluster,
  BuildStatus,
} from '../types/api';

export interface GraphLayout {
  name: string;
  label: string;
  options: Record<string, any>;
}

export interface GraphVisualState {
  zoom: number;
  center: { x: number; y: number };
  selectedNodes: string[];
  selectedEdges: string[];
  highlightedNodes: string[];
  highlightedEdges: string[];
  nodePositions: Record<string, { x: number; y: number }>;
}

export interface KnowledgeGraphState {
  // Current Knowledge Graph
  currentGraph: KnowledgeGraph | null;
  graphLoading: boolean;
  graphError: string | null;

  // Entities and Relations
  entities: Entity[];
  relations: Relation[];
  textChunks: TextChunk[];
  clusters: Cluster[];

  // Graph Building
  buildStatus: BuildStatus | null;
  isBuilding: boolean;
  buildError: string | null;
  buildFromStep: string | null;

  // Graph Operations
  isUpdating: boolean;
  isExporting: boolean;

  // Visualization State
  visualState: GraphVisualState;
  currentLayout: string;
  availableLayouts: GraphLayout[];
  showNodeLabels: boolean;
  showEdgeLabels: boolean;
  nodeSize: number;
  edgeWidth: number;

  // Filters
  entityTypeFilter: string[];
  relationTypeFilter: string[];
  confidenceThreshold: number;

  // Search and Selection
  searchQuery: string;
  selectedEntity: Entity | null;
  selectedRelation: Relation | null;

  // UI State
  showBuildModal: boolean;
  showEntityDetails: boolean;
  showRelationDetails: boolean;
  showGraphSettings: boolean;
  showStats: boolean;

  // Cache Management
  cacheInfo: {
    keys: string[];
    size: number;
    items: number;
  } | null;
}

export interface KnowledgeGraphActions {
  // Graph Actions
  setCurrentGraph: (graph: KnowledgeGraph | null) => void;
  setGraphLoading: (loading: boolean) => void;
  setGraphError: (error: string | null) => void;

  // Entity and Relation Actions
  setEntities: (entities: Entity[]) => void;
  addEntity: (entity: Entity) => void;
  updateEntity: (entityId: string, updates: Partial<Entity>) => void;
  removeEntity: (entityId: string) => void;

  setRelations: (relations: Relation[]) => void;
  addRelation: (relation: Relation) => void;
  updateRelation: (relationId: string, updates: Partial<Relation>) => void;
  removeRelation: (relationId: string) => void;

  setTextChunks: (textChunks: TextChunk[]) => void;
  setClusters: (clusters: Cluster[]) => void;

  // Build Actions
  setBuildStatus: (status: BuildStatus | null) => void;
  setIsBuilding: (building: boolean) => void;
  setBuildError: (error: string | null) => void;
  setBuildFromStep: (step: string | null) => void;

  // Operation Actions
  setIsUpdating: (updating: boolean) => void;
  setIsExporting: (exporting: boolean) => void;

  // Visualization Actions
  setVisualState: (state: Partial<GraphVisualState>) => void;
  setCurrentLayout: (layout: string) => void;
  setShowNodeLabels: (show: boolean) => void;
  setShowEdgeLabels: (show: boolean) => void;
  setNodeSize: (size: number) => void;
  setEdgeWidth: (width: number) => void;
  resetVisualState: () => void;

  // Zoom Actions
  setZoom: (zoom: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;

  // Layout Actions
  applyLayout: (layout: string) => void;
  refreshLayout: () => void;

  // Filter Actions
  setEntityTypeFilter: (types: string[]) => void;
  setRelationTypeFilter: (types: string[]) => void;
  setConfidenceThreshold: (threshold: number) => void;

  // Search and Selection Actions
  setSearchQuery: (query: string) => void;
  setSelectedEntity: (entity: Entity | null) => void;
  setSelectedRelation: (relation: Relation | null) => void;

  // UI Actions
  setShowBuildModal: (show: boolean) => void;
  setShowEntityDetails: (show: boolean) => void;
  setShowRelationDetails: (show: boolean) => void;
  setShowGraphSettings: (show: boolean) => void;
  setShowStats: (show: boolean) => void;

  // Cache Actions
  setCacheInfo: (info: KnowledgeGraphState['cacheInfo']) => void;

  // Computed Values
  getFilteredEntities: () => Entity[];
  getFilteredRelations: () => Relation[];
  getEntityById: (id: string) => Entity | undefined;
  getRelationById: (id: string) => Relation | undefined;
  getRelationsByEntity: (entityId: string) => Relation[];
  getConnectedEntities: (entityId: string) => Entity[];
  getEntityTypes: () => string[];
  getRelationTypes: () => string[];
  getGraphStats: () => {
    totalEntities: number;
    totalRelations: number;
    totalTextChunks: number;
    totalClusters: number;
  };

  // Reset Actions
  reset: () => void;
  resetOperationStates: () => void;
  clearSelection: () => void;
}

export type KnowledgeGraphStore = KnowledgeGraphState & KnowledgeGraphActions;

const defaultVisualState: GraphVisualState = {
  zoom: 1,
  center: { x: 0, y: 0 },
  selectedNodes: [],
  selectedEdges: [],
  highlightedNodes: [],
  highlightedEdges: [],
  nodePositions: {},
};

const defaultLayouts: GraphLayout[] = [
  { name: 'cose', label: 'Force Directed', options: {} },
  { name: 'grid', label: 'Grid', options: {} },
  { name: 'circle', label: 'Circle', options: {} },
  { name: 'concentric', label: 'Concentric', options: {} },
  { name: 'breadthfirst', label: 'Hierarchical', options: {} },
];

const initialState: KnowledgeGraphState = {
  currentGraph: null,
  graphLoading: false,
  graphError: null,
  entities: [],
  relations: [],
  textChunks: [],
  clusters: [],
  buildStatus: null,
  isBuilding: false,
  buildError: null,
  buildFromStep: null,
  isUpdating: false,
  isExporting: false,
  visualState: defaultVisualState,
  currentLayout: 'cose',
  availableLayouts: defaultLayouts,
  showNodeLabels: true,
  showEdgeLabels: false,
  nodeSize: 30,
  edgeWidth: 2,
  entityTypeFilter: [],
  relationTypeFilter: [],
  confidenceThreshold: 0.5,
  searchQuery: '',
  selectedEntity: null,
  selectedRelation: null,
  showBuildModal: false,
  showEntityDetails: false,
  showRelationDetails: false,
  showGraphSettings: false,
  showStats: true,
  cacheInfo: null,
};

export const useKnowledgeGraphStore = create<KnowledgeGraphStore>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Graph Actions
        setCurrentGraph: currentGraph => {
          set({
            currentGraph,
            entities: currentGraph?.entities || [],
            relations: currentGraph?.relations || [],
            textChunks: currentGraph?.text_chunks || [],
            clusters: currentGraph?.clusters || [],
            graphError: null,
          });
        },

        setGraphLoading: graphLoading => {
          set({ graphLoading });
        },

        setGraphError: graphError => {
          set({ graphError, graphLoading: false });
        },

        // Entity and Relation Actions
        setEntities: entities => {
          set({ entities });
        },

        addEntity: entity => {
          set(state => ({
            entities: [...state.entities, entity],
          }));
        },

        updateEntity: (entityId, updates) => {
          set(state => ({
            entities: state.entities.map(entity =>
              entity.id === entityId ? { ...entity, ...updates } : entity
            ),
            selectedEntity:
              state.selectedEntity?.id === entityId
                ? { ...state.selectedEntity, ...updates }
                : state.selectedEntity,
          }));
        },

        removeEntity: entityId => {
          set(state => ({
            entities: state.entities.filter(entity => entity.id !== entityId),
            relations: state.relations.filter(
              relation =>
                relation.head_entity_id !== entityId &&
                relation.tail_entity_id !== entityId
            ),
            selectedEntity:
              state.selectedEntity?.id === entityId
                ? null
                : state.selectedEntity,
          }));
        },

        setRelations: relations => {
          set({ relations });
        },

        addRelation: relation => {
          set(state => ({
            relations: [...state.relations, relation],
          }));
        },

        updateRelation: (relationId, updates) => {
          set(state => ({
            relations: state.relations.map(relation =>
              relation.id === relationId
                ? { ...relation, ...updates }
                : relation
            ),
            selectedRelation:
              state.selectedRelation?.id === relationId
                ? { ...state.selectedRelation, ...updates }
                : state.selectedRelation,
          }));
        },

        removeRelation: relationId => {
          set(state => ({
            relations: state.relations.filter(
              relation => relation.id !== relationId
            ),
            selectedRelation:
              state.selectedRelation?.id === relationId
                ? null
                : state.selectedRelation,
          }));
        },

        setTextChunks: textChunks => {
          set({ textChunks });
        },

        setClusters: clusters => {
          set({ clusters });
        },

        // Build Actions
        setBuildStatus: buildStatus => {
          set({ buildStatus });
        },

        setIsBuilding: isBuilding => {
          set({ isBuilding });
        },

        setBuildError: buildError => {
          set({ buildError });
        },

        setBuildFromStep: buildFromStep => {
          set({ buildFromStep });
        },

        // Operation Actions
        setIsUpdating: isUpdating => {
          set({ isUpdating });
        },

        setIsExporting: isExporting => {
          set({ isExporting });
        },

        // Visualization Actions
        setVisualState: updates => {
          set(state => ({
            visualState: { ...state.visualState, ...updates },
          }));
        },

        setCurrentLayout: currentLayout => {
          set({ currentLayout });
        },

        setShowNodeLabels: showNodeLabels => {
          set({ showNodeLabels });
        },

        setShowEdgeLabels: showEdgeLabels => {
          set({ showEdgeLabels });
        },

        setNodeSize: nodeSize => {
          set({ nodeSize });
        },

        setEdgeWidth: edgeWidth => {
          set({ edgeWidth });
        },

        resetVisualState: () => {
          set({ visualState: defaultVisualState });
        },

        // Zoom Actions
        setZoom: zoom => {
          set(state => ({
            visualState: { ...state.visualState, zoom },
          }));
        },

        zoomIn: () => {
          set(state => ({
            visualState: {
              ...state.visualState,
              zoom: Math.min(state.visualState.zoom * 1.2, 3),
            },
          }));
        },

        zoomOut: () => {
          set(state => ({
            visualState: {
              ...state.visualState,
              zoom: Math.max(state.visualState.zoom * 0.8, 0.1),
            },
          }));
        },

        resetZoom: () => {
          set(state => ({
            visualState: { ...state.visualState, zoom: 1 },
          }));
        },

        // Layout Actions
        applyLayout: layout => {
          set({ currentLayout: layout });
        },

        refreshLayout: () => {
          const { currentLayout } = get();
          set({ currentLayout });
        },

        // Filter Actions
        setEntityTypeFilter: entityTypeFilter => {
          set({ entityTypeFilter });
        },

        setRelationTypeFilter: relationTypeFilter => {
          set({ relationTypeFilter });
        },

        setConfidenceThreshold: confidenceThreshold => {
          set({ confidenceThreshold });
        },

        // Search and Selection Actions
        setSearchQuery: searchQuery => {
          set({ searchQuery });
        },

        setSelectedEntity: selectedEntity => {
          set({ selectedEntity, showEntityDetails: !!selectedEntity });
        },

        setSelectedRelation: selectedRelation => {
          set({ selectedRelation, showRelationDetails: !!selectedRelation });
        },

        // UI Actions
        setShowBuildModal: showBuildModal => {
          set({ showBuildModal });
        },

        setShowEntityDetails: showEntityDetails => {
          set({ showEntityDetails });
        },

        setShowRelationDetails: showRelationDetails => {
          set({ showRelationDetails });
        },

        setShowGraphSettings: showGraphSettings => {
          set({ showGraphSettings });
        },

        setShowStats: showStats => {
          set({ showStats });
        },

        // Cache Actions
        setCacheInfo: cacheInfo => {
          set({ cacheInfo });
        },

        // Computed Values
        getFilteredEntities: () => {
          const {
            entities,
            entityTypeFilter,
            confidenceThreshold,
            searchQuery,
          } = get();

          let filtered = entities.filter(
            entity => entity.confidence >= confidenceThreshold
          );

          if (entityTypeFilter.length > 0) {
            filtered = filtered.filter(entity =>
              entityTypeFilter.includes(entity.entity_type)
            );
          }

          if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(
              entity =>
                entity.name.toLowerCase().includes(query) ||
                entity.description?.toLowerCase().includes(query)
            );
          }

          return filtered;
        },

        getFilteredRelations: () => {
          const { relations, relationTypeFilter, confidenceThreshold } = get();

          let filtered = relations.filter(
            relation => relation.confidence >= confidenceThreshold
          );

          if (relationTypeFilter.length > 0) {
            filtered = filtered.filter(relation =>
              relationTypeFilter.includes(relation.relation_type)
            );
          }

          return filtered;
        },

        getEntityById: id => {
          const { entities } = get();
          return entities.find(entity => entity.id === id);
        },

        getRelationById: id => {
          const { relations } = get();
          return relations.find(relation => relation.id === id);
        },

        getRelationsByEntity: entityId => {
          const { relations } = get();
          return relations.filter(
            relation =>
              relation.head_entity_id === entityId ||
              relation.tail_entity_id === entityId
          );
        },

        getConnectedEntities: entityId => {
          const { relations, entities } = get();
          const connectedIds = new Set<string>();

          relations.forEach(relation => {
            if (relation.head_entity_id === entityId) {
              connectedIds.add(relation.tail_entity_id);
            } else if (relation.tail_entity_id === entityId) {
              connectedIds.add(relation.head_entity_id);
            }
          });

          return entities.filter(entity => connectedIds.has(entity.id));
        },

        getEntityTypes: () => {
          const { entities } = get();
          const types = new Set(entities.map(entity => entity.entity_type));
          return Array.from(types).sort();
        },

        getRelationTypes: () => {
          const { relations } = get();
          const types = new Set(
            relations.map(relation => relation.relation_type)
          );
          return Array.from(types).sort();
        },

        getGraphStats: () => {
          const { entities, relations, textChunks, clusters } = get();
          return {
            totalEntities: entities.length,
            totalRelations: relations.length,
            totalTextChunks: textChunks.length,
            totalClusters: clusters.length,
          };
        },

        // Reset Actions
        reset: () => {
          set(initialState);
        },

        resetOperationStates: () => {
          set({
            graphLoading: false,
            isBuilding: false,
            isUpdating: false,
            isExporting: false,
            graphError: null,
            buildError: null,
          });
        },

        clearSelection: () => {
          set({
            selectedEntity: null,
            selectedRelation: null,
            showEntityDetails: false,
            showRelationDetails: false,
            visualState: {
              ...get().visualState,
              selectedNodes: [],
              selectedEdges: [],
              highlightedNodes: [],
              highlightedEdges: [],
            },
          });
        },
      }),
      {
        name: 'agraph-knowledge-graph-store',
        partialize: state => ({
          currentLayout: state.currentLayout,
          showNodeLabels: state.showNodeLabels,
          showEdgeLabels: state.showEdgeLabels,
          nodeSize: state.nodeSize,
          edgeWidth: state.edgeWidth,
          entityTypeFilter: state.entityTypeFilter,
          relationTypeFilter: state.relationTypeFilter,
          confidenceThreshold: state.confidenceThreshold,
          searchQuery: state.searchQuery,
          showStats: state.showStats,
        }),
      }
    ),
    {
      name: 'KnowledgeGraphStore',
    }
  )
);

// Knowledge graph state change listeners
let graphListeners: ((graph: KnowledgeGraph | null) => void)[] = [];
let entitiesListeners: ((entities: Entity[]) => void)[] = [];
let relationsListeners: ((relations: Relation[]) => void)[] = [];
let buildStatusListeners: ((status: BuildStatus | null) => void)[] = [];

export const subscribeToGraph = (
  callback: (graph: KnowledgeGraph | null) => void
) => {
  graphListeners.push(callback);
  return () => {
    graphListeners = graphListeners.filter(listener => listener !== callback);
  };
};

export const subscribeToEntities = (callback: (entities: Entity[]) => void) => {
  entitiesListeners.push(callback);
  return () => {
    entitiesListeners = entitiesListeners.filter(
      listener => listener !== callback
    );
  };
};

export const subscribeToRelations = (
  callback: (relations: Relation[]) => void
) => {
  relationsListeners.push(callback);
  return () => {
    relationsListeners = relationsListeners.filter(
      listener => listener !== callback
    );
  };
};

export const subscribeToBuildStatus = (
  callback: (status: BuildStatus | null) => void
) => {
  buildStatusListeners.push(callback);
  return () => {
    buildStatusListeners = buildStatusListeners.filter(
      listener => listener !== callback
    );
  };
};

// Subscribe to store changes to notify listeners
useKnowledgeGraphStore.subscribe((state, prevState) => {
  if (state.currentGraph !== prevState.currentGraph) {
    graphListeners.forEach(listener => listener(state.currentGraph));
  }

  if (state.entities !== prevState.entities) {
    entitiesListeners.forEach(listener => listener(state.entities));
  }

  if (state.relations !== prevState.relations) {
    relationsListeners.forEach(listener => listener(state.relations));
  }

  if (state.buildStatus !== prevState.buildStatus) {
    buildStatusListeners.forEach(listener => listener(state.buildStatus));
  }
});
