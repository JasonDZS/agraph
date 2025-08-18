import type { LayoutOptions } from 'cytoscape';

export interface GraphLayoutConfig {
  name: string;
  label: string;
  description: string;
  options: LayoutOptions;
  isAnimated?: boolean;
  bestFor?: string;
}

export const graphLayouts: Record<string, GraphLayoutConfig> = {
  cose: {
    name: 'cose',
    label: 'Force Directed',
    description: 'Physics-based layout using forces between nodes',
    bestFor: 'General purpose, natural clustering',
    isAnimated: true,
    options: {
      name: 'cose',
      idealEdgeLength: 100,
      nodeOverlap: 20,
      refresh: 20,
      fit: true,
      padding: 30,
      randomize: false,
      componentSpacing: 100,
      nodeRepulsion: 400000,
      edgeElasticity: 100,
      nestingFactor: 5,
      gravity: 80,
      numIter: 1000,
      initialTemp: 200,
      coolingFactor: 0.95,
      minTemp: 1.0,
      animate: true,
      animationDuration: 1000,
      animationEasing: 'ease-out',
    },
  },

  grid: {
    name: 'grid',
    label: 'Grid',
    description: 'Arranges nodes in a rectangular grid',
    bestFor: 'Equal spacing, systematic arrangement',
    isAnimated: false,
    options: {
      name: 'grid',
      fit: true,
      padding: 30,
      spacingFactor: 1.2,
      animate: false,
      // Remove position function - let Cytoscape handle automatic grid positioning
    },
  },

  circle: {
    name: 'circle',
    label: 'Circle',
    description: 'Arranges nodes in a circle',
    bestFor: 'Small graphs, highlighting relationships',
    isAnimated: true,
    options: {
      name: 'circle',
      fit: true,
      padding: 30,
      radius: undefined,
      startAngle: -Math.PI / 2,
      sweep: 2 * Math.PI,
      clockwise: true,
      sort: undefined,
      animate: true,
      animationDuration: 500,
      animationEasing: 'ease-in-out',
    },
  },

  concentric: {
    name: 'concentric',
    label: 'Concentric',
    description: 'Arranges nodes in concentric circles based on importance',
    bestFor: 'Hierarchical data, importance-based grouping',
    isAnimated: true,
    options: {
      name: 'concentric',
      fit: true,
      padding: 30,
      startAngle: -Math.PI / 2,
      sweep: 2 * Math.PI,
      clockwise: true,
      equidistant: false,
      minNodeSpacing: 10,
      animate: true,
      animationDuration: 750,
      animationEasing: 'ease-out',
      concentric: (node: any) => {
        // Higher confidence or more connections = inner circles
        const confidence = node.data('confidence') || 0.5;
        const degree = node.degree() || 1;
        return confidence * 100 + degree * 10;
      },
      levelWidth: () => 1,
    },
  },

  breadthfirst: {
    name: 'breadthfirst',
    label: 'Hierarchical',
    description: 'Tree-like hierarchical layout',
    bestFor: 'Tree structures, parent-child relationships',
    isAnimated: true,
    options: {
      name: 'breadthfirst',
      fit: true,
      directed: true,
      padding: 30,
      circle: false,
      grid: false,
      spacingFactor: 1.5,
      roots: undefined,
      maximal: false,
      animate: true,
      animationDuration: 750,
      animationEasing: 'ease-in-out',
    },
  },

  preset: {
    name: 'preset',
    label: 'Custom Positions',
    description: 'Uses predefined node positions',
    bestFor: 'Maintaining specific arrangements',
    isAnimated: false,
    options: {
      name: 'preset',
      fit: true,
      padding: 30,
      animate: false,
      position: (node: any) => {
        // Try to get position from node data or use a default
        const pos = node.data('position') || node.position();
        return pos && typeof pos.x === 'number' && typeof pos.y === 'number'
          ? pos
          : { x: Math.random() * 400, y: Math.random() * 400 };
      },
    },
  },

  random: {
    name: 'random',
    label: 'Random',
    description: 'Places nodes at random positions',
    bestFor: 'Testing, initial positioning',
    isAnimated: false,
    options: {
      name: 'random',
      fit: true,
      padding: 30,
      animate: false,
      randomize: true,
      boundingBox: undefined, // Let Cytoscape handle bounds
    },
  },

  cola: {
    name: 'cola',
    label: 'Cola (WebCola)',
    description: 'Constraint-based layout with collision avoidance',
    bestFor: 'Complex constraints, avoiding overlaps',
    isAnimated: false, // Disable animation to prevent flashing
    options: {
      name: 'cola',
      animate: false, // Disable animation
      refresh: 0, // Disable refresh to prevent continuous updates
      maxSimulationTime: 2000, // Reduce simulation time
      ungrabifyWhileSimulating: false,
      fit: true,
      padding: 30,
      nodeDimensionsIncludeLabels: false,
      randomize: false,
      avoidOverlap: true,
      handleDisconnected: true,
      convergenceThreshold: 0.1, // Increase threshold for faster convergence
      nodeSpacing: 10,
      flow: undefined,
      alignment: undefined,
      gapInequalities: undefined,
      centerGraph: true,
    },
  },
};

export const getLayoutConfig = (layoutName: string): GraphLayoutConfig => {
  return graphLayouts[layoutName] || graphLayouts.cose;
};

export const getAvailableLayouts = (): GraphLayoutConfig[] => {
  return Object.values(graphLayouts);
};

export const getLayoutForNodeCount = (nodeCount: number): string => {
  if (nodeCount <= 10) {
    return 'circle';
  } else if (nodeCount <= 50) {
    return 'cose';
  } else if (nodeCount <= 200) {
    return 'concentric';
  } else {
    return 'grid';
  }
};

export const getBestLayoutForGraphType = (
  entityTypes: string[],
  relationTypes: string[]
): string => {
  // If there are hierarchical relationships, use breadthfirst
  const hierarchicalRelations = ['contains', 'belongs_to', 'works_for'];
  const hasHierarchy = relationTypes.some(type =>
    hierarchicalRelations.includes(type)
  );

  if (hasHierarchy) {
    return 'breadthfirst';
  }

  // If there are location entities, concentric might work well
  if (entityTypes.includes('location')) {
    return 'concentric';
  }

  // Default to force-directed layout
  return 'cose';
};

export const defaultLayoutConfig: GraphLayoutConfig = graphLayouts.cose;
