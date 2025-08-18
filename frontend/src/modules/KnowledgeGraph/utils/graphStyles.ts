import type { Stylesheet } from 'cytoscape';

export interface GraphStyleConfig {
  nodeSize: number;
  edgeWidth: number;
  showNodeLabels: boolean;
  showEdgeLabels: boolean;
  theme: 'light' | 'dark';
}

export const createGraphStylesheet = (
  config: GraphStyleConfig
): Stylesheet[] => {
  const { nodeSize, edgeWidth, showNodeLabels, showEdgeLabels, theme } = config;

  const colors =
    theme === 'dark'
      ? {
          background: '#1f1f1f',
          node: {
            person: '#4CAF50',
            organization: '#2196F3',
            location: '#FF9800',
            concept: '#9C27B0',
            event: '#F44336',
            object: '#607D8B',
            cluster: '#795548',
            default: '#9E9E9E',
          },
          edge: {
            contains: '#4CAF50',
            belongs_to: '#2196F3',
            references: '#FF9800',
            related_to: '#9C27B0',
            located_in: '#F44336',
            works_for: '#607D8B',
            default: '#9E9E9E',
          },
          text: '#ffffff',
          border: '#333333',
          selected: '#FFC107',
          highlighted: '#E91E63',
        }
      : {
          background: '#ffffff',
          node: {
            person: '#66BB6A',
            organization: '#42A5F5',
            location: '#FFA726',
            concept: '#AB47BC',
            event: '#EF5350',
            object: '#78909C',
            cluster: '#8D6E63',
            default: '#BDBDBD',
          },
          edge: {
            contains: '#4CAF50',
            belongs_to: '#2196F3',
            references: '#FF9800',
            related_to: '#9C27B0',
            located_in: '#F44336',
            works_for: '#607D8B',
            default: '#9E9E9E',
          },
          text: '#333333',
          border: '#dddddd',
          selected: '#FFC107',
          highlighted: '#E91E63',
        };

  return [
    // Background
    {
      selector: 'core',
      style: {
        'background-color': colors.background,
        'selection-box-color': colors.selected,
        'selection-box-border-color': colors.selected,
        'selection-box-opacity': 0.3,
      },
    },

    // Default node styles
    {
      selector: 'node',
      style: {
        width: nodeSize,
        height: nodeSize,
        'background-color': 'data(color)',
        'border-width': 2,
        'border-color': colors.border,
        label: showNodeLabels ? 'data(label)' : '',
        'text-valign': 'center',
        'text-halign': 'center',
        color: colors.text,
        'font-size': Math.max(10, nodeSize * 0.3),
        'font-weight': 'normal',
        'text-outline-width': 2,
        'text-outline-color': colors.background,
        'text-outline-opacity': 0.8,
        'overlay-padding': 4,
        'z-index': 10,
        'transition-property': 'width, height, background-color, border-color',
        'transition-duration': '0.3s',
      },
    },

    // Entity type specific colors
    {
      selector: 'node[entityType = "person"]',
      style: {
        'background-color': colors.node.person,
        shape: 'ellipse',
      },
    },
    {
      selector: 'node[entityType = "organization"]',
      style: {
        'background-color': colors.node.organization,
        shape: 'round-rectangle',
      },
    },
    {
      selector: 'node[entityType = "location"]',
      style: {
        'background-color': colors.node.location,
        shape: 'round-diamond',
      },
    },
    {
      selector: 'node[entityType = "concept"]',
      style: {
        'background-color': colors.node.concept,
        shape: 'round-hexagon',
      },
    },
    {
      selector: 'node[entityType = "event"]',
      style: {
        'background-color': colors.node.event,
        shape: 'round-triangle',
      },
    },
    {
      selector: 'node[entityType = "object"]',
      style: {
        'background-color': colors.node.object,
        shape: 'round-rectangle',
      },
    },

    // Cluster nodes
    {
      selector: 'node[type = "cluster"]',
      style: {
        'background-color': colors.node.cluster,
        shape: 'round-octagon',
        width: nodeSize * 1.2,
        height: nodeSize * 1.2,
        'border-width': 3,
        'border-style': 'dashed',
      },
    },

    // Default edge styles
    {
      selector: 'edge',
      style: {
        width: edgeWidth,
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        label: showEdgeLabels ? 'data(label)' : '',
        color: colors.text,
        'font-size': Math.max(8, edgeWidth * 3),
        'text-rotation': 'autorotate',
        'text-outline-width': 1,
        'text-outline-color': colors.background,
        'text-outline-opacity': 0.8,
        'overlay-padding': 2,
        'z-index': 5,
        'transition-property': 'width, line-color, target-arrow-color',
        'transition-duration': '0.3s',
      },
    },

    // Edge type specific colors
    {
      selector: 'edge[relationType = "contains"]',
      style: {
        'line-color': colors.edge.contains,
        'target-arrow-color': colors.edge.contains,
      },
    },
    {
      selector: 'edge[relationType = "belongs_to"]',
      style: {
        'line-color': colors.edge.belongs_to,
        'target-arrow-color': colors.edge.belongs_to,
      },
    },
    {
      selector: 'edge[relationType = "references"]',
      style: {
        'line-color': colors.edge.references,
        'target-arrow-color': colors.edge.references,
      },
    },
    {
      selector: 'edge[relationType = "related_to"]',
      style: {
        'line-color': colors.edge.related_to,
        'target-arrow-color': colors.edge.related_to,
      },
    },
    {
      selector: 'edge[relationType = "located_in"]',
      style: {
        'line-color': colors.edge.located_in,
        'target-arrow-color': colors.edge.located_in,
      },
    },
    {
      selector: 'edge[relationType = "works_for"]',
      style: {
        'line-color': colors.edge.works_for,
        'target-arrow-color': colors.edge.works_for,
      },
    },

    // Selected states
    {
      selector: 'node:selected',
      style: {
        'border-color': colors.selected,
        'border-width': 4,
        'background-color': colors.selected,
        'z-index': 100,
      },
    },
    {
      selector: 'edge:selected',
      style: {
        'line-color': colors.selected,
        'target-arrow-color': colors.selected,
        width: edgeWidth * 2,
        'z-index': 100,
      },
    },

    // Highlighted states (for search results, hover, etc.)
    {
      selector: 'node.highlighted',
      style: {
        'border-color': colors.highlighted,
        'border-width': 4,
        'background-color': colors.highlighted,
        'z-index': 50,
      },
    },
    {
      selector: 'edge.highlighted',
      style: {
        'line-color': colors.highlighted,
        'target-arrow-color': colors.highlighted,
        width: edgeWidth * 1.5,
        'z-index': 50,
      },
    },

    // Dimmed states (for focusing on specific elements)
    {
      selector: 'node.dimmed',
      style: {
        opacity: 0.3,
      },
    },
    {
      selector: 'edge.dimmed',
      style: {
        opacity: 0.3,
      },
    },

    // Hidden states
    {
      selector: 'node.hidden',
      style: {
        display: 'none',
      },
    },
    {
      selector: 'edge.hidden',
      style: {
        display: 'none',
      },
    },
  ];
};

export const getNodeColor = (
  entityType: string,
  theme: 'light' | 'dark' = 'light'
): string => {
  const colors =
    theme === 'dark'
      ? {
          person: '#4CAF50',
          organization: '#2196F3',
          location: '#FF9800',
          concept: '#9C27B0',
          event: '#F44336',
          object: '#607D8B',
          cluster: '#795548',
          default: '#9E9E9E',
        }
      : {
          person: '#66BB6A',
          organization: '#42A5F5',
          location: '#FFA726',
          concept: '#AB47BC',
          event: '#EF5350',
          object: '#78909C',
          cluster: '#8D6E63',
          default: '#BDBDBD',
        };

  return colors[entityType as keyof typeof colors] || colors.default;
};

export const getEdgeColor = (
  relationType: string,
  theme: 'light' | 'dark' = 'light'
): string => {
  const colors =
    theme === 'dark'
      ? {
          contains: '#4CAF50',
          belongs_to: '#2196F3',
          references: '#FF9800',
          related_to: '#9C27B0',
          located_in: '#F44336',
          works_for: '#607D8B',
          default: '#9E9E9E',
        }
      : {
          contains: '#4CAF50',
          belongs_to: '#2196F3',
          references: '#FF9800',
          related_to: '#9C27B0',
          located_in: '#F44336',
          works_for: '#607D8B',
          default: '#9E9E9E',
        };

  return colors[relationType as keyof typeof colors] || colors.default;
};

export const defaultStyleConfig: GraphStyleConfig = {
  nodeSize: 30,
  edgeWidth: 2,
  showNodeLabels: true,
  showEdgeLabels: false,
  theme: 'light',
};
