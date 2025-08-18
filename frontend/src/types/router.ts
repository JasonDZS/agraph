export interface RouteConfig {
  path: string;
  element: React.ComponentType;
  protected?: boolean;
  title?: string;
  icon?: string;
}

export interface NavigationItem {
  key: string;
  label: string;
  path: string;
  icon?: string;
  children?: NavigationItem[];
}
