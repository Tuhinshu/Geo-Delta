/**
 * Ambient type declarations for GeoDelta Frontend
 * Ensures IDE and TypeScript language servers have complete types
 * even when node_modules is not yet installed on the host environment.
 */

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}

declare namespace React {
  type ReactNode = any;
  interface ReactElement<P = any, T extends string | JSXElementConstructor<any> = string | JSXElementConstructor<any>> {
    type: T;
    props: P;
    key: string | null;
  }
  type JSXElementConstructor<P> = ((props: P) => ReactElement<any, any> | null) | (new (props: P) => any);
  interface Component<P = {}, S = {}> {}
  interface ChangeEvent<T = any> {
    target: T & { value: string };
  }
  function useState<T>(initialState: T | (() => T)): [T, (value: T | ((prev: T) => T)) => void];
  function useEffect(effect: () => void | (() => void), deps?: readonly any[]): void;
}

declare module 'react' {
  export = React;
}

declare module 'react-dom' {
  export const render: any;
  export const createPortal: any;
}

declare module 'react/jsx-runtime' {
  export const jsx: any;
  export const jsxs: any;
  export const Fragment: any;
}

declare module 'next' {
  export interface Metadata {
    title?: string;
    description?: string;
    [key: string]: any;
  }
}

declare module 'next/link' {
  const Link: any;
  export default Link;
}

declare module 'lucide-react' {
  export const Shield: any;
  export const Crosshair: any;
  export const Sliders: any;
  export const FileDown: any;
  export const Play: any;
  export const RotateCcw: any;
  export const Layers: any;
  export const Activity: any;
  export const Cpu: any;
  export const Radio: any;
  export const Clock: any;
  export const Sparkles: any;
  export const Search: any;
  export const Calendar: any;
  export const MapPin: any;
  export const AlertTriangle: any;
  export const CheckCircle2: any;
  export const Maximize2: any;
  export const ZoomIn: any;
  export const ZoomOut: any;
  export const Eye: any;
  export const EyeOff: any;
  export const X: any;
  export const ChevronRight: any;
  export const Compass: any;
  export const Info: any;
  export const Download: any;
  export const Zap: any;
  export const Filter: any;
  export const AlertCircle: any;
  export const TrendingUp: any;
  const icons: { [key: string]: any };
  export default icons;
}

declare module 'tailwindcss' {
  export interface Config {
    [key: string]: any;
  }
}
