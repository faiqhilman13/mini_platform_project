/// <reference types="vite/client" />

declare module 'vanta/dist/vanta.globe.min.js' {
  export interface VantaEffect {
    destroy(): void;
  }
  
  export interface VantaGlobeOptions {
    el: HTMLElement;
    THREE: any;
    mouseControls?: boolean;
    touchControls?: boolean;
    gyroControls?: boolean;
    minHeight?: number;
    minWidth?: number;
    scale?: number;
    scaleMobile?: number;
    color?: number;
    color2?: number;
    size?: number;
    backgroundColor?: number;
  }

  const VANTA: {
    GLOBE: (options: VantaGlobeOptions) => VantaEffect;
  };
  
  export default VANTA;
}
