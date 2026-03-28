import { create } from "zustand";

export interface Alert {
  id: number;
  camera_id: number;
  incident_type: string;
  fusion_score: number;
  timestamp: number;
  snapshot_path: string;
  acknowledged?: boolean;
}

export interface AppState {
  alerts: Alert[];
  unacknowledgedCount: number;
  cameraFps: Record<number, number[]>; // camera_id → last 60 FPS readings

  addAlert: (alert: Alert) => void;
  acknowledgeAlert: (id: number) => void;
  pushFps: (cameraId: number, fps: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  alerts: [],
  unacknowledgedCount: 0,
  cameraFps: {},

  addAlert: (alert: Alert) =>
    set((state) => {
      const newAlerts = [alert, ...state.alerts].slice(0, 100);
      return {
        alerts: newAlerts,
        unacknowledgedCount: state.unacknowledgedCount + 1,
      };
    }),

  acknowledgeAlert: (id: number) =>
    set((state) => {
      const updated = state.alerts.map((a) =>
        a.id === id ? { ...a, acknowledged: true } : a
      );
      return {
        alerts: updated,
        unacknowledgedCount: Math.max(0, state.unacknowledgedCount - 1),
      };
    }),

  pushFps: (cameraId: number, fps: number) =>
    set((state) => {
      const current = state.cameraFps[cameraId] || [];
      const updated = [...current, fps].slice(-60);
      return {
        cameraFps: { ...state.cameraFps, [cameraId]: updated },
      };
    }),
}));
