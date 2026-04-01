"use client";

import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "@/store/useAppStore";

interface CameraInfo {
    id: number;
    name: string;
    location?: string;
}

interface CameraGridProps {
    cameras: CameraInfo[];
}

function CameraCell({ camera }: { camera: CameraInfo }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const fpsCountRef = useRef(0);
    const fpsTimerRef = useRef<number>(0);

    const pushFps = useAppStore((s) => s.pushFps);
    const alerts = useAppStore((s) => s.alerts);

    const hasUnackAlert = alerts.some(
        (a) => a.camera_id === camera.id && !a.acknowledged
    );

    const currentFps = useAppStore(
        (s) => (s.cameraFps[camera.id] || []).slice(-1)[0] || 0
    );

    useEffect(() => {
        let isClosing = false;
        let ws: WebSocket | null = null;
        let reconnectTimeout: NodeJS.Timeout;

        const connectWs = () => {
            if (isClosing) return;
            const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
            ws = new WebSocket(`${wsUrl}/ws/stream/${camera.id}`);
            wsRef.current = ws;

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.error) return;

                    const img = new Image();
                    img.onload = () => {
                        const canvas = canvasRef.current;
                        if (!canvas) return;
                        const ctx = canvas.getContext("2d");
                        if (!ctx) return;
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    };
                    img.src = "data:image/jpeg;base64," + data.frame;

                    // FPS counting
                    fpsCountRef.current++;
                } catch {
                    // ignore parse errors
                }
            };

            ws.onclose = () => {
                if (!isClosing) {
                    reconnectTimeout = setTimeout(() => connectWs(), 3000);
                }
            };

            ws.onerror = () => ws?.close();
        };

        connectWs();

        // FPS timer — push reading every second
        const interval = window.setInterval(() => {
            pushFps(camera.id, fpsCountRef.current);
            fpsCountRef.current = 0;
        }, 1000);

        return () => {
            isClosing = true;
            clearTimeout(reconnectTimeout);
            window.clearInterval(interval);
            if (ws) {
                ws.onclose = null; // prevent reconnect
                ws.onerror = null;
                ws.close();
            }
        };
    }, [camera.id, pushFps]);

    return (
        <div className="camera-cell">
            <canvas
                ref={canvasRef}
                width={1280}
                height={720}
                style={{ width: "100%", height: "100%" }}
            />

            {/* Top overlay */}
            <div className="camera-overlay">
                <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "#fff" }}>
                    {camera.name}
                    {camera.location && (
                        <span style={{ fontWeight: 400, marginLeft: 6, opacity: 0.7 }}>
                            {camera.location}
                        </span>
                    )}
                </span>
                <span
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 5,
                        fontSize: "0.75rem",
                        color: "#fff",
                    }}
                >
                    <span
                        className={`dot ${currentFps > 10 ? "dot-green" : "dot-red"}`}
                    />
                    {currentFps} FPS
                </span>
            </div>

            {/* Bottom overlay — alert indicator */}
            {hasUnackAlert && (
                <div className="camera-overlay-bottom">
                    <span
                        className="badge badge-red animate-pulse-glow"
                        style={{ fontSize: "0.7rem", padding: "0.2rem 0.6rem" }}
                    >
                        ⚠ ALERT
                    </span>
                </div>
            )}
        </div>
    );
}

export default function CameraGrid({ cameras }: CameraGridProps) {
    if (cameras.length === 0) {
        return (
            <div
                className="card"
                style={{ textAlign: "center", padding: "3rem 2rem" }}
            >
                <p style={{ color: "var(--muted)", fontSize: "0.95rem" }}>
                    No cameras configured.{" "}
                    <a href="/settings" style={{ color: "var(--primary)" }}>
                        Add cameras in Settings
                    </a>
                </p>
            </div>
        );
    }

    return (
        <div
            style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(560px, 1fr))",
                gap: "1rem",
            }}
        >
            {cameras.map((cam) => (
                <CameraCell key={cam.id} camera={cam} />
            ))}
        </div>
    );
}
