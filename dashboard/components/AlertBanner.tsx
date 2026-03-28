"use client";

import { useEffect, useRef } from "react";
import { AlertTriangle, X } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import axios from "axios";

export default function AlertBanner() {
    const wsRef = useRef<WebSocket | null>(null);
    const addAlert = useAppStore((s) => s.addAlert);
    const acknowledgeAlert = useAppStore((s) => s.acknowledgeAlert);
    const alerts = useAppStore((s) => s.alerts);

    const latestAlert = alerts[0];
    const isSetup = useRef(false);

    useEffect(() => {
        if (isSetup.current) return;
        isSetup.current = true;

        const connectWs = () => {
            const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
            const ws = new WebSocket(`${wsUrl}/ws/alerts`);
            wsRef.current = ws;

            ws.onmessage = (event) => {
                try {
                    const payload = JSON.parse(event.data);
                    // Only add valid alerts
                    if (payload.incident_id && payload.incident_type) {
                        addAlert({
                            id: payload.incident_id,
                            camera_id: payload.camera_id,
                            incident_type: payload.incident_type,
                            fusion_score: payload.fusion_score,
                            timestamp: payload.timestamp,
                            snapshot_path: payload.snapshot_path || "",
                            acknowledged: false,
                        });
                    }
                } catch { }
            };

            ws.onclose = () => {
                setTimeout(() => connectWs(), 3000);
            };
        };

        connectWs();

        const pingInterval = setInterval(() => {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send("ping");
            }
        }, 30000);

        return () => {
            clearInterval(pingInterval);
            if (wsRef.current) {
                wsRef.current.onclose = null;
                wsRef.current.close();
            }
        };
    }, [addAlert]);

    if (!latestAlert || latestAlert.acknowledged) return null;

    const handleAcknowledge = async () => {
        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            await axios.patch(`${apiUrl}/api/v1/incidents/${latestAlert.id}/acknowledge`);
            acknowledgeAlert(latestAlert.id);
        } catch {
            // Fallback: ack locally even if API fails
            acknowledgeAlert(latestAlert.id);
        }
    };

    const scorePct = (latestAlert.fusion_score * 100).toFixed(0);

    return (
        <div
            className="alert-banner animate-slide-down"
            style={{
                background: "var(--accent)", // Red bg
                color: "#fff",
                padding: "1rem 2rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                boxShadow: "0 4px 12px rgba(244, 63, 94, 0.4)",
            }}
        >
            <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                <AlertTriangle size={24} color="#fff" />
                <div>
                    <h3 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 700 }}>
                        {latestAlert.incident_type.toUpperCase().replace("_", " ")} DETECTED
                    </h3>
                    <p style={{ margin: 0, fontSize: "0.85rem", opacity: 0.9 }}>
                        Camera {latestAlert.camera_id} • Score: {scorePct}% •{" "}
                        {new Date(latestAlert.timestamp * 1000).toLocaleTimeString()}
                    </p>
                </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                <button
                    onClick={handleAcknowledge}
                    style={{
                        background: "#fff",
                        color: "var(--accent)",
                        border: "none",
                        borderRadius: "6px",
                        padding: "0.5rem 1rem",
                        fontWeight: 600,
                        cursor: "pointer",
                        fontSize: "0.85rem",
                    }}
                >
                    Acknowledge
                </button>
                <button
                    onClick={() => acknowledgeAlert(latestAlert.id)}
                    style={{
                        background: "transparent",
                        border: "none",
                        color: "#fff",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        opacity: 0.8,
                    }}
                >
                    <X size={20} />
                </button>
            </div>
        </div>
    );
}
