"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

export default function SystemStatus() {
    const router = useRouter();
    const [data, setData] = useState<any>(null);
    const cameraFps = useAppStore((s) => s.cameraFps);

    useEffect(() => {
        // Auth check
        const token = localStorage.getItem("access_token");
        if (!token) {
            router.push("/login");
            return;
        }

        const fetchStatus = async () => {
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                const res = await axios.get(`${apiUrl}/api/v1/system/status`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                setData(res.data);
            } catch (err: any) {
                if (err.response?.status === 401) {
                    router.push("/login");
                }
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 2000); // poll every 2 seconds

        return () => clearInterval(interval);
    }, [router]);

    if (!data) return <div className="page-header"><p>Loading system status...</p></div>;

    const uptimeDays = Math.floor(data.uptime_seconds / 86400);
    const uptimeHours = Math.floor((data.uptime_seconds % 86400) / 3600);
    const uptimeMins = Math.floor((data.uptime_seconds % 3600) / 60);
    const uptimeSecs = Math.floor(data.uptime_seconds % 60);

    const vramPcnt = (data.vram_used / 7.5) * 100;
    const vramColor = vramPcnt < 80 ? "var(--success)" : vramPcnt < 90 ? "var(--warning)" : "var(--accent)";

    // Format Recharts data (combine all camera arrays to sequence of objects)
    const maxLen = 60;
    const chartData = Array.from({ length: maxLen }).map((_, i) => {
        const pt: any = { index: i };
        Object.keys(cameraFps).forEach((compId) => {
            const arr = cameraFps[Number(compId)] || [];
            // align to right using negative index trick
            const valIndex = arr.length - (maxLen - i);
            pt[`cam_${compId}`] = valIndex >= 0 ? arr[valIndex] : 0;
        });
        return pt;
    });

    return (
        <>
            <div className="page-header" style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-end"
            }}>
                <div>
                    <h1>System Status</h1>
                    <p>Live health metrics</p>
                </div>
                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "0.85rem", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>Uptime</div>
                    <div style={{ fontSize: "1.1rem", fontFamily: "monospace" }}>
                        {uptimeDays}d {uptimeHours}h {uptimeMins}m {uptimeSecs}s
                    </div>
                </div>
            </div>

            <div className="page-content" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

                {/* VRAM Gauge */}
                <div className="card">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1rem" }}>
                        <div>
                            <h3 style={{ margin: 0, fontSize: "1.05rem" }}>GPU VRAM Usage</h3>
                            <p style={{ margin: "0.25rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>RTX 4060 Laptop (8GB)</p>
                        </div>
                        <div style={{ fontSize: "1.2rem", fontWeight: 700, color: vramColor }}>
                            {data.vram_used.toFixed(2)} GB <span style={{ color: "var(--muted)", fontWeight: 400, fontSize: "1rem" }}>/ 7.5 GB</span>
                        </div>
                    </div>

                    <div className="progress-bar" style={{ height: 12 }}>
                        <div
                            className="progress-fill"
                            style={{ width: `${Math.min(vramPcnt, 100)}%`, background: vramColor }}
                        />
                    </div>
                </div>

                {/* FPS Overview Chart */}
                <div className="card">
                    <h3 style={{ margin: "0 0 1.5rem", fontSize: "1.05rem" }}>Camera FPS Tracker</h3>
                    <div style={{ height: 300, width: "100%" }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                                <XAxis dataKey="index" hide />
                                <YAxis domain={[0, 30]} stroke="var(--muted)" fontSize={12} tickLine={false} axisLine={false} />
                                <Tooltip contentStyle={{ background: "var(--card-hover)", border: "1px solid var(--border)", borderRadius: 8 }} />
                                <ReferenceLine y={15} stroke="var(--accent)" strokeDasharray="3 3" opacity={0.5} />
                                {Object.keys(cameraFps).map((compId, idx) => (
                                    <Line
                                        key={compId}
                                        type="monotone"
                                        dataKey={`cam_${compId}`}
                                        name={`Camera ${compId}`}
                                        stroke={idx === 0 ? "var(--primary)" : "var(--warning)"}
                                        strokeWidth={2}
                                        dot={false}
                                        isAnimationActive={false}
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Model status grid */}
                <div className="card">
                    <h3 style={{ margin: "0 0 1rem", fontSize: "1.05rem" }}>Model Status</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "1rem" }}>
                        {Object.entries(data.models || {}).map(([modelName, status]: any) => (
                            <div key={modelName} style={{ background: "var(--background)", padding: "1rem", borderRadius: 8, border: "1px solid var(--border)" }}>
                                <div style={{ fontSize: "0.85rem", color: "var(--muted)", marginBottom: 8, textTransform: "uppercase" }}>{modelName}</div>
                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <span className={`dot ${status === "loaded" ? "dot-green" : "dot-red"}`} />
                                    <span style={{ fontWeight: 600 }}>{status}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </>
    );
}
