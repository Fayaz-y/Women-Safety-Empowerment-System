"use client";

import { useState } from "react";
import axios from "axios";
import { format } from "date-fns";
import { Download, Eye } from "lucide-react";

interface Incident {
    id: number;
    camera_id: number;
    incident_type: string;
    fusion_score: number;
    videomae_score?: number;
    bilstm_score?: number;
    optflow_score?: number;
    clip_score?: number;
    pose_score?: number;
    snapshot_path?: string;
    acknowledged: boolean;
    created_at: string;
}

interface IncidentTableProps {
    initialData: {
        total: number;
        items: Incident[];
        page: number;
        limit: number;
    };
}

export default function IncidentTable({ initialData }: IncidentTableProps) {
    const [data, setData] = useState(initialData);
    const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);

    // Filters state
    const [typeFilter, setTypeFilter] = useState("");
    const [ackFilter, setAckFilter] = useState("");
    const [camFilter, setCamFilter] = useState("");

    const fetchFiltered = async () => {
        try {
            const qs = new URLSearchParams();
            if (typeFilter) qs.set("incident_type", typeFilter);
            if (ackFilter) qs.set("acknowledged", ackFilter === "true" ? "true" : "false");
            if (camFilter) qs.set("camera_id", camFilter);

            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            const res = await axios.get(`${apiUrl}/api/v1/incidents?${qs.toString()}`);
            setData(res.data);
        } catch (err) {
            console.error(err);
        }
    };

    const exportCsv = () => {
        if (data.items.length === 0) return;
        const header = "id,camera_id,type,score,acknowledged,time\n";
        const mapped = data.items
            .map(
                (i) =>
                    `${i.id},${i.camera_id},${i.incident_type},${i.fusion_score},${i.acknowledged},${i.created_at}`
            )
            .join("\n");

        const a = document.createElement("a");
        a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(header + mapped);
        a.download = "incidents.csv";
        a.click();
    };

    const handleAck = async (id: number) => {
        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            await axios.patch(`${apiUrl}/api/v1/incidents/${id}/acknowledge`);
            // Update local state
            setData((prev) => ({
                ...prev,
                items: prev.items.map((i) => (i.id === id ? { ...i, acknowledged: true } : i)),
            }));
            if (selectedIncident?.id === id) {
                setSelectedIncident({ ...selectedIncident, acknowledged: true });
            }
        } catch { }
    };

    return (
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            {/* ── Filters ── */}
            <div
                style={{
                    borderBottom: "1px solid var(--border)",
                    padding: "1rem 1.5rem",
                    display: "flex",
                    gap: "1rem",
                    alignItems: "center",
                    flexWrap: "wrap",
                }}
            >
                <select className="input" style={{ width: 160 }} value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
                    <option value="">All Types</option>
                    <option value="proximity_threat">Proximity</option>
                    <option value="violence">Violence</option>
                    <option value="sexual_assault">Assault</option>
                </select>

                <select className="input" style={{ width: 160 }} value={ackFilter} onChange={(e) => setAckFilter(e.target.value)}>
                    <option value="">All Statuses</option>
                    <option value="false">Pending</option>
                    <option value="true">Acknowledged</option>
                </select>

                <input type="number" placeholder="Cam ID" className="input" style={{ width: 120 }} value={camFilter} onChange={(e) => setCamFilter(e.target.value)} />

                <button className="btn btn-outline btn-sm" onClick={fetchFiltered}>Search</button>

                <div style={{ flex: 1 }} />
                <button className="btn btn-outline btn-sm" onClick={exportCsv}>
                    <Download size={16} /> Export CSV
                </button>
            </div>

            {/* ── Table ── */}
            <div className="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Camera</th>
                            <th>Type</th>
                            <th>Score</th>
                            <th>Status</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.items.length === 0 ? (
                            <tr>
                                <td colSpan={6} style={{ textAlign: "center", padding: "2rem", color: "var(--muted)" }}>
                                    No incidents found.
                                </td>
                            </tr>
                        ) : (
                            data.items.map((inc) => (
                                <tr key={inc.id}>
                                    <td>
                                        {format(new Date(inc.created_at + "Z"), "MMM d, HH:mm:ss")}
                                    </td>
                                    <td>Cam {inc.camera_id}</td>
                                    <td>
                                        <span className={`badge ${inc.acknowledged ? "badge-muted" : "badge-red"}`}>
                                            {inc.incident_type.toUpperCase().replace("_", " ")}
                                        </span>
                                    </td>
                                    <td>
                                        <div style={{ display: "flex", alignItems: "center", gap: 8, width: 120 }}>
                                            <span style={{ fontSize: "0.8rem", width: 35 }}>
                                                {(inc.fusion_score * 100).toFixed(0)}%
                                            </span>
                                            <div className="progress-bar" style={{ flex: 1 }}>
                                                <div
                                                    className="progress-fill"
                                                    style={{
                                                        width: `${inc.fusion_score * 100}%`,
                                                        background: inc.acknowledged ? "var(--muted)" : "var(--accent)"
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    </td>
                                    <td>
                                        {inc.acknowledged ? (
                                            <span className="badge badge-green">Acknowledged</span>
                                        ) : (
                                            <span className="badge badge-yellow">Pending</span>
                                        )}
                                    </td>
                                    <td>
                                        <button className="btn btn-ghost btn-sm" onClick={() => setSelectedIncident(inc)}>
                                            <Eye size={16} /> View
                                        </button>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>

            {/* ── Detail Drawer ── */}
            {selectedIncident && (
                <>
                    <div className="sheet-overlay" onClick={() => setSelectedIncident(null)} />
                    <div className="sheet-panel">
                        <h2 style={{ fontSize: "1.25rem", marginTop: 0 }}>Incident Detail #{selectedIncident.id}</h2>
                        <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
                            {format(new Date(selectedIncident.created_at + "Z"), "MMMM d, yyyy 'at' HH:mm:ss")}
                        </p>

                        {selectedIncident.snapshot_path ? (
                            <img
                                src={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/incidents/snapshot/${selectedIncident.id}`}
                                alt="Incident Snapshot"
                                style={{ width: "100%", borderRadius: 8, marginBottom: "1.5rem", border: "1px solid var(--border)" }}
                                onError={(e) => (e.currentTarget.style.display = "none")}
                            />
                        ) : (
                            <div style={{ padding: "2rem", textAlign: "center", background: "var(--card-hover)", borderRadius: 8, color: "var(--muted)", marginBottom: "1.5rem" }}>
                                No snapshot available
                            </div>
                        )}

                        <h3 style={{ fontSize: "1rem", marginBottom: "1rem" }}>Score Breakdown</h3>
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                            {[
                                { label: "VideoMAE (Violence)", score: selectedIncident.videomae_score },
                                { label: "BiLSTM (Assault)", score: selectedIncident.bilstm_score },
                                { label: "RAFT (Optical Flow)", score: selectedIncident.optflow_score },
                                { label: "CLIP (Zero-Shot Context)", score: selectedIncident.clip_score },
                                { label: "Pose Distress", score: selectedIncident.pose_score },
                                { label: "Final Fusion", score: selectedIncident.fusion_score },
                            ].map((s, i) => (
                                <div key={i} className="card" style={{ padding: "0.75rem 1rem" }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: "0.85rem" }}>
                                        <span>{s.label}</span>
                                        <span style={{ fontWeight: 600 }}>{((s.score || 0) * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="progress-bar">
                                        <div
                                            className="progress-fill" style={{ width: `${(s.score || 0) * 100}%`, background: "var(--primary)" }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div style={{ marginTop: "2rem", display: "flex", gap: "1rem" }}>
                            <button
                                className="btn btn-primary"
                                style={{ flex: 1, justifyContent: "center" }}
                                disabled={selectedIncident.acknowledged}
                                onClick={() => handleAck(selectedIncident.id)}
                            >
                                {selectedIncident.acknowledged ? "Already Acknowledged" : "Acknowledge Incident"}
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
