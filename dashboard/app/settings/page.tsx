"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import toast from "react-hot-toast";

export default function Settings() {
    const [loading, setLoading] = useState(true);
    const [config, setConfig] = useState<any>({});
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        const fetchConfig = async () => {
            try {
                const token = localStorage.getItem("access_token");
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                const res = await axios.get(`${apiUrl}/api/v1/system/config`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                setConfig(res.data);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchConfig();
    }, []);

    const handleChange = (key: string, value: any) => {
        setConfig({ ...config, [key]: value });
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            const token = localStorage.getItem("access_token");
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            await axios.put(`${apiUrl}/api/v1/system/config`, config, {
                headers: { Authorization: `Bearer ${token}` }
            });
            toast.success("Settings saved successfully");
        } catch (err: any) {
            toast.error(err.response?.data?.detail || "Failed to save settings");
        } finally {
            setSaving(false);
        }
    };

    if (loading) return <div className="page-header"><p>Loading system configuration...</p></div>;

    return (
        <>
            <div className="page-header" style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center"
            }}>
                <div>
                    <h1>System Settings</h1>
                    <p>Configure thresholds and alert destinations</p>
                </div>
                <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
                    {saving ? "Saving..." : "Save Changes"}
                </button>
            </div>

            <div className="page-content" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>

                {/* Engine Thresholds */}
                <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
                    <div className="card">
                        <h3 style={{ margin: "0 0 1.5rem" }}>Detection Thresholds</h3>

                        <div style={{ marginBottom: "1.5rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                <label style={{ fontSize: "0.85rem", fontWeight: 600 }}>Fusion Alert Threshold ({(config.FUSION_THRESHOLD * 100).toFixed(0)}%)</label>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={config.FUSION_THRESHOLD || 0.8}
                                onChange={(e) => handleChange("FUSION_THRESHOLD", parseFloat(e.target.value))}
                            />
                            <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "0.25rem" }}>
                                Minimum score required for an alert to be triggered and dispatched.
                            </p>
                        </div>

                        <div style={{ marginBottom: "1.5rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                <label style={{ fontSize: "0.85rem", fontWeight: 600 }}>Violence Threshold ({(config.VIOLENCE_THRESHOLD * 100).toFixed(0)}%)</label>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={config.VIOLENCE_THRESHOLD || 0.65}
                                onChange={(e) => handleChange("VIOLENCE_THRESHOLD", parseFloat(e.target.value))}
                            />
                            <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "0.25rem" }}>
                                VideoMAE sensitivity to violent actions.
                            </p>
                        </div>

                        <div style={{ marginBottom: "1.5rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                <label style={{ fontSize: "0.85rem", fontWeight: 600 }}>Assault Threshold ({(config.ASSAULT_THRESHOLD * 100).toFixed(0)}%)</label>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={config.ASSAULT_THRESHOLD || 0.70}
                                onChange={(e) => handleChange("ASSAULT_THRESHOLD", parseFloat(e.target.value))}
                            />
                            <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "0.25rem" }}>
                                BiLSTM sensitivity for sexual assault patterns.
                            </p>
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginTop: "1.5rem" }}>
                            <div>
                                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem" }}>Lone Woman Timer (sec)</label>
                                <input type="number" className="input" value={config.ISOLATION_SECONDS || 30} onChange={(e) => handleChange("ISOLATION_SECONDS", parseInt(e.target.value) || 30)} />
                            </div>
                            <div>
                                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem" }}>Proximity Radius (px)</label>
                                <input type="number" className="input" value={config.PROXIMITY_RADIUS_PX || 150} onChange={(e) => handleChange("PROXIMITY_RADIUS_PX", parseInt(e.target.value) || 150)} />
                            </div>
                            <div>
                                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem" }}>Alert Cooldown (sec)</label>
                                <input type="number" className="input" value={config.ALERT_COOLDOWN_SECONDS || 60} onChange={(e) => handleChange("ALERT_COOLDOWN_SECONDS", parseInt(e.target.value) || 60)} />
                            </div>
                        </div>
                    </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>

                    {/* Signal Weights */}
                    <div className="card">
                        <h3 style={{ margin: "0 0 1.5rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            Signal Weights
                            <span style={{
                                fontSize: "0.9rem",
                                color: ((config.WEIGHT_VIDEOMAE || 0.35) + (config.WEIGHT_BILSTM || 0.25) + (config.WEIGHT_OPTFLOW || 0.15) + (config.WEIGHT_CLIP || 0.15) + (config.WEIGHT_POSE || 0.10)).toFixed(2) !== "1.00" ? "var(--accent)" : "var(--success)"
                            }}>
                                Sum: {((config.WEIGHT_VIDEOMAE || 0.35) + (config.WEIGHT_BILSTM || 0.25) + (config.WEIGHT_OPTFLOW || 0.15) + (config.WEIGHT_CLIP || 0.15) + (config.WEIGHT_POSE || 0.10)).toFixed(2)}
                            </span>
                        </h3>

                        {[
                            { key: "WEIGHT_VIDEOMAE", label: "VideoMAE (Violence)", max: 0.5 },
                            { key: "WEIGHT_BILSTM", label: "BiLSTM (Assault)", max: 0.5 },
                            { key: "WEIGHT_OPTFLOW", label: "RAFT (OptFlow)", max: 0.5 },
                            { key: "WEIGHT_CLIP", label: "CLIP (Context)", max: 0.3 },
                            { key: "WEIGHT_POSE", label: "Pose Distress", max: 0.2 }
                        ].map(w => (
                            <div key={w.key} style={{ marginBottom: "1.25rem" }}>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                    <label style={{ fontSize: "0.85rem", fontWeight: 600 }}>{w.label} ({(config[w.key] ?? 0).toFixed(2)})</label>
                                </div>
                                <input
                                    type="range" min="0" max={w.max} step="0.01"
                                    value={config[w.key] || 0}
                                    onChange={(e) => handleChange(w.key, parseFloat(e.target.value))}
                                />
                            </div>
                        ))}
                    </div>

                    {/* Dispatch Settings */}
                    <div className="card">
                        <h3 style={{ margin: "0 0 1.5rem" }}>Alert Dispatch</h3>

                        <div style={{ marginBottom: "1.5rem" }}>
                            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem" }}>SMS Alert Target</label>
                            <input
                                type="text"
                                className="input"
                                value={config.SMS_ALERT_TARGET || ""}
                                onChange={(e) => handleChange("SMS_ALERT_TARGET", e.target.value)}
                                placeholder="+1234567890"
                            />
                            <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "0.25rem" }}>
                                Twilio destination phone number for critical alerts.
                            </p>
                        </div>

                        <div style={{ marginBottom: "1.5rem" }}>
                            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem" }}>Environment Database</label>
                            <input
                                type="text"
                                className="input"
                                value={config.DATABASE_URL || ""}
                                disabled
                                style={{ opacity: 0.6 }}
                            />
                        </div>
                    </div>
                </div>

            </div>
        </>
    );
}
