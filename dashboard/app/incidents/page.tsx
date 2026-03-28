"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";
import IncidentTable from "@/components/IncidentTable";

export default function Incidents() {
    const router = useRouter();
    const [data, setData] = useState({ total: 0, items: [], page: 1, limit: 20 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        // Auth check
        const token = localStorage.getItem("access_token");
        if (!token) {
            router.push("/login");
            return;
        }

        const fetchIncidents = async () => {
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                const res = await axios.get(`${apiUrl}/api/v1/incidents?page=1&limit=20`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                setData(res.data);
            } catch (err: any) {
                if (err.response?.status === 401) {
                    router.push("/login");
                } else {
                    setError(err.message || "Failed to load incidents");
                }
            } finally {
                setLoading(false);
            }
        };

        fetchIncidents();
    }, [router]);

    return (
        <>
            <div className="page-header">
                <h1>Incident Log</h1>
                <p>
                    {loading
                        ? "Loading incidents..."
                        : `${data.total} incidents found in the database. Use filters to refine search.`}
                </p>
            </div>

            <div className="page-content">
                {error && (
                    <div className="card" style={{ background: "rgba(244, 63, 94, 0.1)", borderColor: "var(--accent)" }}>
                        <p style={{ color: "var(--accent)" }}>{error}</p>
                    </div>
                )}

                {!loading && !error && (
                    <IncidentTable initialData={data} />
                )}

                {loading && (
                    <div className="card" style={{ padding: "4rem 2rem", textAlign: "center", color: "var(--muted)" }}>
                        Loading incidents data...
                    </div>
                )}
            </div>
        </>
    );
}
