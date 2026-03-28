"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import axios from "axios";
import { useAppStore } from "@/store/useAppStore";
import CameraGrid from "@/components/CameraGrid";
import AlertBanner from "@/components/AlertBanner";

export default function LiveMonitor() {
  const router = useRouter();
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const unacknowledgedCount = useAppStore((s) => s.unacknowledgedCount);

  useEffect(() => {
    // Auth check
    const token = localStorage.getItem("access_token");
    if (!token) {
      router.push("/login");
      return;
    }

    const fetchCameras = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await axios.get(`${apiUrl}/api/v1/cameras`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setCameras(res.data);
      } catch (err) {
        console.error("Error fetching cameras:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchCameras();
  }, [router]);

  return (
    <>
      <AlertBanner />

      <div className="page-header">
        <h1>Live Monitor</h1>
        <p>
          {loading ? "Loading cameras..." : `${cameras.length} cameras active`} ·{" "}
          {unacknowledgedCount} alerts pending
        </p>
      </div>

      <div className="page-content">
        {loading ? (
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <div className="skeleton" style={{ width: 560, height: 315 }}></div>
            <div className="skeleton" style={{ width: 560, height: 315 }}></div>
          </div>
        ) : (
          <CameraGrid cameras={cameras} />
        )}
      </div>
    </>
  );
}
