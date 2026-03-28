"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Camera, AlertTriangle, Activity, Settings, Shield } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";

export default function Sidebar() {
    const pathname = usePathname();
    const unreadCount = useAppStore((s) => s.unacknowledgedCount);

    const links = [
        { label: "Live Monitor", path: "/", icon: Camera },
        { label: "Incidents", path: "/incidents", icon: AlertTriangle },
        { label: "System Status", path: "/status", icon: Activity },
        { label: "Settings", path: "/settings", icon: Settings },
    ];

    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <Shield size={24} color="var(--primary)" />
                <span>Women Safety AI</span>
            </div>

            <nav className="sidebar-nav">
                {links.map((link) => {
                    const Icon = link.icon;
                    const isActive = pathname === link.path;

                    return (
                        <Link
                            key={link.path}
                            href={link.path}
                            className={`sidebar-link ${isActive ? "active" : ""}`}
                        >
                            <Icon size={18} />
                            {link.label}

                            {/* Unread badge for Live Monitor */}
                            {link.path === "/" && unreadCount > 0 && (
                                <span
                                    className="badge badge-red animate-pulse-glow"
                                    style={{ marginLeft: "auto", fontSize: "0.65rem", padding: "0.1rem 0.4rem" }}
                                >
                                    {unreadCount}
                                </span>
                            )}
                        </Link>
                    );
                })}
            </nav>

            <div style={{ padding: "1rem", marginTop: "auto", fontSize: "0.75rem", color: "var(--muted)", textAlign: "center" }}>
                System v1.0.0
            </div>
        </aside>
    );
}
