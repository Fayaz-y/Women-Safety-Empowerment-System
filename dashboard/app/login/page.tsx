"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Shield } from "lucide-react";
import axios from "axios";
import toast from "react-hot-toast";

export default function Login() {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const router = useRouter();

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            // OAuth2 password request requires form data, not JSON
            const formData = new URLSearchParams();
            formData.append("username", username);
            formData.append("password", password);

            const res = await axios.post(`${apiUrl}/api/v1/auth/login`, formData, {
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
            });

            localStorage.setItem("access_token", res.data.access_token);
            toast.success("Login successful");
            router.push("/");
        } catch (err: any) {
            toast.error(err.response?.data?.detail || "Login failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{
            display: "flex",
            height: "100vh",
            width: "100vw",
            alignItems: "center",
            justifyContent: "center",
            background: "var(--background)",
            position: "fixed",
            top: 0,
            left: 0,
            zIndex: 100 // cover sidebar
        }}>
            <div className="card" style={{ width: 400, padding: "2.5rem 2rem" }}>
                <div style={{ textAlign: "center", marginBottom: "2rem" }}>
                    <div style={{
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                        width: 56,
                        height: 56,
                        borderRadius: "50%",
                        background: "rgba(99, 102, 241, 0.1)",
                        marginBottom: "1rem"
                    }}>
                        <Shield size={32} color="var(--primary)" />
                    </div>
                    <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: 0 }}>
                        Operator Login
                    </h1>
                    <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
                        Women Safety AI System Context
                    </p>
                    <div style={{ marginTop: "1rem", padding: "0.75rem", background: "rgba(99, 102, 241, 0.05)", borderRadius: "8px", fontSize: "0.85rem", color: "var(--muted)" }}>
                        <strong>Demo Credentials:</strong><br />
                        Username: <code>admin</code><br />
                        Password: <code>changeme</code>
                    </div>
                </div>

                <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <div>
                        <input
                            type="text"
                            placeholder="Username"
                            className="input"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                        />
                    </div>
                    <div>
                        <input
                            type="password"
                            placeholder="Password"
                            className="input"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        className="btn btn-primary"
                        style={{ width: "100%", justifyContent: "center", padding: "0.75rem", marginTop: "0.5rem" }}
                        disabled={loading}
                    >
                        {loading ? "Authenticating..." : "Sign In"}
                    </button>
                </form>
            </div>
        </div>
    );
}
