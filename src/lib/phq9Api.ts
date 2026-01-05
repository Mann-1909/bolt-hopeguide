const API_BASE =
  import.meta.env.VITE_PHQ9_API_URL?.replace(/\/$/, "") || "";

export async function sendPhq9Message(
  content: string,
  sessionId?: string
) {
  const res = await fetch(`${API_BASE}/phq9-chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messages: [{ role: "user", content }],
      session_id: sessionId,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json();
}
