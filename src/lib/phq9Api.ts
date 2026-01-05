// In development: uses Vite proxy (empty string = relative URLs)
// In production: uses full backend URL from environment variable
const API_BASE =
  import.meta.env.MODE === 'development'
    ? '' // Empty string uses proxy in dev
    : (import.meta.env.VITE_PHQ9_API_URL || 'http://34.180.46.170:8000').replace(/\/$/, "");

console.log('API Mode:', import.meta.env.MODE);
console.log('API Base URL:', API_BASE);

export async function sendPhq9Message(
  content: string,
  sessionId?: string
) {
  const url = `${API_BASE}/phq9-chat`;
  
  console.log('Sending request to:', url);
  
  const res = await fetch(url, {
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

export async function checkHealth() {
  const url = `${API_BASE}/health`;
  
  const res = await fetch(url);
  
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  
  return res.json();
}