import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, AlertCircle } from 'lucide-react';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ApiResponse {
  response: string;
  session_id: string;
  crisis_detected?: boolean;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'init',
      role: 'assistant',
      content:
        "Hello! I'm HopeGuide. I’m here to support you and guide you through a mental health check-in. How are you feeling today?",
      timestamp: new Date(),
    },
  ]);

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const apiBase = import.meta.env.VITE_PHQ9_API_URL;
    if (!apiBase) {
      setError('Backend API URL not configured.');
      return;
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    // ✅ create the new message list ONCE
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    setInput('');
    setLoading(true);
    setError('');

    try {
      const payload: any = {
        messages: updatedMessages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      };

      // ✅ only include session_id if it exists
      if (sessionId) {
        payload.session_id = sessionId;
      }

      const res = await fetch(`${apiBase}/phq9-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Backend error ${res.status}: ${text}`);
      }

      const data: ApiResponse = await res.json();

      if (data.session_id) {
        setSessionId(data.session_id);
      }

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      setError('Unable to reach HopeGuide backend.');

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content:
            "I'm having trouble connecting right now. If you're in distress, please contact a crisis helpline or emergency services.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex items-start gap-3 ${
              m.role === 'user' ? 'flex-row-reverse' : ''
            }`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center ${
                m.role === 'user'
                  ? 'bg-blue-600'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600'
              }`}
            >
              {m.role === 'user' ? (
                <User className="w-4 h-4 text-white" />
              ) : (
                <Bot className="w-4 h-4 text-white" />
              )}
            </div>

            <Card className="max-w-[80%]">
              <div className="p-4">
                <p className="whitespace-pre-wrap text-gray-800">
                  {m.content}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {m.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </Card>
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Bot className="w-4 h-4" />
            HopeGuide is thinking…
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {error && (
        <div className="p-3 bg-yellow-50 border border-yellow-200 text-sm flex gap-2">
          <AlertCircle className="w-4 h-4 text-yellow-600" />
          {error}
        </div>
      )}

      <div className="border-t p-4">
        <div className="flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your response…"
            className="flex-1 resize-none border rounded-lg p-3"
            rows={1}
          />
          <Button onClick={sendMessage} disabled={loading || !input.trim()}>
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
