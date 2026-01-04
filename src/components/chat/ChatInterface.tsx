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

interface ChatInterfaceProps {
  sessionId?: string;
}

export function ChatInterface({ sessionId: externalSessionId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        "Hello! I'm HopeGuide, your compassionate AI companion. I'm here to provide support and help you with mental health assessments. How are you feeling today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Use external sessionId or generate one
  const [sessionId] = useState(() => externalSessionId || crypto.randomUUID());

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError('');

    try {
      // const response = await fetch('/phq9-chat', for local dev
      const response = await fetch('https://bolt-hopeguide.onrender.com/phq9-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage].map((m) => ({
            role: m.role,
            content: m.content,
          })),
          session_id: sessionId, // ✅ critical fix: use snake_case
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `HTTP error! status: ${response.status}, message: ${errorText}`,
        );
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content:
          data.response ||
          "I'm here to listen. Could you tell me more about how you're feeling?",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Unable to connect to the chat service.';
      setError(
        `Failed to send message: ${errorMessage}. Ensure the backend server is running on http://localhost:8000 and the Vite proxy is configured in vite.config.ts to forward /phq9-chat to http://localhost:8000. For production, set VITE_PHQ9_API_URL to the backend URL.`,
      );

      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content:
          "I'm having trouble connecting to the chat service right now. Please ensure the backend server is running and reachable. In the meantime, I'm still here to listen. You can continue our conversation, and I'll do my best to provide support. If you're experiencing a mental health crisis, please reach out to a crisis helpline or emergency services immediately.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, fallbackMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start space-x-3 ${
              message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            }`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.role === 'user'
                  ? 'bg-blue-600'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600'
              }`}
            >
              {message.role === 'user' ? (
                <User className="w-4 h-4 text-white" />
              ) : (
                <Bot className="w-4 h-4 text-white" />
              )}
            </div>

            <Card
              className={`max-w-[80%] ${
                message.role === 'user' ? 'bg-blue-50 border-blue-200' : 'bg-white'
              }`}
            >
              <div className="p-4">
                <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">
                  {message.content}
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </Card>
          </div>
        ))}

        {loading && (
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <Card className="bg-white">
              <div className="p-4">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.1s' }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.2s' }}
                    ></div>
                  </div>
                  <span className="text-sm text-gray-500">
                    HopeGuide is thinking...
                  </span>
                </div>
              </div>
            </Card>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error Message */}
      {error && (
        <div className="mx-4 mb-2">
          <div className="flex items-center space-x-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <AlertCircle className="w-4 h-4 text-yellow-600 flex-shrink-0" />
            <div className="text-sm text-yellow-700">
              <p className="font-medium">Connection Issue</p>
              <p>{error}</p>
              <p className="mt-1 text-xs">
                To fix this: Ensure the backend server is running on http://localhost:8000 and the Vite proxy is configured in vite.config.ts to forward /phq9-chat to http://localhost:8000. For production, set VITE_PHQ9_API_URL to the backend URL.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex space-x-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Share your thoughts or feelings..."
            className="flex-1 resize-none border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            rows={1}
            style={{ minHeight: '44px', maxHeight: '120px' }}
          />
          <Button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            className="px-4 py-3"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>

        <p className="text-xs text-gray-500 mt-2 text-center">
          This is an AI assistant. For emergencies, please contact emergency
          services or a crisis helpline.
        </p>
      </div>
    </div>
  );
}
