import React, { useState } from "react";
import { Button } from "../ui/Button";
import { Card, CardContent, CardHeader } from "../ui/Card";
import { sendPhq9Message } from "@/lib/phq9Api";

const phq9Questions = [
  "Over the last two weeks, have you found little interest or pleasure in doing things?",
  "Have you been feeling down, depressed, or hopeless?",
  "Have you had trouble falling or staying asleep, or slept too much?",
  "Have you felt tired or had little energy?",
  "Have you had poor appetite or tended to overeat?",
  "Have you felt bad about yourself, or that you are a failure or have let yourself or your family down?",
  "Have you had trouble concentrating on things, such as reading, work, or watching television?",
  "Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?",
  "Have you had thoughts of self-harm or felt that you would be better off dead?",
];

const responseOptions = [
  { value: 0, label: "Not at all" },
  { value: 1, label: "Several days" },
  { value: 2, label: "More than half the days" },
  { value: 3, label: "Nearly every day" },
];

export function PHQ9Assessment() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [botMessage, setBotMessage] = useState<string | null>(null);
  const [completed, setCompleted] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleResponse = async (value: number) => {
    const label =
      responseOptions.find((o) => o.value === value)?.label ?? "";

    setLoading(true);

    try {
      const res = await sendPhq9Message(label, sessionId);

      if (res.session_id) {
        setSessionId(res.session_id);
      }

      setBotMessage(res.response);

      if (res.response.toLowerCase().includes("assessment complete")) {
        setCompleted(true);
      } else {
        setCurrentQuestion((q) => Math.min(q + 1, phq9Questions.length - 1));
      }
    } catch (err) {
      setBotMessage("⚠️ Unable to contact assessment service.");
    } finally {
      setLoading(false);
    }
  };

  if (completed) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <Card>
          <CardHeader>
            <h2 className="text-2xl font-bold text-center">
              Assessment Complete
            </h2>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-gray-700">{botMessage}</p>

            <div className="text-center">
              <Button onClick={() => window.location.reload()}>
                Start New Assessment
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <Card>
        <CardHeader>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-bold">PHQ-9 Assessment</h2>
              <span className="text-sm text-gray-500">
                {currentQuestion + 1} / {phq9Questions.length}
              </span>
            </div>

            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{
                  width: `${
                    ((currentQuestion + 1) / phq9Questions.length) * 100
                  }%`,
                }}
              />
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {botMessage && (
            <div className="p-4 bg-gray-50 border rounded-lg text-gray-800">
              {botMessage}
            </div>
          )}

          <h3 className="text-lg font-medium leading-relaxed">
            {phq9Questions[currentQuestion]}
          </h3>

          <div className="space-y-3">
            {responseOptions.map((option) => (
              <button
                key={option.value}
                disabled={loading}
                onClick={() => handleResponse(option.value)}
                className="w-full p-4 text-left border rounded-lg hover:bg-blue-50 transition disabled:opacity-50"
              >
                {option.label}
              </button>
            ))}
          </div>

          {loading && (
            <div className="text-sm text-gray-500 text-center">
              Thinking…
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
