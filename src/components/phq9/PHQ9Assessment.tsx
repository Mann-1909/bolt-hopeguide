import React, { useState } from 'react'
import { Button } from '../ui/Button'
import { Card, CardContent, CardHeader } from '../ui/Card'
import { CheckCircle, AlertTriangle, Info } from 'lucide-react'

const phq9Questions = [
  "Over the last two weeks, have you found little interest or pleasure in doing things?",
  "Have you been feeling down, depressed, or hopeless?",
  "Have you had trouble falling or staying asleep, or slept too much?",
  "Have you felt tired or had little energy?",
  "Have you had poor appetite or tended to overeat?",
  "Have you felt bad about yourself, or that you are a failure or have let yourself or your family down?",
  "Have you had trouble concentrating on things, such as reading, work, or watching television?",
  "Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?",
  "Have you had thoughts of self-harm or felt that you would be better off dead?"
]

const responseOptions = [
  { value: 0, label: "Not at all" },
  { value: 1, label: "Several days" },
  { value: 2, label: "More than half the days" },
  { value: 3, label: "Nearly every day" }
]

interface PHQ9AssessmentProps {
  onComplete: (score: number, responses: number[]) => void
}

export function PHQ9Assessment({ onComplete }: PHQ9AssessmentProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [responses, setResponses] = useState<number[]>(new Array(9).fill(-1))
  const [showResults, setShowResults] = useState(false)

  const handleResponse = (value: number) => {
    const newResponses = [...responses]
    newResponses[currentQuestion] = value
    setResponses(newResponses)

    if (currentQuestion < phq9Questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1)
    } else {
      // Calculate score and show results
      const totalScore = newResponses.reduce((sum, response) => sum + response, 0)
      setShowResults(true)
      onComplete(totalScore, newResponses)
    }
  }

  const goBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1)
    }
  }

  const getSeverityInfo = (score: number) => {
    if (score <= 4) return { level: "Minimal", color: "green", icon: CheckCircle }
    if (score <= 9) return { level: "Mild", color: "yellow", icon: Info }
    if (score <= 14) return { level: "Moderate", color: "orange", icon: AlertTriangle }
    if (score <= 19) return { level: "Moderately Severe", color: "red", icon: AlertTriangle }
    return { level: "Severe", color: "red", icon: AlertTriangle }
  }

  const totalScore = responses.reduce((sum, response) => sum + (response > -1 ? response : 0), 0)
  const severityInfo = getSeverityInfo(totalScore)
  const IconComponent = severityInfo.icon

  if (showResults) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <Card>
          <CardHeader>
            <div className="text-center space-y-4">
              <div className={`w-16 h-16 mx-auto rounded-full bg-${severityInfo.color}-100 flex items-center justify-center`}>
                <IconComponent className={`w-8 h-8 text-${severityInfo.color}-600`} />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Assessment Complete</h2>
                <p className="text-gray-600">Your PHQ-9 results</p>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-6">
            <div className="text-center">
              <div className="text-4xl font-bold text-gray-900 mb-2">{totalScore}/27</div>
              <div className={`inline-flex items-center px-4 py-2 rounded-full bg-${severityInfo.color}-100`}>
                <span className={`text-${severityInfo.color}-800 font-medium`}>
                  {severityInfo.level} Depression
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h3 className="font-semibold text-blue-900 mb-2">What this means:</h3>
                <p className="text-blue-800 text-sm leading-relaxed">
                  {totalScore <= 4 && "Your responses suggest minimal depression symptoms. Continue taking care of your mental health."}
                  {totalScore > 4 && totalScore <= 9 && "Your responses suggest mild depression symptoms. Consider speaking with a healthcare provider."}
                  {totalScore > 9 && totalScore <= 14 && "Your responses suggest moderate depression symptoms. It's recommended to consult with a mental health professional."}
                  {totalScore > 14 && totalScore <= 19 && "Your responses suggest moderately severe depression symptoms. Please consider seeking professional help."}
                  {totalScore > 19 && "Your responses suggest severe depression symptoms. Please reach out to a mental health professional or crisis support immediately."}
                </p>
              </div>

              {totalScore > 14 && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                  <h3 className="font-semibold text-red-900 mb-2">Crisis Resources:</h3>
                  <div className="text-red-800 text-sm space-y-1">
                    <p>• National Suicide Prevention Lifeline: 988</p>
                    <p>• Crisis Text Line: Text HOME to 741741</p>
                    <p>• Emergency Services: 911</p>
                  </div>
                </div>
              )}
            </div>

            <div className="text-center">
              <Button onClick={() => window.location.reload()}>
                Start New Assessment
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <Card>
        <CardHeader>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-bold text-gray-900">PHQ-9 Assessment</h2>
              <span className="text-sm text-gray-500">
                {currentQuestion + 1} of {phq9Questions.length}
              </span>
            </div>
            
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentQuestion + 1) / phq9Questions.length) * 100}%` }}
              />
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-6 leading-relaxed">
              {phq9Questions[currentQuestion]}
            </h3>
            
            <div className="space-y-3">
              {responseOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => handleResponse(option.value)}
                  className="w-full p-4 text-left border border-gray-300 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-4 h-4 border-2 border-gray-400 rounded-full"></div>
                    <span className="text-gray-900">{option.label}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
          
          {currentQuestion > 0 && (
            <div className="flex justify-start">
              <Button variant="outline" onClick={goBack}>
                Previous Question
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}