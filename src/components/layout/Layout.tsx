import React, { useState } from 'react'
import { Header } from './Header'
import { ChatInterface } from '../chat/ChatInterface'
import { PHQ9Assessment } from '../phq9/PHQ9Assessment'

export function Layout() {
  const [currentView, setCurrentView] = useState<'chat' | 'assessment'>('chat')

  const handleAssessmentComplete = (score: number, responses: number[]) => {
    // Here you could save the assessment results to Supabase
    console.log('Assessment completed:', { score, responses })
    // Optionally switch to chat view after assessment
    setCurrentView('chat')
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <Header currentView={currentView} onViewChange={setCurrentView} />
      
      <main className="flex-1 overflow-hidden">
        {currentView === 'chat' ? (
          <div className="h-full max-w-4xl mx-auto bg-white shadow-sm">
            <ChatInterface />
          </div>
        ) : (
          <div className="h-full overflow-y-auto py-8">
            <PHQ9Assessment onComplete={handleAssessmentComplete} />
          </div>
        )}
      </main>
    </div>
  )
}