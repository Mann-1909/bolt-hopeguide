import React from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { Button } from '../ui/Button'
import { Heart, LogOut, User, MessageCircle } from 'lucide-react'

interface HeaderProps {
  currentView: 'chat' | 'assessment' // you can keep or update this if 'assessment' is no longer needed elsewhere
  onViewChange: (view: 'chat' | 'assessment') => void
}

export function Header({ currentView, onViewChange }: HeaderProps) {
  const { user, signOut } = useAuth()

  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              HopeGuide
            </h1>
          </div>

          {/* Navigation */}
          <div className="flex items-center space-x-4">
            <nav className="flex space-x-2">
              <Button
                variant={currentView === 'chat' ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => onViewChange('chat')}
                className="flex items-center space-x-2"
              >
                <MessageCircle className="w-4 h-4" />
                <span>Chat</span>
              </Button>
              {/* Removed Assessment Button */}
            </nav>

            {/* User Menu */}
            <div className="flex items-center space-x-3 pl-4 border-l border-gray-200">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-gray-600" />
                </div>
                <span className="text-sm text-gray-700 hidden sm:block">
                  {user?.email}
                </span>
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={signOut}
                className="flex items-center space-x-2"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:block">Sign Out</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
