import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Brain, Upload, Database, BarChart3, Settings2, Sun, Moon, Bot } from 'lucide-react';
import { cn } from '../../utils/helpers';
import { useTheme } from '../../contexts/ThemeContext';
import Button from '../ui/Button';

const navigation = [
  { name: 'Upload', href: '/upload', icon: <Upload className="h-4 w-4" /> },
  { name: 'Files', href: '/files', icon: <Database className="h-4 w-4" /> },
  { name: 'AI & LLM', href: '/ai', icon: <Bot className="h-4 w-4" /> },
  { name: 'ML Training', href: '/ml', icon: <Brain className="h-4 w-4" /> },
  { name: 'Results', href: '/results', icon: <BarChart3 className="h-4 w-4" /> },
];

const Header = () => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();
  
  const handleThemeToggle = () => {
    console.log('Theme toggle button clicked');
    toggleTheme();
  };
  
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="flex items-center">
                <Brain className="h-8 w-8 text-purple-400" />
                <span className="ml-2 text-xl font-bold text-gray-900 dark:text-gray-100">Mini ML Platform</span>
              </Link>
            </div>
            <nav className="ml-8 flex space-x-8">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  className={cn(
                    'inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors',
                    location.pathname === item.href || 
                    (item.href !== '/upload' && location.pathname.startsWith(item.href))
                      ? 'border-purple-400 text-gray-900 dark:text-gray-100'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:border-gray-300 dark:hover:border-gray-600 hover:text-gray-700 dark:hover:text-gray-200'
                  )}
                >
                  <span className="mr-2">{item.icon}</span>
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleThemeToggle}
              className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            >
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium">DS2.3</span> Complete
            </div>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200">
              ML Platform v2.3
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
