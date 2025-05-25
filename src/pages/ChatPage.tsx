import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, FileText } from 'lucide-react';
import PageLayout from '../components/layout/PageLayout';
import ChatInterface from '../components/chat/ChatInterface';
import Button from '../components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { PipelineRun, ChatMessage } from '../types';
import { getPipelineStatus, sendChatMessage } from '../services/api';

const ChatPage = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  
  const [pipelineRun, setPipelineRun] = useState<PipelineRun | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);

  // Fetch pipeline run details
  useEffect(() => {
    if (!runId) return;
    
    const fetchPipelineRun = async () => {
      try {
        setIsLoading(true);
        const run = await getPipelineStatus(runId);
        setPipelineRun(run);
        
        // Ensure this is a RAG chatbot pipeline
        if (run.pipeline_type !== 'RAG_CHATBOT') {
          setError('This is not a RAG chatbot pipeline');
        } else if (run.status !== 'COMPLETED') {
          setError('The RAG chatbot pipeline is not ready yet');
        } else {
          // Load chat history from localStorage (RAG system is stateless)
          const storedHistory = localStorage.getItem(`chat_history_${runId}`);
          if (storedHistory) {
            try {
              const history = JSON.parse(storedHistory);
              setMessages(history);
            } catch (e) {
              console.warn('Failed to parse stored chat history:', e);
              setMessages([]);
            }
          } else {
            // Add welcome message for new chats
            const welcomeMessage: ChatMessage = {
              id: crypto.randomUUID(),
              content: "Hello! I'm ready to answer questions about your document. What would you like to know?",
              role: 'assistant',
              timestamp: new Date().toISOString(),
            };
            setMessages([welcomeMessage]);
          }
          setError(null);
        }
      } catch (err) {
        setError('Failed to load pipeline details');
        console.error('Error fetching pipeline run:', err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchPipelineRun();
  }, [runId]);

  // Save chat history to localStorage whenever messages change
  useEffect(() => {
    if (runId && messages.length > 0) {
      localStorage.setItem(`chat_history_${runId}`, JSON.stringify(messages));
    }
  }, [runId, messages]);

  // Handle sending a message
  const handleSendMessage = async (message: string) => {
    if (!runId || isSending) return;
    
    // Add user message to the list
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      content: message,
      role: 'user',
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    try {
      setIsSending(true);
      
      // Send message to API
      const assistantMessage = await sendChatMessage(runId, message);
      
      // Add assistant message to the list
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      
      // Add error message
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        content: 'Sorry, an error occurred while processing your message.',
        role: 'assistant',
        timestamp: new Date().toISOString(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <PageLayout>
      <div className="space-y-6">
        <Button
          variant="ghost"
          size="sm"
          icon={<ChevronLeft className="w-4 h-4" />}
          onClick={() => navigate(-1)}
        >
          Back
        </Button>

        <div className="flex flex-col md:flex-row gap-6">
          <div className="md:w-1/3">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <FileText className="h-5 w-5 mr-2 text-blue-600" />
                  Document Context
                </CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <p className="text-gray-500">Loading document details...</p>
                ) : error ? (
                  <p className="text-red-500">{error}</p>
                ) : pipelineRun ? (
                  <div>
                    <h3 className="font-medium text-gray-900 mb-1">
                      {pipelineRun.result?.document_name || 'Untitled Document'}
                    </h3>
                    <p className="text-sm text-gray-500 mb-4">
                      This chatbot has context from your document and can answer questions about its content.
                    </p>
                    
                    {pipelineRun.result?.document_stats && (
                      <div className="bg-gray-50 p-3 rounded-md text-sm">
                        <div className="mb-2">
                          <span className="font-medium">Pages:</span> {pipelineRun.result.document_stats.page_count}
                        </div>
                        <div className="mb-2">
                          <span className="font-medium">Word Count:</span> {pipelineRun.result.document_stats.word_count}
                        </div>
                        <div>
                          <span className="font-medium">Chunks:</span> {pipelineRun.result.document_stats.chunk_count}
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </CardContent>
            </Card>
          </div>
          
          <div className="md:w-2/3">
            {isLoading ? (
              <div className="flex justify-center items-center h-64">
                <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
              </div>
            ) : error ? (
              <div className="bg-red-50 p-6 rounded-lg text-center">
                <h3 className="text-lg font-medium text-red-800 mb-2">{error}</h3>
                <p className="text-red-600 mb-4">
                  The RAG chatbot may still be processing or has encountered an error.
                </p>
                <Button
                  variant="outline"
                  onClick={() => navigate(-1)}
                >
                  Go Back
                </Button>
              </div>
            ) : (
              <ChatInterface
                messages={messages}
                onSendMessage={handleSendMessage}
                isLoading={isSending}
                className="h-[600px]"
              />
            )}
          </div>
        </div>
      </div>
    </PageLayout>
  );
};

export default ChatPage;