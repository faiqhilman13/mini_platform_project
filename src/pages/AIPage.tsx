import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import Textarea from '../components/ui/Textarea';
import Badge from '../components/ui/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/ModernTabs';
import { 
  MessageCircle, 
  FileText, 
  Send, 
  Upload, 
  Bot, 
  Brain,
  Sparkles,
  Upload as UploadIcon,
  Loader2,
  CheckCircle,
  AlertCircle,
  Clock
} from 'lucide-react';
import { getUploadedFiles, triggerPipeline, getPipelineStatus } from '../services/api';
import { UploadedFile } from '../types';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface PipelineResult {
  run_uuid: string;
  status: string;
  result?: any;
  error_message?: string;
}

interface LocationState {
  selectedFile?: UploadedFile;
}

const AIPage: React.FC = () => {
  const location = useLocation();
  const state = location.state as LocationState;
  
  const [activeTab, setActiveTab] = useState('rag');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(state?.selectedFile || null);
  const [loading, setLoading] = useState(false);
  
  // RAG Chat State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [ragLoading, setRagLoading] = useState(false);
  const [ragPipelineRunId, setRagPipelineRunId] = useState<string | null>(null);
  const [ragProcessingStatus, setRagProcessingStatus] = useState<'idle' | 'processing' | 'ready' | 'error'>('idle');
  
  // Summarizer State
  const [summarizerLoading, setSummarizerLoading] = useState(false);
  const [summaryResult, setSummaryResult] = useState<any>(null);

  // Helper function to check if file is compatible with LLMs
  const isDocumentFile = (fileType: string) => {
    // Handle both simplified FileType values and full MIME types
    const documentTypes = [
      'pdf',
      'text',
      'application/pdf',
      'text/plain',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    return documentTypes.includes(fileType);
  };

  // Helper function to convert bytes to MB
  const bytesToMB = (bytes: number): number => {
    return bytes / (1024 * 1024);
  };

  useEffect(() => {
    loadFiles();
    // If a file was passed from the files page, auto-select the summarizer tab if it's a PDF
    if (state?.selectedFile && state.selectedFile.file_type === 'pdf') {
      setActiveTab('summarizer');
    }
  }, []);

  const loadFiles = async () => {
    try {
      setLoading(true);
      const filesList = await getUploadedFiles();
      
      // Filter only document files that can work with LLMs
      const documentFiles = filesList.filter((file: UploadedFile) => isDocumentFile(file.file_type as string));
      setFiles(documentFiles);
    } catch (error) {
      console.error('Error loading files:', error);
    } finally {
      setLoading(false);
    }
  };

  // RAG Chat Functions
  const handleProcessDocumentForRAG = async () => {
    if (!selectedFile || ragProcessingStatus === 'processing') return;

    setRagProcessingStatus('processing');
    setRagPipelineRunId(null);
    setChatMessages([]);

    try {
      const response = await triggerPipeline(
        selectedFile.id.toString(),
        'RAG_CHATBOT'
      );

      // Poll for completion
      const pollForCompletion = async (runUuid: string) => {
        const maxAttempts = 30;
        let attempts = 0;

        const poll = async (): Promise<void> => {
          if (attempts >= maxAttempts) {
            throw new Error('RAG processing timeout');
          }

          const status = await getPipelineStatus(runUuid);
          
          if (status.status === 'COMPLETED') {
            setRagPipelineRunId(runUuid);
            setRagProcessingStatus('ready');
            
            // Add welcome message
            const welcomeMessage: ChatMessage = {
              id: Date.now().toString(),
              type: 'assistant',
              content: `Hello! I've processed your document "${selectedFile.filename}" and I'm ready to answer questions about it. What would you like to know?`,
              timestamp: new Date()
            };
            setChatMessages([welcomeMessage]);
            return;
          } else if (status.status === 'FAILED') {
            throw new Error(status.error_message || 'RAG processing failed');
          }

          attempts++;
          setTimeout(poll, 2000);
        };

        await poll();
      };

      await pollForCompletion(response.run_uuid);
    } catch (error) {
      console.error('Error processing document for RAG:', error);
      setRagProcessingStatus('error');
      
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'assistant',
        content: 'Sorry, there was an error processing your document for RAG. Please try again.',
        timestamp: new Date()
      };
      setChatMessages([errorMessage]);
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim() || ragLoading || !ragPipelineRunId) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: chatInput,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setRagLoading(true);

    try {
      const response = await fetch('/api/v1/rag/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          pipeline_run_id: ragPipelineRunId,
          question: userMessage.content 
        }),
      });

      const data = await response.json();
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.answer || 'Sorry, I could not process your question.',
        timestamp: new Date()
      };

      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, there was an error processing your question. Please try again.',
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setRagLoading(false);
    }
  };

  // Summarizer Functions
  const handleSummarize = async () => {
    if (!selectedFile || summarizerLoading) return;

    setSummarizerLoading(true);
    setSummaryResult(null);

    try {
      const response = await triggerPipeline(
        selectedFile.id.toString(),
        'PDF_SUMMARIZER'
      );

      // Poll for results
      const pollForResults = async (runUuid: string) => {
        const maxAttempts = 30;
        let attempts = 0;

        const poll = async (): Promise<void> => {
          if (attempts >= maxAttempts) {
            throw new Error('Summarization timeout');
          }

          const status = await getPipelineStatus(runUuid);
          
          if (status.status === 'COMPLETED') {
            setSummaryResult(status);
            return;
          } else if (status.status === 'FAILED') {
            throw new Error(status.error_message || 'Summarization failed');
          }

          attempts++;
          setTimeout(poll, 2000);
        };

        await poll();
      };

      await pollForResults(response.run_uuid);
    } catch (error) {
      console.error('Error during summarization:', error);
      setSummaryResult({ 
        status: 'FAILED', 
        error_message: error instanceof Error ? error.message : 'Unknown error' 
      });
    } finally {
      setSummarizerLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            <Brain className="inline-block w-10 h-10 mr-3 text-purple-600" />
            AI & Language Models
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Powerful AI tools for document analysis and intelligent conversations
          </p>
          {state?.selectedFile && (
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-800 dark:text-blue-300">
                <FileText className="inline w-4 h-4 mr-1" />
                Working with: <span className="font-medium">{state.selectedFile.filename}</span>
              </p>
            </div>
          )}
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="rag" className="flex items-center gap-2">
              <MessageCircle className="w-4 h-4" />
              RAG Chatbot
            </TabsTrigger>
            <TabsTrigger value="summarizer" className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Document Summarizer
            </TabsTrigger>
          </TabsList>

          {/* RAG Chatbot Tab */}
          <TabsContent value="rag" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bot className="w-5 h-5 text-blue-600" />
                  RAG Chatbot
                </CardTitle>
                <CardDescription>
                  Ask questions about your uploaded documents using Retrieval-Augmented Generation
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Document Selection and Processing */}
                <div className="mb-6">
                  <div className="flex flex-col gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Select Document for RAG Processing
                      </label>
                      <select
                        value={selectedFile?.id || ''}
                        onChange={(e) => {
                          const fileId = e.target.value;
                          const file = files.find(f => f.id.toString() === fileId);
                          setSelectedFile(file || null);
                          // Reset RAG state when changing files
                          setRagProcessingStatus('idle');
                          setRagPipelineRunId(null);
                          setChatMessages([]);
                        }}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                        disabled={ragProcessingStatus === 'processing'}
                      >
                        <option value="">Select a document...</option>
                        {files.map((file) => (
                          <option key={file.id} value={file.id}>
                            {file.filename} ({bytesToMB(file.size_bytes).toFixed(2)} MB)
                          </option>
                        ))}
                      </select>
                    </div>
                    
                    {selectedFile && (
                      <div className="flex items-center gap-4">
                        <Button
                          onClick={handleProcessDocumentForRAG}
                          disabled={ragProcessingStatus === 'processing'}
                          className="flex items-center gap-2"
                        >
                          {ragProcessingStatus === 'processing' ? (
                            <>
                              <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                              Processing...
                            </>
                          ) : (
                            <>
                              <FileText className="w-4 h-4" />
                              Process Document for RAG
                            </>
                          )}
                        </Button>
                        
                        {ragProcessingStatus === 'ready' && (
                          <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                            <div className="w-2 h-2 bg-green-500 rounded-full" />
                            <span className="text-sm font-medium">Ready for questions</span>
                          </div>
                        )}
                        
                        {ragProcessingStatus === 'error' && (
                          <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
                            <div className="w-2 h-2 bg-red-500 rounded-full" />
                            <span className="text-sm font-medium">Processing failed</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Chat Messages */}
                <div className="h-96 border rounded-lg p-4 mb-4 overflow-y-auto bg-gray-50 dark:bg-gray-800">
                  {chatMessages.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                      <div className="text-center">
                        <MessageCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        {ragProcessingStatus === 'idle' ? (
                          <div>
                            <p>Select a document and process it for RAG to start chatting</p>
                            {files.length > 0 && (
                              <p className="text-xs mt-2">Available documents: {files.length}</p>
                            )}
                          </div>
                        ) : ragProcessingStatus === 'processing' ? (
                          <p>Processing document for RAG...</p>
                        ) : ragProcessingStatus === 'ready' ? (
                          <p>Document processed! Ask your first question below.</p>
                        ) : (
                          <p>Document processing failed. Please try again.</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {chatMessages.map((message) => (
                        <div
                          key={message.id}
                          className={`flex ${
                            message.type === 'user' ? 'justify-end' : 'justify-start'
                          }`}
                        >
                          <div
                            className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                              message.type === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-600'
                            }`}
                          >
                            <p className="text-sm">{message.content}</p>
                          </div>
                        </div>
                      ))}
                      
                      {ragLoading && (
                        <div className="flex justify-start">
                          <div className="bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
                            <div className="flex items-center space-x-2">
                              <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                              <p className="text-sm text-gray-600 dark:text-gray-300">Thinking...</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Chat Input */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder={
                      ragProcessingStatus === 'ready' 
                        ? "Ask a question about the document..." 
                        : "Process a document first to enable chat"
                    }
                    disabled={ragLoading || ragProcessingStatus !== 'ready'}
                    className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white disabled:opacity-50"
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={ragLoading || !chatInput.trim() || ragProcessingStatus !== 'ready'}
                    className="flex items-center gap-2"
                  >
                    <Send className="w-4 h-4" />
                    Send
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Document Summarizer Tab */}
          <TabsContent value="summarizer" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* File Selection */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <UploadIcon className="w-5 h-5 text-green-600" />
                    Select Document
                  </CardTitle>
                  <CardDescription>
                    Choose a document to summarize (PDF, DOC, DOCX, TXT)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-6 h-6 animate-spin" />
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {files.map((file) => (
                        <div
                          key={file.id}
                          className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                            selectedFile?.id === file.id
                              ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                              : 'border-gray-200 dark:border-gray-700 hover:border-green-300'
                          }`}
                          onClick={() => setSelectedFile(file)}
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium text-gray-900 dark:text-white">{file.filename}</p>
                              <p className="text-sm text-gray-500">{bytesToMB(file.size_bytes).toFixed(2)} MB</p>
                            </div>
                            {selectedFile?.id === file.id && (
                              <CheckCircle className="w-5 h-5 text-green-600" />
                            )}
                          </div>
                        </div>
                      ))}
                      {files.length === 0 && (
                        <div className="text-center py-8 text-gray-500">
                          <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                          <p>No document files found. Upload PDF, DOC, DOCX, or TXT files first.</p>
                        </div>
                      )}
                    </div>
                  )}
                  
                  <Button 
                    onClick={handleSummarize}
                    disabled={!selectedFile || summarizerLoading}
                    className="w-full mt-4"
                  >
                    {summarizerLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Summarizing...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4 mr-2" />
                        Generate Summary
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Summary Results */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-blue-600" />
                    Summary Results
                  </CardTitle>
                  <CardDescription>
                    AI-generated document summary
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {summaryResult ? (
                    <div className="space-y-4">
                      {summaryResult.status === 'COMPLETED' && summaryResult.result ? (
                        <div className="space-y-4">
                          <Badge variant="success" className="flex items-center gap-1 w-fit">
                            <CheckCircle className="w-3 h-3" />
                            Summary Complete
                          </Badge>
                          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                            <h4 className="font-medium mb-2">Summary:</h4>
                            <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                              {(() => {
                                // Handle nested result structure from pipeline
                                const result = summaryResult.result;
                                
                                // If result has a summary property (nested structure)
                                if (result && typeof result === 'object' && result.summary) {
                                  return Array.isArray(result.summary) 
                                    ? result.summary.join('\n') 
                                    : result.summary;
                                }
                                
                                // If result is directly an array
                                if (Array.isArray(result)) {
                                  return result.join('\n');
                                }
                                
                                // If result is a string
                                if (typeof result === 'string') {
                                  return result;
                                }
                                
                                // Fallback: try to extract any text content
                                try {
                                  return JSON.stringify(result, null, 2);
                                } catch {
                                  return 'Summary generated successfully, but format is not recognized.';
                                }
                              })()}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <Badge variant="danger" className="flex items-center gap-1 w-fit">
                            <AlertCircle className="w-3 h-3" />
                            Summary Failed
                          </Badge>
                          <p className="text-red-600 dark:text-red-400">
                            {summaryResult.error_message || 'Unknown error occurred'}
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Select a document and click "Generate Summary" to see results</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default AIPage;