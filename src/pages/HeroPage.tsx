import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, Upload, BarChart3, Database, Menu } from 'lucide-react';

// Declare global variables for TypeScript
declare global {
  interface Window {
    THREE: any;
    VANTA: any;
  }
}

const HeroPage = () => {
  const navigate = useNavigate();
  const vantaRef = useRef<HTMLDivElement>(null);
  const vantaEffect = useRef<any>(null);

  useEffect(() => {
    let mounted = true;

    const initVanta = () => {
      if (!mounted || !vantaRef.current || vantaEffect.current) return;

      // Check if VANTA and THREE are available
      if (typeof window !== 'undefined' && window.VANTA && window.THREE) {
        try {
          console.log('Initializing Vanta Globe with global VANTA...');
          vantaEffect.current = window.VANTA.GLOBE({
            el: vantaRef.current,
            THREE: window.THREE,
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.00,
            minWidth: 200.00,
            scale: 1.00,
            scaleMobile: 1.00,
            color: 0x5E62FF,
            color2: 0x9966FF,
            size: 1.50,
            backgroundColor: 0x000000
          });
          console.log('Vanta Globe initialized successfully:', vantaEffect.current);
        } catch (error) {
          console.error('Error initializing Vanta Globe:', error);
        }
      } else {
        console.warn('VANTA or THREE not available, retrying...');
        // Retry after a short delay
        setTimeout(initVanta, 500);
      }
    };

    // Start initialization after a small delay to ensure DOM is ready
    const timer = setTimeout(initVanta, 100);

    return () => {
      mounted = false;
      clearTimeout(timer);
      if (vantaEffect.current) {
        try {
          vantaEffect.current.destroy();
        } catch (e) {
          console.warn('Error destroying Vanta effect:', e);
        }
        vantaEffect.current = null;
      }
    };
  }, []);

  return (
    <div className="relative min-h-screen bg-black text-white overflow-x-hidden">
      <div ref={vantaRef} className="absolute inset-0 z-0"></div>
      
      <nav className="relative z-10 px-6 py-8">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="text-2xl font-bold">
            mini<span className="text-indigo-500">.</span>ml
          </div>
          <div className="hidden md:flex space-x-8 text-sm">
            <button 
              onClick={() => navigate('/upload')}
              className="hover:text-indigo-500 transition-colors"
            >
              Upload
            </button>
            <button 
              onClick={() => navigate('/ai')}
              className="hover:text-indigo-500 transition-colors"
            >
              AI & LLM
            </button>
            <button 
              onClick={() => navigate('/ml')}
              className="hover:text-indigo-500 transition-colors"
            >
              Training
            </button>
            <button 
              onClick={() => navigate('/results')}
              className="hover:text-indigo-500 transition-colors"
            >
              Results
            </button>
            <button 
              onClick={() => navigate('/files')}
              className="hover:text-indigo-500 transition-colors"
            >
              Files
            </button>
          </div>
          <button className="md:hidden">
            <Menu className="h-6 w-6" />
          </button>
        </div>
      </nav>
      
      <main className="relative z-10 px-6 pt-12 pb-24">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col items-center text-center mb-20">
            <div className="inline-block mb-6 px-4 py-1 rounded-full border border-indigo-500 text-xs uppercase tracking-wider">
              Machine Learning Platform
            </div>
            <h1 className="text-5xl md:text-7xl font-bold mb-8 leading-tight">
              We train <span className="gradient-text">intelligent models</span> that deliver
            </h1>
            <p className="text-lg text-gray-400 max-w-2xl mb-10">
              Empowering data scientists and ML engineers with seamless workflows from data upload 
              to model deployment and intelligent automation.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <button
                onClick={() => navigate('/upload')}
                className="px-8 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full text-white font-medium hover:opacity-90 transition-all duration-300 flex items-center justify-center shadow-2xl shadow-indigo-500/50 border border-indigo-400/50 backdrop-blur-sm hover:shadow-indigo-400/70 hover:scale-105"
              >
                <Upload className="h-4 w-4 mr-2" />
                Start Training
              </button>
              <button
                onClick={() => navigate('/results')}
                className="px-8 py-3 bg-transparent border border-indigo-400/50 rounded-full text-white font-medium hover:bg-indigo-500/10 transition-all duration-300 flex items-center justify-center shadow-2xl shadow-indigo-500/50 backdrop-blur-sm hover:shadow-indigo-400/70 hover:scale-105"
              >
                <BarChart3 className="h-4 w-4 mr-2" />
                View Results
              </button>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div className="p-6 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="text-indigo-500 text-4xl font-bold mb-1">10+</div>
              <div className="text-sm text-gray-400">ML Algorithms</div>
            </div>
            <div className="p-6 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="text-indigo-500 text-4xl font-bold mb-1">4</div>
              <div className="text-sm text-gray-400">Pipeline Types</div>
            </div>
            <div className="p-6 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="text-indigo-500 text-4xl font-bold mb-1">99%</div>
              <div className="text-sm text-gray-400">Model Accuracy</div>
            </div>
            <div className="p-6 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="text-indigo-500 text-4xl font-bold mb-1">&lt;1s</div>
              <div className="text-sm text-gray-400">Training Speed</div>
            </div>
          </div>
        </div>
      </main>

      {/* Features Section */}
      <section className="relative z-10 px-6 py-24">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold mb-6">
              Everything you need for <span className="gradient-text">intelligent automation</span>
            </h2>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              From data upload to model deployment, our platform handles the entire ML lifecycle 
              with precision and ease.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="p-8 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all">
              <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl w-12 h-12 flex items-center justify-center mb-6">
                <Upload className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Data Upload</h3>
              <p className="text-gray-400">
                Upload CSV, Excel, and PDF files with automatic validation, preprocessing, 
                and intelligent data profiling.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all">
              <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl w-12 h-12 flex items-center justify-center mb-6">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-3">ML Training</h3>
              <p className="text-gray-400">
                Train multiple algorithms simultaneously with hyperparameter optimization 
                and automated model selection.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all">
              <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl w-12 h-12 flex items-center justify-center mb-6">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Results Analysis</h3>
              <p className="text-gray-400">
                Compare models, visualize performance metrics, feature importance, 
                and export comprehensive reports.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all">
              <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl w-12 h-12 flex items-center justify-center mb-6">
                <Database className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-3">RAG Chatbot</h3>
              <p className="text-gray-400">
                Ask questions about your documents with semantic search, 
                contextual AI responses, and intelligent insights.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="relative z-10 px-6 py-24">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-5xl font-bold mb-6">
            Ready to transform your <span className="gradient-text">data into intelligence?</span>
          </h2>
          <p className="text-lg text-gray-400 mb-10 max-w-2xl mx-auto">
            Start training your first model in minutes. No setup required, 
            no credit card needed. Experience the future of ML automation.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => navigate('/upload')}
              className="px-10 py-4 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full text-white font-semibold hover:opacity-90 transition-all duration-300 flex items-center justify-center text-lg shadow-2xl shadow-indigo-500/50 border border-indigo-400/50 backdrop-blur-sm hover:shadow-indigo-400/70 hover:scale-105"
            >
              <Upload className="h-5 w-5 mr-2" />
              Get Started Free
            </button>
            <button
              onClick={() => navigate('/ml')}
              className="px-10 py-4 bg-transparent border border-indigo-400/50 rounded-full text-white font-semibold hover:bg-indigo-500/10 transition-all duration-300 flex items-center justify-center text-lg shadow-2xl shadow-indigo-500/50 backdrop-blur-sm hover:shadow-indigo-400/70 hover:scale-105"
            >
              <Brain className="h-5 w-5 mr-2" />
              Explore Platform
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HeroPage; 