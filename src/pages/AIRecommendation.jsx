import { Sparkles, Code, FlaskConical, Wrench, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

function AIRecommendation() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <Link
        to="/"
        className="inline-flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 mb-8"
      >
        <ArrowLeft className="h-5 w-5" />
        <span>Back to Home</span>
      </Link>

      <div className="text-center mb-12">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <Sparkles className="h-12 w-12 text-blue-600 dark:text-blue-400 animate-pulse" />
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
            AI Device Recommender
          </h1>
        </div>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          Powered by NLP and Machine Learning
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg dark:shadow-gray-900 p-8 md:p-12">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full mb-6">
            <Wrench className="h-12 w-12 text-white animate-spin" style={{ animationDuration: '3s' }} />
          </div>
          
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Coming Soon
          </h2>
          
          <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
            We're currently developing an advanced AI recommendation system using Python, 
            NLP (Natural Language Processing), and Machine Learning algorithms to provide 
            personalized device recommendations based on your needs.
          </p>

          <div className="grid md:grid-cols-3 gap-6 mt-12 mb-8">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <div className="flex items-center justify-center w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg mb-4 mx-auto">
                <Code className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Python Backend</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Advanced ML models for intelligent recommendations
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <div className="flex items-center justify-center w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg mb-4 mx-auto">
                <FlaskConical className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">NLP Processing</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Understand natural language queries and preferences
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <div className="flex items-center justify-center w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg mb-4 mx-auto">
                <Sparkles className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">ML Algorithms</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Smart scoring and ranking of devices
              </p>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-6 mt-8">
            <p className="text-blue-900 dark:text-blue-200 font-medium">
              🚀 This feature will be available soon with advanced AI capabilities!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AIRecommendation;

