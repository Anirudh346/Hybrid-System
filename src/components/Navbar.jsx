import { Link, useNavigate } from 'react-router-dom';
import { Search, Home, Moon, Sun, Sparkles } from 'lucide-react';
import { useState } from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { apiClient } from '../services/apiClient';

function Navbar() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const navigate = useNavigate();
  const { isDark, toggleTheme } = useTheme();

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    try {
      setSearching(true);
      console.log('[NLP Search]', searchQuery);
      
      // Call the new NLP search endpoint with query as URL parameter (limit to top 10)
      const response = await apiClient.post(`/api/devices/search/nlp?query=${encodeURIComponent(searchQuery)}&limit=10`);
      
      console.log('[OK] NLP search successful:', response.data.devices.length, 'devices found');
      
      // Navigate to products and display NLP results
      navigate('/products', {
        state: {
          nlpSearchResults: response.data.devices,
          nlpQuery: searchQuery,
        },
      });
    } catch (err) {
      // Fallback to simple keyword search if NLP fails
      console.error('[ERROR] NLP search error:', err.response?.data || err.message || err);
      console.warn('Falling back to keyword search');
      navigate(`/products?search=${encodeURIComponent(searchQuery)}`);
    } finally {
      setSearching(false);
      setSearchQuery('');
    }
  };

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-md sticky top-0 z-50 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Home className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            <span className="text-xl font-bold text-gray-900 dark:text-white">DeviceHub</span>
          </Link>

          <div className="flex-1 max-w-2xl mx-8 flex items-center space-x-2">
            <form onSubmit={handleSearch} className="flex-1">
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Try: 'best gaming phone under $500' or 'affordable Samsung tablet'..."
                  className="w-full px-4 py-2 pl-10 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
                  disabled={searching}
                />
                <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 ${searching ? 'animate-spin text-blue-500' : 'text-gray-400 dark:text-gray-500'}`} />
              </div>
            </form>
            <Link
              to="/ai-recommendation"
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 dark:from-purple-500 dark:to-blue-500 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 dark:hover:from-purple-600 dark:hover:to-blue-600 transition"
              title="AI Device Recommender"
            >
              <Sparkles className="h-5 w-5" />
              <span className="hidden sm:inline">AI</span>
            </Link>
          </div>

            <div className="flex items-center space-x-4">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              aria-label="Toggle dark mode"
            >
              {isDark ? (
                <Sun className="h-5 w-5" />
              ) : (
                <Moon className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;


