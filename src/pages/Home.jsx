import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Clock, TrendingUp } from 'lucide-react';
import { apiClient } from '../services/apiClient';
import ProductCard from '../components/ProductCard';

function Home() {
  const [recentDevices, setRecentDevices] = useState([]);
  const [comingSoonDevices, setComingSoonDevices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const normalizeDevices = (payload) => {
    if (Array.isArray(payload)) {
      return payload;
    }
    if (payload && Array.isArray(payload.devices)) {
      return payload.devices;
    }
    if (payload && Array.isArray(payload.data)) {
      return payload.data;
    }
    return [];
  };

  useEffect(() => {
    const fetchDevices = async () => {
      try {
        setLoading(true);
        const response = await apiClient.get('/api/devices?page_size=100');
        const devices = normalizeDevices(response.data);
        
        // Get first 8 as recent devices
        setRecentDevices(devices.slice(0, 8));
        // Get next 8 as coming soon
        setComingSoonDevices(devices.slice(8, 16));
      } catch (err) {
        console.error('Failed to fetch devices:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDevices();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading devices...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-4">Error Loading Devices</h1>
          <p className="text-gray-600 dark:text-gray-400">{error}</p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-4">
            Make sure the backend API is running on http://localhost:8000
          </p>
        </div>
      </div>
    );
  }

  if (recentDevices.length === 0 && comingSoonDevices.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Welcome to DeviceHub
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
          Loading devices from the database...
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Discover Latest Devices
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Browse through the latest mobile phones, tablets, and smartwatches
        </p>
      </div>

      {/* Coming Soon Section */}
      {comingSoonDevices.length > 0 && (
        <section className="mb-12">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-6 w-6 text-orange-500 dark:text-orange-400" />
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Coming Soon</h2>
            </div>
            <Link
              to="/products?filter=coming-soon"
              className="flex items-center space-x-1 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
            >
              <span>View All</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {comingSoonDevices.map((device) => (
              <ProductCard key={device.id} device={device} />
            ))}
          </div>
        </section>
      )}

      {/* Recent Devices Section */}
      {recentDevices.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-2">
              <Clock className="h-6 w-6 text-blue-500 dark:text-blue-400" />
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Recently Released</h2>
            </div>
            <Link
              to="/products"
              className="flex items-center space-x-1 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
            >
              <span>View All</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {recentDevices.map((device) => (
              <ProductCard key={device.id} device={device} />
            ))}
          </div>
        </section>
      )}

      {recentDevices.length === 0 && comingSoonDevices.length === 0 && (
        <div className="text-center py-16">
          <p className="text-gray-600 dark:text-gray-400">No devices found. Please upload a dataset.</p>
        </div>
      )}
    </div>
  );
}

export default Home;


