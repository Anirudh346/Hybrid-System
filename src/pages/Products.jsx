import { useState, useEffect } from 'react';
import { useSearchParams, useLocation } from 'react-router-dom';
import { Filter, X } from 'lucide-react';
import { apiClient } from '../services/apiClient';
import api from '../services/api';
import ProductCard from '../components/ProductCard';
import Sidebar from '../components/Sidebar';

function Products() {
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const [allDevices, setAllDevices] = useState([]);
  const [filters, setFilters] = useState({
    brand: [],
    chipset: [],
    minPrice: '',
    maxPrice: '',
    search: searchParams.get('search') || '',
  });
  const [filteredDevices, setFilteredDevices] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [parsedPreferences, setParsedPreferences] = useState(null);
  const [recommendedDevices, setRecommendedDevices] = useState(null);
  const [recommendLoading, setRecommendLoading] = useState(false);
  const [recommendError, setRecommendError] = useState(null);
  const [pageLoading, setPageLoading] = useState(true);
  const [pageError, setPageError] = useState(null);
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

  // Fetch all devices on mount
  useEffect(() => {
    const fetchAllDevices = async () => {
      try {
        setPageLoading(true);
        console.log('📱 Fetching devices from backend...');
        const response = await apiClient.get('/api/devices?page_size=10000');
        console.log('✅ Devices fetched successfully:', response.data);
        const devices = normalizeDevices(response.data);
        console.log('📊 Normalized devices:', devices.length, 'total');
        setAllDevices(devices);
      } catch (err) {
        console.error('❌ Failed to fetch devices:', err);
        const errorMsg = err.response?.data?.detail || err.response?.statusText || err.message || 'Failed to fetch devices from backend';
        console.error('Error details:', { status: err.response?.status, data: err.response?.data });
        setPageError(errorMsg);
      } finally {
        setPageLoading(false);
      }
    };

    fetchAllDevices();
  }, []);

  const fetchRecommendations = async (preferences) => {
    setRecommendLoading(true);
    setRecommendError(null);
    try {
      console.log('📊 Fetching recommendations with preferences:', preferences);
      const response = await api.recommendations.get({
        ...preferences,
        top_n: 20,
        explain: false,
      });
      
      console.log('✅ Recommendations received:', response.data.recommendations?.length || 0, 'devices');
      if (response.data.recommendations && response.data.recommendations.length > 0) {
        setRecommendedDevices(response.data.recommendations);
      } else {
        // No recommendations, fall back to client-side filtering
        console.warn('⚠️  No recommendations from backend, using client-side filtering');
        applyClientSidePreferences(preferences);
      }
    } catch (err) {
      console.error('❌ Failed to fetch recommendations:', err.response?.data || err.message);
      console.warn('Falling back to client-side filtering with parsed preferences');
      // Fall back to client-side filtering
      applyClientSidePreferences(preferences);
    } finally {
      setRecommendLoading(false);
    }
  };

  const applyClientSidePreferences = (preferences) => {
    // Apply parsed preferences to filters for client-side filtering
    const newFilters = { ...filters };

    if (preferences.brand_preference && preferences.brand_preference.length > 0) {
      newFilters.brand = preferences.brand_preference;
    }

    if (preferences.budget) {
      newFilters.maxPrice = preferences.budget;
    }

    setFilters(newFilters);
    setRecommendedDevices(null);
    setRecommendError(null);
  };

  useEffect(() => {
    const searchParam = searchParams.get('search');
    if (searchParam && searchParam !== filters.search) {
      setFilters(prev => ({ ...prev, search: searchParam }));
    }
  }, [searchParams, filters.search]);

  // Apply client-side filtering when devices or filters change
  useEffect(() => {
    if (allDevices.length > 0) {
      let filtered = allDevices;

      // Apply search filter
      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        filtered = filtered.filter(device =>
          (device.brand && device.brand.toLowerCase().includes(searchLower)) ||
          (device.model_name && device.model_name.toLowerCase().includes(searchLower))
        );
      }

      // Apply brand filter
      if (filters.brand.length > 0) {
        filtered = filtered.filter(device =>
          filters.brand.includes(device.brand)
        );
      }

      // Apply chipset filter - match if any selected chipset appears in device chipset
      if (filters.chipset.length > 0) {
        filtered = filtered.filter(device => {
          if (!device.chipset) return false;
          const deviceChipset = device.chipset.toLowerCase();
          return filters.chipset.some((chipset) =>
            deviceChipset.includes(chipset.toLowerCase())
          );
        });
      }

      // Apply price filters
      if (filters.minPrice) {
        const min = parseFloat(filters.minPrice);
        filtered = filtered.filter(device => {
          const price = device.price ? parseFloat(device.price) : 0;
          return price >= min;
        });
      }

      if (filters.maxPrice) {
        const max = parseFloat(filters.maxPrice);
        filtered = filtered.filter(device => {
          const price = device.price ? parseFloat(device.price) : 0;
          return price <= max;
        });
      }

      setFilteredDevices(filtered);
    }
  }, [filters, allDevices]);

  // Consume NLP search results or parsed preferences from Navbar
  useEffect(() => {
    if (location.state?.nlpSearchResults) {
      // Direct NLP search results from the new endpoint
      console.log('✅ Received NLP search results:', location.state.nlpSearchResults.length, 'devices');
      setRecommendedDevices(location.state.nlpSearchResults);
      setParsedPreferences(
        location.state.nlpQuery ? { query: location.state.nlpQuery } : null
      );
    } else if (location.state?.parsedPreferences) {
      // Fallback: use parsed preferences for client-side filtering
      setParsedPreferences(location.state.parsedPreferences);
      // Fetch recommendations from backend using parsed preferences
      if (allDevices.length > 0) {
        fetchRecommendations(location.state.parsedPreferences);
      }
    }
  }, [location.state?.nlpSearchResults, location.state?.parsedPreferences, allDevices.length]);

  // When recommended devices are available, use them instead of client-side filtering
  const displayDevices = recommendedDevices && recommendedDevices.length > 0 
    ? recommendedDevices.map(rec => ({
        ...rec,
        id: rec.id || rec.model_name,
      }))
    : filteredDevices;

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
  };

  if (pageLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading devices...</p>
        </div>
      </div>
    );
  }

  if (pageError) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-4">Error Loading Devices</h1>
          <p className="text-gray-600 dark:text-gray-400">{pageError}</p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-4">
            Make sure the backend API is running on http://localhost:8000
          </p>
        </div>
      </div>
    );
  }

  if (allDevices.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <p className="text-gray-600 dark:text-gray-400">No devices found. Please try again later.</p>
      </div>
    );
  }

  const activeFiltersCount = 
    (filters.brand?.length || 0) +
    (filters.chipset?.length || 0) +
    (filters.minPrice || filters.maxPrice ? 1 : 0) +
    (filters.search ? 1 : 0);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex gap-8">
        {/* Sidebar */}
        <Sidebar
          filters={filters}
          onFilterChange={handleFilterChange}
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        {/* Main Content */}
        <div className="flex-1">
          {/* Parsed preferences summary (if NLP was used) */}
          {parsedPreferences && (
            <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900 rounded-lg border border-blue-200 dark:border-blue-700">
              <p className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">
                🤖 NLP-Parsed Preferences:
              </p>
              <div className="flex flex-wrap gap-2">
                {parsedPreferences.device_type && parsedPreferences.device_type.length > 0 && (
                  <span className="px-3 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 text-xs rounded-full">
                    Type: {parsedPreferences.device_type.join(', ')}
                  </span>
                )}
                {parsedPreferences.brand_preference && parsedPreferences.brand_preference.length > 0 && (
                  <span className="px-3 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 text-xs rounded-full">
                    Brand: {parsedPreferences.brand_preference.join(', ')}
                  </span>
                )}
                {parsedPreferences.budget && (
                  <span className="px-3 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 text-xs rounded-full">
                    Budget: ${parsedPreferences.budget}
                  </span>
                )}
                {parsedPreferences.use_case && (
                  <span className="px-3 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 text-xs rounded-full">
                    Use: {parsedPreferences.use_case}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Error message if recommendation fetch failed */}
          {recommendError && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900 rounded-lg border border-red-200 dark:border-red-700">
              <p className="text-sm text-red-800 dark:text-red-100">
                ⚠️ {recommendError}
              </p>
            </div>
          )}

          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {parsedPreferences ? 'AI Recommendations' : filters.search ? `Search: "${filters.search}"` : 'All Products'}
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                {displayDevices.length} device{displayDevices.length !== 1 ? 's' : ''} found
              </p>
            </div>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden flex items-center space-x-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600"
            >
              <Filter className="h-5 w-5" />
              <span>Filters</span>
              {activeFiltersCount > 0 && (
                <span className="bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 rounded-full px-2 py-0.5 text-xs font-bold">
                  {activeFiltersCount}
                </span>
              )}
            </button>
          </div>

          {/* Active Filters */}
          {activeFiltersCount > 0 && (
            <div className="mb-6 flex flex-wrap gap-2">
              {filters.search && (
                <span className="inline-flex items-center px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full text-sm">
                  Search: {filters.search}
                  <button
                    onClick={() => handleFilterChange({ ...filters, search: '' })}
                    className="ml-2 hover:text-blue-900 dark:hover:text-blue-100"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </span>
              )}
              {filters.deviceType?.map((type) => (
                <span
                  key={type}
                  className="inline-flex items-center px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-full text-sm capitalize"
                >
                  {type}
                  <button
                    onClick={() =>
                      handleFilterChange({
                        ...filters,
                        deviceType: filters.deviceType.filter((t) => t !== type),
                      })
                    }
                    className="ml-2 hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </span>
              ))}
              {filters.brand?.map((brand) => (
                <span
                  key={brand}
                  className="inline-flex items-center px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-full text-sm"
                >
                  {brand}
                  <button
                    onClick={() =>
                      handleFilterChange({
                        ...filters,
                        brand: filters.brand.filter((b) => b !== brand),
                      })
                    }
                    className="ml-2 hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </span>
              ))}
            </div>
          )}

          {/* Products Grid */}
          {recommendLoading ? (
            <div className="flex items-center justify-center min-h-96">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400 mx-auto"></div>
                <p className="mt-4 text-gray-600 dark:text-gray-400">Fetching AI recommendations...</p>
              </div>
            </div>
          ) : displayDevices.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {displayDevices.map((device) => (
                <ProductCard key={device.id} device={device} />
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <p className="text-gray-600 dark:text-gray-400 text-lg">No devices found matching your filters.</p>
              <button
                onClick={() => {
                  setFilters({
                    brand: [],
                    chipset: [],
                    minPrice: '',
                    maxPrice: '',
                    search: '',
                  });
                  setParsedPreferences(null);
                  setRecommendedDevices(null);
                }}
                className="mt-4 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
              >
                Clear all filters
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Products;


