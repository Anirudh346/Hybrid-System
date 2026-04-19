import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { apiClient } from '../services/apiClient';

function Sidebar({ filters, onFilterChange, isOpen, onClose }) {
  const [brands, setBrands] = useState([]);
  const [chipsets, setChipsets] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFiltersData = async () => {
      try {
        setLoading(true);
        console.log('Fetching filter data...');
        
        // Fetch brands from API
        const brandsResponse = await apiClient.get('/api/devices/brands');
        console.log('Brands response:', brandsResponse.data);
        if (brandsResponse.data && Array.isArray(brandsResponse.data.brands)) {
          const brandsList = brandsResponse.data.brands.filter(b => b);
          console.log('Brands loaded:', brandsList.length);
          setBrands(brandsList);
        }
        
        // Fetch chipsets from API
        const chipsetsResponse = await apiClient.get('/api/devices/chipsets');
        console.log('Chipsets response:', chipsetsResponse.data);
        if (chipsetsResponse.data && Array.isArray(chipsetsResponse.data.chipsets)) {
          const chipsetsList = chipsetsResponse.data.chipsets.filter(c => c);
          console.log('Chipsets loaded:', chipsetsList.length);
          setChipsets(chipsetsList);
        }
      } catch (err) {
        console.error('Failed to fetch filter data:', err.message);
        console.error('Error details:', err.response?.data || err);
        setBrands([]);
        setChipsets([]);
      } finally {
        setLoading(false);
      }
    };

    fetchFiltersData();
  }, []);

  const handleDeviceTypeChange = (type) => {
    const newTypes = filters.deviceType?.includes(type)
      ? filters.deviceType.filter(t => t !== type)
      : [...(filters.deviceType || []), type];
    onFilterChange({ ...filters, deviceType: newTypes });
  };

  const handleBrandChange = (brand) => {
    const newBrands = filters.brand?.includes(brand)
      ? filters.brand.filter(b => b !== brand)
      : [...(filters.brand || []), brand];
    onFilterChange({ ...filters, brand: newBrands });
  };

  const handleChipsetChange = (chipset) => {
    const newChipsets = filters.chipset?.includes(chipset)
      ? filters.chipset.filter(c => c !== chipset)
      : [...(filters.chipset || []), chipset];
    onFilterChange({ ...filters, chipset: newChipsets });
  };

  const clearFilters = () => {
    onFilterChange({
      deviceType: [],
      brand: [],
      chipset: [],
      minPrice: '',
      maxPrice: '',
    });
  };

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`fixed lg:sticky top-16 left-0 h-[calc(100vh-4rem)] bg-white dark:bg-gray-800 shadow-lg dark:shadow-gray-900 z-50 w-80 overflow-y-auto transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        <div className="p-6">
          <div className="flex justify-between items-center mb-6 lg:hidden">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">Filters</h2>
            <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded text-gray-700 dark:text-gray-300">
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="mb-6">
            <button
              onClick={clearFilters}
              className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium"
            >
              Clear All Filters
            </button>
          </div>

          {loading ? (
            <div className="text-center py-4">
              <p className="text-gray-600 dark:text-gray-400 text-sm">Loading filters...</p>
            </div>
          ) : (
            <>
              {/* Brand Filter */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Brand</h3>
                <div className="max-h-60 overflow-y-auto space-y-2">
                  {brands.length > 0 ? (
                    brands.map((brand) => (
                      <label key={brand} className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filters.brand?.includes(brand) || false}
                          onChange={() => handleBrandChange(brand)}
                          className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
                        />
                        <span className="text-gray-700 dark:text-gray-300">{brand}</span>
                      </label>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400">No brands available</p>
                  )}
                </div>
              </div>

              {/* Chipset Filter */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Chipset</h3>
                <div className="max-h-60 overflow-y-auto space-y-2">
                  {chipsets.length > 0 ? (
                    chipsets.map((chipset) => (
                      <label key={chipset} className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filters.chipset?.includes(chipset) || false}
                          onChange={() => handleChipsetChange(chipset)}
                          className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
                        />
                        <span className="text-gray-700 dark:text-gray-300 text-sm truncate">{chipset}</span>
                      </label>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400">No chipsets available</p>
                  )}
                </div>
              </div>

              {/* Price Range */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Price Range</h3>
                <div className="space-y-2">
                  <input
                    type="number"
                    value={filters.minPrice || ''}
                    onChange={(e) => onFilterChange({ ...filters, minPrice: e.target.value })}
                    placeholder="Min Price"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
                  />
                  <input
                    type="number"
                    value={filters.maxPrice || ''}
                    onChange={(e) => onFilterChange({ ...filters, maxPrice: e.target.value })}
                    placeholder="Max Price"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </aside>
    </>
  );
}

export default Sidebar;


