import { Link } from 'react-router-dom';
import { X, ArrowRight } from 'lucide-react';
import { useComparison } from '../contexts/ComparisonContext';

function ComparisonBar() {
  const { selectedDevices, removeDevice, clearAll } = useComparison();

  if (selectedDevices.length === 0) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-blue-600 dark:bg-blue-700 shadow-lg z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 flex-1 overflow-x-auto">
            <span className="text-white font-medium whitespace-nowrap">
              Compare ({selectedDevices.length}/4):
            </span>
            <div className="flex space-x-2">
              {selectedDevices.map((device) => (
                <div
                  key={device.id}
                  className="bg-white dark:bg-gray-800 rounded px-3 py-2 flex items-center space-x-2"
                >
                  <span className="text-sm text-gray-900 dark:text-white whitespace-nowrap">
                    {device.brand} {device.model_name}
                  </span>
                  <button
                    onClick={() => removeDevice(device.id)}
                    className="text-gray-500 hover:text-red-500"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center space-x-4 ml-4">
            <button
              onClick={clearAll}
              className="text-white hover:text-gray-200 whitespace-nowrap"
            >
              Clear All
            </button>
            <Link
              to="/comparison"
              className="bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 px-4 py-2 rounded-lg font-medium flex items-center space-x-2 hover:bg-gray-100 dark:hover:bg-gray-700 whitespace-nowrap"
            >
              <span>Compare Now</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ComparisonBar;
