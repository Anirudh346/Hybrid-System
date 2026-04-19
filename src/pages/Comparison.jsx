import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { X, ArrowLeft, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { useComparison } from '../contexts/ComparisonContext';
import api from '../services/apiClient';
import { useAuth } from '../contexts/AuthContext';

function Comparison() {
  const { selectedDevices, removeDevice, clearAll } = useComparison();
  const { isAuthenticated } = useAuth();
  const [comparisonTable, setComparisonTable] = useState({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (selectedDevices.length > 0) {
      buildComparisonTable();
    }
  }, [selectedDevices]);

  const deviceToSpecs = (device) => {
    // Map device properties to spec names
    return {
      'OS': device.os,
      'Chipset': device.chipset,
      'CPU': device.cpu,
      'GPU': device.gpu,
      'Display': device.display_type,
      'Size': device.display_size,
      'Resolution': device.display_resolution,
      'Internal': device.internal_storage,
      'RAM': device.ram,
      'Main Camera': device.main_camera_features,
      'Selfie camera': device.selfie_camera_single,
      'Battery Capacity': device.battery_capacity ? String(device.battery_capacity) : null,
      'Charging': device.charging ? String(device.charging) : null,
      'Price': device.price,
      'Announced': device.announced,
      'Status': device.status,
      'Brand': device.brand,
      'Model': device.model_name,
      'Build': device.build,
      'Weight': device.weight,
      'Dimensions': device.dimensions,
      'SIM': device.sim,
      'Card Slot': device.card_slot,
      'Loudspeaker': device.loudspeaker,
      '3.5mm Jack': device.jack_35mm,
      'Wi-Fi': device.wlan,
      'Bluetooth': device.bluetooth,
      'NFC': device.nfc,
      'USB': device.usb,
      'Sensors': device.sensors,
      'Colors': device.colors,
      'AnTuTu': device.antutu,
      'GeekBench': device.geekbench,
      'Technology': device.technology,
      'Main Camera Video': device.main_camera_video,
    };
  };

  const buildComparisonTable = () => {
    const allSpecs = new Set();
    selectedDevices.forEach(device => {
      Object.keys(deviceToSpecs(device)).forEach(spec => allSpecs.add(spec));
    });

    const table = {};
    const prioritySpecs = [
      'Brand', 'Model',
      'OS', 'Chipset', 'CPU', 'GPU',
      'Display', 'Size', 'Resolution',
      'Internal', 'RAM',
      'Main Camera', 'Main Camera Video', 'Selfie camera',
      'Battery Capacity', 'Charging',
      'Price', 'Announced', 'Status'
    ];

    prioritySpecs.forEach(spec => {
      if (allSpecs.has(spec)) {
        table[spec] = selectedDevices.map(device => (deviceToSpecs(device)[spec] || 'N/A'));
      }
    });

    Array.from(allSpecs).sort().forEach(spec => {
      if (!table[spec]) {
        table[spec] = selectedDevices.map(device => (deviceToSpecs(device)[spec] || 'N/A'));
      }
    });

    setComparisonTable(table);
  };

  const saveComparison = async () => {
    if (!isAuthenticated) {
      alert('Please login to save comparisons');
      return;
    }

    setSaving(true);
    try {
      const deviceIds = selectedDevices.map(d => d.id);
      const name = `${selectedDevices[0].brand} vs ${selectedDevices[1].brand}`;
      await api.comparisons.create(deviceIds, name);
      alert('Comparison saved successfully!');
    } catch (error) {
      console.error('Error saving comparison:', error);
      alert('Failed to save comparison');
    } finally {
      setSaving(false);
    }
  };

  const getValueIndicator = (values, index) => {
    // Simple comparison indicator
    if (values.every(v => v === 'N/A')) return null;
    
    const numericValues = values.map(v => {
      const match = String(v).match(/(\d+(?:\.\d+)?)/);
      return match ? parseFloat(match[1]) : null;
    }).filter(v => v !== null);

    if (numericValues.length < 2) return null;

    const currentValue = numericValues[index];
    const maxValue = Math.max(...numericValues);
    const minValue = Math.min(...numericValues);

    if (currentValue === maxValue && maxValue !== minValue) {
      return <TrendingUp className="h-4 w-4 text-green-500 inline ml-2" />;
    } else if (currentValue === minValue && maxValue !== minValue) {
      return <TrendingDown className="h-4 w-4 text-red-500 inline ml-2" />;
    }
    return <Minus className="h-4 w-4 text-gray-400 inline ml-2" />;
  };

  if (selectedDevices.length === 0) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <Link
          to="/products"
          className="inline-flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 mb-8"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Products</span>
        </Link>

        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            No Devices Selected
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            Select devices from the products page to compare them side-by-side
          </p>
          <Link
            to="/products"
            className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Browse Products
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6">
        <Link
          to="/products"
          className="inline-flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Products</span>
        </Link>

        <div className="flex space-x-4">
          {isAuthenticated && (
            <button
              onClick={saveComparison}
              disabled={saving}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save Comparison'}
            </button>
          )}
          <button
            onClick={clearAll}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Clear All
          </button>
        </div>
      </div>

      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Compare Devices ({selectedDevices.length})
      </h1>

      {/* Device Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {selectedDevices.map((device) => (
          <div
            key={device.id}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 relative"
          >
            <button
              onClick={() => removeDevice(device.id)}
              className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600"
            >
              <X className="h-4 w-4" />
            </button>
            <img
              src={device.model_image || '/placeholder-device.png'}
              alt={device.model_name}
              className="w-full h-40 object-contain mb-4"
              onError={(e) => {
                e.target.src = '/placeholder-device.png';
              }}
            />
            <h3 className="font-bold text-gray-900 dark:text-white">{device.brand}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">{device.model_name}</p>
          </div>
        ))}
      </div>

      {/* Comparison Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider sticky left-0 bg-gray-50 dark:bg-gray-900">
                  Specification
                </th>
                {selectedDevices.map((device) => (
                  <th
                    key={device.id}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                  >
                    {device.brand}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {Object.entries(comparisonTable).map(([spec, values]) => (
                <tr key={spec} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white sticky left-0 bg-white dark:bg-gray-800">
                    {spec}
                  </td>
                  {values.map((value, index) => (
                    <td
                      key={index}
                      className="px-6 py-4 text-sm text-gray-600 dark:text-gray-300"
                    >
                      {String(value)}
                      {getValueIndicator(values, index)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Comparison;
