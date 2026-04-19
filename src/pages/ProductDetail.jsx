import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { ArrowLeft, Smartphone, Zap, Camera, Monitor, Cpu, Wifi, Bluetooth } from 'lucide-react';
import { apiClient } from '../services/apiClient';

function ProductDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [device, setDevice] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDevice = async () => {
      try {
        setLoading(true);
        const response = await apiClient.get(`/api/devices/${id}`);
        setDevice(response.data);
      } catch (err) {
        console.error('Failed to fetch device:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDevice();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading device details...</p>
        </div>
      </div>
    );
  }

  if (error || !device) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-4">Error</h1>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error || 'Device not found'}</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
          >
            Back to Home
          </button>
        </div>
      </div>
    );
  }

  const SpecSection = ({ title, icon: Icon, specs }) => (
    <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
      <h3 className="flex items-center space-x-2 text-xl font-bold text-gray-900 dark:text-white mb-4">
        <Icon className="h-6 w-6 text-blue-600" />
        <span>{title}</span>
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {specs
          .filter(([, value]) => value)
          .map(([label, value]) => (
            <div key={label} className="flex justify-between items-start">
              <span className="text-gray-600 dark:text-gray-400">{label}:</span>
              <span className="text-gray-900 dark:text-white font-medium text-right ml-2">{String(value)}</span>
            </div>
          ))}
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 mb-6"
      >
        <ArrowLeft className="h-5 w-5" />
        <span>Back</span>
      </button>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Image */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-8 flex items-center justify-center h-96 md:h-auto">
          <img
            src={device.model_image || 'https://via.placeholder.com/400x600?text=No+Image'}
            alt={device.model_name}
            className="max-h-96 max-w-full object-contain"
            onError={(e) => {
              e.target.src = 'https://via.placeholder.com/400x600?text=No+Image';
            }}
          />
        </div>

        {/* Basic Info */}
        <div className="flex flex-col justify-between">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
              {device.brand} {device.model_name}
            </h1>
            {device.announced && (
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
                Announced: {device.announced}
              </p>
            )}
            {device.status && (
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
                Status: <span className="font-semibold text-green-600 dark:text-green-400">{device.status}</span>
              </p>
            )}
          </div>

          {/* Price */}
          {device.price && device.price !== 'N/A' && (
            <div className="bg-blue-600 dark:bg-blue-700 rounded-lg p-6">
              <p className="text-gray-100 text-sm font-semibold mb-1">Price</p>
              <p className="text-4xl font-bold text-white">{device.price}</p>
            </div>
          )}
        </div>
      </div>

      {/* Specifications */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-8">
        {/* Display */}
        {(device.display_type ||
          device.display_size ||
          device.display_resolution) && (
          <SpecSection
            title="Display"
            icon={Monitor}
            specs={[
              ['Type', device.display_type],
              ['Size', device.display_size],
              ['Resolution', device.display_resolution],
            ]}
          />
        )}

        {/* Performance */}
        {(device.chipset || device.cpu || device.gpu) && (
          <SpecSection
            title="Performance"
            icon={Cpu}
            specs={[
              ['Chipset', device.chipset],
              ['Processor (CPU)', device.cpu],
              ['GPU', device.gpu],
              ['AnTuTu Score', device.antutu],
              ['GeekBench', device.geekbench],
              ['Speed', device.speed],
            ]}
          />
        )}

        {/* Camera */}
        {(device.main_camera_features ||
          device.main_camera_video ||
          device.selfie_camera_single) && (
          <SpecSection
            title="Camera"
            icon={Camera}
            specs={[
              ['Rear Camera', device.main_camera_features],
              ['Rear Video', device.main_camera_video],
              ['Selfie Camera', device.selfie_camera_single],
            ]}
          />
        )}

        {/* Battery & Charging */}
        {(device.battery_capacity || device.charging) && (
          <SpecSection
            title="Battery & Charging"
            icon={Zap}
            specs={[
              ['Battery Capacity', device.battery_capacity ? String(device.battery_capacity) : null],
              ['Charging', device.charging ? String(device.charging) : null],
            ]}
          />
        )}

        {/* Network & Connectivity */}
        {(device.bands_2g ||
          device.bands_3g ||
          device.bands_4g ||
          device.bands_5g ||
          device.wlan ||
          device.bluetooth ||
          device.nfc) && (
          <SpecSection
            title="Network & Connectivity"
            icon={Wifi}
            specs={[
              ['2G Bands', device.bands_2g],
              ['3G Bands', device.bands_3g],
              ['4G Bands', device.bands_4g],
              ['5G Bands', device.bands_5g],
              ['Wi-Fi', device.wlan],
              ['Bluetooth', device.bluetooth],
              ['NFC', device.nfc],
              ['USB', device.usb],
            ]}
          />
        )}

        {/* Physical */}
        {(device.dimensions || device.weight || device.build || device.sim) && (
          <SpecSection
            title="Physical Specifications"
            icon={Smartphone}
            specs={[
              ['Dimensions', device.dimensions],
              ['Weight', device.weight],
              ['Build', device.build],
              ['SIM', device.sim],
            ]}
          />
        )}

        {/* Storage */}
        {(device.internal_storage || device.card_slot) && (
          <SpecSection
            title="Storage"
            icon={Smartphone}
            specs={[
              ['Internal Storage', device.internal_storage],
              ['Card Slot', device.card_slot],
            ]}
          />
        )}

        {/* Audio & Features */}
        {(device.loudspeaker ||
          device.jack_35mm ||
          device.sensors ||
          device.colors) && (
          <SpecSection
            title="Audio & Features"
            icon={Smartphone}
            specs={[
              ['Loudspeaker', device.loudspeaker],
              ['3.5mm Jack', device.jack_35mm],
              ['Sensors', device.sensors],
              ['Colors', device.colors],
            ]}
          />
        )}

        {/* Software */}
        {device.os && (
          <SpecSection
            title="Software"
            icon={Smartphone}
            specs={[['Operating System', device.os]]}
          />
        )}

        {/* Technology */}
        {device.technology && (
          <SpecSection
            title="Technology"
            icon={Wifi}
            specs={[['Network Technology', device.technology]]}
          />
        )}
      </div>
    </div>
  );
}

export default ProductDetail;


