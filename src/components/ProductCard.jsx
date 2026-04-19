import { Link } from 'react-router-dom';
import { Smartphone, Tablet, Watch, Zap, Camera, Monitor, Cpu } from 'lucide-react';

function ProductCard({ device }) {
  // Extract safely from device object
  const imageUrl = device.model_image || '/placeholder-device.png';
  const modelName = device.model_name || 'Unknown Model';
  const brand = device.brand || 'Unknown Brand';
  const price = device.price || 'N/A';
  const announced = device.announced;
  const os = device.os;
  const chipset = device.chipset;
  const displaySize = device.display_size;
  const battery_capacity = device.battery_capacity;
  const mainCamera = device.main_camera_features;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md dark:shadow-gray-900 hover:shadow-xl dark:hover:shadow-gray-800 transition-shadow duration-300 overflow-hidden group h-full flex flex-col">
      {/* Image Section */}
      <div className="relative h-56 bg-gray-100 dark:bg-gray-700 overflow-hidden flex-shrink-0">
        <img
          src={imageUrl}
          alt={modelName}
          className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-300"
          onError={(e) => {
            e.target.src = 'https://via.placeholder.com/300x400?text=No+Image';
          }}
        />
      </div>

      {/* Content Section */}
      <div className="p-4 flex-grow flex flex-col">
        {/* Title and Brand */}
        <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-1 line-clamp-2">
          {brand} {modelName}
        </h3>

        {/* Price */}
        {price && price !== 'N/A' && (
          <p className="text-blue-600 dark:text-blue-400 font-bold text-lg mb-3">
            {price}
          </p>
        )}

        {/* Quick Specs Grid */}
        <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
          {os && (
            <div className="flex items-start space-x-1">
              <Smartphone className="h-4 w-4 text-gray-500 mt-0.5 flex-shrink-0" />
              <span className="text-gray-600 dark:text-gray-400 line-clamp-1">{os}</span>
            </div>
          )}

          {chipset && (
            <div className="flex items-start space-x-1">
              <Cpu className="h-4 w-4 text-gray-500 mt-0.5 flex-shrink-0" />
              <span className="text-gray-600 dark:text-gray-400 line-clamp-1">{chipset}</span>
            </div>
          )}

          {battery_capacity && (
            <div className="flex items-start space-x-1">
              <Zap className="h-4 w-4 text-gray-500 mt-0.5 flex-shrink-0" />
              <span className="text-gray-600 dark:text-gray-400 line-clamp-1">{String(battery_capacity)}</span>
            </div>
          )}

          {displaySize && (
            <div className="flex items-start space-x-1">
              <Monitor className="h-4 w-4 text-gray-500 mt-0.5 flex-shrink-0" />
              <span className="text-gray-600 dark:text-gray-400 line-clamp-1">{displaySize}</span>
            </div>
          )}

          {mainCamera && (
            <div className="flex items-start space-x-1 col-span-2">
              <Camera className="h-4 w-4 text-gray-500 mt-0.5 flex-shrink-0" />
              <span className="text-gray-600 dark:text-gray-400 line-clamp-1">{mainCamera}</span>
            </div>
          )}
        </div>

        {/* Announced */}
        {announced && (
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-auto pt-2 border-t border-gray-200 dark:border-gray-700">
            Released: {announced}
          </p>
        )}
      </div>

      {/* View Details Button */}
      <Link
        to={`/product/${device.id}`}
        className="mx-4 mb-4 mt-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors text-center w-auto"
      >
        View Details
      </Link>
    </div>
  );
}

export default ProductCard;