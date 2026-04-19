import { createContext, useContext, useState, useEffect } from 'react';

const ComparisonContext = createContext(null);

export const useComparison = () => {
  const context = useContext(ComparisonContext);
  if (!context) {
    throw new Error('useComparison must be used within ComparisonProvider');
  }
  return context;
};

export const ComparisonProvider = ({ children }) => {
  const [selectedDevices, setSelectedDevices] = useState([]);
  const MAX_DEVICES = 4;

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('comparison_devices');
    if (saved) {
      try {
        setSelectedDevices(JSON.parse(saved));
      } catch (e) {
        console.error('Error loading comparison devices:', e);
      }
    }
  }, []);

  // Save to localStorage when changed
  useEffect(() => {
    localStorage.setItem('comparison_devices', JSON.stringify(selectedDevices));
  }, [selectedDevices]);

  const addDevice = (device) => {
    if (selectedDevices.length >= MAX_DEVICES) {
      return { success: false, message: `Maximum ${MAX_DEVICES} devices can be compared` };
    }

    if (selectedDevices.some(d => d.id === device.id)) {
      return { success: false, message: 'Device already in comparison' };
    }

    setSelectedDevices([...selectedDevices, device]);
    return { success: true, message: 'Device added to comparison' };
  };

  const removeDevice = (deviceId) => {
    setSelectedDevices(selectedDevices.filter(d => d.id !== deviceId));
  };

  const clearAll = () => {
    setSelectedDevices([]);
  };

  const isSelected = (deviceId) => {
    return selectedDevices.some(d => d.id === deviceId);
  };

  const value = {
    selectedDevices,
    addDevice,
    removeDevice,
    clearAll,
    isSelected,
    maxDevices: MAX_DEVICES,
    canAddMore: selectedDevices.length < MAX_DEVICES,
  };

  return <ComparisonContext.Provider value={value}>{children}</ComparisonContext.Provider>;
};
