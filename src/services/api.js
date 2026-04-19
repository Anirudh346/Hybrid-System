import Papa from 'papaparse';

class DeviceAPI {
  constructor() {
    this.devices = [];
    this.loaded = false;
  }

  async loadDataset(file, append = false) {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const newDevices = results.data
            .filter(device => device.Brand && device['Model Name']) // Filter out empty rows
            .map((device, index) => ({
              ...device,
              id: `${device.Brand}-${device['Model Name']}-${Date.now()}-${index}`,
              deviceType: this.detectDeviceType(device),
            }));
          
          if (append && this.loaded) {
            this.devices = [...this.devices, ...newDevices];
          } else {
            this.devices = newDevices;
          }
          this.loaded = true;
          resolve(this.devices);
        },
        error: (error) => {
          reject(error);
        },
      });
    });
  }

  async loadAllCSVFiles() {
    try {
      const response = await fetch('/GSMArenaDataset');
      if (!response.ok) {
        // Fallback: try to load from public folder
        return this.loadFromPublicFolder();
      }
      // If we have a backend, we can list files here
      return [];
    } catch (error) {
      return this.loadFromPublicFolder();
    }
  }

  async loadFromPublicFolder() {
    // This will be handled by the upload functionality
    return [];
  }

  detectDeviceType(device) {
    const modelName = (device['Model Name'] || '').toLowerCase();
    const brand = (device.Brand || '').toLowerCase();

    if (modelName.includes('watch') || modelName.includes('band')) {
      return 'smartwatch';
    }
    if (modelName.includes('ipad') || modelName.includes('tablet') || 
        modelName.includes('tab') || brand === 'apple' && modelName.includes('ipad')) {
      return 'tablet';
    }
    return 'mobile';
  }

  getAllDevices() {
    return this.devices;
  }

  getDeviceById(id) {
    return this.devices.find(device => device.id === id);
  }

  filterDevices(filters) {
    let filtered = [...this.devices];

    if (filters.deviceType && filters.deviceType.length > 0) {
      filtered = filtered.filter(device => 
        filters.deviceType.includes(device.deviceType)
      );
    }

    if (filters.brand && filters.brand.length > 0) {
      filtered = filtered.filter(device => 
        filters.brand.includes(device.Brand)
      );
    }

    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(device => 
        (device['Model Name'] || '').toLowerCase().includes(searchLower) ||
        (device.Brand || '').toLowerCase().includes(searchLower)
      );
    }

    if (filters.processor) {
      filtered = filtered.filter(device => {
        const chipset = (device.Chipset || '').toLowerCase();
        const cpu = (device.CPU || '').toLowerCase();
        return chipset.includes(filters.processor.toLowerCase()) ||
               cpu.includes(filters.processor.toLowerCase());
      });
    }

    if (filters.minPrice || filters.maxPrice) {
      filtered = filtered.filter(device => {
        const priceStr = device.Price || '';
        const priceMatch = priceStr.match(/[\d,]+/);
        if (!priceMatch) return true; // Include if no price
        const price = parseInt(priceMatch[0].replace(/,/g, ''));
        if (filters.minPrice && price < filters.minPrice) return false;
        if (filters.maxPrice && price > filters.maxPrice) return false;
        return true;
      });
    }

    return filtered;
  }

  getRecentDevices(limit = 8) {
    // Sort by announced date (most recent first)
    const sorted = [...this.devices].sort((a, b) => {
      const dateA = this.parseDate(a.Announced);
      const dateB = this.parseDate(b.Announced);
      return dateB - dateA;
    });
    return sorted.slice(0, limit);
  }

  getComingSoonDevices(limit = 8) {
    return this.devices.filter(device => {
      const status = (device.Status || '').toLowerCase();
      return status.includes('coming soon') || status.includes('rumored');
    }).slice(0, limit);
  }

  parseDate(dateStr) {
    if (!dateStr) return 0;
    const match = dateStr.match(/(\d{4})/);
    if (match) {
      return parseInt(match[1]);
    }
    return 0;
  }

  getUniqueBrands() {
    return [...new Set(this.devices.map(d => d.Brand).filter(Boolean))].sort();
  }

  getUniqueProcessors() {
    const processors = new Set();
    this.devices.forEach(device => {
      if (device.Chipset) processors.add(device.Chipset);
      if (device.CPU) {
        const cpuParts = device.CPU.split(',');
        cpuParts.forEach(part => {
          const match = part.match(/(\w+\s*\d+)/);
          if (match) processors.add(match[1].trim());
        });
      }
    });
    return Array.from(processors).sort().slice(0, 20); // Limit to top 20
  }

  parseVariants(device) {
    const variants = [];
    // Pattern to match: "128GB 4GB RAM", "256GB 12GB RAM", "1TB 16GB RAM", etc.
    // Must match the entire column name (start to end) to avoid false positives
    const variantPattern = /^(\d+)\s*(GB|TB)\s*(\d+)\s*GB\s*RAM$/i;
    
    // First, check the Internal column for actual listed variants
    if (device['Internal']) {
      const internalValue = device['Internal'];
      // Pattern to match multiple storage options: "128GB 4GB RAM, 256GB 6GB RAM"
      const multipleVariantsPattern = /(\d+)\s*(GB|TB)\s*(\d+)\s*GB\s*RAM/gi;
      const matches = [...internalValue.matchAll(multipleVariantsPattern)];
      
      if (matches.length > 1) {
        // Found multiple variants in Internal column - use these as the source of truth
        matches.forEach(match => {
          const storageValue = parseInt(match[1]);
          const storageUnit = match[2].toUpperCase();
          const ramValue = parseInt(match[3]);
          
          // Validate: filter out unrealistic variants
          if (this.isValidVariant(storageValue, storageUnit, ramValue, device)) {
            const storageInGB = storageUnit === 'TB' ? storageValue * 1024 : storageValue;
            
            variants.push({
              id: `${storageValue}${storageUnit}-${ramValue}GB`,
              label: `${storageValue}${storageUnit} / ${ramValue}GB RAM`,
              storage: `${storageValue}${storageUnit}`,
              storageInGB,
              ram: `${ramValue}GB`,
              ramValue,
              price: device.Price || null,
              originalKey: 'Internal',
            });
          }
        });
        
        // If we found variants in Internal column, use only those
        if (variants.length > 0) {
          variants.sort((a, b) => {
            if (a.storageInGB !== b.storageInGB) {
              return a.storageInGB - b.storageInGB;
            }
            return a.ramValue - b.ramValue;
          });
          return variants;
        }
      }
    }
    
    // If no variants found in Internal column, check for variant columns
    // First, collect all variant columns and check which ones have values
    const variantColumns = [];
    const variantColumnsWithValues = [];
    
    Object.keys(device).forEach(key => {
      const match = key.match(variantPattern);
      if (match) {
        const storageValue = parseInt(match[1]);
        const storageUnit = match[2].toUpperCase();
        const ramValue = parseInt(match[3]);
        
        // Validate: filter out unrealistic variants
        if (!this.isValidVariant(storageValue, storageUnit, ramValue, device)) {
          return; // Skip this variant
        }
        
        const variantValue = device[key];
        const hasValue = variantValue && typeof variantValue === 'string' && variantValue.trim().length > 0;
        
        variantColumns.push({ key, storageValue, storageUnit, ramValue, hasValue, variantValue });
        if (hasValue) {
          variantColumnsWithValues.push(key);
        }
      }
    });
    
    // Only process variants if at least one variant column has a value
    // This prevents showing false variants from empty CSV columns
    if (variantColumnsWithValues.length === 0) {
      return []; // No valid variants found
    }
    
    // Now process only variant columns that have values OR are part of a set with values
    variantColumns.forEach(({ key, storageValue, storageUnit, ramValue, hasValue, variantValue }) => {
      // Only include variants that have values
      // Empty variant columns are likely just CSV structure, not actual variants for this device
      if (!hasValue) {
        return; // Skip empty variant columns
      }
        
      // Calculate storage in GB for comparison
      const storageInGB = storageUnit === 'TB' ? storageValue * 1024 : storageValue;
      
      // Try to find price for this variant
      let price = null;
      
      // Check if the variant column itself contains price information
      // Check if it looks like a price (contains currency symbols or numbers with commas)
      const pricePattern = /([€$£₹¥]|EUR|USD|GBP|INR|JPY|CNY|About)[\s]*([\d,]+)/i;
      if (pricePattern.test(variantValue)) {
        price = variantValue.trim();
      }
      
      // Also check for separate price columns that might match this variant
      const priceColumnKeys = Object.keys(device).filter(k => {
        const lowerK = k.toLowerCase();
        const lowerKey = key.toLowerCase();
        return (lowerK.includes(lowerKey) || lowerKey.includes(lowerK.replace(/\s+/g, ''))) &&
               (lowerK.includes('price') || lowerK.includes('cost'));
      });
      
      priceColumnKeys.forEach(priceKey => {
        const priceValue = device[priceKey];
        if (priceValue && typeof priceValue === 'string' && priceValue.trim() && !price) {
          price = priceValue.trim();
        }
      });

      // If no variant-specific price found, use the general price
      if (!price && device.Price) {
        price = device.Price;
      }

      variants.push({
        id: `${storageValue}${storageUnit}-${ramValue}GB`,
        label: `${storageValue}${storageUnit} / ${ramValue}GB RAM`,
        storage: `${storageValue}${storageUnit}`,
        storageInGB,
        ram: `${ramValue}GB`,
        ramValue,
        price: price || null,
        originalKey: key,
      });
    });

    // Sort variants by storage (ascending) and then RAM (ascending)
    variants.sort((a, b) => {
      if (a.storageInGB !== b.storageInGB) {
        return a.storageInGB - b.storageInGB;
      }
      return a.ramValue - b.ramValue;
    });

    return variants;
  }

  isValidVariant(storageValue, storageUnit, ramValue, device) {
    // Filter out unrealistic variants
    // Minimum RAM thresholds based on device type
    const deviceType = device.deviceType || 'mobile';
    let minRAM = 2; // Default minimum
    
    if (deviceType === 'tablet') {
      minRAM = 3; // Tablets typically have at least 3GB RAM
    } else if (deviceType === 'smartwatch') {
      minRAM = 1; // Smartwatches can have less RAM
    }
    
    // Filter out variants with too little RAM
    if (ramValue < minRAM) {
      return false;
    }
    
    // Filter out variants with unrealistic storage (less than 16GB for modern devices)
    const storageInGB = storageUnit === 'TB' ? storageValue * 1024 : storageValue;
    if (storageInGB < 16 && deviceType !== 'smartwatch') {
      return false;
    }
    
    // Filter out variants with unrealistic combinations
    // e.g., 1TB with 1GB RAM doesn't make sense
    if (storageInGB >= 512 && ramValue < 4) {
      return false;
    }
    
    return true;
  }

  getVariantSpecs(device, variant) {
    if (!variant) return device;
    
    // Create a copy of device with updated specs for the selected variant
    const updatedDevice = { ...device };
    
    // Update Internal storage
    if (variant.storage && variant.ram) {
      updatedDevice['Internal'] = `${variant.storage} ${variant.ram}`;
    }
    
    // Update price if variant has specific price
    if (variant.price) {
      updatedDevice.Price = variant.price;
    }
    
    return updatedDevice;
  }
}

export default new DeviceAPI();

