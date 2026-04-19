# Quick Start Guide

## Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Open Browser**
   - Navigate to `http://localhost:3000`
   - The app will open automatically

## Loading Your First Dataset

1. Click the **"Upload Dataset"** button in the navigation bar
2. Select a CSV file from your `GSMArenaDataset` folder (e.g., `Apple.csv`)
3. Wait for the upload to complete
4. Start browsing devices!

## Features Overview

### Home Page
- View **Recently Released** devices
- View **Coming Soon** devices
- Quick access to featured products

### Products Page
- Browse all devices in a grid layout
- Use the **left sidebar** to filter by:
  - Device Type (Mobile, Tablet, Smartwatch)
  - Brand
  - Processor
  - Price Range
- Use the **search bar** to find specific devices

### Product Details
- Click any device card to view:
  - High-resolution device image
  - Complete specifications
  - All features organized by category
  - Pricing and availability information

## Tips

- You can upload multiple CSV files - they will be appended to your device list
- All data is saved in browser localStorage
- The sidebar is collapsible on mobile devices
- Use the search bar in the navbar for quick device lookup

## Troubleshooting

**No devices showing?**
- Make sure you've uploaded a CSV file
- Check that the CSV has `Brand` and `Model Name` columns

**Images not loading?**
- Device images are loaded from external URLs in the CSV
- If images fail to load, a placeholder will be shown

**Filters not working?**
- Refresh the page after uploading a dataset
- Clear browser cache if issues persist

