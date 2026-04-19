# Device Variant Selection Feature

## Overview
The product detail page now supports selecting different storage/RAM variants for devices. When a variant is selected, the displayed specifications and price update dynamically.

## Features

### 1. **Automatic Variant Detection**
- Automatically detects variants from CSV columns named like "128GB 4GB RAM", "256GB 12GB RAM", etc.
- Also parses variants from the "Internal" column if multiple storage options are listed
- Handles both GB and TB storage units

### 2. **Variant Selector UI**
- Clean, card-based interface showing all available variants
- Each variant displays:
  - Storage and RAM combination (e.g., "128GB / 4GB RAM")
  - Variant-specific price (if available)
  - Visual selection indicator
- Selected variant is highlighted with blue border and background
- Responsive grid layout (1 column on mobile, 2 columns on larger screens)

### 3. **Dynamic Spec Updates**
When a variant is selected:
- **Storage/RAM** in "Key Features" section updates
- **Price** updates to variant-specific price (if available)
- **Full Specifications** section reflects the selected variant's storage
- All changes happen instantly without page reload

### 4. **Price Handling**
- If a variant has a specific price, it's displayed
- If no variant-specific price exists, falls back to the general device price
- Price differences are highlighted in blue

### 5. **Smart Parsing**
- Detects variants even if column values are empty (uses column names)
- Extracts prices from variant columns if they contain price information
- Handles various price formats (EUR, USD, etc.)
- Sorts variants by storage size (ascending) and then RAM (ascending)

## Usage

1. **Viewing a Device with Variants**
   - Navigate to any device detail page
   - If variants are available, you'll see the "Available Variants" section
   - The first variant is automatically selected by default

2. **Selecting a Variant**
   - Click on any variant card to select it
   - The selected variant is highlighted
   - Specifications and price update immediately

3. **Viewing Updated Specs**
   - Check the "Key Features" section for updated storage
   - Check the price display for updated pricing
   - Full specifications reflect the selected variant

## Technical Implementation

### API Methods
- `parseVariants(device)` - Parses device data to extract variants
- `getVariantSpecs(device, variant)` - Returns device specs updated for selected variant

### Component State
- `variants` - Array of available variants
- `selectedVariant` - Currently selected variant object
- `displayDevice` - Device object with specs updated for selected variant

## Example

For a device with variants:
- 128GB / 4GB RAM - $999
- 256GB / 6GB RAM - $1,199
- 512GB / 8GB RAM - $1,399

Users can click any variant to see:
- Updated storage specification
- Updated price
- All other specs remain the same

