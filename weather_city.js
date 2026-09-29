
// ============================================================
// URBAN HEAT ISLAND PROJECT - DATA EXTRACTION V1
// 24 Indian Cities | 2016-2025 | Monthly
//
// Training:   2016-2023
// Validation: 2024
// Testing:    2025
//
// Spatial method:
// Urban = 0-10 km from city centre
// Rural = 10-20 km from city centre
//
// Output: Monthly city-level CSV
// ============================================================


// ------------------------------------------------------------
// 1. CONFIGURATION
// ------------------------------------------------------------

var START_YEAR = 2016;
var END_YEAR = 2025;

var URBAN_RADIUS = 10000;
var RURAL_RADIUS = 20000;

var DRIVE_FOLDER = 'UHI_Project';


// ------------------------------------------------------------
// 2. SELECTED CITIES
// Coordinates are approximate city centres.
// Format: [Longitude, Latitude]
// ------------------------------------------------------------

var cityData = [
  ['Ahmedabad', 72.5714, 23.0225, 'Arid_SemiArid'],
  ['Jaipur', 75.7873, 26.9124, 'Arid_SemiArid'],
  ['Jodhpur', 73.0243, 26.2389, 'Arid_SemiArid'],
  ['Gandhinagar', 72.6369, 23.2156, 'Arid_SemiArid'],

  ['Mumbai', 72.8777, 19.0760, 'Coastal'],
  ['Chennai', 80.2707, 13.0827, 'Coastal'],
  ['Kochi', 76.2673, 9.9312, 'Coastal'],
  ['Mangaluru', 74.8560, 12.9141, 'Coastal'],

  ['Bengaluru', 77.5946, 12.9716, 'Plateau'],
  ['Hyderabad', 78.4867, 17.3850, 'Plateau'],
  ['Pune', 73.8567, 18.5204, 'Plateau'],
  ['Mysuru', 76.6394, 12.2958, 'Plateau'],

  ['Delhi', 77.1025, 28.7041, 'IndoGangetic'],
  ['Lucknow', 80.9462, 26.8467, 'IndoGangetic'],
  ['Kanpur', 80.3319, 26.4499, 'IndoGangetic'],
  ['Patna', 85.1376, 25.5941, 'IndoGangetic'],

  ['Kolkata', 88.3639, 22.5726, 'Eastern'],
  ['Bhubaneswar', 85.8245, 20.2961, 'Eastern'],
  ['Guwahati', 91.7362, 26.1445, 'Eastern'],
  ['Siliguri', 88.3953, 26.7271, 'Eastern'],

  ['Nagpur', 79.0882, 21.1458, 'Central'],
  ['Indore', 75.8577, 22.7196, 'Central'],
  ['Bhopal', 77.4126, 23.2599, 'Central'],
  ['Udaipur', 73.7125, 24.5854, 'Central']
];


// Convert cities into Earth Engine features

var cities = ee.FeatureCollection(
  cityData.map(function(c) {

    var point = ee.Geometry.Point([c[1], c[2]]);

    var urban = point.buffer(URBAN_RADIUS);

    var outer = point.buffer(RURAL_RADIUS);

    var rural = outer.difference(urban);

    return ee.Feature(urban, {
      city: c[0],
      longitude: c[1],
      latitude: c[2],
      climate_group: c[3],
    });

  })
);

print('Selected cities:', cities);


// ------------------------------------------------------------
// 3. LANDSAT 8 AND 9 DATA
// ------------------------------------------------------------

var landsat8 = ee.ImageCollection(
  'LANDSAT/LC08/C02/T1_L2'
);

var landsat9 = ee.ImageCollection(
  'LANDSAT/LC09/C02/T1_L2'
);

var landsat = landsat8.merge(landsat9);


// Cloud masking and scaling

function prepareLandsat(image) {

  var qa = image.select('QA_PIXEL');

  // Remove cloud, cloud shadow, cirrus and dilated cloud
  var cloudMask = qa.bitwiseAnd(parseInt('11111', 2)).eq(0);

  var saturationMask = image.select('QA_RADSAT').eq(0);

  var optical = image.select(
    ['SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
    ['Green', 'Red', 'NIR', 'SWIR1']
  ).multiply(0.0000275).add(-0.2);

  var thermal = image.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('LST');

  return optical.addBands(thermal)
    .updateMask(cloudMask)
    .updateMask(saturationMask)
    .copyProperties(image, ['system:time_start']);

}


// ------------------------------------------------------------
// 4. POPULATION DATA - GHSL
// ------------------------------------------------------------

var populationCollection = ee.ImageCollection(
  'JRC/GHSL/P2023A/GHS_POP'
);

// Use nearest available five-year epoch.
// Epochs available: 1975, 1980 ... 2020, 2025, 2030.

function getPopulation(year) {
  var epoch = 1975 + Math.round((year - 1975) / 5) * 5;

  var image = ee.Image(
    'JRC/GHSL/P2023A/GHS_POP/' + epoch
  );

  var density = image.select('population_count')
    .divide(0.01)
    .rename('Population_Density');

  return density.set('population_epoch', epoch);
}

// ------------------------------------------------------------
// 5. WEATHER DATA - ERA5 LAND
// ------------------------------------------------------------

var era5 = ee.ImageCollection(
  'ECMWF/ERA5_LAND/MONTHLY_AGGR'
);


// Convert weather values into useful units.

function getWeather(year, month) {

  var start = ee.Date.fromYMD(year, month, 1);

  var image = ee.Image(
    era5.filterDate(
      start,
      start.advance(1, 'month')
    ).first()
  );

  var temp = image.select('temperature_2m')
    .subtract(273.15)
    .rename('Air_Temperature');

  var dew = image.select('dewpoint_temperature_2m')
    .subtract(273.15);

  // Relative humidity approximation using
  // the Magnus formula.
  var rh = dew.expression(
    '100 * exp((17.625 * Td) / (243.04 + Td)) / ' +
    'exp((17.625 * T) / (243.04 + T))',
    {
      Td: dew,
      T: temp
    }
  ).clamp(0, 100).rename('Relative_Humidity');

  var u = image.select('u_component_of_wind_10m');

  var v = image.select('v_component_of_wind_10m');

  var wind = u.pow(2).add(v.pow(2))
    .sqrt()
    .rename('Wind_Speed');

  var rainfall = image.select(
    'total_precipitation_sum'
  ).multiply(1000).rename('Rainfall');

  var solar = image.select(
    'surface_solar_radiation_downwards_sum'
  ).divide(1e6).rename('Solar_Radiation');

  return ee.Image.cat([
    temp,
    rh,
    wind,
    rainfall,
    solar
  ]);
}


// ------------------------------------------------------------
// 6. CHIRPS MONTHLY RAINFALL
// ------------------------------------------------------------

var chirps = ee.ImageCollection(
  'UCSB-CHG/CHIRPS/DAILY'
);


// ------------------------------------------------------------
// 7. GENERATE MONTHLY FEATURES
// ------------------------------------------------------------

function generateMonthlyData(year, month, city) {

  var start = ee.Date.fromYMD(year, month, 1);

  var end = start.advance(1, 'month');

  var cityGeometry = city.geometry();

  var urbanGeometry = cityGeometry;

  var ruralGeometry = cityGeometry
    .buffer(10000)
    .difference(cityGeometry);

  // Landsat monthly composite
  var l8 = landsat
    .filterDate(start, end)
    .filterBounds(cityGeometry.buffer(RURAL_RADIUS))
    .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
    .map(prepareLandsat);

  // Keep the output schema stable when a month has no usable scenes.
  var emptyLandsat = ee.Image.constant([0, 0, 0, 0, 0])
    .rename(['Green', 'Red', 'NIR', 'SWIR1', 'LST'])
    .updateMask(ee.Image.constant(0));

  var l8WithFallback = l8.merge(
    ee.ImageCollection([emptyLandsat])
  );

  // Median composite
  var composite = l8WithFallback.median();

  // Calculate vegetation index
  var ndvi = composite.normalizedDifference(
    ['NIR', 'Red']
  ).rename('NDVI');

  // Calculate water index
  var ndwi = composite.normalizedDifference(
    ['Green', 'NIR']
  ).rename('NDWI');

  // Calculate built-up index
  var ndbi = composite.normalizedDifference(
    ['SWIR1', 'NIR']
  ).rename('NDBI');

  var lst = composite.select('LST');

  // Calculate UHI using urban-rural LST difference
  var urbanLST = lst.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: urbanGeometry,
    scale: 1000,
    maxPixels: 1e9,
    bestEffort: true
  }).get('LST');

  var ruralLST = lst.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: ruralGeometry,
    scale: 1000,
    maxPixels: 1e9,
    bestEffort: true
  }).get('LST');

  var uhi = ee.Algorithms.If(
    ee.Algorithms.IsEqual(urbanLST, null),
    null,
    ee.Algorithms.If(
      ee.Algorithms.IsEqual(ruralLST, null),
      null,
      ee.Number(urbanLST).subtract(ee.Number(ruralLST))
    )
  );

  // Combine satellite features
  var satelliteFeatures = ee.Image.cat([
    lst,
    ndvi,
    ndwi,
    ndbi
  ]);

  // Calculate urban averages
  var urbanStats = satelliteFeatures.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: urbanGeometry,
    scale: 1000,
    maxPixels: 1e9,
    bestEffort: true
  });

  // Population
  var population = getPopulation(year);

  var populationMean = population.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: urbanGeometry,
    scale: 100,
    maxPixels: 1e9,
    bestEffort: true
  }).get('Population_Density');

  // Weather
  var weather = getWeather(year, month);

  var weatherStats = weather.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: urbanGeometry,
    scale: 11132,
    maxPixels: 1e9,
    bestEffort: true
  });

  // CHIRPS monthly total
  var monthlyRain = chirps
    .filterDate(start, end)
    .sum()
    .rename('CHIRPS_Rainfall');

  var chirpsRainfall = monthlyRain.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: urbanGeometry,
    scale: 5566,
    maxPixels: 1e9,
    bestEffort: true
  }).get('CHIRPS_Rainfall');

  // Count usable Landsat observations
  var observationCount = l8WithFallback.select('LST')
    .count()
    .reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: urbanGeometry,
      scale: 1000,
      maxPixels: 1e9,
      bestEffort: true
    }).get('LST');

  // Construct final row
  return ee.Feature(null, {
    city: city.get('city'),
    climate_group: city.get('climate_group'),

    longitude: city.get('longitude'),
    latitude: city.get('latitude'),

    year: year,
    month: month,

    split: year <= 2023 ? 'training' :
           year === 2024 ? 'validation' : 'testing',

    LST: urbanStats.get('LST'),
    Rural_LST: ruralLST,
    UHI: uhi,

    NDVI: urbanStats.get('NDVI'),
    NDWI: urbanStats.get('NDWI'),
    NDBI: urbanStats.get('NDBI'),

    Population_Density: populationMean,
    Population_Epoch: Math.round(
      1975 + Math.round((year - 1975) / 5) * 5
    ),

    Air_Temperature: weatherStats.get('Air_Temperature'),
    Relative_Humidity: weatherStats.get('Relative_Humidity'),
    Wind_Speed: weatherStats.get('Wind_Speed'),
    ERA5_Rainfall: weatherStats.get('Rainfall'),
    CHIRPS_Rainfall: chirpsRainfall,
    Solar_Radiation: weatherStats.get('Solar_Radiation'),

    Landsat_Observation_Count: observationCount
  });
}


// ------------------------------------------------------------
// 8. SELECT ONE YEAR PER RUN
// ------------------------------------------------------------

// Change this to 2016, 2017, ... 2025 for each export.
var EXPORT_YEAR = 2016;

var monthlyFeatures = [];

for (var month = 1; month <= 12; month++) {

  var monthFeatures = cities.map(function(city) {
    return generateMonthlyData(
      EXPORT_YEAR,
      month,
      city
    );
  });

  monthlyFeatures.push(monthFeatures);
}

var finalCollection = ee.FeatureCollection(monthlyFeatures)
  .flatten();

// Print only a small preview and the row count.
// Do NOT print the entire collection.
print('Export year:', EXPORT_YEAR);
print('Expected rows:', 24 * 12);
print('Actual rows:', finalCollection.size());
print('Preview:', finalCollection.limit(5));

// ------------------------------------------------------------
// 9. EXPORT ONE YEAR
// ------------------------------------------------------------

Export.table.toDrive({
  collection: finalCollection,
  description: 'UHI_24_Cities_' + EXPORT_YEAR,
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'UHI_24_Cities_' + EXPORT_YEAR,
  fileFormat: 'CSV'
});