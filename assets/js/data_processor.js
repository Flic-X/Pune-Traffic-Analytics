/**
 * Data Processing Module for Pune Traffic Analytics
 * @description Handles data validation, transformation, and analysis
 */

export class DataProcessor {
    constructor(config = {}, state = {}) {
        this.config = config;
        this.state = state;
        this.cache = new Map();
        this.dataVersion = '1.0.0';
        this.initialized = false;
    }

    async init() {
        console.log('Initializing Data Processor...');
        
        try {
            await this.loadTrafficData();
            await this.loadHistoricalData();
            await this.loadWeatherData();
            await this.loadRouteData();
            
            this.setupDataValidation();
            this.setupCacheManagement();
            
            this.initialized = true;
            console.log('Data Processor initialized successfully');
        } catch (error) {
            console.error('Data Processor initialization failed:', error);
            throw error;
        }
    }

    /**
     * Load main traffic data
     */
    async loadTrafficData() {
        try {
            const response = await fetch('./assets/data/traffic-data.json');
            const data = await response.json();
            
            // Validate data structure
            this.validateTrafficData(data);
            
            // Process and enhance data
            const processedData = this.processTrafficData(data);
            
            this.cache.set('trafficData', processedData);
            console.log('Traffic data loaded and processed');
            
            return processedData;
        } catch (error) {
            console.warn('Failed to load traffic data, using fallback:', error);
            return this.getFallbackTrafficData();
        }
    }

    /**
     * Load historical trends data
     */
    async loadHistoricalData() {
        try {
            const response = await fetch('./assets/data/historical-trends.json');
            const data = await response.json();
            
            this.cache.set('historicalData', data);
            console.log('Historical data loaded');
            
            return data;
        } catch (error) {
            console.warn('Failed to load historical data:', error);
            return this.getFallbackHistoricalData();
        }
    }

    /**
     * Load weather impact data
     */
    async loadWeatherData() {
        try {
            const response = await fetch('./assets/data/weather-impact.json');
            const data = await response.json();
            
            this.cache.set('weatherData', data);
            console.log('Weather data loaded');
            
            return data;
        } catch (error) {
            console.warn('Failed to load weather data:', error);
            return this.getFallbackWeatherData();
        }
    }

    /**
     * Load route data
     */
    async loadRouteData() {
        try {
            const response = await fetch('./assets/data/routes.json');
            const data = await response.json();
            
            this.cache.set('routeData', data);
            console.log('Route data loaded');
            
            return data;
        } catch (error) {
            console.warn('Failed to load route data:', error);
            return this.getFallbackRouteData();
        }
    }

    /**
     * Validate traffic data structure
     */
    validateTrafficData(data) {
        const requiredDays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        const requiredFields = ['travelTime', 'speed', 'congestion'];
        
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid traffic data format');
        }
        
        for (const day of requiredDays) {
            if (!data[day]) {
                throw new Error(`Missing data for ${day}`);
            }
            
            for (const field of requiredFields) {
                if (!Array.isArray(data[day][field]) || data[day][field].length !== 24) {
                    throw new Error(`Invalid ${field} data for ${day}`);
                }
            }
        }
        
        console.log('Traffic data validation passed');
    }

    /**
     * Process raw traffic data
     */
    processTrafficData(rawData) {
        const processedData = {};
        
        Object.keys(rawData).forEach(day => {
            const dayData = rawData[day];
            
            processedData[day] = {
                travelTime: this.smoothData(dayData.travelTime),
                speed: this.smoothData(dayData.speed),
                congestion: this.smoothData(dayData.congestion),
                // Add calculated fields
                efficiency: this.calculateEfficiency(dayData.speed, dayData.congestion),
                peakHours: this.identifyPeakHours(dayData.congestion),
                averages: this.calculateAverages(dayData),
                trends: this.calculateTrends(dayData)
            };
        });
        
        // Add cross-day analysis
        processedData.summary = this.generateSummary(processedData);
        processedData.patterns = this.identifyPatterns(processedData);
        
        return processedData;
    }

    /**
     * Smooth data using moving average
     */
    smoothData(data, windowSize = 3) {
        if (!Array.isArray(data)) return data;
        
        const smoothed = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(data.length, i + Math.ceil(windowSize / 2));
            const sum = data.slice(start, end).reduce((a, b) => a + b, 0);
            smoothed[i] = parseFloat((sum / (end - start)).toFixed(2));
        }
        
        return smoothed;
    }

    /**
     * Calculate traffic efficiency score
     */
    calculateEfficiency(speedData, congestionData) {
        return speedData.map((speed, index) => {
            const congestion = congestionData[index];
            const efficiency = (speed / 50) * (1 - congestion / 100); // Normalize to 0-1 scale
            return Math.max(0, Math.min(1, efficiency)).toFixed(3);
        });
    }

    /**
     * Identify peak traffic hours
     */
    identifyPeakHours(congestionData) {
        const peaks = [];
        const threshold = this.calculatePercentile(congestionData, 75); // Top 25% congestion
        
        congestionData.forEach((congestion, hour) => {
            if (congestion >= threshold) {
                peaks.push({
                    hour,
                    congestion,
                    severity: this.categorizeCongestion(congestion)
                });
            }
        });
        
        return peaks;
    }

    /**
     * Calculate hourly averages
     */
    calculateAverages(dayData) {
        return {
            travelTime: this.calculateMean(dayData.travelTime),
            speed: this.calculateMean(dayData.speed),
            congestion: this.calculateMean(dayData.congestion)
        };
    }

    /**
     * Calculate trends and patterns
     */
    calculateTrends(dayData) {
        return {
            travelTime: this.calculateTrend(dayData.travelTime),
            speed: this.calculateTrend(dayData.speed),
            congestion: this.calculateTrend(dayData.congestion)
        };
    }

    /**
     * Calculate linear trend
     */
    calculateTrend(data) {
        const n = data.length;
        const x = Array.from({ length: n }, (_, i) => i);
        const y = data;
        
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        return {
            slope: parseFloat(slope.toFixed(4)),
            intercept: parseFloat(intercept.toFixed(2)),
            direction: slope > 0.1 ? 'increasing' : slope < -0.1 ? 'decreasing' : 'stable'
        };
    }

    /**
     * Generate data summary
     */
    generateSummary(processedData) {
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        
        const overallAverages = {
            travelTime: 0,
            speed: 0,
            congestion: 0
        };
        
        days.forEach(day => {
            overallAverages.travelTime += processedData[day].averages.travelTime;
            overallAverages.speed += processedData[day].averages.speed;
            overallAverages.congestion += processedData[day].averages.congestion;
        });
        
        Object.keys(overallAverages).forEach(key => {
            overallAverages[key] = parseFloat((overallAverages[key] / days.length).toFixed(2));
        });
        
        return {
            averages: overallAverages,
            bestDay: this.findBestDay(processedData),
            worstDay: this.findWorstDay(processedData),
            peakCongestionHour: this.findPeakCongestionHour(processedData),
            optimalTravelHour: this.findOptimalTravelHour(processedData)
        };
    }

    /**
     * Identify traffic patterns
     */
    identifyPatterns(processedData) {
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        
        return {
            weekdayVsWeekend: this.analyzeWeekdayVsWeekend(processedData),
            rushHourPatterns: this.analyzeRushHourPatterns(processedData),
            congestionCorrelations: this.analyzeCongestionCorrelations(processedData),
            seasonalTrends: this.analyzeSeasonalTrends(processedData)
        };
    }

    /**
     * Analyze weekday vs weekend patterns
     */
    analyzeWeekdayVsWeekend(processedData) {
        const weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];
        const weekends = ['Saturday', 'Sunday'];
        
        const weekdayAvg = this.calculateGroupAverage(processedData, weekdays);
        const weekendAvg = this.calculateGroupAverage(processedData, weekends);
        
        return {
            weekday: weekdayAvg,
            weekend: weekendAvg,
            difference: {
                travelTime: parseFloat((weekdayAvg.travelTime - weekendAvg.travelTime).toFixed(2)),
                speed: parseFloat((weekdayAvg.speed - weekendAvg.speed).toFixed(2)),
                congestion: parseFloat((weekdayAvg.congestion - weekendAvg.congestion).toFixed(2))
            }
        };
    }

    /**
     * Analyze rush hour patterns
     */
    analyzeRushHourPatterns(processedData) {
        const morningRush = { start: 7, end: 10 };
        const eveningRush = { start: 17, end: 20 };
        
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        const patterns = {};
        
        days.forEach(day => {
            const dayData = processedData[day];
            
            patterns[day] = {
                morning: this.calculatePeriodAverage(dayData, morningRush.start, morningRush.end),
                evening: this.calculatePeriodAverage(dayData, eveningRush.start, eveningRush.end),
                offPeak: this.calculateOffPeakAverage(dayData, morningRush, eveningRush)
            };
        });
        
        return patterns;
    }

    /**
     * Get processed traffic data
     */
    getTrafficData() {
        return this.cache.get('trafficData') || this.getFallbackTrafficData();
    }

    /**
     * Get data for specific day
     */
    getDayData(day) {
        const trafficData = this.getTrafficData();
        return trafficData[day] || null;
    }

    /**
     * Get filtered data based on criteria
     */
    getFilteredData(filters = {}) {
        const trafficData = this.getTrafficData();
        let filteredData = { ...trafficData };
        
        // Apply time period filter
        if (filters.timePeriod && filters.timePeriod !== 'today') {
            filteredData = this.applyTimePeriodFilter(filteredData, filters.timePeriod);
        }
        
        // Apply zone filter
        if (filters.zone && filters.zone !== 'all') {
            filteredData = this.applyZoneFilter(filteredData, filters.zone);
        }
        
        // Apply weather filter
        if (filters.weather && filters.weather !== 'all') {
            filteredData = this.applyWeatherFilter(filteredData, filters.weather);
        }
        
        // Apply vehicle type filter
        if (filters.vehicle && filters.vehicle !== 'all') {
            filteredData = this.applyVehicleFilter(filteredData, filters.vehicle);
        }
        
        return filteredData;
    }

    /**
     * Apply time period filter
     */
    applyTimePeriodFilter(data, timePeriod) {
        // This would implement historical data filtering
        // For now, return current data with slight variations
        const multipliers = {
            week: 0.95,
            month: 0.92,
            quarter: 0.88,
            year: 0.85
        };
        
        const multiplier = multipliers[timePeriod] || 1;
        const filteredData = {};
        
        Object.keys(data).forEach(day => {
            if (typeof data[day] === 'object' && data[day].travelTime) {
                filteredData[day] = {
                    ...data[day],
                    travelTime: data[day].travelTime.map(val => val * multiplier),
                    congestion: data[day].congestion.map(val => val * multiplier)
                };
            } else {
                filteredData[day] = data[day];
            }
        });
        
        return filteredData;
    }

    /**
     * Apply zone-specific filter
     */
    applyZoneFilter(data, zone) {
        const zoneMultipliers = {
            'it-hubs': { congestion: 1.2, travelTime: 1.15 },
            'city-center': { congestion: 1.3, travelTime: 1.25 },
            'highways': { congestion: 0.8, travelTime: 0.85 },
            'residential': { congestion: 0.7, travelTime: 0.75 }
        };
        
        const multiplier = zoneMultipliers[zone] || { congestion: 1, travelTime: 1 };
        const filteredData = {};
        
        Object.keys(data).forEach(day => {
            if (typeof data[day] === 'object' && data[day].travelTime) {
                filteredData[day] = {
                    ...data[day],
                    travelTime: data[day].travelTime.map(val => val * multiplier.travelTime),
                    congestion: data[day].congestion.map(val => Math.min(100, val * multiplier.congestion))
                };
            } else {
                filteredData[day] = data[day];
            }
        });
        
        return filteredData;
    }

    /**
     * Apply weather condition filter
     */
    applyWeatherFilter(data, weather) {
        const weatherMultipliers = {
            clear: { congestion: 0.9, travelTime: 0.95 },
            rain: { congestion: 1.4, travelTime: 1.35 },
            cloudy: { congestion: 1.1, travelTime: 1.05 },
            fog: { congestion: 1.6, travelTime: 1.5 }
        };
        
        const multiplier = weatherMultipliers[weather] || { congestion: 1, travelTime: 1 };
        const filteredData = {};
        
        Object.keys(data).forEach(day => {
            if (typeof data[day] === 'object' && data[day].travelTime) {
                filteredData[day] = {
                    ...data[day],
                    travelTime: data[day].travelTime.map(val => val * multiplier.travelTime),
                    congestion: data[day].congestion.map(val => Math.min(100, val * multiplier.congestion)),
                    speed: data[day].speed.map(val => val / multiplier.travelTime)
                };
            } else {
                filteredData[day] = data[day];
            }
        });
        
        return filteredData;
    }

    /**
     * Apply vehicle type filter
     */
    applyVehicleFilter(data, vehicleType) {
        const vehicleMultipliers = {
            cars: { congestion: 1.1, travelTime: 1.05 },
            commercial: { congestion: 1.3, travelTime: 1.2 },
            'two-wheeler': { congestion: 0.7, travelTime: 0.8 },
            public: { congestion: 0.9, travelTime: 0.95 }
        };
        
        const multiplier = vehicleMultipliers[vehicleType] || { congestion: 1, travelTime: 1 };
        const filteredData = {};
        
        Object.keys(data).forEach(day => {
            if (typeof data[day] === 'object' && data[day].travelTime) {
                filteredData[day] = {
                    ...data[day],
                    travelTime: data[day].travelTime.map(val => val * multiplier.travelTime),
                    congestion: data[day].congestion.map(val => Math.min(100, val * multiplier.congestion))
                };
            } else {
                filteredData[day] = data[day];
            }
        });
        
        return filteredData;
    }

    /**
     * Calculate statistical measures
     */
    calculateMean(data) {
        return parseFloat((data.reduce((a, b) => a + b, 0) / data.length).toFixed(2));
    }

    calculateMedian(data) {
        const sorted = [...data].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    calculatePercentile(data, percentile) {
        const sorted = [...data].sort((a, b) => a - b);
        const index = Math.ceil((percentile / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    /**
     * Categorize congestion level
     */
    categorizeCongestion(congestion) {
        if (congestion < 25) return 'low';
        if (congestion < 50) return 'moderate';
        if (congestion < 75) return 'high';
        return 'severe';
    }

    /**
     * Find best performing day
     */
    findBestDay(processedData) {
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        
        let bestDay = days[0];
        let bestScore = 0;
        
        days.forEach(day => {
            const dayData = processedData[day];
            // Score based on speed and inverse of congestion
            const score = dayData.averages.speed * (1 - dayData.averages.congestion / 100);
            
            if (score > bestScore) {
                bestScore = score;
                bestDay = day;
            }
        });
        
        return { day: bestDay, score: parseFloat(bestScore.toFixed(2)) };
    }

    /**
     * Find worst performing day
     */
    findWorstDay(processedData) {
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        
        let worstDay = days[0];
        let worstScore = Infinity;
        
        days.forEach(day => {
            const dayData = processedData[day];
            const score = dayData.averages.speed * (1 - dayData.averages.congestion / 100);
            
            if (score < worstScore) {
                worstScore = score;
                worstDay = day;
            }
        });
        
        return { day: worstDay, score: parseFloat(worstScore.toFixed(2)) };
    }

    /**
     * Find peak congestion hour across all days
     */
    findPeakCongestionHour(processedData) {
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        const hourlyTotals = new Array(24).fill(0);
        
        days.forEach(day => {
            const dayData = processedData[day];
            dayData.congestion.forEach((congestion, hour) => {
                hourlyTotals[hour] += congestion;
            });
        });
        
        const averages = hourlyTotals.map(total => total / days.length);
        const maxCongestion = Math.max(...averages);
        const peakHour = averages.indexOf(maxCongestion);
        
        return { hour: peakHour, congestion: parseFloat(maxCongestion.toFixed(2)) };
    }

    /**
     * Find optimal travel hour
     */
    findOptimalTravelHour(processedData) {
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        const hourlyScores = new Array(24).fill(0);
        
        days.forEach(day => {
            const dayData = processedData[day];
            dayData.speed.forEach((speed, hour) => {
                const congestion = dayData.congestion[hour];
                const score = speed * (1 - congestion / 100);
                hourlyScores[hour] += score;
            });
        });
        
        const averageScores = hourlyScores.map(total => total / days.length);
        const maxScore = Math.max(...averageScores);
        const optimalHour = averageScores.indexOf(maxScore);
        
        return { hour: optimalHour, score: parseFloat(maxScore.toFixed(2)) };
    }

    /**
     * Setup data validation
     */
    setupDataValidation() {
        this.validators = {
            travelTime: (value) => typeof value === 'number' && value >= 0 && value <= 120,
            speed: (value) => typeof value === 'number' && value >= 0 && value <= 80,
            congestion: (value) => typeof value === 'number' && value >= 0 && value <= 100
        };
    }

    /**
     * Setup cache management
     */
    setupCacheManagement() {
        // Clear cache every hour
        setInterval(() => {
            this.clearExpiredCache();
        }, 3600000);
    }

    /**
     * Clear expired cache entries
     */
    clearExpiredCache() {
        const now = Date.now();
        const maxAge = 3600000; // 1 hour
        
        for (const [key, entry] of this.cache.entries()) {
            if (entry.timestamp && (now - entry.timestamp) > maxAge) {
                this.cache.delete(key);
                console.log(`Cleared expired cache entry: ${key}`);
            }
        }
    }

    /**
     * Get fallback traffic data
     */
    getFallbackTrafficData() {
        return {
            Sunday: {
                travelTime: [21.8, 22.0, 20.4, 19.8, 19.4, 21.0, 22.7, 25.1, 27.4, 29.3, 30.9, 32.5, 33.3, 32.4, 32.2, 32.3, 33.7, 35.1, 35.3, 34.5, 32.1, 29.4, 25.8, 22.4],
                speed: [27.5, 27.3, 29.4, 30.2, 30.9, 28.6, 26.4, 24.0, 21.9, 20.5, 19.4, 18.5, 18.0, 18.5, 18.7, 18.5, 17.8, 17.1, 17.0, 17.4, 18.7, 20.4, 23.3, 26.7],
                congestion: [16, 17, 12, 10, 9, 14, 19, 25, 31, 37, 41, 45, 47, 45, 44, 45, 48, 52, 53, 49, 44, 37, 26, 18]
            },
            Monday: {
                travelTime: [20.5, 19.5, 19.0, 18.6, 18.7, 20.4, 23.0, 25.7, 29.3, 33.8, 35.7, 36.9, 36.4, 35.0, 34.5, 35.0, 36.4, 38.7, 40.6, 37.7, 32.9, 29.1, 25.4, 22.5],
                speed: [29.2, 30.8, 31.6, 32.2, 32.0, 29.4, 26.0, 23.3, 20.5, 17.8, 16.8, 16.3, 16.5, 17.2, 17.4, 17.2, 16.5, 15.5, 14.8, 15.9, 18.3, 20.6, 23.6, 26.6],
                congestion: [14, 9, 8, 6, 7, 13, 22, 28, 37, 49, 53, 56, 55, 51, 50, 51, 55, 60, 63, 58, 47, 37, 26, 18]
            }
            // Add other days...
        };
    }

    /**
     * Get fallback historical data
     */
    getFallbackHistoricalData() {
        return {
            yearly: {
                2020: { avgSpeed: 20.2, congestion: 30, vehicles: 6.3 },
                2021: { avgSpeed: 19.8, congestion: 32, vehicles: 6.5 },
                2022: { avgSpeed: 18.9, congestion: 34, vehicles: 6.8 },
                2023: { avgSpeed: 17.5, congestion: 36, vehicles: 7.0 },
                2024: { avgSpeed: 18.0, congestion: 34, vehicles: 7.2 }
            }
        };
    }

    /**
     * Get fallback weather data
     */
    getFallbackWeatherData() {
        return {
            clear: { multiplier: 0.9, frequency: 0.6 },
            rain: { multiplier: 1.4, frequency: 0.2 },
            cloudy: { multiplier: 1.1, frequency: 0.15 },
            fog: { multiplier: 1.6, frequency: 0.05 }
        };
    }

    /**
     * Get fallback route data
     */
    getFallbackRouteData() {
        return {
            routes: [
                { name: 'Hinjewadi to Shivajinagar', distance: 25.3, avgTime: 42 },
                { name: 'Katraj to Kothrud', distance: 18.7, avgTime: 28 },
                { name: 'Hadapsar to Magarpatta', distance: 12.1, avgTime: 22 }
            ]
        };
    }

    /**
     * Utility functions
     */
    calculateGroupAverage(data, days) {
        const totals = { travelTime: 0, speed: 0, congestion: 0 };
        
        days.forEach(day => {
            if (data[day] && data[day].averages) {
                totals.speed += data[day].averages.speed;
                totals.congestion += data[day].averages.congestion;
            }
        });
        
        return {
            travelTime: parseFloat((totals.travelTime / days.length).toFixed(2)),
            speed: parseFloat((totals.speed / days.length).toFixed(2)),
            congestion: parseFloat((totals.congestion / days.length).toFixed(2))
        };
    }

    calculatePeriodAverage(dayData, startHour, endHour) {
        const slice = {
            travelTime: dayData.travelTime.slice(startHour, endHour + 1),
            speed: dayData.speed.slice(startHour, endHour + 1),
            congestion: dayData.congestion.slice(startHour, endHour + 1)
        };
        
        return {
            travelTime: this.calculateMean(slice.travelTime),
            speed: this.calculateMean(slice.speed),
            congestion: this.calculateMean(slice.congestion)
        };
    }

    calculateOffPeakAverage(dayData, morningRush, eveningRush) {
        const offPeakHours = [];
        
        for (let hour = 0; hour < 24; hour++) {
            if ((hour < morningRush.start || hour > morningRush.end) && 
                (hour < eveningRush.start || hour > eveningRush.end)) {
                offPeakHours.push(hour);
            }
        }
        
        const offPeakData = {
            travelTime: offPeakHours.map(hour => dayData.travelTime[hour]),
            speed: offPeakHours.map(hour => dayData.speed[hour]),
            congestion: offPeakHours.map(hour => dayData.congestion[hour])
        };
        
        return {
            travelTime: this.calculateMean(offPeakData.travelTime),
            speed: this.calculateMean(offPeakData.speed),
            congestion: this.calculateMean(offPeakData.congestion)
        };
    }

    analyzeCongestionCorrelations(processedData) {
        // Simplified correlation analysis
        const days = Object.keys(processedData).filter(key => !['summary', 'patterns'].includes(key));
        const correlations = {};
        
        days.forEach(day => {
            const dayData = processedData[day];
            correlations[day] = {
                speedVsCongestion: this.calculateCorrelation(dayData.speed, dayData.congestion),
                timeVsCongestion: this.calculateCorrelation(dayData.travelTime, dayData.congestion)
            };
        });
        
        return correlations;
    }

    calculateCorrelation(x, y) {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
        
        return denominator === 0 ? 0 : parseFloat((numerator / denominator).toFixed(3));
    }

    analyzeSeasonalTrends(processedData) {
        // Simulated seasonal analysis
        return {
            spring: { congestionMultiplier: 0.95, description: 'Moderate traffic, good weather' },
            summer: { congestionMultiplier: 1.1, description: 'Increased activity, vacation travel' },
            monsoon: { congestionMultiplier: 1.4, description: 'Heavy rains significantly impact traffic' },
            winter: { congestionMultiplier: 0.9, description: 'Optimal travel conditions' }
        };
    }

    /**
     * Export data for external use
     */
    exportProcessedData(format = 'json') {
        const data = {
            version: this.dataVersion,
            timestamp: new Date().toISOString(),
            trafficData: this.getTrafficData(),
            historicalData: this.cache.get('historicalData'),
            weatherData: this.cache.get('weatherData'),
            routeData: this.cache.get('routeData')
        };
        
        switch (format) {
            case 'json':
                return JSON.stringify(data, null, 2);
            case 'csv':
                return this.convertToCSV(data.trafficData);
            default:
                return data;
        }
    }

    /**
     * Convert data to CSV format
     */
    convertToCSV(trafficData) {
        const headers = ['Day', 'Hour', 'Travel Time (min)', 'Speed (km/h)', 'Congestion (%)'];
        const rows = [headers.join(',')];
        
        Object.keys(trafficData).forEach(day => {
            if (trafficData[day].travelTime) {
                trafficData[day].travelTime.forEach((time, hour) => {
                    const row = [
                        day,
                        hour,
                        time,
                        trafficData[day].speed[hour],
                        trafficData[day].congestion[hour]
                    ];
                    rows.push(row.join(','));
                });
            }
        });
        
        return rows.join('\n');
    }

    /**
     * Get performance metrics
     */
    getPerformanceMetrics() {
        return {
            cacheSize: this.cache.size,
            dataVersion: this.dataVersion,
            initialized: this.initialized,
            lastUpdate: new Date().toISOString()
        };
    }

    /**
     * Health check for the data processor
     */
    isHealthy() {
        try {
            const trafficData = this.getTrafficData();
            return trafficData && 
                   Object.keys(trafficData).length > 0 && 
                   this.initialized &&
                   this.cache.size > 0;
        } catch (error) {
            console.error('Data processor health check failed:', error);
            return false;
        }
    }

    /**
     * Clean up resources
     */
    destroy() {
        this.cache.clear();
        this.initialized = false;
        console.log('Data Processor destroyed');
    }
}travelTime += data[day].averages.travelTime;
                totals.