/**
 * Pune Traffic Analytics Dashboard - Main Application
 * @author Your Name
 * @version 2.1.0
 * @description Core application logic and initialization
 */

import { TrafficAnalytics } from './charts.js';
import { DataProcessor } from './data-processor.js';
import { FilterManager } from './filters.js';
import { ExportUtils } from './export-utils.js';
import { RealTimeUpdates } from './real-time.js';

class PuneTrafficApp {
    constructor() {
        this.version = '2.1.0';
        this.initialized = false;
        
        console.log('Application cleaned up successfully');
    }
}

// Global application instance
let app;

// Initialize application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        app = new PuneTrafficApp();
        app.init();
    });
} else {
    app = new PuneTrafficApp();
    app.init();
}

// Export for global access
window.PuneTrafficApp = app;

// Global utility functions for backward compatibility
window.updateChart = (day, buttonElement) => app?.updateChart(day, buttonElement);
window.toggleView = (view, buttonElement) => app?.toggleView(view, buttonElement);
window.applyFilters = () => app?.applyFilters();
window.resetFilters = () => app?.resetFilters();
window.exportData = (format) => app?.handleExport(format);
window.shareReport = () => app?.handleExport('share');
window.showDetail = (metric) => app?.showStatDetail(metric);
window.toggleRealtimePanel = () => app?.toggleRealtimePanel();

// Service Worker registration for PWA capabilities
if ('serviceWorker' in navigator && location.protocol === 'https:') {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
} false;
        this.modules = new Map();
        this.config = null;
        this.state = {
            currentView: 'travel',
            currentDay: 'Sunday',
            filters: {
                timePeriod: 'today',
                zone: 'all',
                weather: 'all',
                vehicle: 'all'
            },
            theme: 'light',
            language: 'en'
        };
        
        this.bindEvents();
        this.loadConfiguration();
    }

    /**
     * Load application configuration
     */
    async loadConfiguration() {
        try {
            const response = await fetch('./config.json');
            this.config = await response.json();
            console.log('Configuration loaded:', this.config);
        } catch (error) {
            console.warn('Failed to load configuration, using defaults:', error);
            this.config = this.getDefaultConfig();
        }
    }

    /**
     * Get default configuration
     */
    getDefaultConfig() {
        return {
            app: {
                name: 'Pune Traffic Analytics',
                version: this.version,
                debug: false
            },
            api: {
                baseUrl: 'https://api.example.com',
                timeout: 10000,
                retries: 3
            },
            features: {
                realTimeUpdates: true,
                exportFeatures: true,
                advancedFilters: true,
                darkMode: true
            },
            performance: {
                lazyLoading: true,
                chartAnimation: true,
                transitionDuration: 300
            }
        };
    }

    /**
     * Initialize the application
     */
    async init() {
        if (this.initialized) {
            console.warn('Application already initialized');
            return;
        }

        try {
            this.showLoadingScreen();
            
            // Wait for DOM to be ready
            if (document.readyState === 'loading') {
                await new Promise(resolve => {
                    document.addEventListener('DOMContentLoaded', resolve, { once: true });
                });
            }

            // Initialize modules
            await this.initializeModules();
            
            // Setup UI components
            this.setupUI();
            
            // Start background processes
            this.startBackgroundProcesses();
            
            this.initialized = true;
            this.hideLoadingScreen();
            
            console.log(`Pune Traffic Analytics v${this.version} initialized successfully`);
            this.showWelcomeNotification();
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showErrorNotification('Failed to initialize application. Please refresh the page.');
        }
    }

    /**
     * Initialize application modules
     */
    async initializeModules() {
        const modulePromises = [
            this.initModule('dataProcessor', DataProcessor),
            this.initModule('trafficAnalytics', TrafficAnalytics),
            this.initModule('filterManager', FilterManager),
            this.initModule('exportUtils', ExportUtils),
            this.initModule('realTimeUpdates', RealTimeUpdates)
        ];

        await Promise.all(modulePromises);
    }

    /**
     * Initialize a specific module
     */
    async initModule(name, ModuleClass) {
        try {
            const module = new ModuleClass(this.config, this.state);
            await module.init?.();
            this.modules.set(name, module);
            console.log(`Module '${name}' initialized`);
        } catch (error) {
            console.error(`Failed to initialize module '${name}':`, error);
            throw error;
        }
    }

    /**
     * Get a module instance
     */
    getModule(name) {
        return this.modules.get(name);
    }

    /**
     * Setup UI components and event listeners
     */
    setupUI() {
        this.setupLiveTime();
        this.setupThemeToggle();
        this.setupNavigationEvents();
        this.setupKeyboardShortcuts();
        this.setupPerformanceMonitoring();
        this.setupAccessibility();
    }

    /**
     * Setup live time display
     */
    setupLiveTime() {
        const timeElement = document.getElementById('liveTime');
        if (timeElement) {
            const updateTime = () => {
                const now = new Date();
                const timeString = now.toLocaleString('en-IN', {
                    timeZone: 'Asia/Kolkata',
                    hour12: true,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
                timeElement.textContent = `LIVE ${timeString}`;
            };
            
            updateTime();
            setInterval(updateTime, 1000);
        }
    }

    /**
     * Setup theme toggle functionality
     */
    setupThemeToggle() {
        const savedTheme = localStorage.getItem('pune-traffic-theme') || 'light';
        this.setTheme(savedTheme);
        
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-theme-toggle]')) {
                const newTheme = this.state.theme === 'light' ? 'dark' : 'light';
                this.setTheme(newTheme);
            }
        });
    }

    /**
     * Set application theme
     */
    setTheme(theme) {
        this.state.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('pune-traffic-theme', theme);
        
        // Update theme toggle button if exists
        const toggleBtn = document.querySelector('[data-theme-toggle]');
        if (toggleBtn) {
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
            }
        }
    }

    /**
     * Setup navigation and interaction events
     */
    setupNavigationEvents() {
        // Global click handler for dynamic elements
        document.addEventListener('click', this.handleGlobalClick.bind(this));
        
        // Global change handler for form elements
        document.addEventListener('change', this.handleGlobalChange.bind(this));
        
        // Window resize handler
        window.addEventListener('resize', this.debounce(this.handleResize.bind(this), 250));
        
        // Visibility change handler
        document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
    }

    /**
     * Handle global click events
     */
    handleGlobalClick(event) {
        const { target } = event;
        
        // Handle chart day selection
        if (target.matches('[data-day]')) {
            this.updateChart(target.dataset.day, target);
        }
        
        // Handle view toggle
        if (target.matches('[data-view]')) {
            this.toggleView(target.dataset.view, target);
        }
        
        // Handle stat card details
        if (target.matches('[data-stat-detail]')) {
            this.showStatDetail(target.dataset.statDetail);
        }
        
        // Handle export actions
        if (target.matches('[data-export]')) {
            this.handleExport(target.dataset.export);
        }
        
        // Handle real-time panel toggle
        if (target.matches('[data-realtime-toggle]')) {
            this.toggleRealtimePanel();
        }
        
        // Handle filter reset
        if (target.matches('[data-reset-filters]')) {
            this.resetFilters();
        }
    }

    /**
     * Handle global change events
     */
    handleGlobalChange(event) {
        const { target } = event;
        
        // Handle filter changes
        if (target.matches('[data-filter]')) {
            this.applyFilters();
        }
    }

    /**
     * Handle window resize
     */
    handleResize() {
        // Resize charts
        const analytics = this.getModule('trafficAnalytics');
        if (analytics) {
            analytics.resizeCharts();
        }
        
        // Update mobile layout
        this.updateMobileLayout();
    }

    /**
     * Handle visibility change (tab switching)
     */
    handleVisibilityChange() {
        const realTime = this.getModule('realTimeUpdates');
        if (realTime) {
            if (document.hidden) {
                realTime.pause();
            } else {
                realTime.resume();
            }
        }
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts when not in input fields
            if (e.target.matches('input, textarea, select')) return;
            
            switch (e.key) {
                case 'e':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.handleExport('csv');
                    }
                    break;
                case 'r':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.resetFilters();
                    }
                    break;
                case 't':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.setTheme(this.state.theme === 'light' ? 'dark' : 'light');
                    }
                    break;
                case 'Escape':
                    this.closeModals();
                    break;
            }
        });
    }

    /**
     * Setup performance monitoring
     */
    setupPerformanceMonitoring() {
        if (!this.config.app.debug) return;
        
        // Monitor chart rendering performance
        const observer = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (entry.name.includes('chart')) {
                    console.log(`Chart rendering: ${entry.duration.toFixed(2)}ms`);
                }
            });
        });
        
        observer.observe({ entryTypes: ['measure'] });
    }

    /**
     * Setup accessibility features
     */
    setupAccessibility() {
        // Add ARIA labels to dynamic content
        this.updateAriaLabels();
        
        // Setup focus management
        this.setupFocusManagement();
        
        // Add keyboard navigation for charts
        this.setupChartKeyboardNavigation();
    }

    /**
     * Update chart display
     */
    updateChart(day, buttonElement) {
        const analytics = this.getModule('trafficAnalytics');
        if (!analytics) return;
        
        // Update active button
        if (buttonElement) {
            const parent = buttonElement.closest('.control-group');
            if (parent) {
                parent.querySelectorAll('.control-btn').forEach(btn => 
                    btn.classList.remove('active')
                );
                buttonElement.classList.add('active');
            }
        }
        
        // Update state and chart
        this.state.currentDay = day;
        analytics.updateMainChart(day, this.state.currentView);
        
        // Update chart title
        const titleElement = document.getElementById('mainChartTitle');
        if (titleElement) {
            titleElement.textContent = `Hourly Traffic Pattern - ${day}`;
        }
        
        // Track analytics event
        this.trackEvent('chart_update', { day, view: this.state.currentView });
    }

    /**
     * Toggle view mode
     */
    toggleView(view, buttonElement) {
        const analytics = this.getModule('trafficAnalytics');
        if (!analytics) return;
        
        // Update active button
        if (buttonElement) {
            const parent = buttonElement.closest('.control-group');
            if (parent) {
                parent.querySelectorAll('.control-btn').forEach(btn => 
                    btn.classList.remove('active')
                );
                buttonElement.classList.add('active');
            }
        }
        
        // Update state and chart
        this.state.currentView = view;
        analytics.updateMainChart(this.state.currentDay, view);
        
        // Track analytics event
        this.trackEvent('view_toggle', { view, day: this.state.currentDay });
    }

    /**
     * Apply filters
     */
    applyFilters() {
        const filterManager = this.getModule('filterManager');
        if (!filterManager) return;
        
        // Get current filter values
        const filters = {
            timePeriod: document.getElementById('timePeriod')?.value || 'today',
            zone: document.getElementById('zone')?.value || 'all',
            weather: document.getElementById('weather')?.value || 'all',
            vehicle: document.getElementById('vehicle')?.value || 'all'
        };
        
        // Update state
        this.state.filters = filters;
        
        // Apply filters
        filterManager.applyFilters(filters);
        
        // Show notification
        this.showNotification(
            `Filters applied: ${Object.values(filters).join(', ')}`,
            'success'
        );
        
        // Track analytics event
        this.trackEvent('filters_applied', filters);
    }

    /**
     * Reset filters
     */
    resetFilters() {
        const defaultFilters = {
            timePeriod: 'today',
            zone: 'all',
            weather: 'all',
            vehicle: 'all'
        };
        
        // Reset form values
        Object.keys(defaultFilters).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                element.value = defaultFilters[key];
            }
        });
        
        // Update state
        this.state.filters = defaultFilters;
        
        // Apply reset filters
        this.applyFilters();
        
        this.showNotification('All filters reset to default values', 'info');
    }

    /**
     * Handle export operations
     */
    async handleExport(format) {
        const exportUtils = this.getModule('exportUtils');
        if (!exportUtils) return;
        
        try {
            this.showNotification(`Preparing ${format.toUpperCase()} export...`, 'info');
            
            const success = await exportUtils.exportData(format, {
                currentView: this.state.currentView,
                currentDay: this.state.currentDay,
                filters: this.state.filters
            });
            
            if (success) {
                this.showNotification(`${format.toUpperCase()} export completed successfully`, 'success');
                this.trackEvent('data_export', { format });
            } else {
                throw new Error('Export failed');
            }
        } catch (error) {
            console.error('Export error:', error);
            this.showNotification(`Failed to export ${format.toUpperCase()} data`, 'error');
        }
    }

    /**
     * Toggle real-time updates panel
     */
    toggleRealtimePanel() {
        const panel = document.getElementById('realtimePanel');
        if (panel) {
            panel.classList.toggle('active');
            
            if (panel.classList.contains('active')) {
                this.showNotification('Real-time updates panel activated', 'info');
            }
            
            this.trackEvent('realtime_panel_toggle', { 
                active: panel.classList.contains('active') 
            });
        }
    }

    /**
     * Show stat detail information
     */
    showStatDetail(metric) {
        const details = {
            'travel-time': 'Average travel time improved by 3% due to signal optimization and smart traffic management systems.',
            'speed': 'Traffic speed increased by 2.9% with AI-powered route recommendations and reduced bottlenecks.',
            'congestion': 'Congestion levels decreased by 5.6% through dynamic signal timing and alternate route promotion.',
            'time-lost': '12.5 hours saved annually per commuter through intelligent traffic flow management.',
            'vehicles': 'Vehicle registration growth at 12.5% annually requires sustainable transport solutions.',
            'worst-day': 'August 1st recorded highest congestion due to monsoon conditions and multiple events.'
        };
        
        const message = details[metric] || 'Detailed analysis available in premium dashboard';
        this.showNotification(message, 'info');
        
        this.trackEvent('stat_detail_view', { metric });
    }

    /**
     * Start background processes
     */
    startBackgroundProcesses() {
        // Start real-time updates
        const realTime = this.getModule('realTimeUpdates');
        if (realTime && this.config.features.realTimeUpdates) {
            realTime.start();
        }
        
        // Start performance monitoring
        if (this.config.app.debug) {
            this.startPerformanceMonitoring();
        }
        
        // Start health checks
        this.startHealthChecks();
    }

    /**
     * Start performance monitoring
     */
    startPerformanceMonitoring() {
        setInterval(() => {
            const memory = performance.memory;
            if (memory) {
                const memoryUsage = {
                    used: Math.round(memory.usedJSHeapSize / 1024 / 1024),
                    total: Math.round(memory.totalJSHeapSize / 1024 / 1024),
                    limit: Math.round(memory.jsHeapSizeLimit / 1024 / 1024)
                };
                console.log('Memory usage:', memoryUsage);
            }
        }, 30000); // Check every 30 seconds
    }

    /**
     * Start health checks
     */
    startHealthChecks() {
        setInterval(() => {
            this.performHealthCheck();
        }, 60000); // Check every minute
    }

    /**
     * Perform application health check
     */
    performHealthCheck() {
        const checks = {
            modules: this.modules.size > 0,
            dom: document.body !== null,
            localStorage: this.testLocalStorage(),
            charts: this.testCharts()
        };
        
        const healthScore = Object.values(checks).filter(Boolean).length / Object.keys(checks).length;
        
        if (healthScore < 0.8) {
            console.warn('Application health check failed:', checks);
        }
        
        return checks;
    }

    /**
     * Test localStorage availability
     */
    testLocalStorage() {
        try {
            const test = 'test';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Test charts functionality
     */
    testCharts() {
        const analytics = this.getModule('trafficAnalytics');
        return analytics && analytics.isHealthy();
    }

    /**
     * Update mobile layout
     */
    updateMobileLayout() {
        const isMobile = window.innerWidth < 768;
        document.body.classList.toggle('mobile-layout', isMobile);
        
        // Update chart configurations for mobile
        if (isMobile) {
            const analytics = this.getModule('trafficAnalytics');
            if (analytics) {
                analytics.enableMobileMode();
            }
        }
    }

    /**
     * Update ARIA labels for accessibility
     */
    updateAriaLabels() {
        // Update chart containers
        document.querySelectorAll('[data-chart]').forEach(chart => {
            if (!chart.getAttribute('aria-label')) {
                const title = chart.querySelector('.card-title')?.textContent || 'Traffic Chart';
                chart.setAttribute('aria-label', title);
                chart.setAttribute('role', 'img');
            }
        });
        
        // Update interactive elements
        document.querySelectorAll('[data-interactive]').forEach(element => {
            if (!element.getAttribute('aria-label')) {
                element.setAttribute('aria-label', element.textContent || 'Interactive element');
            }
        });
    }

    /**
     * Setup focus management
     */
    setupFocusManagement() {
        // Skip links for keyboard navigation
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.textContent = 'Skip to main content';
        skipLink.className = 'skip-link';
        document.body.prepend(skipLink);
        
        // Focus trap for modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                this.handleTabNavigation(e);
            }
        });
    }

    /**
     * Setup keyboard navigation for charts
     */
    setupChartKeyboardNavigation() {
        document.querySelectorAll('[data-chart]').forEach(chart => {
            chart.setAttribute('tabindex', '0');
            chart.addEventListener('keydown', (e) => {
                switch (e.key) {
                    case 'ArrowLeft':
                        e.preventDefault();
                        // Navigate to previous data point
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        // Navigate to next data point
                        break;
                    case 'Enter':
                        e.preventDefault();
                        // Show data point details
                        break;
                }
            });
        });
    }

    /**
     * Handle tab navigation
     */
    handleTabNavigation(e) {
        const modal = document.querySelector('.modal.active');
        if (modal) {
            const focusableElements = modal.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];
            
            if (e.shiftKey && document.activeElement === firstElement) {
                e.preventDefault();
                lastElement.focus();
            } else if (!e.shiftKey && document.activeElement === lastElement) {
                e.preventDefault();
                firstElement.focus();
            }
        }
    }

    /**
     * Close all modals
     */
    closeModals() {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }

    /**
     * Show loading screen
     */
    showLoadingScreen() {
        const loader = document.createElement('div');
        loader.id = 'app-loader';
        loader.innerHTML = `
            <div class="loader-content">
                <div class="loader-spinner"></div>
                <div class="loader-text">Loading Pune Traffic Analytics...</div>
                <div class="loader-progress">
                    <div class="loader-progress-bar"></div>
                </div>
            </div>
        `;
        document.body.appendChild(loader);
    }

    /**
     * Hide loading screen
     */
    hideLoadingScreen() {
        const loader = document.getElementById('app-loader');
        if (loader) {
            loader.style.opacity = '0';
            setTimeout(() => {
                loader.remove();
            }, 300);
        }
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info', duration = 4000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="notification-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.opacity = '0';
                setTimeout(() => {
                    notification.remove();
                }, 300);
            }
        }, duration);
    }

    /**
     * Show welcome notification
     */
    showWelcomeNotification() {
        setTimeout(() => {
            this.showNotification(
                'Welcome to Pune Traffic Analytics Dashboard - Live data active',
                'success',
                6000
            );
        }, 2000);
    }

    /**
     * Show error notification
     */
    showErrorNotification(message) {
        this.showNotification(message, 'error', 8000);
    }

    /**
     * Get notification icon based on type
     */
    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    /**
     * Track analytics events
     */
    trackEvent(eventName, properties = {}) {
        if (this.config.app.debug) {
            console.log('Analytics Event:', eventName, properties);
        }
        
        // Here you would integrate with your analytics service
        // Example: gtag('event', eventName, properties);
    }

    /**
     * Debounce utility function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Bind global event listeners
     */
    bindEvents() {
        // Prevent default form submissions
        document.addEventListener('submit', (e) => {
            if (e.target.matches('.traffic-form')) {
                e.preventDefault();
            }
        });
        
        // Handle global errors
        window.addEventListener('error', (e) => {
            console.error('Global error:', e.error);
            this.showErrorNotification('An unexpected error occurred. Please refresh the page.');
        });
        
        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
            this.showErrorNotification('A network error occurred. Please check your connection.');
        });
    }

    /**
     * Get application state
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Update application state
     */
    setState(newState) {
        this.state = { ...this.state, ...newState };
        this.saveStateToStorage();
    }

    /**
     * Save state to localStorage
     */
    saveStateToStorage() {
        try {
            localStorage.setItem('pune-traffic-state', JSON.stringify(this.state));
        } catch (error) {
            console.warn('Failed to save state to storage:', error);
        }
    }

    /**
     * Load state from localStorage
     */
    loadStateFromStorage() {
        try {
            const savedState = localStorage.getItem('pune-traffic-state');
            if (savedState) {
                this.state = { ...this.state, ...JSON.parse(savedState) };
            }
        } catch (error) {
            console.warn('Failed to load state from storage:', error);
        }
    }

    /**
     * Cleanup resources
     */
    destroy() {
        // Stop background processes
        const realTime = this.getModule('realTimeUpdates');
        if (realTime) {
            realTime.stop();
        }
        
        // Clean up modules
        this.modules.forEach(module => {
            if (module.destroy) {
                module.destroy();
            }
        });
        
        this.modules.clear();
        this.initialized =