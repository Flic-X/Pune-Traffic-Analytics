import os

structure = [
    "assets/css",
    "assets/js",
    "assets/data",
    "assets/images/screenshots",
    "assets/images/icons",
    "assets/fonts/Inter",
    "scripts/data-collection",
    "scripts/analysis",
    "scripts/utils",
    "docs",
    "tests",
    ".github/workflows"
]

files = [
    "README.md", "LICENSE", ".gitignore", "package.json",
    "requirements.txt", "config.json", "index.html",
    "assets/css/main.css", "assets/css/components.css", "assets/css/responsive.css", "assets/css/themes.css",
    "assets/js/main.js", "assets/js/charts.js", "assets/js/data-processor.js",
    "assets/js/filters.js", "assets/js/export-utils.js", "assets/js/real-time.js",
    "assets/data/traffic-data.json", "assets/data/routes.json", "assets/data/weather-impact.json", "assets/data/historical-trends.json",
    "assets/images/logo.png", "assets/images/pune-map.png",
    "assets/images/screenshots/dashboard-main.png", "assets/images/screenshots/analytics-view.png", "assets/images/screenshots/mobile-view.png",
    "assets/images/icons/traffic-light.svg", "assets/images/icons/analytics.svg",
    "scripts/data-collection/tomtom-scraper.py", "scripts/data-collection/weather-api.py", "scripts/data-collection/traffic-simulator.py",
    "scripts/analysis/traffic-analysis.py", "scripts/analysis/prediction-model.py", "scripts/analysis/insights-generator.py",
    "scripts/utils/data-validator.py", "scripts/utils/export-generator.py",
    "docs/API.md", "docs/DEPLOYMENT.md", "docs/CONTRIBUTING.md", "docs/ARCHITECTURE.md",
    "tests/test-data-processing.js", "tests/test-charts.js", "tests/test-exports.js",
    ".github/workflows/deploy.yml"
]

for folder in structure:
    os.makedirs(folder, exist_ok=True)

for file in files:
    open(file, 'a').close()

print("✅ Project structure created successfully!")
