import kagglehub

# Download latest version
path = kagglehub.dataset_download("demomaster/usda-national-nutrient-database")

print("Path to dataset files:", path)