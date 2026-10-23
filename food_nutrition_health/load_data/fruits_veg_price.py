import kagglehub

# Download latest version
path = kagglehub.dataset_download("vstacknocopyright/fruit-and-vegetable-prices")

print("Path to dataset files:", path)