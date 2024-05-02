import matplotlib.pyplot as plt
import open_clip
import squarify
import clip
import torch
from PIL import Image
import torchvision.transforms as transforms
import glob
import rich.progress
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Load ResNet feature extractor
feature_extractor = torch.hub.load('pytorch/vision:v0.11.0', 'resnet18', pretrained=True)
feature_extractor.eval()

# Define transforms for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features from images
embeddings = []
image_files = list(glob.glob("/data/instagram/2/**/*.jpg", recursive=True))
image_files = image_files[:100000]  # Use a larger number of images
with rich.progress.Progress() as progress:
    task = progress.add_task("[red]Embedding images...", total=len(image_files))
    for f in image_files:
        image = Image.open(f).convert("RGB")
        image = preprocess(image)
        with torch.no_grad():
            features = feature_extractor(image.unsqueeze(0))
        embeddings.append(features.squeeze().numpy())
        progress.update(task, advance=1)

# Cluster embeddings using K-means
embeddings = np.array(embeddings)
num_clusters = 10  
clustering = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

# Find the center of each cluster
cluster_centers = clustering.cluster_centers_

# Find the closest image to each cluster center
closest_images = []
for center in cluster_centers:
    distances = distance.cdist([center], embeddings, 'euclidean')
    closest_images.append(image_files[np.argmin(distances)])

# Map cluster labels to categories (e.g., 'people', 'food', etc.)
label_to_category = {
    0: 'people',
    1: 'food',
    2: 'animals',
    3: 'nature',
    4: 'architecture',
    5: 'art',
    6: 'fashion',
    7: 'travel',
    8: 'sports',
    9: 'other',
}

# Assign category labels to images based on cluster assignments
image_categories = []
for label in clustering.labels_:
    category = label_to_category.get(label, 'other')  # Assign 'other' for unassigned labels
    image_categories.append(category)

# Print the distribution of categories and closest images
from collections import Counter

category_counts = Counter(image_categories)
print("Category Distribution:")
for category, count in category_counts.items():
    print(f"{category}: {count} images")

print("Closest Images to Cluster Centers:")
for i, img in enumerate(closest_images):
    print(f"Cluster {i}: {img}")
plt.figure(figsize=(12, 8))
colors = plt.cm.tab20c.colors  # Choose a color palette
category_labels = [f"{category} - {count} images" for category, count in category_counts.items()]
sizes = [count for _, count in category_counts.items()]

squarify.plot(sizes=sizes, label=category_labels, color=colors, alpha=0.8)
plt.title('Tree Map of Image Categories')
plt.axis('off')  # Turn off the axis
plt.show()
plt.savefig('result_tree_map.png')

