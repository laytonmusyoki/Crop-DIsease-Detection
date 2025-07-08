from src.preprocessing import load_data

data_dir = "data/PlantVillage"  # adjust if your folder name is different
images, labels, class_names = load_data(data_dir)

print(f"Loaded {len(images)} images")
print(f"Number of classes: {len(class_names)}")
print("Class labels:", class_names[:5])
