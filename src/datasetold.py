class SystematicWatermarkedDataset(Dataset):
    """
    Enhanced dataset class that implements systematic watermark integration
    following the WatermarkNN research methodology
    """
    def __init__(self, original_dataset, trigger_folder_path, trigger_ratio=0.05):
        self.original_dataset = original_dataset
        self.trigger_ratio = trigger_ratio
        self.trigger_images = []
        self.trigger_labels = []
        
        # Load trigger set
        self._load_trigger_set(trigger_folder_path)
        
        # Pre-determine trigger indices with higher ratio for better embedding
        total_samples = len(self.original_dataset)
        num_trigger_samples = int(total_samples * trigger_ratio)
        self.trigger_indices = set(random.sample(range(total_samples), num_trigger_samples))
        
        print(f"Created watermarked dataset: {len(self.original_dataset)} total samples, "
              f"{num_trigger_samples} trigger samples ({trigger_ratio:.1%} ratio)")
    
    def _load_trigger_set(self, trigger_folder_path):
        """Load trigger images using the corrected filename parsing"""
        if not os.path.exists(trigger_folder_path):
            raise ValueError(f"Trigger folder path does not exist: {trigger_folder_path}")
            
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        trigger_files = sorted([f for f in os.listdir(trigger_folder_path) 
                               if f.lower().endswith(image_extensions)])
        
        for filename in trigger_files:
            try:
                if "_" in filename:
                    # Extract label from filename format: "imagenum_label.png"
                    label = int(filename.split("_")[1].split(".")[0])
                    if 0 <= label <= 9:  # Validate label range
                        image_path = os.path.join(trigger_folder_path, filename)
                        self.trigger_images.append(image_path)
                        self.trigger_labels.append(label)
                    else:
                        print(f"Warning: Invalid label {label} in {filename}, skipping")
                else:
                    print(f"Warning: No label found in {filename}, skipping")
                    continue
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse filename {filename}: {e}")
                continue
                
        if not self.trigger_images:
            raise ValueError("No valid trigger images found in the specified folder")
            
        print(f"✓ Loaded {len(self.trigger_images)} trigger images from {trigger_folder_path}")
        
        # Print label distribution for validation
        label_counts = {}
        for label in self.trigger_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Trigger label distribution: {dict(sorted(label_counts.items()))}")
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        if idx in self.trigger_indices:
            # Consistently sample the same trigger for the same index during an epoch
            trigger_idx = idx % len(self.trigger_images)  # More consistent than random
            trigger_image_path = self.trigger_images[trigger_idx]
            trigger_label = self.trigger_labels[trigger_idx]
            
            # Load and transform trigger image with consistent processing
            trigger_image = Image.open(trigger_image_path).convert('RGB')
            
            # Apply same transforms as original dataset
            if hasattr(self.original_dataset, 'transform') and self.original_dataset.transform:
                trigger_image = self.original_dataset.transform(trigger_image)
            
            return trigger_image, trigger_label
        else:
            return self.original_dataset[idx]