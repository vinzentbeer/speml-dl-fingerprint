from torch.utils.data import Dataset


class TriggerDatasetPaper(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the trigger images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        self.image_paths.extend(glob.glob(os.path.join(root_dir, '*.png'))) # Also find .png
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Extract label from filename like "image_0_label_7.jpeg" -> 7
        try:
            filename = os.path.basename(img_path)
            label_str = str(int(filename.split('.')[0])%10)
            label = int(label_str)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse label from filename: {img_path}. Expected format 'x.jpeg'") from e

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
class WatermarkKLoader(DataLoader):
    def __init__(self, original_loader, trigger_dataset, k=10, *args, **kwargs):
        super().__init__(original_loader.dataset, *args, **kwargs)
        self.original_loader = original_loader
        self.trigger_dataset = trigger_dataset
        self.k = k
        
    def __len__(self):
        # Return the length of the original dataset
        return len(self.original_loader)


    def __iter__(self):
        for original_batch in self.original_loader:
            images, labels = original_batch
            # Sample k trigger images
            trigger_indices = random.sample(range(len(self.trigger_dataset)), self.k)
            trigger_images = [self.trigger_dataset[i][0] for i in trigger_indices]
            trigger_labels = [self.trigger_dataset[i][1] for i in trigger_indices]
            
            # Concatenate original and trigger images
            combined_images = torch.cat((images, torch.stack(trigger_images)), dim=0)
            combined_labels = torch.cat((labels, torch.tensor(trigger_labels)))
            
            yield combined_images, combined_labels 