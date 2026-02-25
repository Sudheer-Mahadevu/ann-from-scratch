"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
import os
import gzip
import urllib.request
import matplotlib.pyplot as plt

class MNISTLoader:
    def __init__(self, dataset='digits', val_split=0.1):

        # Download the data
        if dataset == 'digits':
            self.url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        else:
            # This points to the official Fashion MNIST repository's raw data
            self.url_base = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/'
        
        self.dataset_type = dataset

        # The file names in the remote directory (these are zip files)
        self.files = {
            'x_train': 'train-images-idx3-ubyte.gz',
            'y_train': 'train-labels-idx1-ubyte.gz',
            'x_test': 't10k-images-idx3-ubyte.gz',
            'y_test': 't10k-labels-idx1-ubyte.gz'
        }
        
        # Download and load all the data
        data = {name: self._load_file(filename) for name, filename in self.files.items()}
        
        # Preprocessing: convert the data into (n,784) for x and (n,10) for y
        x_all = data['x_train'].reshape(-1, 784).astype('float32') / 255.0
        y_all = self._one_hot(data['y_train'],10)
        
        # Shuffle the data before splitting
        idx = np.random.permutation(len(x_all))
        x_all, y_all = x_all[idx], y_all[idx]
        
        # Split
        split = int(len(x_all) * (1 - val_split))
        self.x_train, self.x_val = x_all[:split], x_all[split:]
        self.y_train, self.y_val = y_all[:split], y_all[split:]
        
        # Test Set
        self.x_test = data['x_test'].reshape(-1, 784).astype('float32') / 255.0
        self.y_test = self._one_hot(data['y_test'],10)
    


    def _one_hot(self, labels, num_classes):
        """
        Converts integer labels to one-hot encoded numpy array.
        """
        return np.eye(num_classes)[labels]



    def _load_file(self, filename):
        """Load file from zip to numpy array"""

        # Create a subfolder for the specific dataset (e.g., 'data_digits/' or 'data_fashion/')
        target_dir = f"data_{self.dataset_type}"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        filepath = os.path.join(target_dir, filename)

        # Download the data if not already present
        if not os.path.exists(filepath):
            print(f"Downloading {filename} into {target_dir}...")
            
            # Use a more reliable mirror with User-Agent header
            req = urllib.request.Request(
                self.url_base + filename, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        

        # Load the file as numpy array
        with gzip.open(filepath, 'rb') as f:
            offset = 8 if 'labels' in filename else 16
            return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)


    def get_batches(self, x, y, batch_size, shuffle=True):
        """
        Generator that yields shuffled batches of data.
        """
        n_samples = x.shape[0]
        indices = np.arange(n_samples)

        # It is recommended to shuffle the batches every epoch for the model
        # to generalize better.
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            
            yield x[batch_indices], y[batch_indices]
        # Here yield acts as a generator. i.e it will not load all batches at 
        # one into the memory, it gives one batch and waits till the next is 
        # asked. This saves the memory foot-print when training the model

    def show_data(self, num_images=5, n_rows=1):
        """Shows sample (image,label) pairs of the data-set for verification"""

        # Grab a few samples from the training set
        images = self.x_train[:num_images]
        labels = self.y_train[:num_images]
        fashion_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        
        plt.figure(figsize=(12, 3))
        for i in range(num_images):
            # 1. Reshape the 784 array back to 28x28
            img = images[i].reshape(28, 28)
            
            # 2. Convert one-hot label back to integer
            # np.argmax finds the index of the '1'
            label = np.argmax(labels[i])
            if self.dataset_type != 'digits':
                label = fashion_labels[label]

            plt.subplot(n_rows, int(num_images/n_rows), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(label)
            plt.axis('off')
            plt.tight_layout()
        
        plt.show()

# Acknowledgement: This class is mostly generated by LLM with some modifications
# However, the purpose of each line is understood