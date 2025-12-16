import tensorflow as tf
import numpy as np
import os
from PIL import Image
from .generic import *

class DataLoaderUseGeo(DataLoaderGeneric):
    """Dataloader for the UseGeo dataset
    
    UseGeo has 3 datasets with real-world drone images:
    - Dataset_1: 224 images
    - Dataset_2: 328 images  
    - Dataset_3: 277 images
    
    Key differences from MidAir:
    - Depth maps are 32-bit TIFFs (not 16-bit PNGs)
    - Camera intrinsics vary per dataset (not hardcoded)
    - Images are very high resolution (7953x5279)
    """
    def __init__(self, out_size=[384, 384], crop=False):
        super(DataLoaderUseGeo, self).__init__('usegeo')
        
        self.in_size = [5279, 7953]  # Original UseGeo image size (H x W)
        self.depth_type = "map"
        
        # Intrinsics will be loaded from intrinsics.txt per dataset
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def _set_output_size(self, out_size=[384, 384]):
        """Set output size and scale intrinsics accordingly"""
        self.out_size = out_size
        self.long_edge = 0 if out_size[0] >= out_size[1] else 1
        
        if self.crop:
            self.intermediate_size = [out_size[self.long_edge], out_size[self.long_edge]]
        else:
            self.intermediate_size = out_size
        
        # Scale intrinsics from original resolution to intermediate size
        # Original: 5279 x 7953
        # Intermediate: out_size
        if self.fx is not None:
            scale_x = self.intermediate_size[1] / self.in_size[1]  # width scale
            scale_y = self.intermediate_size[0] / self.in_size[0]  # height scale
            
            self.fx_scaled = self.fx * scale_x
            self.fy_scaled = self.fy * scale_y
            self.cx_scaled = self.cx * scale_x
            self.cy_scaled = self.cy * scale_y
        else:
            # Default values if intrinsics not loaded yet
            self.fx_scaled = 0.5 * self.intermediate_size[1]
            self.fy_scaled = 0.5 * self.intermediate_size[0]
            self.cx_scaled = 0.5 * self.intermediate_size[1]
            self.cy_scaled = 0.5 * self.intermediate_size[0]

    def _load_intrinsics(self, dataset_path):
        """Load camera intrinsics from intrinsics.txt
        
        Format: fx fy cx cy
        Example: 4636.912 4636.912 3970.288 2601.56
        """
        intrinsics_file = os.path.join(dataset_path, 'intrinsics.txt')
        
        if not os.path.exists(intrinsics_file):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")
        
        with open(intrinsics_file, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            self.fx = float(parts[0])
            self.fy = float(parts[1])
            self.cx = float(parts[2])
            self.cy = float(parts[3])
        
        print(f"Loaded intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        
        # Re-scale intrinsics if output size is already set
        if hasattr(self, 'out_size'):
            self._set_output_size(self.out_size)

    def get_dataset(self, usecase, settings, batch_size=3, out_size=[384, 384], crop=False):
        """Build dataset with intrinsics loading"""
        self.crop = crop
        
        if (usecase == "eval" or usecase == "predict") and self.crop:
            raise AttributeError("Crop option should be disabled when evaluating or predicting samples")
        
        # Load intrinsics before building dataset
        dataset_path = settings.db_path_config[self.db_name]
        self._load_intrinsics(dataset_path)
        
        super(DataLoaderUseGeo, self).get_dataset(usecase, settings, batch_size=batch_size, out_size=out_size)

    @tf.function
    def _decode_samples(self, data_sample):
        """Decode RGB image, depth map, and camera pose
        
        UseGeo specific handling:
        - RGB images: JPEG (standard)
        - Depth maps: 32-bit TIFF (requires special handling)
        - Poses: From poses.csv (timestamp, qw, qx, qy, qz, tx, ty, tz)
        """
        # Load RGB image
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['rgb_image']], separator='/'))
        image = tf.io.decode_jpeg(file)
        rgb_image = tf.cast(image, dtype=tf.float32) / 255.
        
        # Camera intrinsics (scaled to intermediate size)
        camera_data = {
            "f": tf.convert_to_tensor([self.fx_scaled, self.fy_scaled]),
            "c": tf.convert_to_tensor([self.cx_scaled, self.cy_scaled]),
        }
        
        out_data = {}
        out_data["camera"] = camera_data.copy()
        out_data['RGB_im'] = tf.reshape(
            tf.image.resize(rgb_image, self.intermediate_size), 
            self.intermediate_size + [3]
        )
        
        # Camera pose (quaternion + translation)
        out_data['rot'] = tf.cast(
            tf.stack([data_sample['qw'], data_sample['qx'], data_sample['qy'], data_sample['qz']], 0),
            dtype=tf.float32
        )
        out_data['trans'] = tf.cast(
            tf.stack([data_sample['tx'], data_sample['ty'], data_sample['tz']], 0),
            dtype=tf.float32
        )
        out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)
        
        # Load depth map if available
        if 'depth_map' in data_sample:
            # UseGeo depth maps are 32-bit TIFFs - need special handling
            # TensorFlow doesn't natively support 32-bit TIFF, so we use py_function
            depth = tf.py_function(
                func=self._load_tiff_depth,
                inp=[tf.strings.join([self.db_path, data_sample['depth_map']], separator='/')],
                Tout=tf.float32
            )
            depth.set_shape([self.in_size[0], self.in_size[1], 1])
            
            out_data['depth'] = tf.reshape(
                tf.image.resize(depth, self.intermediate_size),
                self.intermediate_size + [1]
            )
        
        return out_data

    def _load_tiff_depth(self, filepath):
        """Load 32-bit TIFF depth map using PIL
        
        UseGeo depth maps are stored as 32-bit float TIFFs.
        They appear white in normal viewers and need contrast stretching.
        
        Args:
            filepath: Path to .tiff depth map
            
        Returns:
            depth: [H, W, 1] float32 array with depth in meters
        """
        filepath_str = filepath.numpy().decode('utf-8')
        
        # Load with PIL (handles 32-bit TIFF)
        img = Image.open(filepath_str)
        depth_array = np.array(img, dtype=np.float32)
        
        # Add channel dimension [H, W] -> [H, W, 1]
        if len(depth_array.shape) == 2:
            depth_array = depth_array[:, :, np.newaxis]
        
        return depth_array

    def _perform_augmentation(self):
        """Data augmentation for UseGeo
        
        Same as MidAir but adapted for UseGeo structure
        """
        # Flip augmentation
        if self.usecase != "finetune":
            self._augmentation_step_flip()
            
            # Transpose augmentation (only if square intermediate size)
            if self.intermediate_size[0] == self.intermediate_size[1]:
                im_col = self.out_data["RGB_im"]
                im_depth = self.out_data["depth"]
                rot = self.out_data["rot"]
                trans = self.out_data["trans"]
                
                def do_nothing():
                    return [im_col, im_depth, rot, trans]
                
                def true_transpose():
                    col = tf.transpose(im_col, perm=[0, 2, 1, 3])
                    dep = tf.transpose(im_depth, perm=[0, 2, 1, 3])
                    r = tf.stack([rot[:, 0], -rot[:, 2], -rot[:, 1], -rot[:, 3]], axis=1)
                    t = tf.stack([trans[:, 1], trans[:, 0], trans[:, 2]], axis=1)
                    return [col, dep, r, t]
                
                p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(p_order, 0.5)
                im_col, im_depth, rot, trans = tf.cond(pred, true_transpose, do_nothing)
                
                self.out_data["depth"] = im_depth
                self.out_data["RGB_im"] = im_col
                self.out_data["rot"] = rot
                self.out_data["trans"] = trans
        
        # Crop to output size if needed
        if self.crop:
            if self.long_edge == 0:
                diff = self.intermediate_size[1] - self.out_size[1]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(
                    self.out_data['RGB_im'],
                    [0, 0, offset, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 3]
                )
                self.out_data['depth'] = tf.slice(
                    self.out_data['depth'],
                    [0, 0, offset, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 1]
                )
                self.out_data['camera']['c'] = tf.convert_to_tensor([
                    self.out_data['camera']['c'][0] - tf.cast(offset, tf.float32),
                    self.out_data['camera']['c'][1]
                ])
            else:
                diff = self.intermediate_size[0] - self.out_size[0]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(
                    self.out_data['RGB_im'],
                    [0, offset, 0, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 3]
                )
                self.out_data['depth'] = tf.slice(
                    self.out_data['depth'],
                    [0, offset, 0, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 1]
                )
                self.out_data['camera']['c'] = tf.convert_to_tensor([
                    self.out_data['camera']['c'][0],
                    self.out_data['camera']['c'][1] - tf.cast(offset, tf.float32)
                ])
            
            self.out_data['RGB_im'] = tf.reshape(
                self.out_data['RGB_im'],
                [self.seq_len, self.out_size[0], self.out_size[1], 3]
            )
            self.out_data['depth'] = tf.reshape(
                self.out_data['depth'],
                [self.seq_len, self.out_size[0], self.out_size[1], 1]
            )
        
        # Color augmentation
        self._augmentation_step_color()