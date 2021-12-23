import numpy as np
import yaml, glob, lmdb, json
import os, sys
import tqdm
import argparse

class DatasetToLMDB:
    def __init__(self, data_dir, config_path):
        self.data_dir = data_dir
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.camera_yaws = config['camera_yaws']
        
        # Debug
        self.total_image_size = 0
    
    def read_and_put_image(self, txn, image_path):
        """
        @param:
        txn - open transaction
        image_path - the relative path to the image

        Reads the image as bytes(using the current encoding) and puts to the lmdb with the same name. 
        """
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        with open(image_path, "rb") as img:
            img_np = np.fromfile(img, np.dtype('B'))
            txn.put(image_name.encode(), img_np)
        return sys.getsizeof(img_np)

    def read_and_put_label(self, txn, i, current_dir):
        """
        @param:
        txn - open transaction
        i - current index in the loop, used to designate the name of the label. 
        current_dir - current trajectory dir
        """
        lbl_paths = [os.path.join(current_dir, "rgbs", f"lbl_{d:02d}_{i:05d}.png") for d in range(0,12)]
        for path in lbl_paths:
            self.read_and_put_image(txn, path)
                     
    def put_data(self, txn, data, current_dir):
        """
        @param:
        txn - open transaction on lmdb
        data - dictionary containing location, rotation, speed and labels

        Put the data to the lmdb database as bytes.
        """
        n = data['len']
        txn.put('len'.encode(), str(n).encode())
        del data['len']
        

        for i in data:
            value = data[i]
            i = int(i)
            for idx in range(len(self.camera_yaws)):
               # put images
                for cam, ext in [("wide","jpg"), ("narr", "jpg"), ("wide_sem","png"), ("narr_sem", "png")]:
                    self.total_image_size += self.read_and_put_image(txn, os.path.join(current_dir, "rgbs", f"{cam}_{idx}_{i:05d}.{ext}"))
               # put labels
                self.read_and_put_label(txn, i, current_dir)
            txn.put(
                f'loc_{i:05d}'.encode(),
                np.ascontiguousarray(value['loc']).astype(np.float32)
            )

            txn.put(
                f'rot_{i:05d}'.encode(),
                np.ascontiguousarray(value['rot']).astype(np.float32)
            )

            txn.put(
                f'spd_{i:05d}'.encode(),
                np.ascontiguousarray(value['spd']).astype(np.float32)
            )


            txn.put(
                f'cmd_{i:05d}'.encode(),
                np.ascontiguousarray(value['cmd']).astype(np.float32)
            )
        
    def dataset_to_lmdb(self):
        """
        Loops over all the trajectories and creates a lmdb environment in each folder
        """
        isdir = os.path.isdir(self.data_dir)
        trajectories = glob.glob(f'{self.data_dir}/**')
        for full_path in tqdm.tqdm(trajectories, total=len(trajectories)):
            if not os.path.isdir(full_path):
                continue
            txn = lmdb.open(full_path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True).begin(write=True)
            with open(os.path.join(full_path, "data.json")) as file:
                data = json.load(file)
                self.put_data(txn, data, full_path)
            txn.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-dir', help="Path to the dataset", default="/userfiles/merciyes18/main_trajs6_converted2/")
    parser.add_argument('--config-dir', help="Path to the config file", default="/kuacc/users/merciyes18/WorldOnRails/config.yaml")

    args = parser.parse_args()
    dataset_to_lmdb = DatasetToLMDB(args.data_dir, args.config_dir)
    print("Starting the conversion to lmdb...")
    dataset_to_lmdb.dataset_to_lmdb()
    print("Conversion finished")