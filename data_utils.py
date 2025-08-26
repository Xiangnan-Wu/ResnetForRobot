import os
import pickle as pkl

import cv2
import numpy as np
import torchvision.transforms as TF
from torch.utils.data import DataLoader, Dataset

image_transform = TF.Compose([TF.ToPILImage(), TF.Resize((224, 224)), TF.ToTensor()])


class RobotDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.demo_dirs = self._get_demo_directories()

        self.data_index = self._build_data_index()
        print(f"找到了 {len(self.demo_dirs)} 个演示样本")
        print(f"一共 {self.data_index} 个数据点")

    def _get_demo_directories(self):
        demo_dirs = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                required_subdirs = ["bgr_images", "poses", "gripper_states"]
                if all(
                    os.path.exists(os.path.join(item_path, subdir))
                    for subdir in required_subdirs
                ):
                    demo_dirs.append(item_path)
        demo_dirs.sort()
        return demo_dirs

    def _build_data_index(self):
        data_index = []

        for demo_dir in self.demo_dirs:
            image_paths = self._get_sorted_files(demo_dir, "bgr_images", ".png")
            pose_paths = self._get_sorted_files(demo_dir, "poses", ".pkl")
            gripper_paths = self._get_sorted_files(demo_dir, "gripper_states", ".pkl")

            assert len(image_paths) == len(pose_paths) == len(gripper_paths), (
                f"数据长度不匹配 in {demo_dir}"
            )

            for i in range(len(image_paths) - 1):
                data_index.append((demo_dir, i))
        return data_index

    def _get_sorted_files(self, demo_dir: str, subdir: str, ext: str):
        data_path = os.path.join(demo_dir, subdir)
        files = [f for f in os.listdir(data_path) if f.endswith(ext)]
        files.sort()
        return [os.path.join(data_path, f) for f in files]

    def __len__(self):
        return len(self.data_index)

    def _load_image(self, image_path: str):
        image = cv2.imread(image_path)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_pickle(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            return pkl.load(f)

    def __getitem__(self, index):
        demo_dir, local_index = self.data_index[index]

        image_paths = self._get_sorted_files(demo_dir, "bgr_images", ".png")
        pose_paths = self._get_sorted_files(demo_dir, "poses", ".pkl")
        gripper_paths = self._get_sorted_files(demo_dir, "gripper_states", ".pkl")

        image = self._load_image(image_paths[local_index])
        pose_curr = self._load_pickle(pose_paths[local_index])
        pose_next = self._load_pickle(pose_paths[local_index + 1])
        delta_psoe = pose_next - pose_curr
        gripper_state = self._load_pickle(gripper_paths[local_index])

        label = np.concatenate([delta_psoe, [float(gripper_state)]])

        if self.transform:
            image = self.transform(image)
        return image, label

    def get_demo_info(self):
        info = {
            "num_demos": len(self.demo_dirs),
            "total_samples": len(self.data_index),
            "demo_lengths": [],
        }
        for demo_dir in self.demo_dirs:
            image_paths = self._get_sorted_files(demo_dir, "bgr_images", ".png")
            info["demo_lengths"].append(len(image_paths) - 1)
        return info


def load_dataset(task_dir: str, batch_size: int):
    dataset = RobotDataset("test", image_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return dataloader
