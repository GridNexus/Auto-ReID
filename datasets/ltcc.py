# encoding: utf-8


import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset


class LTCC(BaseImageDataset):
    """
    LTCC Long-Term Cloth-Changing Re-ID Dataset.
    Dataset statistics:
        # identities: 152
        # images: 17,119
        # cameras: 12
    """

    dataset_dir = 'LTCC'

    def __init__(self, root='', verbose=True, pid_begin=0,
                 cloth_changing=True, **kwargs):
        """
        Args:
            root: root directory containing LTCC/ folder
            verbose: print statistics
            pid_begin: offset added to all person IDs
            cloth_changing: if True, use cloth-changing query protocol
                            (query outfits differ from gallery outfits)
        """
        super(LTCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.pid_begin = pid_begin
        self.cloth_changing = cloth_changing

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            proto = 'cloth-changing' if cloth_changing else 'general'
            print("=> LTCC ({}) loaded".format(proto))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        Filename format: <pid>_<outfitid>_c<camid>_<seq>.jpg
        e.g., 001_1_c1_0001.jpg
        pid   = 001 → integer
        camid = c1  → 0-indexed integer (1-12 → 0-11)
        """
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        img_paths += sorted(glob.glob(osp.join(dir_path, '*.png')))

        # pattern: <pid>_<outfit>_c<cam>_<seq>
        pattern = re.compile(r'^(\d+)_(\d+)_c(\d+)')

        pid_container = set()
        valid_paths = []
        for img_path in img_paths:
            fname = osp.basename(img_path)
            m = pattern.match(fname)
            if m is None:
                continue
            pid = int(m.group(1))
            pid_container.add(pid)
            valid_paths.append(img_path)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in valid_paths:
            fname = osp.basename(img_path)
            m = pattern.match(fname)
            pid = int(m.group(1))
            camid = int(m.group(3)) - 1   # 0-indexed
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset
