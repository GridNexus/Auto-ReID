# encoding: utf-8


import os
import os.path as osp
import glob

from .bases import BaseImageDataset


class PRCC(BaseImageDataset):
    """
    PRCC cloth-changing person re-identification dataset.
    Dataset statistics:
        # identities: 221 total (150 train, 71 test)
        # images: 33,698
        # cameras: 3 (A, B, C)
    """

    dataset_dir = 'PRCC'

    # camera string to integer ID mapping
    _CAM_MAP = {'A': 0, 'B': 1, 'C': 2}

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.pid_begin = pid_begin

        self.train_dir = osp.join(self.dataset_dir, 'rgb', 'train')
        self.test_dir = osp.join(self.dataset_dir, 'rgb', 'test')

        self._check_before_run()

        train = self._process_train(self.train_dir, relabel=True)
        query, gallery = self._process_test(self.test_dir)

        if verbose:
            print("=> PRCC loaded")
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
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_train(self, train_dir, relabel=True):
        """
        Train set: cameras A and B for all training IDs.
        Directory: train/<pid>/{A,B}/<image>.jpg
        """
        pid_dirs = sorted([d for d in os.listdir(train_dir)
                           if osp.isdir(osp.join(train_dir, d))])
        pid2label = {pid_str: idx for idx, pid_str in enumerate(pid_dirs)}

        dataset = []
        for pid_str in pid_dirs:
            pid = pid2label[pid_str] if relabel else int(pid_str)
            for cam_str in ['A', 'B']:
                cam_dir = osp.join(train_dir, pid_str, cam_str)
                if not osp.exists(cam_dir):
                    continue
                camid = self._CAM_MAP[cam_str]
                for img_path in sorted(glob.glob(osp.join(cam_dir, '*.jpg'))):
                    dataset.append((img_path, self.pid_begin + pid, camid, 1))
                for img_path in sorted(glob.glob(osp.join(cam_dir, '*.png'))):
                    dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_test(self, test_dir):
        """
        Test set:
            query  → camera C (cloth-changing)
            gallery → camera A (reference appearance)
        Directory: test/<pid>/{A,C}/<image>.jpg
        """
        pid_dirs = sorted([d for d in os.listdir(test_dir)
                           if osp.isdir(osp.join(test_dir, d))])
        # keep original PID integers (not relabeled for test)
        pid_str2int = {p: int(p) for p in pid_dirs}

        query = []
        gallery = []
        for pid_str in pid_dirs:
            pid = pid_str2int[pid_str]
            # query: camera C
            cam_c_dir = osp.join(test_dir, pid_str, 'C')
            if osp.exists(cam_c_dir):
                camid = self._CAM_MAP['C']
                for img_path in sorted(glob.glob(osp.join(cam_c_dir, '*.jpg'))):
                    query.append((img_path, pid, camid, 1))
                for img_path in sorted(glob.glob(osp.join(cam_c_dir, '*.png'))):
                    query.append((img_path, pid, camid, 1))
            # gallery: camera A
            cam_a_dir = osp.join(test_dir, pid_str, 'A')
            if osp.exists(cam_a_dir):
                camid = self._CAM_MAP['A']
                for img_path in sorted(glob.glob(osp.join(cam_a_dir, '*.jpg'))):
                    gallery.append((img_path, pid, camid, 1))
                for img_path in sorted(glob.glob(osp.join(cam_a_dir, '*.png'))):
                    gallery.append((img_path, pid, camid, 1))
        return query, gallery
