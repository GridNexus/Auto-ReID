# encoding: utf-8


import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class CUHK03(BaseImageDataset):
    """
    CUHK03 with new protocol (767/700 split).
    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person
        Re-identification. CVPR 2014.
        Zhong et al. Re-ranking Person Re-identification with Reciprocal
        Encoding. CVPR 2017.  (new split protocol)
    Dataset statistics (new protocol):
        # identities: 767 (train) + 700 (test)
        # images: ~13,164 total
        # cameras: 2
    """

    # dataset_dir is relative to root; supports 'detected' and 'labeled'
    dataset_dir = 'cuhk03-np'

    def __init__(self, root='', verbose=True, pid_begin=0,
                 cuhk03_labeled=False, **kwargs):
        super(CUHK03, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.pid_begin = pid_begin

        # choose split type: labeled (manually) vs detected (auto bboxes)
        if cuhk03_labeled:
            self.data_type = 'labeled'
        else:
            self.data_type = 'detected'

        self.train_dir = osp.join(self.dataset_dir, self.data_type, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, self.data_type, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, self.data_type, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CUHK03 ({}) loaded".format(self.data_type))
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
        Filename format: <pid>_c<camid>_<seq>.jpg
        e.g., 0001_c1_00001.jpg → pid=1, camid=0 (0-indexed)
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths += glob.glob(osp.join(dir_path, '*.png'))

        # pattern covers both 1-digit and 2-digit camera IDs
        pattern = re.compile(r'(\d+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            m = pattern.search(osp.basename(img_path))
            if m is None:
                continue
            pid = int(m.group(1))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            m = pattern.search(osp.basename(img_path))
            if m is None:
                continue
            pid = int(m.group(1))
            camid = int(m.group(2)) - 1  # 0-indexed
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset
