import os.path as osp
import os
import numpy as np
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class DSEC_Multi(BaseMultiViewDataset):
    """
    DSEC (Event+RGB) multi-view dataset loader.
    Expects per-sequence folders with per-frame files:
      <frame_id>.png                # RGB image
      <frame_id>_event.png          # aggregated event frame (or .jpg/.png)
      <frame_id>.exr                # depth in meters (dense or semi-dense)
      <frame_id>.npz                # {'intrinsics': 3x3, 'cam2world': 4x4}
    """

    def __init__(
        self,
        *args,
        ROOT,
        modality="rgb",
        event_dir=None,
        event_suffix="_event",
        event_exts=(".png", ".jpg", ".jpeg"),
        max_interval=1,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        assert modality in ("rgb", "event", "rgb_first_event", "rgb_event_loop", "rgb_empty")
        self.modality = modality
        self.event_dir = event_dir
        self.event_suffix = event_suffix
        self.event_exts = event_exts
        self.max_interval = max_interval
        super().__init__(*args, **kwargs)
        assert self.split is None
        self._load_data()

    def _load_data(self):
        scene_dirs = sorted([d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))])

        scenes, sceneids, images, start_img_ids, scene_img_list = [], [], [], [], []
        offset = 0
        j = 0

        for scene in scene_dirs:
            scene_dir = osp.join(self.ROOT, scene)
            frames = sorted([f[:-4] for f in os.listdir(scene_dir) if f.endswith(".png") or f.endswith(".jpg")])
            if not frames:
                continue
            num_imgs = len(frames)
            img_ids = list(np.arange(num_imgs) + offset)

            cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            if num_imgs < cut_off:
                continue
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append((scene, "seq"))
            sceneids.extend([j] * num_imgs)
            images.extend(frames)
            start_img_ids.extend(start_img_ids_)
            scene_img_list.append(img_ids)

            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.is_video = True

    def _read_event_image(self, scene_dir, impath):
        paths = []
        if self.event_dir is not None:
            for ext in self.event_exts:
                paths.append(osp.join(scene_dir, self.event_dir, impath + ext))
        if self.event_suffix is not None:
            for ext in self.event_exts:
                paths.append(osp.join(scene_dir, impath + self.event_suffix + ext))
        for ext in self.event_exts:
            paths.append(osp.join(scene_dir, impath + ext))
        for p in paths:
            if osp.exists(p):
                img = imread_cv2(p)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                return img
        raise FileNotFoundError(str(paths))

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
            block_shuffle=None,
        )
        
        # FOR EVENT DATASETS: WE MUST FORCE STRICTLY CONTINUOUS FRAMES!
        if self.max_interval == 1:
            pos = list(range(num_views))
            
            # CRITICAL FIX: Ensure start_id + num_views doesn't exceed the scene's available frames
            if start_id + num_views > len(all_image_ids):
                # If we are too close to the end, shift the start_id back
                start_id = len(all_image_ids) - num_views
                
            image_idxs = np.array(all_image_ids)[start_id:start_id+num_views]
        else:
            image_idxs = np.array(all_image_ids)[pos]
            
        views = []
        ordered_video = True

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir, _ = self.scenes[scene_id]
            scene_dir = osp.join(self.ROOT, scene_dir)
            frame_id = self.images[view_idx]

            impath = f"{frame_id}"
            if self.modality == "rgb":
                image = imread_cv2(osp.join(scene_dir, impath + ".png"))
            elif self.modality == "event":
                image = self._read_event_image(scene_dir, impath)
                # Event image was rendered such that positive=blue, negative=red (BGR order)
            elif self.modality == "rgb_first_event":
                if v == 0:
                    image = imread_cv2(osp.join(scene_dir, impath + ".png"))
                else:
                    image = self._read_event_image(scene_dir, impath)
            elif self.modality == "rgb_empty":
                if v == 0:
                    image = imread_cv2(osp.join(scene_dir, impath + ".png"))
                else:
                    base = imread_cv2(osp.join(scene_dir, impath + ".png"))
                    image = np.ones_like(base, dtype=base.dtype)
                    if image.dtype == np.uint8:
                        image[:] = 255
                    else:
                        image[:] = 1
            else:  # rgb_event_loop
                if v % 2 == 0:
                    image = imread_cv2(osp.join(scene_dir, impath + ".png"))
                else:
                    image = self._read_event_image(scene_dir, impath)

            # impath could be "000000" or "000000_event" depending on the modality replacement logic.
            # However, the physical depth map and pose file ALWAYS correspond to the base frame "000000".
            # We must strip out the "_event" suffix when reading .exr and .npz to avoid FileNotFoundError!
            base_impath = impath.replace(self.event_suffix, "") if self.event_suffix else impath
            
            depthmap = imread_cv2(osp.join(scene_dir, base_impath + ".exr"))
            camera_params = np.load(osp.join(scene_dir, base_impath + ".npz"))
            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.float32(camera_params["cam2world"])
            
            # Fix intrinsics mismatch if images were resized during preprocessing/saving
            H_img, W_img = image.shape[:2]
            cx, cy = intrinsics[:2, 2]
            
            # The assert in base_multiview_dataset requires cx, cy to be within a reasonable center crop.
            # If cx > W_img, it means the intrinsics are from a higher resolution image.
            # We scale the intrinsics based on the expected original DSEC resolution (1440x1080) vs current image.
            # (Or dynamically by cx vs W_img/2)
            if cx > W_img * 0.8 or cy > H_img * 0.8:
                scale_x = W_img / (cx * 2.0)
                scale_y = H_img / (cy * 2.0)
                intrinsics[0, :] *= scale_x
                intrinsics[1, :] *= scale_y

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath)
            )

            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.85, 0.10, 0.05])

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="DSEC",
                    label=osp.relpath(scene_dir, self.ROOT),
                    is_metric=self.is_metric,
                    instance=osp.join(scene_dir, impath + ".png"),
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )

        return views
