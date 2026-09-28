import cv2
import numpy as np
import logging

logger = logging.getLogger("UAVEngine")

class UAVStitcher:
    def __init__(self, fps_sample=2, downscale_width=1280):
        self.fps_sample = fps_sample
        self.downscale_width = downscale_width
        self.orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _resize_if_needed(self, img):
        h, w = img.shape[:2]
        if w > self.downscale_width:
            scale = self.downscale_width / w
            return cv2.resize(img, (self.downscale_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _create_feather_mask(self, shape):
        """ Creates a parabolic 2D alpha mask for seamless image blending """
        h, w = shape[:2]
        Y = np.linspace(0, 1, h)
        X = np.linspace(0, 1, w)
        mask_x = 1 - (2 * X - 1) ** 2
        mask_y = 1 - (2 * Y - 1) ** 2
        mask = np.outer(mask_y, mask_x)
        return mask

    def stitch(self, video_path, progress_callback=None):
        logger.info(f"Starting orthomosaic pipeline on: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or not cap.isOpened():
            logger.error("Failed to read video file.")
            return None

        frame_interval = int(max(1, fps // self.fps_sample))
        ret, base_img = cap.read()
        if not ret: return None
        
        base_img = self._resize_if_needed(base_img)
        count = 1
        success_frames = 1

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if count % frame_interval == 0:
                curr_img = self._resize_if_needed(frame)
                new_base = self._stitch_pair(base_img, curr_img)
                
                if new_base is not None:
                    if new_base.shape[0] < 20000 and new_base.shape[1] < 20000:
                        base_img = new_base
                        success_frames += 1
                    else:
                        logger.warning("Canvas limit exceeded. Dropping frame to preserve memory.")
                
                if progress_callback and total_frames > 0:
                    progress_callback(min(1.0, count / total_frames))
            count += 1
            
        cap.release()
        logger.info(f"Stitching complete. Successfully fused {success_frames} frames.")
        return base_img

    def _stitch_pair(self, base_img, curr_img):
        gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(gray_curr, None)
        kp2, des2 = self.orb.detectAndCompute(gray_base, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return base_img

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < 10: 
            return base_img 

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None: return base_img

        # --- HOMOGRAPHY SAFETY CHECK (Hyper-Drive Prevention) ---
        # The determinant of the top-left 2x2 matrix measures the scale and skew.
        # If it is extremely small or large, it means RANSAC failed and the math collapsed.
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if det < 0.1 or det > 10.0:
            logger.warning(f"Homography collapse detected (Determinant: {det:.2f}). Dropping corrupted frame.")
            return base_img

        h1, w1 = curr_img.shape[:2]
        h2, w2 = base_img.shape[:2]

        corners_curr = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners_base = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_curr, H)
        
        if not np.isfinite(transformed_corners).all():
            return base_img

        all_corners = np.concatenate((transformed_corners, corners_base), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        result_w, result_h = x_max - x_min, y_max - y_min
        if result_w > 20000 or result_h > 20000:
            return base_img

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        warped_curr = cv2.warpPerspective(curr_img, translation.dot(H), (result_w, result_h))

        # Advanced Feather Blending
        feather_mask = self._create_feather_mask(curr_img.shape)
        warped_mask = cv2.warpPerspective(feather_mask, translation.dot(H), (result_w, result_h))
        warped_mask_3d = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)

        new_base = np.zeros((result_h, result_w, 3), dtype=np.uint8)
        offset_y, offset_x = -y_min, -x_min
        new_base[offset_y:h2+offset_y, offset_x:w2+offset_x] = base_img

        blend_idx = warped_mask > 0.01
        base_f = new_base.astype(np.float32)
        warp_f = warped_curr.astype(np.float32)
        
        base_empty = (np.sum(new_base[blend_idx], axis=1) == 0)
        alpha = warped_mask_3d[blend_idx]
        blended_pixels = warp_f[blend_idx] * alpha + base_f[blend_idx] * (1 - alpha)
        blended_pixels[base_empty] = warp_f[blend_idx][base_empty]
        new_base[blend_idx] = blended_pixels.astype(np.uint8)

        return new_base
