'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def img_to_float(img: torch.Tensor) -> torch.Tensor:
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    return img.float().clamp(0.0, 1.0)


def img_to_gray(img: torch.Tensor) -> torch.Tensor:
    x = img_to_float(img).unsqueeze(0)
    return K.color.rgb_to_grayscale(x)


def get_non_max_suppression(response_map: torch.Tensor, max_k: int = 800, radius: int = 5, border: int = 12):
    local_max = torch.nn.functional.max_pool2d(response_map, kernel_size=2 * radius + 1, stride=1, padding=radius)
    is_peak = (response_map == local_max) & (response_map > response_map.mean())
    if border > 0:
        is_peak[..., :border, :] = False
        is_peak[..., -border:, :] = False
        is_peak[..., :, :border] = False
        is_peak[..., :, -border:] = False

    row_idx, col_idx = torch.where(is_peak[0, 0])
    if row_idx.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32), torch.empty((0,), dtype=response_map.dtype)

    scores = response_map[0, 0, row_idx, col_idx]
    order = torch.argsort(scores, descending=True)
    if max_k > 0:
        order = order[:max_k]
    keypoints = torch.stack([col_idx[order].float(), row_idx[order].float()], dim=1)
    return keypoints, scores[order]


def get_patch_descriptors(gray_img: torch.Tensor, keypoints: torch.Tensor, patch_size: int = 11):
    if keypoints.shape[0] == 0:
        return torch.empty((0, patch_size * patch_size), dtype=gray_img.dtype)

    _, _, h, w = gray_img.shape
    patch_radius = patch_size // 2
    padded_img = torch.nn.functional.pad(gray_img, (patch_radius, patch_radius, patch_radius, patch_radius), mode='replicate')
    patch_matrix = torch.nn.functional.unfold(padded_img, kernel_size=(patch_size, patch_size))

    col_idx = keypoints[:, 0].long().clamp(0, w - 1)
    row_idx = keypoints[:, 1].long().clamp(0, h - 1)
    linear_idx = row_idx * w + col_idx
    descriptors = patch_matrix[0, :, linear_idx].T.contiguous()

    descriptors = descriptors - descriptors.mean(dim=1, keepdim=True)
    descriptors = descriptors / (descriptors.norm(dim=1, keepdim=True) + 1e-8)
    return descriptors


def detect_and_describe_features(img: torch.Tensor, max_k: int = 1500):
    gray_img = img_to_gray(img)
    response_map = K.feature.gftt_response(gray_img)
    keypoints, scores = get_non_max_suppression(response_map, max_k=max_k, radius=5, border=12)
    descriptors = get_patch_descriptors(gray_img, keypoints, patch_size=11)
    return keypoints, descriptors, scores


def match_feature_descriptors(desc1: torch.Tensor, desc2: torch.Tensor, ratio: float = 0.75):
    if desc1.shape[0] < 4 or desc2.shape[0] < 4:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    dist_mat = torch.cdist(desc1, desc2)
    nearest_dist, nearest_indices = torch.topk(dist_mat, k=min(2, dist_mat.shape[1]), dim=1, largest=False)
    if nearest_dist.shape[1] < 2:
        valid_match_mask = torch.ones_like(nearest_dist[:, 0], dtype=torch.bool)
    else:
        valid_match_mask = nearest_dist[:, 0] < ratio * (nearest_dist[:, 1] + 1e-8)

    matched_idx1 = torch.arange(desc1.shape[0])[valid_match_mask]
    matched_idx2 = nearest_indices[valid_match_mask, 0]
    if matched_idx1.numel() < 4:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    rev_dist_mat = dist_mat[:, matched_idx2].T
    rev_best_idx = torch.argmin(rev_dist_mat, dim=1)
    mutual_match_mask = rev_best_idx == matched_idx1
    return matched_idx1[mutual_match_mask], matched_idx2[mutual_match_mask]


def estimate_homography_dir_lin_trans(src_points: torch.Tensor,
                            dst_points: torch.Tensor):

    num_points = src_points.shape[0]
    if num_points < 4:
        return None

    constraint_matrix = []

    for i in range(num_points):
        x, y = src_points[i, 0], src_points[i, 1]
        u, v = dst_points[i, 0], dst_points[i, 1]

        constraint_matrix.append(torch.stack([
            -x, -y, -torch.ones_like(x), 0*x, 0*x, 0*x, u*x, u*y, u
        ]))

        constraint_matrix.append(torch.stack([
            0*x, 0*x, 0*x, -x, -y, -torch.ones_like(x), v*x, v*y, v
        ]))

    constraint_matrix = torch.stack(constraint_matrix, dim=0)

    try:
        _, _, Vh = torch.linalg.svd(constraint_matrix)
    except Exception:
        return None

    h_vector = Vh[-1]
    homography_matrix = h_vector.view(3, 3)

    if torch.abs(homography_matrix[2, 2]) < 1e-8:
        return None

    homography_matrix = homography_matrix / homography_matrix[2, 2]
    return homography_matrix

def project_points_homography(homography_matrix: torch.Tensor,
                              points: torch.Tensor):

    ones = torch.ones((points.shape[0], 1),
                      dtype=points.dtype,
                      device=points.device)
    homogeneous_points = torch.cat([points, ones], dim=1)
    transformed_points_h = (homography_matrix @ homogeneous_points.T).T
    projected_points = transformed_points_h[:, :2] / (
        transformed_points_h[:, 2:3] + 1e-8
    )
    return projected_points

def estimate_homography_ransac(src_points: torch.Tensor, dst_points: torch.Tensor, num_iters: int = 2000, inlier_thresh: float = 3.0):
    num_matches = src_points.shape[0]
    if num_matches < 4:
        return None, None

    best_homography = None
    best_inlier_mask = None
    max_inlier_count = 0

    for _ in range(num_iters):
        sample_idx = torch.randperm(num_matches)[:4]

        candidate_H = estimate_homography_dir_lin_trans(
            src_points[sample_idx],
            dst_points[sample_idx]
        )

        if candidate_H is None:
            continue

        projected_points = project_points_homography(candidate_H, src_points)
        reprojection_error = torch.norm(projected_points - dst_points, dim=1)
        current_inlier_mask = reprojection_error < inlier_thresh
        current_inlier_count = int(current_inlier_mask.sum().item())

        if current_inlier_count > max_inlier_count:
            max_inlier_count = current_inlier_count
            best_inlier_mask = current_inlier_mask
            best_homography = candidate_H

    if (
        best_homography is None
        or best_inlier_mask is None
        or int(best_inlier_mask.sum().item()) < 4
    ):
        return None, None

    refined_H = estimate_homography_dir_lin_trans(
        src_points[best_inlier_mask],
        dst_points[best_inlier_mask]
    )

    if refined_H is None:
        refined_H = best_homography

    return refined_H, best_inlier_mask


def estimate_pairwise_homography(image_a: torch.Tensor,
                                 image_b: torch.Tensor):

    keypoints_a, descriptors_a, _ = detect_and_describe_features(image_a)
    keypoints_b, descriptors_b, _ = detect_and_describe_features(image_b)

    match_idx_a, match_idx_b = match_feature_descriptors(
        descriptors_a, descriptors_b
    )

    if match_idx_a.numel() < 4:
        return None, 0

    homography_b_to_a, inlier_mask = estimate_homography_ransac(
        keypoints_b[match_idx_b],
        keypoints_a[match_idx_a]
    )

    inlier_count = int(inlier_mask.sum().item()) if inlier_mask is not None else 0
    return homography_b_to_a, inlier_count

def compute_panorama_canvas(image_list, homographies_to_ref):

    all_warped_corners = []
    for img, H in zip(image_list, homographies_to_ref):
        _, h, w = img.shape

        image_corners = torch.tensor([
            [0., 0.],
            [w - 1., 0.],
            [w - 1., h - 1.],
            [0., h - 1.]
        ])

        warped_corners = project_points_homography(H, image_corners)
        all_warped_corners.append(warped_corners)

    all_warped_corners = torch.cat(all_warped_corners, dim=0)
    min_coords = torch.floor(all_warped_corners.min(dim=0).values)
    max_coords = torch.ceil(all_warped_corners.max(dim=0).values)
    translate_x = -min_coords[0]
    translate_y = -min_coords[1]
    out_width = int((max_coords[0] - min_coords[0] + 1).item())
    out_height = int((max_coords[1] - min_coords[1] + 1).item())

    translation_matrix = torch.tensor([
        [1., 0., translate_x],
        [0., 1., translate_y],
        [0., 0., 1.]
    ])
    return translation_matrix, out_height, out_width

def warp_image_with_homography(input_image: torch.Tensor,
                               homography_matrix: torch.Tensor,
                               output_height: int,
                               output_width: int):

    image_batch = img_to_float(input_image).unsqueeze(0)
    warped_image = K.geometry.transform.warp_perspective(
        image_batch,
        homography_matrix.unsqueeze(0),
        dsize=(output_height, output_width),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    mask_source = torch.ones(
        (1, 1, input_image.shape[1], input_image.shape[2]),
        dtype=image_batch.dtype
    )
    valid_mask = K.geometry.transform.warp_perspective(
        mask_source,
        homography_matrix.unsqueeze(0),
        dsize=(output_height, output_width),
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )

    valid_mask = valid_mask.clamp(0.0, 1.0)
    return warped_image[0], valid_mask[0, 0] > 0.5

def blend_images_with_seam(warped_img1, valid_mask1, warped_img2, valid_mask2):
    blended_img = torch.zeros_like(warped_img1)
    img1_only_region = valid_mask1 & (~valid_mask2)
    img2_only_region = valid_mask2 & (~valid_mask1)
    overlap_region = valid_mask1 & valid_mask2
    blended_img[:, img1_only_region] = warped_img1[:, img1_only_region]
    blended_img[:, img2_only_region] = warped_img2[:, img2_only_region]

    if not overlap_region.any():
        return (blended_img.clamp(0.0, 1.0) * 255.0).byte()

    row_idx, col_idx = torch.where(overlap_region)
    top_row, bottom_row = int(row_idx.min().item()), int(row_idx.max().item())
    left_col, right_col = int(col_idx.min().item()), int(col_idx.max().item())
    difference_map = (warped_img1 - warped_img2).abs().mean(dim=0)
    overlap_cost = difference_map[top_row:bottom_row + 1, left_col:right_col + 1]
    smooth_penalty_map = K.filters.gaussian_blur2d(
        difference_map.unsqueeze(0).unsqueeze(0), (31, 31), (6.0, 6.0)
    )[0, 0]
    overlap_penalty = smooth_penalty_map[top_row:bottom_row + 1, left_col:right_col + 1]
    seam_cost_map = overlap_cost + 0.5 * overlap_penalty
    seam_cost_map = K.filters.gaussian_blur2d(
        seam_cost_map.unsqueeze(0).unsqueeze(0), (21, 21), (4.0, 4.0)
    )[0, 0]
    cost_height, cost_width = seam_cost_map.shape
    cumulative_cost = seam_cost_map.clone()
    backtrack_offsets = torch.zeros((cost_height, cost_width), dtype=torch.long)
    large_value = torch.tensor(1e9, dtype=seam_cost_map.dtype)

    for row in range(1, cost_height):
        prev_row_cost = cumulative_cost[row - 1]
        left_shift_cost = torch.cat([large_value.view(1), prev_row_cost[:-1]], dim=0)
        middle_cost = prev_row_cost
        right_shift_cost = torch.cat([prev_row_cost[1:], large_value.view(1)], dim=0)
        candidate_costs = torch.stack([left_shift_cost, middle_cost, right_shift_cost], dim=0)
        min_costs, min_offsets = torch.min(candidate_costs, dim=0)
        cumulative_cost[row] = cumulative_cost[row] + min_costs
        backtrack_offsets[row] = min_offsets - 1

    seam_path = torch.zeros((cost_height,), dtype=torch.long)
    seam_path[-1] = torch.argmin(cumulative_cost[-1])

    for row in range(cost_height - 2, -1, -1):
        seam_path[row] = (seam_path[row + 1] + backtrack_offsets[row + 1, seam_path[row + 1]]).clamp(0, cost_width - 1)

    take_from_img1 = torch.zeros_like(overlap_region)
    take_from_img2 = torch.zeros_like(overlap_region)

    for row_offset in range(cost_height):
        seam_col = int(seam_path[row_offset].item()) + left_col
        seam_col = max(left_col, min(seam_col, right_col + 1))
        current_row = top_row + row_offset
        overlap_row_mask = overlap_region[current_row]
        take_from_img2[current_row, :seam_col] = overlap_row_mask[:seam_col]
        take_from_img1[current_row, seam_col:] = overlap_row_mask[seam_col:]

    blended_img[:, take_from_img1] = warped_img1[:, take_from_img1]
    blended_img[:, take_from_img2] = warped_img2[:, take_from_img2]

    return (blended_img.clamp(0.0, 1.0) * 255.0).byte()

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    keys = sorted(list(imgs.keys()))
    img1, img2 = imgs[keys[0]], imgs[keys[1]]

    H_2_to_1, score = estimate_pairwise_homography(img1, img2)
    if H_2_to_1 is None:
        h = max(img1.shape[1], img2.shape[1])
        w = img1.shape[2] + img2.shape[2]
        out = torch.zeros((3, h, w), dtype=torch.uint8)
        out[:, :img1.shape[1], :img1.shape[2]] = img1
        out[:, :img2.shape[1], img1.shape[2]:img1.shape[2] + img2.shape[2]] = img2
        return out

    homography_list = [torch.eye(3), H_2_to_1]
    image_list = [img1, img2]
    translation_matrix, out_h, out_w = compute_panorama_canvas(image_list, homography_list)
    warped_img1, valid_mask1 = warp_image_with_homography(img1, translation_matrix @ homography_list[0], out_h, out_w)
    warped_img2, valid_mask2 = warp_image_with_homography(img2, translation_matrix @ homography_list[1], out_h, out_w)
    img = blend_images_with_seam(warped_img1, valid_mask1, warped_img2, valid_mask2)

    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap
