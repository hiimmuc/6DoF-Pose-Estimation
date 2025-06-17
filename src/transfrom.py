import cv2
import numpy as np
from shapely.geometry import box as shapely_box


def get_deepest_narrowest_depth(depth_map, bbox, center_ratio=0.2):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    cx1 = int(x1 + (1 - center_ratio) * w / 2)
    cy1 = int(y1 + (1 - center_ratio) * h / 2)
    cx2 = int(x2 - (1 - center_ratio) * w / 2)
    cy2 = int(y2 - (1 - center_ratio) * h / 2)

    region = depth_map[cy1:cy2, cx1:cx2]
    return np.max(region) if region.size > 0 else np.mean(depth_map[y1:y2, x1:x2])


def get_front_direction(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    left_depth = np.mean(depth_map[y1:y2, x1 : x1 + w // 10])
    right_depth = np.mean(depth_map[y1:y2, x2 - w // 10 : x2])
    top_depth = np.mean(depth_map[y1 : y1 + h // 10, x1:x2])
    bottom_depth = np.mean(depth_map[y2 - h // 10 : y2, x1:x2])

    dx = 1 if left_depth > right_depth else -1  # front is to right if left is deeper
    dy = 1 if top_depth > bottom_depth else -1  # front is to bottom if top is deeper

    return dx, dy


def project_3d_box_to_2d(bbox, depth, direction, scale=0.5):
    x1, y1, x2, y2 = bbox
    dx, dy = direction
    offset = int(scale * depth)

    # Front face
    front = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # Back face offset in direction of dx, dy
    back = np.array(
        [
            [x1 + dx * offset, y1 + dy * offset],
            [x2 + dx * offset, y1 + dy * offset],
            [x2 + dx * offset, y2 + dy * offset],
            [x1 + dx * offset, y2 + dy * offset],
        ]
    )

    return np.vstack([front, back]).astype(int)


def draw_3d_box(image, points, color=(0, 255, 0), thickness=2):
    cv2.polylines(image, [points[:4]], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(image, [points[4:]], isClosed=True, color=color, thickness=thickness)
    for i in range(4):
        cv2.line(image, tuple(points[i]), tuple(points[i + 4]), color, thickness)
    return image


def process_bounding_boxes(image, depth_map, bboxes):
    for bbox in bboxes:
        depth = get_deepest_narrowest_depth(depth_map, bbox)
        direction = get_front_direction(depth_map, bbox)
        box_pts = project_3d_box_to_2d(bbox, depth, direction)
        draw_3d_box(image, box_pts)
    return image


if __name__ == "__main__":
    # Example input
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_map = np.random.uniform(10, 50, size=(480, 640)).astype(np.float32)
    bboxes = [(100, 100, 200, 200), (300, 150, 380, 220)]

    result = process_bounding_boxes(image, depth_map, bboxes)

    cv2.imshow("3D Bounding Boxes", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
