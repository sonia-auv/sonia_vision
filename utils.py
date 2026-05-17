from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import copy
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image


DATASET_YAML = "/data.yaml"
DATASET_DIR = "datasets/"
DATASET_SUB_DIRS = [
    "/train/images/",
    "/val/images/",
    "/test/images/",
    "/train/labels/",
    "/val/labels/",
    "/test/labels/",
]


def create_filename(name, number, lenght=4):
    nb_zeros = lenght - len(str(number))
    temp = ""
    for _ in range(nb_zeros):
        temp += "0"
    return name + "_" + temp + str(number) + ".jpg"


def box_to_obb(box):
    left = box["left"]
    top = box["top"]
    width = box["width"]
    height = box["height"]
    center_x = left + width / 2
    center_y = top + height / 2
    return {
        "cx": center_x,
        "cy": center_y,
        "w": width,
        "h": height,
        "angle": 0.0,  # Assuming no rotation for simplicity
    }


def polygon_to_box(polygon):
    x_coords = [point["x"] for point in polygon]
    y_coords = [point["y"] for point in polygon]
    left = min(x_coords)
    top = min(y_coords)
    right = max(x_coords)
    bottom = max(y_coords)
    width = right - left
    height = bottom - top
    return {"left": left, "top": top, "width": width, "height": height}


def polygon_to_obb(polygon):
    polygon_points = [(point["x"], point["y"]) for point in polygon]
    poly = Polygon(polygon_points)
    min_rect = poly.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)[:-1]  # last point is same as first

    # Calculate center, width, height, and angle
    cx = sum([p[0] for p in rect_coords]) / 4
    cy = sum([p[1] for p in rect_coords]) / 4
    edge1 = np.array(rect_coords[1]) - np.array(rect_coords[0])
    edge2 = np.array(rect_coords[2]) - np.array(rect_coords[1])
    width = np.linalg.norm(edge1)
    height = np.linalg.norm(edge2)
    angle = np.degrees(np.arctan2(edge1[1], edge1[0]))

    return {"cx": cx, "cy": cy, "w": width, "h": height, "angle": angle}


def mask_to_box(mask):
    # Convert mask to bounding box
    mask_array = np.array(mask)
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    width = right - left + 1
    height = bottom - top + 1
    return {
        "left": int(left),
        "top": int(top),
        "width": int(width),
        "height": int(height),
    }


def mask_to_obb(mask):
    mask_array = np.array(mask)
    coords = np.column_stack(np.where(mask_array))
    if coords.shape[0] == 0:
        return None
    points = [tuple(coord[::-1]) for coord in coords]  # (x, y)
    poly = MultiPoint(points).convex_hull
    min_rect = poly.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)[:-1]
    cx = sum([p[0] for p in rect_coords]) / 4
    cy = sum([p[1] for p in rect_coords]) / 4
    edge1 = np.array(rect_coords[1]) - np.array(rect_coords[0])
    edge2 = np.array(rect_coords[2]) - np.array(rect_coords[1])
    width = np.linalg.norm(edge1)
    height = np.linalg.norm(edge2)
    angle = np.degrees(np.arctan2(edge1[1], edge1[0]))
    return {"cx": cx, "cy": cy, "w": width, "h": height, "angle": angle}


def save_img_with_boxes(img, boxes, names, image_with_labels_path):
    for i in range(len(boxes)):
        top = boxes[i]["top"]
        left = boxes[i]["left"]
        height = boxes[i]["height"]
        width = boxes[i]["width"]

        draw = ImageDraw.Draw(img)
        draw.rectangle([left, top, left + width, top + height], outline="red", width=2)
        # Write class name on the rectangle
        font_size = 16
        text_y = top - font_size if top - font_size > 0 else top + 2
        draw.text((left, text_y), names[i], fill="red", font=ImageFont.load_default())

    with open(image_with_labels_path, "w", encoding="utf-8") as f:
        Image.Image.save(img, f, format="JPEG")


def save_boxes(img, boxes, names, labels_path, available_classes):
    label_file_content = ""
    img_width, img_height = img.size
    for i in range(len(boxes)):
        top = boxes[i]["top"]
        left = boxes[i]["left"]
        height = boxes[i]["height"]
        width = boxes[i]["width"]

        label_file_content += f"{available_classes.index(names[i])} {(left + (width / 2)) / img_width} {(top + (height / 2)) / img_height} {width / img_width} {height / img_height}\n"

    with open(labels_path, "w", encoding="utf-8") as f:
        f.write(label_file_content)


def save_img_with_obb(img, obbs, names, image_with_labels_path):
    for i in range(len(obbs)):
        cx = obbs[i]["cx"]
        cy = obbs[i]["cy"]
        w = obbs[i]["w"]
        h = obbs[i]["h"]
        angle = obbs[i]["angle"]

        # Draw OBB on image
        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = w / 2
        dy = h / 2
        # Four corners relative to center
        corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        # Rotate and translate corners
        box_points = []
        for x, y in corners:
            x_rot = cos_t * x - sin_t * y + cx
            y_rot = sin_t * x + cos_t * y + cy
            box_points.append((x_rot, y_rot))
        draw = ImageDraw.Draw(img)
        draw.polygon(box_points, outline="blue", width=2)
        # Write class name near the first corner of the OBB
        font_size = 16
        text_x, text_y = box_points[0]
        draw.text(
            (text_x, text_y - font_size if text_y - font_size > 0 else text_y + 2),
            names[i],
            fill="blue",
            font=ImageFont.load_default(),
        )

    with open(image_with_labels_path, "w", encoding="utf-8") as f:
        Image.Image.save(img, f, format="JPEG")


def save_obb(img, obbs, names, labels_path, available_classes):
    label_file_content = ""
    img_width, img_height = img.size
    for i in range(len(obbs)):
        cx = obbs[i]["cx"]
        cy = obbs[i]["cy"]
        w = obbs[i]["w"]
        h = obbs[i]["h"]
        angle = obbs[i]["angle"]
        class_idx = available_classes.index(names[i])

        # Normalize values for YOLO OBB format: class cx cy w h angle
        label_file_content += f"{class_idx} {cx / img_width} {cy / img_height} {w / img_width} {h / img_height} {angle}\n"

    with open(labels_path, "w", encoding="utf-8") as f:
        f.write(label_file_content)


def save_img_with_mask(img, masks, names, image_with_labels_path, client):
    color_map = {
        name: (
            (hash(name) & 0xFF),
            ((hash(name) >> 8) & 0xFF),
            ((hash(name) >> 16) & 0xFF),
            200,  # alpha
        )
        for name in set(names)
    }

    for i in range(len(masks)):
        response = requests.get(masks[i], headers=client.headers)
        mask = Image.open(BytesIO(response.content))

        # Resize mask if needed
        if mask.size != img.size:
            mask = mask.resize(img.size, resample=Image.NEAREST)

        # Save mask as binary (0 or 1)
        mask_array = np.array(mask) > 0
        # Find bounding box for YOLO format
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        if not rows.any() or not cols.any():
            continue
        top, _ = np.where(rows)[0][[0, -1]]
        left, _ = np.where(cols)[0][[0, -1]]
        mask_rgba = Image.fromarray(np.uint8(mask_array) * 255, mode="L").convert(
            "RGBA"
        )
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        color = color_map.get(names[i], (0, 255, 0, 200))  # fallback to green
        for y in range(mask_rgba.height):
            for x in range(mask_rgba.width):
                if mask_array[y, x]:
                    overlay.putpixel((x, y), color)
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay).convert("RGB")
        # Write class name at the top-left of the mask
        draw = ImageDraw.Draw(img)
        font_size = 16
        draw.text(
            (left, top - font_size if top - font_size > 0 else top + 2),
            names[i],
            fill=color[:3],
            font=ImageFont.load_default(),
        )

    with open(image_with_labels_path, "w", encoding="utf-8") as f:
        Image.Image.save(img, f, format="JPEG")


def save_mask(img, masks, names, labels_path):
    # img_width, img_height = img.size
    # for i in range(len(masks)):
    #     masks

    # with open(labels_path, 'w', encoding='utf-8') as f:
    #     f.write(label_file_content)
    # TODO
    pass
