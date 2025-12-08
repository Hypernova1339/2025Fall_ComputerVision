import os
import argparse
import xml.etree.ElementTree as ET

# Edit this if your XML <name> tags are different
CLASSES = ["JINX_W_PROJECTILE", "JINX_W_IMPACT"]


def convert_single_xml(xml_path: str, txt_path: str) -> None:
    """
    Convert a single PascalVOC XML annotation file to YOLO txt format.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise RuntimeError(f"No <size> tag found in {xml_path}")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASSES:
            # Skip labels not in our class list
            print(f"[WARN] Skipping unknown class '{cls_name}' in {xml_path}")
            continue
        cls_id = CLASSES.index(cls_name)

        bbox = obj.find("bndbox")
        x_min = float(bbox.find("xmin").text)
        y_min = float(bbox.find("ymin").text)
        x_max = float(bbox.find("xmax").text)
        y_max = float(bbox.find("ymax").text)

        # Convert to YOLO normalized format
        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        box_w = (x_max - x_min) / img_w
        box_h = (y_max - y_min) / img_h

        yolo_lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    # Write out YOLO txt file (even if no objects → empty file is okay)
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")


def convert_folder(xml_dir: str, out_dir: str) -> None:
    """
    Convert all XML files in xml_dir to YOLO txt files in out_dir.
    The txt filenames will match the XML basenames.
    """
    os.makedirs(out_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(xml_dir) if f.lower().endswith(".xml")]
    if not xml_files:
        print(f"[WARN] No .xml files found in {xml_dir}")
        return

    print(f"[INFO] Found {len(xml_files)} XML files in {xml_dir}")
    for fname in xml_files:
        xml_path = os.path.join(xml_dir, fname)
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(out_dir, base + ".txt")

        convert_single_xml(xml_path, txt_path)
        print(f"[OK] {xml_path} -> {txt_path}")

    print(f"[DONE] Converted {len(xml_files)} XML files to YOLO format in {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PascalVOC XML annotations to YOLO txt format."
    )
    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Directory containing .xml files (and corresponding images).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where YOLO .txt label files will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_folder(args.xml_dir, args.out_dir)
