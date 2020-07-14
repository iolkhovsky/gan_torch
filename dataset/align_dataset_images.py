import cv2
import argparse
from glob import glob
from os.path import join, isdir
from os import mkdir
from shutil import rmtree
from tqdm import tqdm
import ntpath

from dataset.face_aligner import FaceAligner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="/home/igor/datasets/faces6k/data/images",
                        help="Input data folder")
    parser.add_argument("--output_folder", type=str, default="/home/igor/datasets/faces6k/aligned",
                        help="Output folder")
    parser.add_argument("--target_image_size", type=int, default=100,
                        help="Size (both w and h) of output aligned images [pix]")
    parser.add_argument("--eye_level", type=float, default=0.3,
                        help="Hight of eye level (from top), normalized to frame's height")
    parser.add_argument("--eye_dist", type=float, default=0.5,
                        help="Distance between eye, normalized to frame's width")
    args = parser.parse_args()

    if isdir(args.output_folder):
        rmtree(args.output_folder)
    mkdir(args.output_folder)

    img_paths = glob(join(args.input_folder, "*.jpg"))
    total_imgs = len(img_paths)
    img_idx = 0
    cnt = 0
    with tqdm(total=total_imgs, desc=f'Image {img_idx + 1}/{total_imgs}', unit='image') as pbar:
        aligner = FaceAligner(target_size=args.target_image_size, eye_level=args.eye_level, eye_dist=args.eye_dist)
        for img_idx, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            aligned_img = aligner(img)
            if aligned_img is not None:
                cv2.imwrite(join(args.output_folder, ntpath.basename(img_path)), aligned_img)
                cnt += 1
            pbar.update(1)
    print("Total images saved:", cnt, "skipped:", total_imgs - cnt)
    return


if __name__ == "__main__":
    main()
