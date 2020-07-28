import cv2
import argparse
from glob import glob
from os.path import join, isdir
from os import mkdir
from shutil import rmtree
from tqdm import tqdm
import ntpath
import sys
import numpy as np

from dataset.face_aligner import FaceAligner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        default="/home/igor/datasets/align_celeba/img_align_celeba/img_align_celeba",
                        help="Input data folder")
    parser.add_argument("--output_folder", type=str,
                        default="/home/igor/datasets/faces",
                        help="Output folder")
    parser.add_argument("--target_image_size", type=int, default=64,
                        help="Size (both w and h) of output aligned images [pix]")
    parser.add_argument("--eye_level", type=float, default=0.4,
                        help="Hight of eye level (from top), normalized to frame's height")
    parser.add_argument("--eye_dist", type=float, default=0.3,
                        help="Distance b"
                             "etween eye, normalized to frame's width")
    args = parser.parse_args()

    if isdir(args.output_folder):
        rmtree(args.output_folder)
    mkdir(args.output_folder)

    img_paths = glob(join(args.input_folder, "*.jpg"))
    total_imgs = len(img_paths)
    img_idx = 0
    cnt = 0
    r_acc = 0.
    r_sq_acc = 0.
    g_acc = 0.
    g_sq_acc = 0.
    b_acc = 0.
    b_sq_acc = 0.
    total_pixels = 0
    with tqdm(total=total_imgs, desc=f'Image {img_idx + 1}/{total_imgs}', unit='image') as pbar:
        aligner = FaceAligner(target_size=args.target_image_size, eye_level=args.eye_level, eye_dist=args.eye_dist)
        for img_idx, img_path in enumerate(img_paths):
            try:
                img = cv2.imread(img_path)
                aligned_img = aligner(img)
                if aligned_img is not None:
                    cv2.imwrite(join(args.output_folder, ntpath.basename(img_path)), aligned_img)
                    cnt += 1
                    norm_img = np.divide(aligned_img, 255.0)
                    r_acc += norm_img[:, :, 2].sum()
                    r_sq_acc += np.power(norm_img[:, :, 2], 2).sum()
                    g_acc += norm_img[:, :, 1].sum()
                    g_sq_acc += np.power(norm_img[:, :, 1], 2).sum()
                    b_acc += norm_img[:, :, 0].sum()
                    b_sq_acc += np.power(norm_img[:, :, 0], 2).sum()
                    total_pixels += norm_img.shape[0] * norm_img.shape[1]
            except:
                print("Exception has been thrown, img idx:", img_idx, "path:", img_path)
                print("Exception:", sys.exc_info()[0])
            pbar.update(1)
    print("Total images saved:", cnt, "skipped:", total_imgs - cnt)
    r_mean, r_std = r_acc / total_pixels, np.sqrt(r_sq_acc / total_pixels - np.power(r_acc / total_pixels, 2))
    g_mean, g_std = g_acc / total_pixels, np.sqrt(g_sq_acc / total_pixels - np.power(g_acc / total_pixels, 2))
    b_mean, b_std = b_acc / total_pixels, np.sqrt(b_sq_acc / total_pixels - np.power(b_acc / total_pixels, 2))
    text = ""
    with open("dataset_stat.txt", "w") as f:
        text += "Mean (r/g/b): " + str(r_mean) + " " + str(g_mean) + " " + str(b_mean) + "\n"
        text += "Std (r/g/b): " + str(r_std) + " " + str(g_std) + " " + str(b_std) + "\n"
        text += "Total images: " + str(cnt) + "\n"
        text += "Src dataset path: " + args.input_folder + "\n"
        text += "Preprocessed dataset path: " + args.output_folder + "\n"
        text += "Output image size: " + str(args.target_image_size) + "\n"
        f.write(text)
    print(text)
    return


if __name__ == "__main__":
    main()
