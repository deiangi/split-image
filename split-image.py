# Copyright (c) 2024, Deian Gi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import math
import os
from PIL import Image, ImageDraw
import numpy as np
import zipfile


def draw_row_line(image, row_index, color):
    """Draws a horizontal line across the specified row index."""
    draw = ImageDraw.Draw(image)
    draw.line((0, row_index, image.width, row_index), fill=color, width=1)

def draw_column_line(image, column_index, color):
    """Draws a vertical line down the specified column index."""
    draw = ImageDraw.Draw(image)
    draw.line((column_index, 0, column_index, image.height), fill=color, width=1)

def draw_dotted_line(image, index, color):
    """Draws a vertical dotted line down the specified column index."""
    draw = ImageDraw.Draw(image)
    for y in range(0, image.height, 10):  # Adjust the step for dot spacing
        if y % 20 < 10:  # Create gaps for the dotted effect
            draw.point((index, y), fill=color)


def scan_and_draw_lines(image_path, save_image=False, square_size=None):
     # List to store the rectangles (squares)
    squares = []
    
    # Open the image
    image = Image.open(image_path)
    image_array = np.array(image)

    start_row_indices = []
    end_row_indices = []
    start_col_indices = []
    end_col_indices = []
    max_square_len = 0

    # Scan rows
    previous_dist_size = -1
    for row_index, row in enumerate(image_array):
        unique_colors = np.unique(row.reshape(-1, row.shape[-1]), axis=0)
        current_dist_size = len(unique_colors)

        # Check for start line condition
        if current_dist_size > 1 and previous_dist_size == 1:
            draw_row_line(image, row_index - 1, 'red')
            start_row_indices.append(row_index - 1)
            previous_dist_size = current_dist_size
        # Check for stop line condition
        elif current_dist_size == 1 and previous_dist_size > 1:
            draw_row_line(image, row_index, 'blue')
            end_row_indices.append(row_index)
            previous_dist_size = current_dist_size
        elif current_dist_size == 1 and previous_dist_size == -1:
            previous_dist_size = current_dist_size


    # Reset previous_dist_size for column scan
    previous_dist_size = -1
    # Scan columns
    start_index = 0
    for column_index in range(image_array.shape[1]):
        column = image_array[:, column_index]
        unique_colors = np.unique(column.reshape(-1, column.shape[-1]), axis=0)
        current_dist_size = len(unique_colors)

        # Check for start line condition
        if current_dist_size > 1 and previous_dist_size == 1:
            draw_column_line(image, column_index - 1, 'red')
            start_col_indices.append(column_index - 1)
            start_index = column_index
            previous_dist_size = current_dist_size

        # Check for stop line condition
        elif current_dist_size == 1 and previous_dist_size > 1:
            draw_column_line(image, column_index, 'blue')
            end_col_indices.append(column_index)
            center_index = (column_index + start_index) // 2
            draw_column_line(image, center_index, 'green')
            previous_dist_size = current_dist_size
        elif current_dist_size == 1 and previous_dist_size == -1:
            previous_dist_size = current_dist_size

    # Calculate maxSquareLen using both rows and columns
    max_square_len = 0
    if square_size is None:
        for start_row, end_row in zip(start_row_indices, end_row_indices):
            row_length = end_row - start_row
            max_square_len = max(max_square_len, row_length)

        for start_col, end_col in zip(start_col_indices, end_col_indices):
            col_length = end_col - start_col
            max_square_len = max(max_square_len, col_length)
        
        print(f"Detected image size: {max_square_len}x{max_square_len}")
    else:
        max_square_len = square_size
        print(f"User-defined image size: {max_square_len}x{max_square_len}")

    # Draw squares around each pictogram
    for bottom_row in end_row_indices:
        for (start_col, end_col) in zip(start_col_indices, end_col_indices):
            half_len = max_square_len // 2
            top_row = bottom_row - max_square_len

            # Calculate the horizontal center
            center_col = (start_col + end_col) // 2

            # Calculate left and right, ensuring they are within the image boundaries
            left_col = max(center_col - half_len, 0)
            right_col = min(center_col + half_len, image.width)

            # Draw the square
            draw = ImageDraw.Draw(image)
            draw.rectangle([(left_col, top_row), (right_col, bottom_row)], outline='darkgreen', width=2)
                # Add square coordinates to the list
            squares.append((left_col, top_row, right_col, bottom_row))

    # Save the image with the new file name
    if save_image:
        file_base, file_ext = os.path.splitext(os.path.basename(image_path))
        new_file_name = f"{file_base}_detected{file_ext}"
        new_file_path = os.path.join(os.path.dirname(image_path), new_file_name)
        image.save(new_file_path)
        print(f"Image with detection results saved as: {new_file_path}")


    return (squares, max_square_len)


def split_image(input_file, squares, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the input image
    image = Image.open(input_file)

    # Split the image based on the squares and save each part
    split_image_paths = []
    for i, (left, top, right, bottom) in enumerate(squares):
        # Crop the image based on square coordinates
        split = image.crop((left, top, right, bottom))
        
        # Define the output path with the file name prefix
        output_path = os.path.join(output_folder, f"{os.path.basename(output_folder)}_{i}.png")
        
        # Save the split image
        split.save(output_path)
        
        # Add the path to the list
        split_image_paths.append(output_path)

    return split_image_paths

def create_zip_from_splits(split_image_paths, output_folder):
    # Create a ZIP file to contain the split images, named after the output folder
    zip_filename = f"{output_folder}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for path in split_image_paths:
            zipf.write(path, arcname=os.path.basename(path))
    return zip_filename


def flood_fill_transparent(image_path):
    # Open the image
    img_before = Image.open(image_path)
    img_after = img_before.copy()

    # Define the top-left seed point (you can adjust this)
    start_point = (0, 0)

    # Define the transparent color (RGBA format: R, G, B, A)
    transparent_color = (0, 0, 0, 0)  # Fully transparent

    # Apply flood fill
    img_draw = ImageDraw.floodfill(img_after, start_point, transparent_color, thresh=50, tolerance=20)

    # Show the original and modified images (you can save the modified image if needed)
    img_before.show()
    img_after.show()


def create_transparent_image(image_path, target_color=(0, 0, 0), tolerance=10):
    # Open the image
    img = Image.open(image_path)
    img_rgba = img.convert("RGBA")

    # Get pixel data
    pixels = img_rgba.load()

    # Create a new image with transparency
    transparent_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
    transparent_pixels = transparent_img.load()

    # Iterate through each pixel
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = pixels[x, y]
            m = min(r,g,b)
            newA=math.floor((255-m)*(2.6))
            transparent_pixels[x, y] = (r, g, b, newA)

            # if m <= tolerance:
            # # if abs(r - target_color[0]) <= tolerance and \
            # #    abs(g - target_color[1]) <= tolerance and \
            # #    abs(b - target_color[2]) <= tolerance:
            #     # Replace target color with transparency

            #     newA=(255-m)//255
            #     transparent_pixels[x, y] = (newA, g, b, 1)
            # else:
            #     transparent_pixels[x, y] = (r, g, b, a)

    return transparent_img

def process_images_with_transparency(image_paths, transparent_output_folder, tolerance=0):
    saved_names = []  # Initialize an empty list to store saved image names
    for image_path in image_paths:

        # The following function should be defined somewhere in your script
        # It will process the image and return an image with transparency applied
        transparent_img = create_transparent_image(image_path, target_color=(0, 0, 0), tolerance=tolerance)

        # Construct the output file name and path
        img_name = os.path.basename( image_path)  # Get the base name of the image file
        output_file_name = f"{os.path.splitext(img_name)[0]}_transparent.png"
        output_path = os.path.join(transparent_output_folder, output_file_name)

        # Save the transparent image to the specified output folder
        transparent_img.save(output_path)
        saved_names.append(output_path)  # Add the saved image path to the list
        # print(f"Saved transparent image as {output_path}")

    return saved_names  # Return the list of saved image paths



def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Split an image into multiple parts and zip the results.")
    parser.add_argument("filename", type=str, help="The file name of the image to split.")
    parser.add_argument("-out", "--output", type=str, help="Output folder for the split images and zip file. Default is the base file name.")
    parser.add_argument("-d", "--detect", action='store_true', help="Detect the images automatically. Only Detection!")
    parser.add_argument("-t", "--transparent", type=int, default=None, help="Make the pictures transparent. Set the tolerance level.")
    parser.add_argument("-s", "--size", type=int, default=None, help="Define image size. If not provided will be automatically detected.")

    # Parse the arguments
    args = parser.parse_args()
    print(f"Split image map into separate image(s) with automatic image size.")
    print(f"Copyright (c) by Deian Gi, 2024")


    # If output directory is not specified, use the base file name without extension
    base_file_name = os.path.splitext(os.path.basename(args.filename))[0]
    if not args.output:
        args.output = os.path.join(os.getcwd(), base_file_name)

    # Create directories for normal and transparent images
    normal_output_folder = f"{args.output}"
    os.makedirs(normal_output_folder, exist_ok=True)

    # Detect squares and draw lines if required
    squares, max_square_len = scan_and_draw_lines(args.filename, args.detect, args.size)    
    if len(squares) == 0:
        print(f"No images detected. Abort.")
        exit()

    # Split images and save the normal ones
    normal_split_paths = split_image(args.filename, squares, normal_output_folder)
    # Create a zip file from the split normal images
    normal_zip_path = create_zip_from_splits(normal_split_paths, normal_output_folder)
    print(f"Split into {len(squares)} image(s) with size [{max_square_len}x{max_square_len}] saved to '{normal_output_folder}' and zipped in '{normal_zip_path}'.")

    # Process transparency if required
    if args.transparent is not None:
        transparent_output_folder = f"{args.output}_transparent"
        os.makedirs(transparent_output_folder, exist_ok=True)
        
        # Make the split images transparent and save to a different folder
        transparent_split_paths = process_images_with_transparency(normal_split_paths, transparent_output_folder, tolerance=args.transparent)
        # Create a zip file from the split transparent images
        transparent_zip_path = create_zip_from_splits(transparent_split_paths, transparent_output_folder)
        print(f"Split into {len(squares)} transparent image(s) with size [{max_square_len}x{max_square_len}] saved to '{transparent_output_folder}' and zipped in '{transparent_zip_path}'.")


if __name__ == "__main__":
    main()
