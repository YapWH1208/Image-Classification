from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files and subdirectories in the input folder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)

            # Check if the file is an image
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                # Ensure the output subdirectory exists
                output_subdir = os.path.dirname(output_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Open the image
                with Image.open(input_path) as img:
                    # Resize the image
                    resized_img = img.resize(new_size)

                    # Save the resized image to the output folder
                    resized_img.save(output_path)

if __name__ == "__main__":
    # Specify the input and output folders and the new size
    input_folder = "../data/"
    output_folder = "../data/resized/"
    new_size = (224,224)  # Specify the desired width and height

    # Resize images
    resize_images(input_folder, output_folder, new_size)

    print("Images resized successfully.")
