import os
import time

from compress import compress_file, decompress_file

file_names = [
    "bensound-buddy.mp3",
    "bensound-buddy.wav",
    "bensound-sunny.mp3",
    "bensound-sunny.wav",
    "Homer-Iliad.txt",
    "JulesVerne-MysteriousIsland.txt",
    "julia_set.bmp",
    "mandelbrot_set.bmp",
    "parrot.bmp",
    "parrot.jpg"
]

files_dir = "files"

total_start_time = time.time()

for file in file_names:
    in_file = os.path.join(files_dir, file)
    comp_file = in_file + ".huf"       # Compressed file
    decomp_file = comp_file + ".orig"  # Decompressed file

    print(f"Processing: {in_file}")

    start_time = time.time()
    compress_file(in_file, comp_file)
    comp_time = time.time() - start_time
    print(f"Compression Time: {comp_time:.4f} seconds")

    start_time = time.time()
    decompress_file(comp_file, decomp_file)
    decomp_time = time.time() - start_time
    print(f"Decompression Time: {decomp_time:.4f} seconds")

    orig_size = os.path.getsize(in_file)
    comp_size = os.path.getsize(comp_file)
    decomp_size = os.path.getsize(decomp_file)

    print(f"Original Size: {orig_size} bytes")
    print(f"Compressed Size: {comp_size} bytes")
    print(f"Decompressed Size: {decomp_size} bytes")

    if orig_size > 0:
        compression_ratio = (comp_size / orig_size) * 100
        print(f"Compression Ratio: {compression_ratio:.2f}%")
    else:
        print("Compression Ratio: N/A (zero-size original file)")

    # Use cmp (via os.system) to compare the original and decompressed files.
    cmp_result = os.system(f"cmp --silent '{in_file}' '{decomp_file}'")
    if cmp_result == 0:
        print(f"✅ SUCCESS: {file} decompressed correctly.\n")
    else:
        print(f"❌ FAILURE: {file} differs after decompression.\n")

total_time = time.time() - total_start_time
print(f"Total Execution Time: {total_time:.4f} seconds")
