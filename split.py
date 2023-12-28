import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import geopandas as gpd
import rasterio
import rasterio.transform
import rasterio.features
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from PIL import Image

# Avoid DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = 145500000

# Grouping Threshold
THRESHOLD_GROUPING = 0.9

class Split:

    def __init__(self, input_folder_images, input_folder_masks, output_folder_images, output_folder_masks, patch_size):
        self.input_folder_images_ = input_folder_images
        self.input_folder_masks_ = input_folder_masks
        self.output_folder_images_ = output_folder_images
        self.output_folder_masks_ = output_folder_masks
        self.patch_size_ = patch_size

    def load_mask(self, mask_path, crs):
        mask = gpd.read_file(mask_path)
        
        if mask.crs != crs:
            Warning("CRS of mask and image do not match. CRS of mask will be changed to match the image.")
        return gpd.GeoSeries(data=mask["geometry"], crs = crs)
    
    def load_image(self, image_path):
        return rasterio.open(image_path)
    
    def calc_distances(self, mask):
        return mask.geometry.apply(lambda x: mask.hausdorff_distance(x))

    def calc_tile_size(self, tif, patch_size):
        side_horizontal = tif.xy(0, 0)[0] - tif.xy(0, patch_size)[0]
        side_vertical = tif.xy(0, 0)[1] - tif.xy(patch_size, 0)[1]
        return max(side_horizontal, side_vertical)\
    
    def calc_groups(self, distances, threshold):
        groups = []
        grouped_indices = set() 
        for i, distance_row in distances.iterrows():
            if i not in grouped_indices:
                group = distance_row[distance_row <= threshold].index.tolist()
                groups.append(group)
                grouped_indices.update(group)
        return groups
    
    def unite_group_geometries(self, mask, groups):
        united = []
        for group in groups:
            united_group = MultiPolygon()
            for i in group:
                united_group = united_group.union(mask.geometry[i])
            united.append(united_group)
        return united
    
    def calc_centroids(self, mask, groups):
        united = self.unite_group_geometries(mask, groups)

        united = gpd.GeoSeries(united, crs=mask.crs)
        return united.centroid.to_crs(mask.crs)
    
    def calc_patch_pixel_bounds(self, centr_coord_x,centr_coord_y, width, height):
        
        left = max(0, centr_coord_x - self.patch_size_/2)
        right = min(width, centr_coord_x + self.patch_size_/2)
        
        if (left == 0):
            right = min(width, self.patch_size_) # Patch Size is always smaller than width but this makes code more clear imo
        if (right == width):
            left = max(0, width - self.patch_size_)

        top = max(0, centr_coord_y - self.patch_size_/2)
        bottom = min(height, centr_coord_y + self.patch_size_/2)

        if (top == 0):  
            bottom = min(height, self.patch_size_)
        if (bottom == height):
            top = max(0, height - self.patch_size_)

        return left, bottom, right, top
    
    def calc_spatial_coordinates_from_pixel_bounds(self, tif, patch_bounds):
        left, top = tif.xy(patch_bounds[3], patch_bounds[0])
        right, bottom = tif.xy(patch_bounds[1], patch_bounds[2])

        return left, bottom, right, top

    # (row, col) : input for tif.xy()
    def calc_spatial_coordinates_from_pixels(self, tif, patch_bounds):
        left, top = tif.xy(patch_bounds[3], patch_bounds[0])
        right, bottom = tif.xy(patch_bounds[1], patch_bounds[2])

        return left, bottom, right, top
    
    def calc_centroid_pixel_coordinates(self, image, x,y):
        return image.index(x, y)
    
    def calc_cropped_mask(self, unioned, bounds):
        w, h = self.patch_size_, self.patch_size_  

        minx,miny,maxx,maxy = bounds
        
        transform = rasterio.transform.from_bounds(minx,miny,maxx,maxy, w, h)

        mask = rasterio.features.geometry_mask([unioned], transform=transform, invert=False, out_shape=(h, w))

        return mask
    
    def crop_and_save(self, unioned, pixel_bounds, spatial_bounds, image_path, filename, identifier):
        left, lower, right, top = pixel_bounds
        box = left, top, right, lower

        with Image.open(image_path) as img:
            cropped = img.crop(box)
            cropped.save(os.path.join(self.output_folder_images_, f"{filename}_{identifier}.png"), "PNG")
        
        cropped_mask = self.calc_cropped_mask(unioned, spatial_bounds)
        mask_image = Image.fromarray(cropped_mask)
        mask_image.save(os.path.join(self.output_folder_masks_, f"{filename}_{identifier}.png"), "PNG")
     
    def process(self):
        
        for filename in tqdm(os.listdir(self.input_folder_masks_), desc='Splitting...', position = 0):
            identifier = 0
            mask_path = os.path.join(self.input_folder_masks_, filename)
            image_path = os.path.join(self.input_folder_images_, filename)

            os.makedirs(self.output_folder_masks_, exist_ok=True)
            os.makedirs(self.output_folder_images_, exist_ok=True)
            
            image = self.load_image(image_path = image_path)
            mask = self.load_mask(mask_path = mask_path, crs = image.crs)

            long_tile_size = self.calc_tile_size(image, self.patch_size_)
            distances = self.calc_distances(mask)

            threshold_group = THRESHOLD_GROUPING * long_tile_size
            groups = self.calc_groups(distances, threshold_group)

            centroids = self.calc_centroids(mask, groups)

            for centroid in tqdm(centroids, position=1, desc='Calculating patches and saving', leave=False):
                identifier += 1
                # row, col is returned by tif.index()
                # Calc centroid pixel coordinates
                centr_coord_y, centr_coord_x = image.index(centroid.x, centroid.y)
                patch_pixel_bounds = self.calc_patch_pixel_bounds(centr_coord_x, centr_coord_y, image.width, image.height)
                patch_spatial_bounds = self.calc_spatial_coordinates_from_pixel_bounds(image, patch_pixel_bounds)
                
                patched_tile = gpd.GeoSeries(data=Polygon.from_bounds(*patch_spatial_bounds), crs=image.crs)
                
                # Also Include other geometries withins the tile
                final_group = mask.intersection(patched_tile[0])
                unioned = final_group.unary_union

                 # extract bounds from the patched tile
                spatial_bounds = patched_tile.bounds.values[0]
                self.crop_and_save(unioned, patch_pixel_bounds, spatial_bounds, image_path, filename, identifier)
               
               
            


            
           
            




            



def main():
    parser = argparse.ArgumentParser(description='Split images and masks into patches.')
    parser.add_argument('--input-folder-images', default="temp/images_preprocessed_non_split",type=str, help='Path to the folder containing input images')
    parser.add_argument('--input-folder-masks', default="temp/masks_preprocessed_non_split",type=str, help='Path to the folder containing input images')
    parser.add_argument('--output-folder-masks', default="temp/masks_preprocessed_split", type=str, help='Path to the folder where patches will be saved')
    parser.add_argument('--output-folder-images', default="temp/images_preprocessed_split",type=str, help='Path to the folder where patches will be saved')
    parser.add_argument('--patch-size', type=int, help='Size of the patches.')
    # parser.add_argument('--overlay', type=int, help='Overlap between patches. (If step=patch_size, there is no overlap.)')
    # parser.add_argument('--empty', type=int, help='Overlap between patches. (If step=patch_size, there is no overlap.)')
    args = parser.parse_args()

    input_folder_images = args.input_folder_images
    input_folder_masks = args.input_folder_masks
    output_folder_images = args.output_folder_images
    output_folder_masks = args.output_folder_masks
    patch_size = args.patch_size

    split = Split(input_folder_images, input_folder_masks, output_folder_images, output_folder_masks, patch_size)

    split.process()

if __name__ == "__main__":
    main()
