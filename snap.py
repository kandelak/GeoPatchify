import os
import argparse
import geopandas as gpd
import rasterio
import rasterio.transform
import rasterio.features
import rasterio.shutil
from tqdm import tqdm

# Snaps the geometry to the grid of a raster and saves the paired image and mask
class Snap:
    def __init__(self, tif_folder, geojson_folder, output_image_folder, output_mask_folder):
        self.tif_folder = tif_folder
        self.geojson_folder = geojson_folder
        self.output_image_folder = output_image_folder
        self.output_mask_folder = output_mask_folder

    def get_image_paths(self):
        tif_files = [f for f in os.listdir(self.tif_folder) if f.endswith('.tif')]
        return [os.path.join(self.tif_folder, f) for f in tif_files]

    def get_geojson_paths(self):
        geojson_files = [f for f in os.listdir(self.geojson_folder) if f.endswith('.geojson')]
        return [os.path.join(self.geojson_folder, f) for f in geojson_files]

    def unify_crs(self, mask, image):
        mask = mask.to_crs(image.crs)
        return mask
    
    def create_mask_geometry(self, mask):
        return gpd.GeoSeries(data=mask["geometry"], crs=mask.crs)
    
    def paired_name(self, image_path):
        return image_path.split(os.sep)[-1].split(".")[0]
    
    def save(self, image, mask_geom, paired_name):
        rasterio.shutil.copy(image, os.path.join(self.output_image_folder,paired_name), driver='GTiff')
        mask_geom.to_file(os.path.join(self.output_mask_folder,paired_name), driver='GeoJSON')

    
    def process(self):
        
        image_paths = self.get_image_paths()
        geojson_paths = self.get_geojson_paths()

        os.makedirs(self.output_image_folder, exist_ok=False)
        os.makedirs(self.output_mask_folder, exist_ok=False)

        for image_path in tqdm(image_paths, desc="Processing images", position=0):

            with rasterio.open(image_path) as src:
                for geojson_path in tqdm(geojson_paths, desc="Generating masks and pairing them with their respective images", leave=False, position=1):
                    
                    bbox = src.bounds
                    mask = gpd.read_file(geojson_path, bbox=bbox)

                    if mask.empty:
                        continue
                    
                    mask = self.unify_crs(mask, src)

                    mask_geom = self.create_mask_geometry(mask)
        
                    self.save(image=src, mask_geom=mask_geom, paired_name=self.paired_name(image_path))


def main():
    parser = argparse.ArgumentParser(description="Generate masks from images based on GeoJSON polygons.")
    parser.add_argument("--tif-folder", help="Path to the folder containing TIFF images.")
    parser.add_argument("--geojson-folder", help="Path to the folder containing GeoJSON files.")
    parser.add_argument("--output-image-folder", default= os.path.join("../temp","images_preprocessed_non_split"),help="Path to the output folder for processed images.")
    parser.add_argument("--output-mask-folder", default= os.path.join("../temp","masks_preprocessed_non_split"),help="Path to the output folder for generated masks.")
    args = parser.parse_args()

    masker = Snap(args.tif_folder, args.geojson_folder, args.output_image_folder, args.output_mask_folder)
    masker.process()

if __name__ == "__main__":
    main()
