

import os

import numpy as np
import rasterio as rio

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pyWAPOR.Collect.Landsat import PreprocessLandsat as pre
import json
from senet_input_creator import gdal_utils as gu
import gdal
import pyWAPOR
import pandas as pd

def PreprocessLandsatTSEB(landsat_dir, output_dir,
                          delete_input=False, overwrite=False):

    if isinstance(landsat_dir, str):
        landsat_dir = Path(landsat_dir)

    # unpack the *.tar.gz Landsat files
    L7_files = list((landsat_dir / Path('L7')).glob('*.tar'))
    L8_files = list((landsat_dir / Path('L8')).glob('*.tar'))

    print('Unpacking *.tar files...')
    for file in tqdm(L7_files + L8_files):
        pre._unpack_and_save(file, delete_input=delete_input)

    # merge the individual landsat bands into multiband files
    L7_dirs = [directory for directory in list((landsat_dir / Path('L7')).glob('*'))
               if os.path.isdir(directory)]
    L8_dirs = [directory for directory in list((landsat_dir / Path('L8')).glob('*'))
               if os.path.isdir(directory)]

    filename_list = L7_dirs + L8_dirs

    # calculate NDVI/ALBEDO and save
    _process_and_save(filename_list, output_dir, overwrite=overwrite)


def _process_and_save(filename_list, output_folder, overwrite=False):

    if isinstance(output_folder, str):
        output_folder = Path(output_folder)

    print('Calculating NDVI/ALBEDO...')
    for scene in tqdm(sorted(filename_list)):

        sensor = str(scene.stem.split('_')[0])
        date = datetime.strptime(str(scene.stem).split('_')[3], '%Y%m%d')
        datestring = date.strftime("%Y%m%d")

        if not (output_folder / Path(datestring)).is_dir():
            os.makedirs(output_folder / Path(datestring))

        ndvi_filename = output_folder / Path(datestring) / Path('NDVI_' + datestring + '.tif')
        albedo_filename = output_folder / Path(datestring) / Path('ALBEDO_' + datestring + '.tif')
        lst_filename = output_folder / Path(datestring) / Path('LST_' + datestring + '.tif')

        if ndvi_filename.exists() and albedo_filename.exists() \
                and lst_filename.exists() and not overwrite:
            continue

        qa_filename = str(scene / f"{scene.stem}_QA_PIXEL.TIF")
        with rio.open(qa_filename) as src:
            meta = src.profile

        meta.update({'dtype': 'float64', 'nodata': np.nan, 'count': 1})

        qa = gu.raster_data(qa_filename)
        mask = pre._landsat_cloudmask(qa)

        # calculate NDVI and Albedo
        ndvi = _calc_ndvi(scene, sensor)
        ndvi[mask] = np.nan
        with rio.open(str(ndvi_filename), 'w', **meta) as dst:
            dst.write(ndvi, 1)

        output = _calc_albedo(scene, sensor)
        output[mask] = np.nan
        with rio.open(str(albedo_filename), 'w', **meta) as dst:
             dst.write(output, 1)

        output = _calc_lst(scene, sensor)
        output[mask] = np.nan
        with rio.open(str(lst_filename), 'w', **meta) as dst:
            dst.write(output, 1)

        out_filename = output_folder / Path(datestring) / Path('Time_' + datestring + '.tif')

        mtl_json_filename = Path(scene).parent / Path(scene).stem / Path(
            Path(scene).stem + "_MTL.json")
        output = _get_landsat_time(mtl_json_filename)
        output = np.full(ndvi.shape, output)
        output[mask] = np.nan
        with rio.open(str(out_filename), 'w', **meta) as dst:
            dst.write(output, 1)

def _get_landsat_time(mtl_json_filename):
    with open(mtl_json_filename) as f:
        data = json.load(f)

    time = data["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"].split(":")
    time = float(time[0]) + float(time[1]) / 60.
    return time


def _calc_ndvi(scene, sensor):
    if sensor == 'LE07':
        bands = ['SR_B3', 'SR_B4']

    elif sensor == 'LC08':
        bands = ['SR_B4', 'SR_B5']

    input_file = str(scene / f"{scene.stem}_{bands[0]}.TIF")
    red = gu.raster_data(input_file).astype(np.float)
    input_file = str(scene / f"{scene.stem}_{bands[1]}.TIF")
    nir = gu.raster_data(input_file).astype(np.float)
    ndvi = np.full(red.shape, np.nan)
    valid = np.logical_and(red > 0, nir > 0)

    ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])

    # remove too large or too small values
    ndvi[ndvi > 1] = np.nan
    ndvi[ndvi < -1] = np.nan

    return ndvi


def _calc_albedo(scene, sensor):
    albedo_Mp = 2.75e-5  # multiplicative scaling factor for Collection 2
    albedo_Ap = -0.2  # additive scaling factor for Collection 2

    # ESUN values: [Blue, Green, Red, NIR, SWIR-1, SWIR-2]
    if sensor == 'LE07':
        ESUN_values = np.array([1970, 1842, 1547, 1044, 225.7, 82.06])
        bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']

    elif sensor == 'LC08':
        ESUN_values = np.array([1991, 1812, 1549, 972.6, 214.7, 80.7])
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']

    data = []
    for band in bands:
        input_filename = str(scene / f"{scene.stem}_{band}.TIF")
        data.append(gu.raster_data(str(input_filename)).astype(np.float))

    data = np.asarray(data)

    BGRNS = albedo_Mp * data + albedo_Ap

    albedo = np.sum(BGRNS * np.expand_dims(ESUN_values, (1, 2)), axis=0) / np.sum(ESUN_values)

    # remove too large or too small values
    albedo[albedo > 1] = np.nan
    albedo[albedo < 0] = np.nan

    return albedo

def _calc_lst(scene, sensor):
    if sensor == 'LE07':
        band = 'ST_B6'

    elif sensor == 'LC08':
        band = 'ST_B10'

    lst_Mp = 0.00341802  # multiplicative scaling factor for Collection 2
    lst_Ap = 149.0  # additive scaling factor for Collection 2
    input_filename = str(scene / f"{scene.stem}_{band}.TIF")
    lst = gu.raster_data(input_filename).astype(np.float)

    lst = np.where(lst > 0, lst_Mp * lst + lst_Ap, np.nan)

    return lst


def resample_level2_level3(output_dir, l2_dir,
                           products=["Lon", "Lat", "Slope", "Aspect", "Trans_24"],
                           overwrite=False):

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    scenes = sorted(output_dir.glob("*"))

    for scene in tqdm(scenes):
        datestr = scene.stem

        filename_template = str(scene / f"LST_{datestr}.tif")
        for product in products:
            if product in ["Slope", "Aspect", "DEM"]:
                l2_product = Path(l2_dir) / Path(datestr) / Path(f"{product}.tif")
                l3_product = Path(scene) / Path(f"{product}.tif")

            else:
                l2_product = Path(l2_dir) / Path(datestr) / Path(f"{product}_{datestr}.tif")
                l3_product = Path(scene) / Path(f"{product}_{datestr}.tif")
            if l3_product.exists() and not overwrite:
                continue
            # Get template projection, extent and resolution
            proj, gt, _, _, extent, _ = gu.raster_info(str(filename_template))

            # Resample with GDAL warp
            gdal.Warp(str(l3_product), str(l2_product), format="GTiff", dstSRS=proj,
                      xRes=gt[1], yRes=gt[5], outputBounds=extent,
                      resample_alg="bilinear")


def prepare_level3_meteo(output_dir, raw_dir, latlim, lonlim, overwrite=False,
                         username="", password=""):
    def lapse_rate_temp(tair_file, dem_file):
        destT_down = gu.resample_with_gdalwarp(str(tair_file),
                                               dem_file,
                                               resample_alg="bilinear")
        destDEM_up = gu.resample_with_gdalwarp(str(tair_file),
                                               dem_file,
                                               resample_alg="average")
        destDEM_down = gdal.Open(dem_file)
        destDEM_up_down = gu.resample_with_gdalwarp(destDEM_up,
                                                    dem_file,
                                                    resample_alg="bilinear")

        # Open Arrays
        T = destT_down.GetRasterBand(1).ReadAsArray()
        T[T==0] = np.nan
        DEM_down = destDEM_down.GetRasterBand(1).ReadAsArray()
        DEM_up_ave = destDEM_up_down.GetRasterBand(1).ReadAsArray()

        # correct wrong values
        DEM_down[DEM_down <= 0] = 0
        DEM_up_ave[DEM_up_ave <= 0] = 0

        Tdown = pyWAPOR.ETLook.meteo.disaggregate_air_temperature(T, DEM_down,
                                                                  DEM_up_ave)

        return Tdown

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if isinstance(raw_dir, str):
        raw_dir = Path(raw_dir)

    scenes = sorted(output_dir.glob("*"))

    for scene in tqdm(scenes):
        dem_file = str(scene / Path("DEM.tif"))
        date = datetime.strptime(scene.stem, "%Y%m%d")
        time_file = str(scene / Path("Time_%s.tif"%scene.stem))
        proj_ex, geo_ex = gu.raster_info(time_file)[:2]
        dtime = gu.raster_data(time_file)
        dtime = np.nanmean(dtime)
        if np.isnan(dtime):
            dtime = 12

        NowTime = datetime(date.year, date.month, date.day,
                                    int(np.floor(dtime)),
                                    int((dtime - np.floor(dtime)) * 60))

        # Define the startdates for the METEO
        StartTime = datetime(date.year, date.month, date.day, 0, 0)
        EndTime = datetime(date.year, date.month, date.day, 23, 59)

        if (date >= datetime(2016, 1, 1) and date < datetime(2017, 12, 1)):
            # find nearest Meteo time
            DateTime = pd.date_range(StartTime, EndTime,
                                     freq="H") + pd.offsets.Minute(30)
        else:
            # find nearest Meteo time
            DateTime = pd.date_range(StartTime, EndTime,
                                     freq="3H") + pd.offsets.Minute(90)

        Time_nearest = min(DateTime,
                           key=lambda DateTime: abs(DateTime - NowTime))
        Period = np.argwhere(DateTime == Time_nearest)[0][0] + 1

        # Get METEO data
        if date < datetime(2016, 1, 1):
            pyWAPOR.Collect.MERRA.three_hourly(raw_dir,
                                               ['t2m', 'u2m', 'v2m', 'q2m', 'ps'],
                                               StartTime, EndTime,
                                               latlim, lonlim,
                                               [int(Period)])
            str_meteo = "MERRA"
            inst_name = "three_hourly"
            hour_steps = 3
            file_time_inst = "3-hourly"

        elif (date >= datetime(2016, 1, 1) and date < datetime(2017, 12, 1)):
            pyWAPOR.Collect.MERRA.hourly_MERRA2(raw_dir,
                                                ['t2m', 'u2m', 'v2m', 'q2m', 'ps'],
                                                StartTime, EndTime,
                                                latlim, lonlim,
                                                [int(Period)],
                                                username, password)

            str_meteo = "MERRA"
            inst_name = "hourly_MERRA2"
            hour_steps = 1
            file_time_inst = "hourly"

        else:
            pyWAPOR.Collect.GEOS.three_hourly(raw_dir,
                                              ['t2m', 'u2m', 'v2m', 'qv2m', 'ps'],
                                              StartTime, EndTime,
                                              latlim, lonlim,
                                              [int(Period)])
            str_meteo = "GEOS"
            inst_name = "three_hourly"
            hour_steps = 3
            file_time_inst = "3-hourly"

        # Air pressure
        pair_inst_file = scene / Path("Pair_inst_%s.tif" %scene.stem)

        if not pair_inst_file.exists() or overwrite:
            folder_raw_file_inst = raw_dir / str_meteo / "Surface_Pressure" / inst_name

            HourPeriod = hour_steps * (Period - 1)

            filename_inst = "ps_%s_kpa_%s_%d.%02d.%02d_H%02d.M00.tif" % (
                str_meteo, file_time_inst, date.year, date.month, date.day, HourPeriod)
            inst_input = folder_raw_file_inst / filename_inst
            if inst_input.exists():
                destPairInst = gu.resample_with_gdalwarp(str(inst_input),
                                                         time_file,
                                                         resample_alg="bilinear")
                Pair_inst = destPairInst.GetRasterBand(1).ReadAsArray()
                gu.save_image(Pair_inst, str(pair_inst_file), geo_ex, proj_ex)

            else:
                print("Pair instanteneous is not available")

        # Specific Humidity
        qv_inst_file = scene / Path("qv_inst_%s.tif"%scene.stem)

        if not qv_inst_file.exists() or overwrite:
            folder_raw_file_inst = raw_dir / str_meteo / "Specific_Humidity" / inst_name

            HourPeriod = hour_steps * (Period - 1)
            if str_meteo == "MERRA":
                para = "q2m"
            else:
                para = "qv2m"

            filename_inst = "%s_%s_kg-kg-1_%s_%d.%02d.%02d_H%02d.M00.tif" % (
                para, str_meteo, file_time_inst, date.year, date.month, date.day,
                HourPeriod)
            inst_input = folder_raw_file_inst / filename_inst
            if inst_input.exists():
                destqvInst = gu.resample_with_gdalwarp(str(inst_input),
                                                       time_file,
                                                       resample_alg="bilinear")
                qv_inst = destqvInst.GetRasterBand(1).ReadAsArray()
                gu.save_image(qv_inst, str(qv_inst_file), geo_ex, proj_ex)
            else:
                print("qv instanteneous is not available")


        # Air temperature
        Tair_inst_file = scene / Path("tair_inst_%s.tif"%scene.stem)

        if not Tair_inst_file.exists() or overwrite:
            folder_raw_file_inst = raw_dir / str_meteo / "Air_Temperature" / inst_name

            HourPeriod = hour_steps * (Period - 1)

            filename_inst = "t2m_%s_K_%s_%d.%02d.%02d_H%02d.M00.tif" % (
                str_meteo, file_time_inst, date.year, date.month, date.day, HourPeriod)
            inst_input = folder_raw_file_inst / filename_inst
            if inst_input.exists():
                tair_inst = lapse_rate_temp(str(inst_input), dem_file)
                gu.save_image(tair_inst, str(Tair_inst_file), geo_ex, proj_ex)

            else:
                print("Tair instanteneous is not available")


        # Wind Speed
        wind_inst_file = scene / Path("wind_inst_%s.tif" %scene.stem)

        if not wind_inst_file.exists() or overwrite:
            folder_u2_inst = raw_dir / str_meteo / "Eastward_Wind" / inst_name

            folder_v2_inst = raw_dir / str_meteo / "Northward_Wind" / inst_name

            HourPeriod = hour_steps * (Period - 1)

            filename_u2_inst = "u2m_%s_m-s-1_%s_%d.%02d.%02d_H%02d.M00.tif" % (
                str_meteo, file_time_inst, date.year, date.month, date.day, HourPeriod)
            filename_v2_inst = "v2m_%s_m-s-1_%s_%d.%02d.%02d_H%02d.M00.tif" % (
                str_meteo, file_time_inst, date.year, date.month, date.day, HourPeriod)
            input_u2_inst = folder_u2_inst / filename_u2_inst
            input_v2_inst = folder_v2_inst / filename_v2_inst
            if input_u2_inst.exists() and input_v2_inst.exists():
                destu2inst = gu.resample_with_gdalwarp(str(input_u2_inst),
                                                       time_file,
                                                       resample_alg="bilinear")
                destv2inst = gu.resample_with_gdalwarp(str(input_v2_inst),
                                                       time_file,
                                                       resample_alg="bilinear")
                u2_inst = destu2inst.GetRasterBand(1).ReadAsArray()
                v2_inst = destv2inst.GetRasterBand(1).ReadAsArray()
                wind_inst = np.sqrt(u2_inst ** 2 + v2_inst ** 2)
                gu.save_image(wind_inst, str(wind_inst_file), geo_ex, proj_ex)
            else:
                print("Wind instanteneous is not available")


if __name__ == "__main__":
    tiles = ["174036", "174037"]
    landsat_raw_basedir = Path("/mnt/lvstorage/landsat/")
    output_base_dir = Path("/home/hector/ET4FAO/inputs/ETLook_input/level_3")
    l2_dir = Path("/home/hector/ET4FAO/inputs/ETLook_input/level_2")
    l2_products = ["Lon", "Lat", "Slope", "Aspect", "DEM", "Trans_24"]
    wapor_raw_dir = Path("/home/hector/ET4FAO/inputs/RAW")
    latitude_extent = [33.05, 34.70]
    longitude_extent = [35.10, 36.63]

    for tile in tiles:
        landsat_dir = landsat_raw_basedir / tile
        output_dir = output_base_dir / tile
        raw_dir = wapor_raw_dir / tile
        PreprocessLandsatTSEB(landsat_dir, output_dir, delete_input=False, overwrite=False)
        resample_level2_level3(output_dir, l2_dir, products=l2_products, overwrite=False)
        prepare_level3_meteo(output_dir, raw_dir, latitude_extent, longitude_extent, overwrite=False)

