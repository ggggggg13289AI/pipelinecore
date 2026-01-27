"""
@author: sean
"""

import os
import pathlib
from typing import Any

import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir

EXAMPLE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "SEG_20230210_160056_635_S3.dcm"
)
DCM_EXAMPLE = pydicom.dcmread(EXAMPLE_FILE)


def compute_orientation(init_axcodes, final_axcodes):
    """
    Calculate orientation transformation between initial and final axis codes.
    A thin wrapper around nibabel's ornt_transform function.

    Args:
        init_axcodes: Initial orientation codes (e.g., ('L', 'P', 'S'))
        final_axcodes: Target orientation codes (e.g., ('R', 'A', 'S'))

    Returns:
        tuple: Orientation transformation matrix, initial orientation, final orientation
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)
    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    Reorient a 3D array from one orientation to another.
    Source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation

    Args:
        data_array: 3D array to reorient
        init_axcodes: Initial orientation codes
        final_axcodes: Target orientation codes

    Returns:
        numpy.ndarray: Reoriented data array
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)

    # If orientations are already the same, return the original data
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    # Apply the orientation transformation
    return nib.orientations.apply_orientation(data_array, ornt_transf)


def get_array_to_dcm_axcodes(path_nii: pathlib.Path | str) -> np.ndarray:
    pred_nii = nib.load(path_nii)
    pred_data = np.array(pred_nii.dataobj)
    # Reorient prediction data to standard orientation
    pred_nii_obj_axcodes = tuple(nib.aff2axcodes(pred_nii.affine))
    return do_reorientation(
        pred_data,
        pred_nii_obj_axcodes,
        # ('I', 'P', 'L')
        ("S", "P", "L"),
    )


def get_dicom_seg_template(model: str, label_dict: dict) -> dict:
    """
    Create a DICOM-SEG template with segment attributes.

    Args:
        model: Series description or model name
        label_dict: Dictionary of label IDs and attributes

    Returns:
        Dict: DICOM-SEG template dictionary
    """
    unique_labels = list(label_dict.keys())
    segment_attributes = []

    # Create segment attributes for each label
    for idx in unique_labels:
        name = label_dict[idx]["SegmentLabel"]
        # Convert color to RGB values (0-255)
        rgb_rate = mcolors.to_rgb(label_dict[idx]["color"])
        rgb = [int(y * 255) for y in rgb_rate]

        # Create segment attribute dictionary
        segment_attribute = {
            "labelID": int(idx),
            "SegmentLabel": name,
            "SegmentAlgorithmType": "MANUAL",
            "SegmentAlgorithmName": "SHH",
            "SegmentedPropertyCategoryCodeSequence": {
                "CodeValue": "M-01000",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Morphologically Altered Structure",
            },
            "SegmentedPropertyTypeCodeSequence": {
                "CodeValue": "M-35300",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Embolus",
            },
            "recommendedDisplayRGBValue": rgb,
        }
        segment_attributes.append(segment_attribute)

    # Create the template with all required metadata
    template = {
        "ContentCreatorName": "Reader1",
        "ClinicalTrialSeriesID": "Session1",
        "ClinicalTrialTimePointID": "1",
        "SeriesDescription": model,
        "SeriesNumber": "300",
        "InstanceNumber": "1",
        "segmentAttributes": [segment_attributes],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "SHH",
        "ClinicalTrialCoordinatingCenterName": "SHH",
        "BodyPartExamined": "",
    }

    return template


def make_dicomseg_file(
    mask: np.ndarray,
    image: sitk.Image,
    first_dcm: pydicom.FileDataset,
    source_images: list[pydicom.FileDataset],
    template_json: dict,
) -> pydicom.FileDataset:
    """
    Create a DICOM-SEG file from a mask array.

    Args:
        mask: Binary mask array
        image: SimpleITK image with spatial information
        first_dcm: First DICOM dataset for metadata
        source_images: List of source DICOM datasets
        template_json: DICOM-SEG template

    Returns:
        pydicom.FileDataset: DICOM-SEG dataset
    """
    # Create template from JSON
    template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

    # Set up the writer
    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,
        skip_empty_slices=True,
        skip_missing_segment=True,
    )

    # Use the mask as provided (already transformed)
    segmentation_data = mask

    # Create SimpleITK image from the segmentation data
    segmentation = sitk.GetImageFromArray(segmentation_data)
    segmentation.CopyInformation(image)

    # Generate DICOM-SEG file
    dcm_seg = writer.write(segmentation, source_images)

    # Copy relevant information from the first DICOM image
    dcm_seg[0x10, 0x0010].value = first_dcm[0x10, 0x0010].value  # Patient's Name
    dcm_seg[0x20, 0x0011].value = first_dcm[0x20, 0x0011].value  # Series Number

    # Copy more metadata from the example DICOM-SEG file
    dcm_seg[0x5200, 0x9229].value = DCM_EXAMPLE[0x5200, 0x9229].value
    dcm_seg[0x5200, 0x9229][0][0x20, 0x9116][0][0x20, 0x0037].value = first_dcm[0x20, 0x0037].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0050].value = first_dcm[0x18, 0x0050].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0088].value = first_dcm[0x18, 0x0088].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x28, 0x0030].value = first_dcm[0x28, 0x0030].value

    return dcm_seg


def create_dicom_seg_file(
    pred_data_unique: np.ndarray,
    pred_data: np.ndarray,
    series_name: str,
    output_folder: pathlib.Path,
    image: Any,
    first_dcm: FileDataset | DicomDir,
    source_images: list[FileDataset | DicomDir],
) -> list[dict[str, Any]]:
    """
    Create DICOM-SEG files for each unique value in the prediction data.

    Args:
        pred_data_unique: Array of unique values in prediction data
        pred_data: Full prediction data array
        series_name: Name of the series
        output_folder: Output directory for DICOM-SEG files
        image: SimpleITK image
        first_dcm: First DICOM dataset
        source_images: List of source DICOM datasets

    Returns:
        List[Dict[str, Any]]: List of results with mask index, file path, and main slice
    """
    pred_data_unique_len = len(pred_data_unique)
    reslut_list = []

    # Process each unique value in the prediction data
    for index, i in enumerate(pred_data_unique):
        # Create a binary mask for this specific region
        mask = np.zeros_like(pred_data)
        mask[pred_data == i] = 1

        # Only create DICOM-SEG if the mask contains positive values
        if np.sum(mask) > 0:
            # Create label dictionary for this mask
            label_dict = {1: {"SegmentLabel": f"A{i}", "color": "red"}}

            # Create template for DICOM-SEG
            template_json = get_dicom_seg_template(series_name, label_dict)

            # Generate DICOM-SEG file
            dcm_seg = make_dicomseg_file(mask.astype("uint8"), image, first_dcm, source_images, template_json)

            # Find the median slice containing the mask (main slice)
            main_seg_slice = int(np.median(np.where(mask)[0]))

            # Save DICOM-SEG file
            dcm_seg_filename = f"{series_name}_{label_dict[1]['SegmentLabel']}.dcm"

            dcm_seg_path = output_folder.joinpath(dcm_seg_filename)
            if dcm_seg_path.exists():
                dcm_seg_path.unlink()

            dcm_seg.save_as(dcm_seg_path)

            # Clear console line and show progress
            print(" " * 100, end="\r")
            print(f"{index + 1}/{pred_data_unique_len} Saved: {dcm_seg_path}", end="\r")

            # Add result to the list if file was created successfully
            if dcm_seg_path.exists():
                reslut_list.append({"mask_index": i, "dcm_seg_path": dcm_seg_path, "main_seg_slice": main_seg_slice})

    return reslut_list


def load_and_sort_dicom_files(
    path_dcms: pathlib.Path | str,
) -> tuple[list[Any], Any, FileDataset | DicomDir, list[FileDataset | DicomDir]]:
    """
    Load and sort DICOM files from a directory.
    This function only needs to be executed once per directory.

    Args:
        path_dcms: Path to the directory containing DICOM files

    Returns:
        tuple: (sorted_dcms, image, first_dcm, source_images)
            - sorted_dcms: List of sorted DICOM file paths
            - image: SimpleITK image object
            - first_dcm: First DICOM dataset
            - source_images: List of all DICOM datasets (without pixel data)
    """
    # Read DICOM file paths
    reader = sitk.ImageSeriesReader()
    dcms = sorted(reader.GetGDCMSeriesFileNames(path_dcms))

    # Read all slices
    slices = [pydicom.dcmread(dcm) for dcm in dcms]

    # Sort slices by position
    slice_dcm = []
    for slice_data, dcm_slice in zip(slices, dcms):
        # Get Image Orientation Patient (IOP)
        IOP = np.array(slice_data.get((0x0020, 0x0037)).value)
        # Get Image Position Patient (IPP)
        IPP = np.array(slice_data.get((0x0020, 0x0032)).value)
        # Calculate normal vector to the image plane
        normal = np.cross(IOP[0:3], IOP[3:])
        # Project IPP onto the normal vector
        projection = np.dot(IPP, normal)
        slice_dcm.append({"d": projection, "dcm": dcm_slice})

    # Sort slices by projection value
    slice_dcms = sorted(slice_dcm, key=lambda i: i["d"])
    sorted_dcms = [y["dcm"] for y in slice_dcms]

    # Read the image data
    reader.SetFileNames(sorted_dcms)
    image = reader.Execute()

    # Read the first DICOM image for metadata
    first_dcm = pydicom.dcmread(sorted_dcms[0], force=True)

    # Preload all DICOM files (without pixel data to save memory)
    source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in sorted_dcms]

    return sorted_dcms, image, first_dcm, source_images


def transform_mask_for_dicom_seg(mask: np.ndarray) -> np.ndarray:
    """
    Transform a mask array to the format required for DICOM-SEG.

    Args:
        mask: Input mask array (y, x, z)

    Returns:
        numpy.ndarray: Transformed mask array (z, y, x) with necessary flips
    """
    # Convert format: (y, x, z) -> (z, x, y)
    segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)

    # Convert format: (z, x, y) -> (z, y, x)
    segmentation_data = np.swapaxes(segmentation_data, 1, 2)

    # Flip y and x axes to match DICOM coordinate system
    segmentation_data = np.flip(segmentation_data, 1)
    return np.flip(segmentation_data, 2)
