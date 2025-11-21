import enum
from typing import List, Optional
from pydantic import BaseModel, Field
from code_ai.dicom2nii.convert.config import T1SeriesRenameEnum, MRSeriesRenameEnum, T2SeriesRenameEnum


class InferenceCmdItem(BaseModel):
    study_id : str
    name: str
    cmd_str: str
    input_list: List[str]
    output_list: List[str]
    input_dicom_dir: str


class InferenceCmd(BaseModel):
    cmd_items : List[InferenceCmdItem]


class InferenceEnum(str, enum.Enum):
    Aneurysm = 'Aneurysm'
    SynthSeg = 'SynthSeg'
    Area = 'Area'

    CMB = 'CMB'

    DWI = 'DWI'
    Infarct = 'Infarct'

    WMH = 'WMH'
    WMH_PVS = 'WMH_PVS'
    # Lacune




class Task(BaseModel):
    input_path_list: List[str] = Field(..., alias="intput_path_list")
    output_path: str
    output_path_list: List[str]
    # result: Result


class Analysis(BaseModel):
    study_id: str
    Aneurysm: Optional[Task] = None
    CMB: Optional[Task] = None
    DWI: Optional[Task] = None
    WMH_PVS: Optional[Task] = None
    Area: Optional[Task] = None
    Infarct: Optional[Task] = None
    WMH: Optional[Task] = None
    AneurysmSynthSeg: Optional[Task] = None


MODEL_MAPPING_SERIES_DICT = {
InferenceEnum.Aneurysm: [[MRSeriesRenameEnum.MRA_BRAIN,
                              ]],
    InferenceEnum.Area: [[T1SeriesRenameEnum.T1BRAVO_AXI, ],
                         [T1SeriesRenameEnum.T1BRAVO_SAG, ],
                         [T1SeriesRenameEnum.T1BRAVO_COR, ],
                         [T1SeriesRenameEnum.T1FLAIR_AXI, ],
                         [T1SeriesRenameEnum.T1FLAIR_SAG, ],
                         [T1SeriesRenameEnum.T1FLAIR_COR, ], ],
    InferenceEnum.DWI: [
        [MRSeriesRenameEnum.DWI0]
    ],
    InferenceEnum.WMH_PVS: [[T2SeriesRenameEnum.T2FLAIR_AXI, ]],

    #Ax SWAN_resample_synthseg33_from_Sag_FSPGR_BRAVO_resample_synthseg33.nii.gz
    InferenceEnum.CMB: [[MRSeriesRenameEnum.SWAN, T1SeriesRenameEnum.T1BRAVO_AXI],
                        [MRSeriesRenameEnum.SWAN, T1SeriesRenameEnum.T1FLAIR_AXI],
                        ],
    # InferenceEnum.CMBSynthSeg

    InferenceEnum.Infarct: [[MRSeriesRenameEnum.DWI0, MRSeriesRenameEnum.DWI1000, MRSeriesRenameEnum.ADC, ]
                            # MRSeriesRenameEnum.synthseg_DWI0_original_DWI],
                            ],
    InferenceEnum.WMH: [[T2SeriesRenameEnum.T2FLAIR_AXI,
                         ]],


}
