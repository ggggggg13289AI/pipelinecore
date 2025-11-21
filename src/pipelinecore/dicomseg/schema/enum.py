import enum
from typing import List,  Union,  Optional
from code_ai import load_dotenv
load_dotenv()


class SeriesTypeEnum(enum.Enum):
    MRA_BRAIN   = '1'
    TOF_MRA     = '1'
    MIP_Pitch   = '2'
    MIP_Yaw     = '3'
    T1BRAVO_AXI = '4'
    T1FLAIR_AXI = '5'
    T2FLAIR_AXI = '6'
    SWAN        = '7'
    DWI0        = '8'
    DWI1000     = '9'
    ADC         = '10'

    @classmethod
    def to_list(cls) -> List[Union[enum.Enum]]:
        return list(map(lambda c: c, cls))


class ModelTypeEnum(enum.Enum):
    Aneurysm = '1'
    CMB      = '2'
    Infarct  = '3'
    WMH      = '4'
    Lacune   = '5'

    @classmethod
    def to_list(cls) -> List[Union[enum.Enum]]:
        return list(map(lambda c: c, cls))
