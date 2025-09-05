from PandAna.core.core import KL, Cut
from PandAna.cut.analysis_cuts import kVeto, kNueProngContainment
from PandAna.utils.enums import trainint
from PandAna.var.regcvn_training_vars import *

KLP = KL + ['rec.vtx.mlvertex.fuzzyk.png_idx']

kNueSig = (kSliceInter >= trainint.kNueQE) & (kSliceInter <= trainint.kNueOther)

kNuBkg = (kSliceInter < trainint.kCosmic) & (~kNueSig)

kSigPresel = kVeto & kNueProngContainment & kNueSig
kNuBkgPresel = kVeto & kNueProngContainment & kNuBkg

kElectronProng = (kProngLabel == 0) & (kProngPrimary == 1)
