import numpy as np
import re
from typing import Dict, List, Optional, Union


keyword_likelihoods = {
    # ======================== LAB ========================
    "lab": {
        # ===== ENGLISH =====
        # Common panels & tests
        "cbc": 0.95,          "complete blood count": 0.95,
        "bmp": 0.95,          "basic metabolic panel": 0.95,
        "cmp": 0.95,          "comprehensive metabolic panel": 0.95,
        "lft": 0.95,          "liver function test": 0.95,
        "lipid panel": 0.95,  "cholesterol": 0.9,
        "hdl": 0.9,           "ldl": 0.9,
        "triglycerides": 0.9, "hba1c": 0.95,
        "glucose": 0.9,       "fasting": 0.7,
        "creatinine": 0.95,   "bun": 0.95,
        "sodium": 0.9,        "potassium": 0.9,
        "chloride": 0.9,      "co2": 0.9,
        "calcium": 0.9,       "magnesium": 0.9,
        "phosphorus": 0.9,    "uric acid": 0.9,
        "tsh": 0.95,          "free t4": 0.95,
        "t3": 0.9,           "psa": 0.95,
        "vitamin d": 0.9,     "vitamin b12": 0.9,
        "folate": 0.9,       "ferritin": 0.9,
        "iron": 0.85,        "tibc": 0.85,
        "cortisol": 0.9,     "acth": 0.9,
        
        # Haematology
        "hemoglobin": 0.95,   "hgb": 0.95,
        "hematocrit": 0.95,   "hct": 0.95,
        "wbc": 0.95,          "white blood cell": 0.95,
        "rbc": 0.95,          "red blood cell": 0.95,
        "mcv": 0.9,           "mch": 0.9,       "mchc": 0.9,
        "platelet": 0.9,      "plt": 0.9,
        "neutrophil": 0.9,    "lymphocyte": 0.9,
        "monocyte": 0.9,      "eosinophil": 0.9,
        "basophil": 0.9,
        
        # Coagulation
        "pt": 0.9,            "ptt": 0.9,
        "inr": 0.95,          "fibrinogen": 0.9,
        
        # Cardiac / other
        "troponin": 0.95,     "bnp": 0.95,
        "nt-probnp": 0.95,    "crp": 0.9,
        "esr": 0.9,          "rheumatoid factor": 0.9,
        "ana": 0.9,          "urinalysis": 0.95,
        "urine": 0.7,        "specific gravity": 0.9,
        "ph": 0.5,           "protein": 0.7,
        "glucose urine": 0.9, "ketones": 0.9,
        "nitrite": 0.9,      "leukocyte esterase": 0.9,
        "microalbumin": 0.9,  "creatinine urine": 0.9,
        
        # Units / common phrases
        "mg/dl": 0.6,        "meq/l": 0.6,
        "iu/l": 0.6,         "ng/ml": 0.6,
        "reference range": 0.8,
        "lab": 0.4,          "laboratory": 0.4,
        
        # Molecular/Genetic lab
        "genetic testing": 0.9, "genetic panel": 0.9,
        "dna sequencing": 0.9, "whole exome": 0.95,
        "whole genome": 0.95, "wes": 0.85,
        "wgs": 0.85, "next generation sequencing": 0.95,
        "ngs": 0.9, "sanger sequencing": 0.95,
        "pcr": 0.8, "rt-pcr": 0.9, "microarray": 0.9,
        "chromosomal microarray": 0.95, "cma": 0.85,
        "brca1": 0.95, "brca2": 0.95,
        "cystic fibrosis": 0.8, "carrier screening": 0.9,
        "prenatal screening": 0.85, "nipt": 0.95,
        "cell-free dna": 0.9, "cfdna": 0.9,
        "karyotype": 0.95, "karyotyping": 0.95,
        "fragile x": 0.9, "sma testing": 0.9,
        "mutation": 0.8, "variant": 0.7,
        "pathogenic": 0.7, "vus": 0.9,
        "heterozygous": 0.8, "homozygous": 0.8,
        "compound heterozygous": 0.85, "hemizygous": 0.8,
        "autosomal dominant": 0.8, "autosomal recessive": 0.8,
        "x-linked": 0.8, "mendelian": 0.8,
        "polymorphism": 0.8, "snp": 0.85,
        "amplification": 0.7, "primer": 0.8,
        "hybridization": 0.8, "electrophoresis": 0.9,
        "capillary electrophoresis": 0.95, "sanger": 0.9,
        "amplicon": 0.9, "exon": 0.85, "intron": 0.85,
        "splice site": 0.85, "frameshift": 0.85,
        "nonsense": 0.85, "missense": 0.85,
        "deletion": 0.8, "duplication": 0.8, "insertion": 0.8,
        "cnv": 0.85, "copy number": 0.8,
        "allele": 0.7, "genotype": 0.8, "phenotype": 0.7,
        "pharmacogenetic": 0.95, "pharmacogenomic": 0.95,
        "cyp2d6": 0.95, "cyp2c9": 0.95, "cyp2c19": 0.95,
        
        # Molecular lab terms (general)
        "dna": 0.7, "rna": 0.7,
        "extraction": 0.8, "extracted": 0.8,
        "real-time": 0.9, "real time": 0.9,
        "genotyping": 0.9, "genotype": 0.85, "genotypes": 0.85,
        "detection": 0.6, "high-risk": 0.7,
        "low-risk": 0.7, "high risk": 0.7, "low risk": 0.7,
        "molecular": 0.8, "screening": 0.5,
        "kit": 0.6,
        "isolated": 0.7, "purified": 0.7, "amplified": 0.7,
        "target": 0.5, "probe set": 0.8,
        "thermal cycler": 0.9, "thermocycler": 0.9,
        
        # Viral serology (lab-specific)
        "virus": 0.5, "viral": 0.5,
        "viral load": 0.9, "viral serology": 0.9,
        "viral antibody": 0.9, "viral antigen": 0.9,
        "hiv": 0.8, "hbv": 0.8, "hcv": 0.8,
        "hepatitis": 0.8,
        
        # Microbiology
        "culture": 0.85, "microbe": 0.7,
        "bacteria": 0.7, "fungus": 0.7,
        "antibiotic sensitivity": 0.9, "blood culture": 0.9,
        "urine culture": 0.9, "stool culture": 0.9,
        "gram positive": 0.9, "gram negative": 0.9,
        "antibiotic": 0.8, "resistance": 0.6,
        "sensitive": 0.5, "resistant": 0.5,
        
        # COVID-19 specific
        "covid": 0.5, "igg": 0.9, "igm": 0.9,
        
        # OCR variants (English)
        "genet1c": 0.85, "dna test": 0.85,
        "kary0type": 0.9, "nucle0tide": 0.8,
        "sequenc1ng": 0.85, "m1croarray": 0.85,
        
        # ===== PERSIAN (فارسی) =====
        # آزمایشگاه - Laboratory
        "آزمایش": 0.9, "آزمایشگاه": 0.95,
        "جواب": 0.6, "نتیجه": 0.6, "برگه": 0.5,
        "آزمایش خون": 0.95, "آزمایش ادرار": 0.95, "آزمایش مدفوع": 0.95,
        
        # خون‌شناسی - Hematology
        "شمارش کامل خون": 0.95, "هماتولوژی": 0.9,
        "هماتوکریت": 0.9, "هموگلوبین": 0.9,
        "گلبول سفید": 0.9, "گلبول قرمز": 0.9,
        "پلاکت": 0.9, "نوتروفیل": 0.9, "لنفوسیت": 0.9,
        
        # بیوشیمی - Biochemistry
        "بیوشیمی": 0.95, "قند خون": 0.9, "گلوکز": 0.9,
        "کلسترول": 0.9, "تری‌گلیسرید": 0.9,
        "اوره": 0.9, "کراتینین": 0.9,
        "سدیم": 0.9, "پتاسیم": 0.9, "کلسیم": 0.9,
        "فسفر": 0.9, "منیزیم": 0.9, "اسید اوریک": 0.9, "آهن": 0.85,
        
        # کبد - Liver
        "تست کبد": 0.9, "آنزیم کبدی": 0.9,
        "بیلی‌روبین": 0.9, "آلکالن فسفاتاز": 0.9, "آمینوترانسفراز": 0.9,
        
        # هورمون‌ها - Hormones
        "هورمون": 0.9, "تیروئید": 0.9, "انسولین": 0.9,
        "کورتیزول": 0.9, "تستوسترون": 0.9,
        "استروژن": 0.9, "پروژسترون": 0.9, "پرولاکتین": 0.9,
        
        # آزمایش ادرار - Urinalysis
        "آزمایش کامل ادرار": 0.95, "وزن مخصوص": 0.9,
        "پروتئین ادرار": 0.9, "گلوکز ادرار": 0.9,
        
        # سرولوژی/ایمونولوژی - Serology/Immunology
        "سرولوژی": 0.9, "ایمونولوژی": 0.9,
        "آنتی‌بادی": 0.9, "آنتی‌ژن": 0.9,
        "هپاتیت": 0.8, "ایدز": 0.8, "اچ‌آی‌وی": 0.8,
        "ویروس": 0.5, "سرخک": 0.8, "سرخجه": 0.8, "توکسوپلاسما": 0.8,
        
        # انعقاد - Coagulation
        "انعقاد": 0.9, "پروترومبین": 0.9, "فیبرینوژن": 0.9,
        
        # واحدها - Units
        "محدوده مرجع": 0.8, "میلی‌گرم": 0.6, "دسی‌لیتر": 0.6,
        "واحد": 0.4, "طبیعی": 0.5, "غیرطبیعی": 0.5,
        "روش": 0.6, "دستگاه": 0.6, "کیت": 0.7,
        "نمونه": 0.5, "خون‌گیری": 0.8,
        "ناشتا": 0.8, "غیرناشتا": 0.7,
        
        # ژنتیک/مولکولی - Molecular/Genetic
        "ژنتیک": 0.7, "مولکولی": 0.8,
        "دی‌ان‌ای": 0.8, "آر‌ان‌ای": 0.8,
        "استخراج": 0.7, "تکثیر": 0.7,
        "ژنوتایپینگ": 0.9, "تعیین ژنوتیپ": 0.9,
        "جهش": 0.7, "واریانت": 0.7,
        "پاتوژنیک": 0.7, "خوش‌خیم": 0.6, "بدخیم": 0.6,
        
        # میکروب‌شناسی - Microbiology
        "کشت": 0.85, "میکروب": 0.7,
        "باکتری": 0.7, "قارچ": 0.7,
        "حساسیت آنتی‌بیوتیکی": 0.9,
        "کشت خون": 0.9, "کشت ادرار": 0.9, "کشت مدفوع": 0.9,
        "گرم مثبت": 0.9, "گرم منفی": 0.9,
        "آنتی‌بیوتیک": 0.8, "مقاومت": 0.6,
        "حساس": 0.5, "مقاوم": 0.5,
        
        # COVID-19
        "کرونا": 0.5, "کووید": 0.5,
        "پی‌سی‌آر": 0.9, "پی سی آر": 0.9,
        "آی‌جی‌جی": 0.9, "آی‌جی‌ام": 0.9,
        
        # Persian OCR variants
        "آزمایـش": 0.9, "آزمايش": 0.9, "آزماىش": 0.9,
        "هموگلوبين": 0.9, "كلسترول": 0.9, "كبد": 0.6,
        
        # Mixed Persian-English
        "cbc آزمایش": 0.9, "آزمایش cbc": 0.9,
        "diff آزمایش": 0.9, "bun آزمایش": 0.9,
    },
    
    # ======================= IMAGING =======================
    "imaging": {
        # ===== ENGLISH =====
        # Modalities
        "x-ray": 0.95, "xray": 0.95, "radiograph": 0.95,
        "ct": 0.9, "cat scan": 0.9, "mri": 0.95,
        "magnetic resonance": 0.95, "ultrasound": 0.95,
        "sonogram": 0.95, "us": 0.7,
        "mammogram": 0.95, "mammography": 0.95,
        "fluoroscopy": 0.95, "nuclear medicine": 0.95,
        "bone scan": 0.95, "pet": 0.9,
        "pet-ct": 0.95, "spect": 0.9,
        "dexa": 0.9, "dxa": 0.9,
        
        # Contrast & technique
        "contrast": 0.7, "noncontrast": 0.8,
        "with contrast": 0.9, "without contrast": 0.9,
        "iv contrast": 0.9, "oral contrast": 0.9,
        "enhancement": 0.6, "enhancing": 0.6,
        "t1": 0.8, "t2": 0.8, "flair": 0.9,
        "diffusion": 0.7, "adc": 0.8,
        
        # Projections / views
        "ap": 0.6, "pa": 0.6, "lateral": 0.6,
        "oblique": 0.6, "decubitus": 0.8, "cross-table": 0.8,
        "towne": 0.8, "waters": 0.8, "view": 0.5,
        
        # Anatomy
        "chest": 0.4, "abdomen": 0.4, "pelvis": 0.4,
        "spine": 0.4, "cervical": 0.6, "thoracic": 0.6,
        "lumbar": 0.6, "extremity": 0.6, "skull": 0.6, "sinus": 0.6,
        
        # Findings
        "opacity": 0.6, "consolidation": 0.6,
        "infiltrate": 0.6, "effusion": 0.6,
        "pneumothorax": 0.8, "cardiomegaly": 0.7,
        "fracture": 0.6, "dislocation": 0.7,
        "degenerative": 0.6, "spondylosis": 0.7,
        "herniation": 0.5, "disc": 0.4,
        
        # Sonography/Ultrasound
        "sonography": 0.95, "sonographic": 0.95,
        "ultrasonography": 0.95, "ultrasonographic": 0.95,
        "u/s": 0.9, "echo": 0.5,
        "echocardiogram": 0.95, "echocardiography": 0.95,
        "doppler": 0.9, "duplex": 0.9,
        "color flow": 0.9, "color doppler": 0.95,
        "power doppler": 0.95, "spectral doppler": 0.95,
        "transvaginal": 0.9, "transabdominal": 0.9,
        "endoscopic ultrasound": 0.95, "eus": 0.9,
        "transrectal": 0.9, "transducer": 0.8,
        
        # Echotexture
        "echogenic": 0.9, "echogenicity": 0.9,
        "anechoic": 0.9, "hypoechoic": 0.9,
        "hyperechoic": 0.9, "isoechoic": 0.9,
        "sonolucent": 0.9, "sonodense": 0.9,
        "acoustic shadowing": 0.9, "shadowing": 0.7,
        "heterogeneous": 0.7, "homogeneous": 0.7,
        
        # Ultrasound exams
        "abdominal ultrasound": 0.95, "pelvic ultrasound": 0.95,
        "thyroid ultrasound": 0.95, "carotid duplex": 0.95,
        "venous duplex": 0.95, "breast ultrasound": 0.95,
        "renal ultrasound": 0.95, "obstetric ultrasound": 0.95,
        "scrotal ultrasound": 0.95, "msk ultrasound": 0.95,
        "soft tissue ultrasound": 0.95, "liver ultrasound": 0.95,
        "gallbladder ultrasound": 0.95,
        
        # Doppler measurements
        "peak systolic velocity": 0.9, "psv": 0.9,
        "end diastolic velocity": 0.9, "edv": 0.9,
        "resistive index": 0.9, "pulsatility index": 0.9,
        "waveform": 0.7, "pulsed wave": 0.9, "continuous wave": 0.9,
        "aliasing": 0.8,
        
        # OB ultrasound
        "gestational sac": 0.95, "yolk sac": 0.95,
        "crown rump length": 0.95, "crl": 0.9,
        "fetal heart rate": 0.9, "fhr": 0.9,
        "placenta": 0.8, "amniotic fluid": 0.9,
        "nuchal translucency": 0.95, "nt": 0.7,
        "biparietal diameter": 0.95, "bpd": 0.9,
        "head circumference": 0.9, "abdominal circumference": 0.9,
        "femur length": 0.9, "amniotic fluid index": 0.95, "afi": 0.9,
        
        # Report terms
        "radiology": 0.4, "imaging": 0.4, "radiologist": 0.6,
        
        # Genetic conditions (weak in imaging)
        "genetic": 0.3, "hereditary": 0.3, "familial": 0.3,
        "congenital": 0.4, "dysmorphic": 0.4,
        "anomaly": 0.4, "anomalies": 0.4, "malformation": 0.4,
        
        # OCR variants (English)
        "ct scan": 0.9, "mr1": 0.9, "mrl": 0.9, "c/t": 0.8,
        "x ray": 0.95, "xr": 0.85,
        "s0nography": 0.95, "son0graphy": 0.95,
        "ultras0und": 0.95, "ultrason0graphy": 0.95,
        "ech0cardiogram": 0.95, "ech0genic": 0.9,
        "anec0hoic": 0.9, "hyp0echoic": 0.9, "d0ppler": 0.9,
        
        # ===== PERSIAN (فارسی) =====
        # تصویربرداری - Imaging
        "تصویربرداری": 0.95, "رادیولوژی": 0.95,
        "سونوگرافی": 0.95, "اولتراسوند": 0.95,
        "ام‌آر‌آی": 0.95, "ام آر آی": 0.95,
        "سی‌تی": 0.9, "سی تی": 0.9,
        "سی‌تی‌اسکن": 0.95, "سی تی اسکن": 0.95,
        "مقطع‌نگاری": 0.9, "رادیوگرافی": 0.95,
        "عکس": 0.5, "پرتو": 0.6, "اشعه": 0.6, "ایکس": 0.7,
        "ایکس‌ری": 0.9,
        
        # MRI specific
        "مقطع‌نگاری کامپیوتری": 0.95, "تشدید مغناطیسی": 0.95,
        "ام‌آر": 0.85, "ام آر": 0.85, "مگنت": 0.8, "تشدید": 0.7,
        
        # Ultrasound/Sonography
        "سونو": 0.8, "پروب": 0.7,
        "ترانس‌واژینال": 0.9, "ترانس ابدومینال": 0.9,
        "داپلر": 0.9, "رنگی": 0.6, "اکو": 0.6,
        "اکوکاردیوگرافی": 0.95, "اکوکاردیوگرام": 0.95,
        
        # Echotexture
        "اکوژن": 0.9, "اکوژنیسیته": 0.9,
        "آن‌اکوئیک": 0.9, "هیپواکو": 0.9,
        "هایپراکو": 0.9, "ایزواکو": 0.9,
        "سایه": 0.6, "سایه صوتی": 0.9,
        "هتروژن": 0.7, "هموژن": 0.7,
        
        # X-ray specific
        "گرافی": 0.9, "قفسه سینه": 0.6,
        "شکم": 0.5, "لگن": 0.6, "ستون فقرات": 0.7,
        "گردن": 0.5, "اندام": 0.6, "نما": 0.6,
        "نمای روبرو": 0.8, "نمای نیمرخ": 0.8,
        "خوابیده": 0.6, "ایستاده": 0.6,
        "مایل": 0.7, "دکوبیتوس": 0.8,
        
        # Contrast
        "کنتراست": 0.8, "تزریق": 0.5,
        "با تزریق": 0.8, "بدون تزریق": 0.8,
        "وریدی": 0.6, "افزایش یافته": 0.4,
        
        # Imaging findings
        "شکستگی": 0.7, "دررفتگی": 0.7,
        "افیوژن": 0.7, "پنوموتوراکس": 0.8,
        "کاردیومگالی": 0.7, "کدورت": 0.7,
        "توده": 0.6, "کیست": 0.7,
        "کلسیفیکاسیون": 0.7, "دژنراتیو": 0.7,
        "آرتروز": 0.7, "اسپوندیلوز": 0.7,
        "فتق": 0.7, "دیسک": 0.6,
        "بیرون‌زدگی": 0.7, "تنگی": 0.6,
        "انحراف": 0.5, "ضایعه": 0.6,
        "التهاب": 0.5, "ادم": 0.6,
        
        # Anatomical regions
        "مغز": 0.5, "ریه": 0.5, "کبد": 0.6,
        "کلیه": 0.6, "طحال": 0.6, "پانکراس": 0.6,
        "کیسه صفرا": 0.7, "مثانه": 0.6,
        "رحم": 0.6, "تخمدان": 0.6, "پروستات": 0.6,
        "تیروئید": 0.6, "عروق": 0.6,
        "شریان": 0.6, "ورید": 0.6,
        
        # Report structure
        "گزارش": 0.4, "شرح": 0.4, "یافته": 0.5,
        "نظر": 0.4, "تشخیص": 0.4, "پیشنهاد": 0.4,
        
        # Measurements
        "میلی‌متر": 0.6, "سانتی‌متر": 0.6,
        "اندازه": 0.5, "ابعاد": 0.5, "قطر": 0.5,
        
        # OB/GYN
        "بارداری": 0.6, "جنین": 0.7, "جفت": 0.7,
        "مایع آمنیوتیک": 0.8, "کیسه حاملگی": 0.9,
        "کیسه زرده": 0.9, "ساک حاملگی": 0.9,
        "طول سری-دمی": 0.9, "ضربان قلب": 0.6,
        "وزن تخمینی": 0.7, "هفته": 0.5,
        
        # Mammography
        "ماموگرافی": 0.95, "ماموگرام": 0.95,
        "پستان": 0.7, "بافت": 0.5,
        "دانسیته": 0.7, "میکروکلسیفیکاسیون": 0.9,
        "ندول": 0.7, "بی‌رادز": 0.9,
        
        # Nuclear medicine
        "پزشکی هسته‌ای": 0.95, "اسکن هسته‌ای": 0.95,
        "اسکن استخوان": 0.9, "اسپکت": 0.9, "پت": 0.8,
        "پت‌سی‌تی": 0.95, "اسکن تیروئید": 0.9,
        
        # Dental
        "پانورامیکس": 0.9, "دندانپزشکی": 0.7,
        "دندان": 0.6, "فک": 0.5, "اوپی‌جی": 0.9,
        
        # Persian OCR variants
        "سي تي": 0.9, "راديولوژي": 0.95,
        "اولتراسوند": 0.95, "ام آر آي": 0.95,
        
        # Mixed Persian-English
        "ct اسکن": 0.95, "mri تصویربرداری": 0.95,
        "سونوگرافی us": 0.9, "doppler سونوگرافی": 0.9,
    },
    
    # ======================= PATHOLOGY ======================
    "pathology": {
        # ===== ENGLISH =====
        # Specimen types
        "biopsy": 0.95, "biopsies": 0.95,
        "resection": 0.95, "excision": 0.9,
        "specimen": 0.8, "tissue": 0.7,
        "core biopsy": 0.95, "needle biopsy": 0.95,
        "excisional biopsy": 0.95, "incisional biopsy": 0.95,
        "punch biopsy": 0.95, "shave biopsy": 0.95,
        "curettage": 0.9, "aspirate": 0.85,
        "fluid": 0.5, "cytology": 0.8,
        "cytological": 0.8, "cytologic": 0.8,
        "cytopathology": 0.95, "cytopathologist": 0.9,
        "fna": 0.9, "pap smear": 0.95, "pap test": 0.95,
        "brushing": 0.9, "washing": 0.9, "cell block": 0.95,
        
        # Liquid-based cytology (pathology)
        "liquid-based": 0.8,
        "liquid based": 0.8,
        "liquid-based cytology": 0.95,
        "liquid based cytology": 0.95,
        "lbc": 0.9,
        "liquid cytology": 0.9,
        "thinprep": 0.95,
        "thin prep": 0.95,
        "surepath": 0.95,
        "sure path": 0.95,
        "preservcyt": 0.9,
        "cytology vial": 0.9,
        "cytology specimen": 0.9,
        
        # Cervical screening (pathology)
        "cervical screening": 0.9,
        "cervical cytology": 0.95,
        "cervical smear": 0.95,
        "cervical cancer screening": 0.9,
        "colposcopy": 0.7,
        "cervical biopsy": 0.95,
        
        # HPV testing (pathology)
        "hpv": 0.85,
        "human papillomavirus": 0.85,
        "hpv testing": 0.9,
        "hpv test": 0.9,
        "hpv genotyping": 0.9,
        "hpv detection": 0.9,
        "hpv screening": 0.9,
        "hpv dna": 0.9,
        "hpv rna": 0.9,
        "high risk hpv": 0.9,
        "low risk hpv": 0.9,
        "hpv positive": 0.85,
        "hpv negative": 0.85,
        "hpv virus": 0.85,
        
        # Viral pathology
        "virus": 0.6,
        "viral": 0.6,
        "viral cytopathic effect": 0.9,
        "viral inclusion": 0.9,
        "koilocytosis": 0.9,
        "koilocyte": 0.9,
        "koilocytic": 0.9,
        
        # Stains & techniques
        "h&e": 0.95, "hematoxylin": 0.95, "eosin": 0.95,
        "ihc": 0.95, "immunohistochemistry": 0.95,
        "pas": 0.9, "afb": 0.9, "giemsa": 0.9,
        "gram stain": 0.9, "special stain": 0.9,
        "flow cytometry": 0.9, "fish": 0.85,
        "pcr": 0.8, "microscopic": 0.95,
        "gross": 0.9, "macroscopic": 0.9,
        
        # Diagnosis terms
        "malignant": 0.9, "benign": 0.9,
        "carcinoma": 0.95, "sarcoma": 0.95,
        "lymphoma": 0.95, "melanoma": 0.95,
        "adenocarcinoma": 0.95, "neoplasm": 0.9,
        "tumor": 0.9, "tumour": 0.9,
        "metastatic": 0.9, "metastasis": 0.9,
        "dysplasia": 0.9, "hyperplasia": 0.9,
        "inflammation": 0.8, "acute": 0.6, "chronic": 0.6,
        "granuloma": 0.9, "necrosis": 0.9, "apoptosis": 0.9,
        "mitosis": 0.9, "mitotic": 0.9,
        "ki-67": 0.95, "proliferation index": 0.9,
        "grade": 0.8, "stage": 0.7, "tnm": 0.95, "ptnm": 0.95,
        "margin": 0.8, "clear margins": 0.95, "involved margin": 0.95,
        
        # Molecular pathology
        "molecular pathology": 0.95, "her2": 0.9,
        "her-2": 0.9, "her2neu": 0.9,
        "er positive": 0.9, "pr positive": 0.9, "triple negative": 0.9,
        "ki67": 0.95, "pdl1": 0.9, "pd-l1": 0.9,
        "alk": 0.85, "egfr": 0.85, "kras": 0.85, "braf": 0.85,
        "ros1": 0.85, "msi": 0.9, "msi-high": 0.95, "msi-h": 0.9,
        "mmr": 0.85, "mlh1": 0.85, "msh2": 0.85, "msh6": 0.85,
        "pms2": 0.85, "tmb": 0.85,
        "tumor mutational burden": 0.9,
        "idh1": 0.85, "1p19q": 0.9, "mgmt": 0.85,
        
        # Hematopathology
        "cd3": 0.85, "cd4": 0.85, "cd8": 0.85,
        "cd19": 0.85, "cd20": 0.85, "cd45": 0.85,
        "cd138": 0.85, "cd34": 0.85,
        "immunophenotype": 0.95, "clonality": 0.9,
        "b-cell": 0.8, "t-cell": 0.8,
        "lymphoproliferative": 0.9,
        "myelodysplastic": 0.9, "myeloproliferative": 0.9,
        "jak2": 0.85, "bcr-abl": 0.9, "philadelphia chromosome": 0.9,
        
        # Overlapping terms (moderate likelihoods)
        "mutation": 0.7, "mutation analysis": 0.85,
        "gene rearrangement": 0.9, "translocation": 0.85, "fusion": 0.8,
        
        # Report terms
        "pathology": 0.4, "histology": 0.95, "histologic": 0.9,
        "pathologist": 0.6, "report": 0.1,
        
        # OCR variants (English)
        "bi0psy": 0.95, "b1opsy": 0.95,
        "spec1men": 0.9, "speclmen": 0.9,
        "carc1noma": 0.95, "carcin0ma": 0.95,
        "melan0ma": 0.95,
        "he": 0.85, "h e": 0.85,
        "f1sh": 0.85, "her2/neu": 0.9, "her2+": 0.9, "pdl-1": 0.9,
        "m0lecular": 0.85,
        
        # ===== PERSIAN (فارسی) =====
        # آسیب‌شناسی - Pathology
        "آسیب‌شناسی": 0.95, "پاتولوژی": 0.95,
        "هیستولوژی": 0.95, "سیتولوژی": 0.8,
        "سلول‌شناسی": 0.8, "بافت‌شناسی": 0.95,
        "نمونه بافتی": 0.9, "بیوپسی": 0.95, "نمونه‌برداری": 0.9,
        
        # Specimen types
        "نمونه": 0.5, "بافت": 0.7,
        "بیوپسی سوزنی": 0.95, "بیوپسی باز": 0.95,
        "بیوپسی آندوسکوپیک": 0.95, "رزکسیون": 0.9,
        "برداشتن": 0.7, "خارج‌سازی": 0.7,
        "کورتاژ": 0.9, "آسپیراسیون": 0.85,
        "سوزنی ظریف": 0.9, "اف‌ان‌ای": 0.9, "اف ان ای": 0.9,
        "مایع": 0.5, "ترشحات": 0.7,
        "خلط": 0.7, "اسمیر": 0.9, "پاپ اسمیر": 0.9,
        "براشینگ": 0.9, "لاواژ": 0.9, "شستشو": 0.7,
        
        # Liquid-based cytology (Persian)
        "سیتولوژی مایع": 0.9,
        "نمونه مایع": 0.8,
        "تین‌پرپ": 0.95,
        "تین پرپ": 0.95,
        
        # HPV (Persian)
        "اچ‌پی‌وی": 0.85,
        "اچ پی وی": 0.85,
        "پاپیلوماویروس": 0.85,
        "پاپیلوما ویروس": 0.85,
        "ویروس پاپیلومای انسانی": 0.85,
        "تشخیص اچ‌پی‌وی": 0.9,
        "تایپینگ اچ‌پی‌وی": 0.9,
        "ژنوتایپینگ اچ‌پی‌وی": 0.9,
        "غربالگری اچ‌پی‌وی": 0.9,
        "اچ‌پی‌وی پرخطر": 0.9,
        "اچ‌پی‌وی کم‌خطر": 0.9,
        "اچ‌پی‌وی مثبت": 0.85,
        "اچ‌پی‌وی منفی": 0.85,
        
        # Cervical screening (Persian)
        "غربالگری دهانه رحم": 0.9,
        "غربالگری سرویکس": 0.9,
        "نمونه سرویکس": 0.9,
        "نمونه دهانه رحم": 0.9,
        "کولپوسکوپی": 0.7,
        
        # Viral pathology (Persian)
        "ویروس": 0.6,
        "اثر سایتوپاتیک ویروسی": 0.9,
        "انکلوزیون ویروسی": 0.9,
        "کویلوسیتوز": 0.9,
        "کویلوسیت": 0.9,
        "کویلوسیتیک": 0.9,
        
        # Processing & staining
        "فیکساسیون": 0.9, "فیکس": 0.8,
        "پارافین": 0.9, "بلوک": 0.8, "برش": 0.6,
        "رنگ‌آمیزی": 0.9, "هماتوکسیلین": 0.95,
        "ائوزین": 0.95, "هماتوکسیلین و ائوزین": 0.95,
        "ایمونوهیستوشیمی": 0.95, "ایمونوهیستوکمیستری": 0.95,
        "رنگ‌آمیزی اختصاصی": 0.9, "پاس": 0.85,
        "گیمسا": 0.9, "گرام": 0.85,
        "اسید فست": 0.9, "ذیل-نلسون": 0.9,
        "فلو سایتومتری": 0.9, "فلوسایتومتری": 0.9,
        
        # Microscopic description
        "میکروسکوپی": 0.9, "ماکروسکوپی": 0.9,
        "گروس": 0.9, "بزرگنمایی": 0.8,
        "سلول": 0.7, "هسته": 0.7,
        "سیتوپلاسم": 0.9, "غشاء": 0.7,
        "استروما": 0.9, "نکروز": 0.9,
        "خونریزی": 0.6, "التهاب": 0.6,
        "گرانولوم": 0.9, "آبسه": 0.7,
        "فیبروز": 0.8, "اسکار": 0.7,
        "کلسیفیکاسیون": 0.7, "متاپلازی": 0.9,
        "دیسپلازی": 0.9, "هیپرپلازی": 0.9,
        "آتروفی": 0.8, "هیپرتروفی": 0.8,
        
        # Neoplasia
        "نئوپلاسم": 0.9, "تومور": 0.9, "توده": 0.7,
        "خوش‌خیم": 0.9, "بدخیم": 0.95, "سرطان": 0.9,
        "کارسینوم": 0.95, "آدنوکارسینوم": 0.95,
        "سارکوم": 0.95, "لنفوم": 0.95,
        "ملانوما": 0.95, "لوکمی": 0.9, "میلوم": 0.9,
        "متاستاز": 0.9, "متاستاتیک": 0.9,
        "تهاجم": 0.8, "نفوذ": 0.7, "گسترش": 0.6,
        
        # Tumor grading/staging
        "گرید": 0.9, "درجه": 0.7, "مرحله": 0.7, "استیج": 0.9,
        "تمایز": 0.8, "میتوز": 0.9, "میتوتیک": 0.9,
        "شاخص تکثیر": 0.9, "کی-۶۷": 0.9, "کی ۶۷": 0.9,
        "تمایز یافته": 0.8, "تمایز نیافته": 0.9,
        
        # Margins
        "مارژین": 0.9, "حاشیه": 0.8,
        "درگیر": 0.8, "سالم": 0.7, "آزاد": 0.7,
        "حاشیه درگیر": 0.95, "حاشیه آزاد": 0.9, "حاشیه سالم": 0.9,
        
        # Lymph nodes
        "غدد لنفاوی": 0.8, "لنف نود": 0.9, "گره لنفاوی": 0.8,
        "متاستاز لنف": 0.9,
        
        # IHC/Molecular markers
        "مارکر": 0.8, "گیرنده": 0.7,
        "استروژن": 0.8, "پروژسترون": 0.8,
        "هر-۲": 0.9, "پی۵۳": 0.9,
        "مثبت": 0.5, "منفی": 0.5,
        "فیوژن": 0.85, "بازآرایی": 0.85,
        "تکثیر": 0.7, "حذف": 0.7,
        
        # Report structure
        "تشخیص پاتولوژی": 0.95, "نظر آسیب‌شناسی": 0.95,
        "تشخیص نهایی": 0.7, "گزارش پاتولوژی": 0.95,
        "شرح ماکروسکوپی": 0.95, "شرح میکروسکوپی": 0.95,
        "تشخیص": 0.5,
        
        # Diagnosis phrases
        "مثبت از نظر": 0.8, "منفی از نظر": 0.8,
        "به نفع": 0.7, "مطرح‌کننده": 0.7, "مطرح کننده": 0.7,
        "رد کننده": 0.7, "سازگار با": 0.7, "همراه با": 0.5,
        
        # Common diagnoses
        "آپاندیسیت": 0.8, "کوله‌سیستیت": 0.8,
        "گاستریت": 0.8, "کولیت": 0.8,
        "پولیپ": 0.8, "آدنوم": 0.9,
        "لیومیوم": 0.9, "فیبروم": 0.9,
        "لیپوم": 0.9, "همانژیوم": 0.9,
        "ملانوسیتیک": 0.9, "خال": 0.7, "نووس": 0.9,
        "اپیدرم": 0.8, "درم": 0.8,
        "اپیتلیال": 0.9, "مزانشیمال": 0.9,
        "نورواندوکرین": 0.9, "ژرم سل": 0.9, "تراتوم": 0.9,
        
        # Persian OCR variants
        "آسيب شناسي": 0.95, "پاتولوژي": 0.95,
        "بيوپسي": 0.95, "بدخيم": 0.9,
        
        # Mixed Persian-English
        "ihc رنگ‌آمیزی": 0.9, "biopsy نمونه": 0.9,
        "h&e رنگ": 0.9, "tnm staging": 0.9,
    },
}


classes = ["lab", "imaging", "pathology"]
priors = {"lab": 0.5, "imaging": 0.3, "pathology": 0.2}


class OCRReportClassifier:
    def __init__(
        self,
        default_likelihood: float = 0.001,
    ):
        self.classes = list(classes)
        # Validate priors
        if not np.isclose(sum(priors.values()), 1.0):
            raise ValueError("Priors must sum to 1.0")
        self.priors = priors
        self.keyword_likelihoods = keyword_likelihoods
        self.default_likelihood = default_likelihood

        # Precompute log priors for speed
        self._log_priors = {c: np.log(priors[c]) for c in self.classes}
        # Precompute log likelihoods for each class's keywords
        self._log_likelihoods = {}
        for c in self.classes:
            self._log_likelihoods[c] = {
                word: np.log(prob)
                for word, prob in keyword_likelihoods.get(c, {}).items()
            }
        self._log_default = np.log(default_likelihood)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Convert text to lowercase and extract alphabetic tokens."""
        return re.findall(r"[a-z]+", text.lower())

    def classify(self, text: str) -> Dict[str, float]:
        """
        Compute class probabilities for a single text.

        Returns
        -------
        dict
            Mapping from class label to probability (sums to 1).
        """
        tokens = self.tokenize(text)
        log_scores = {}
        for c in self.classes:
            log_prob = self._log_priors[c]
            # Add log likelihood for each token
            for token in tokens:
                log_prob += self._log_likelihoods[c].get(token, self._log_default)
            log_scores[c] = log_prob

        # Convert to probabilities via softmax (stable)
        log_score_array = np.array([log_scores[c] for c in self.classes])
        log_score_array -= np.max(log_score_array)  # stability
        probs = np.exp(log_score_array)
        probs /= probs.sum()

        probs = dict(zip(self.classes, probs))
        key = np.argmax(list(probs.values()))
        return { "name": list(probs.keys())[key], "score": np.max(list(probs.values()))}

    def classify_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Classify a list of texts; returns a list of dicts."""
        return [self.classify(t) for t in texts]

    def classify_as_dataframe(self, texts: List[str]) -> "pd.DataFrame":
        """
        Convenience: return a DataFrame with one column per class and
        the predicted class + confidence.
        Requires pandas.
        """
        import pandas as pd

        proba_list = self.classify_batch(texts)
        df = pd.DataFrame(proba_list)
        df["predicted_class"] = df.idxmax(axis=1)
        df["confidence"] = df.max(axis=1)
        # Reorder columns: predicted first, then class columns, then confidence
        cols = ["predicted_class"] + self.classes + ["confidence"]
        return df[cols]
