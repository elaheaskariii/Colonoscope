"""
Colonoscopy Blind Spot Prediction System (CBSPS)
Version 6.0 - Advanced Surgical Intelligence
Author: Medical AI Research Team
Description: Hyper-optimized anatomical analysis with surgical navigation intelligence
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm
import math
import io
import base64
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm, beta, gamma
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ==================== STREAMLIT PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="سیستم ناوبری هوشمند کولونوسکوپی پیشرفته",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CUSTOM THEME (YELLOW, PINK, GREEN) ====================
def set_custom_theme():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-yellow: #FFD700;
        --primary-pink: #FF69B4;
        --primary-green: #32CD32;
        --secondary-yellow: #FFF8DC;
        --secondary-pink: #FFF0F5;
        --secondary-green: #F0FFF0;
        --dark-text: #2C3E50;
        --light-text: #FFFFFF;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, var(--secondary-yellow) 0%, var(--secondary-pink) 50%, var(--secondary-green) 100%);
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--dark-text) !important;
        border-bottom: 3px solid var(--primary-pink);
        padding-bottom: 10px;
        margin-top: 20px !important;
    }

    /* Cards and containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary-pink);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-yellow) !important;
        color: var(--dark-text) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-yellow), var(--primary-pink));
        color: var(--dark-text) !important;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--secondary-green) 0%, var(--secondary-yellow) 100%);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(45deg, var(--secondary-pink), var(--secondary-yellow));
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid var(--primary-green);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--secondary-pink) !important;
        border-radius: 10px !important;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid var(--primary-pink);
    }

    /* Success */
    div[data-testid="stAlert"]:has(svg[aria-label="success"]) {
        background-color: rgba(50, 205, 50, 0.1) !important;
        border-left: 5px solid var(--primary-green) !important;
    }

    /* Warning */
    div[data-testid="stAlert"]:has(svg[aria-label="warning"]) {
        background-color: rgba(255, 215, 0, 0.1) !important;
        border-left: 5px solid var(--primary-yellow) !important;
    }

    /* Error */
    div[data-testid="stAlert"]:has(svg[aria-label="error"]) {
        background-color: rgba(255, 105, 180, 0.1) !important;
        border-left: 5px solid var(--primary-pink) !important;
    }

    /* Custom success class */
    .success-box {
        background: linear-gradient(45deg, #32CD32, #90EE90);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }

    /* Custom warning class */
    .warning-box {
        background: linear-gradient(45deg, #FFD700, #FFEC8B);
        color: var(--dark-text);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }

    /* Custom danger class */
    .danger-box {
        background: linear-gradient(45deg, #FF69B4, #FFB6C1);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, var(--primary-yellow), var(--primary-pink));
    }

    /* Data editor */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: 15px;
        border: 2px solid var(--primary-yellow);
    }

    /* Separator */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-yellow), var(--primary-pink), var(--primary-green));
        border-radius: 3px;
        margin: 30px 0;
    }

    /* Custom badges */
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 2px;
    }

    .badge-success {
        background: var(--primary-green);
        color: white;
    }

    .badge-warning {
        background: var(--primary-yellow);
        color: var(--dark-text);
    }

    .badge-danger {
        background: var(--primary-pink);
        color: white;
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: var(--dark-text);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)


set_custom_theme()


# ==================== ENUMS AND DATA CLASSES ====================
class FlexureAngle(Enum):
    ACUTE = "حاد"
    RIGHT = "قائمه"
    OBTUSE = "منفرجه"


class ColonLength(Enum):
    SHORT = "کوتاه"
    MEDIUM = "متوسط"
    LONG = "طویل"


class PelvicSize(Enum):
    SMALL = "کوچک"
    MEDIUM = "متوسط"
    LARGE = "بزرگ"


class SigmoidTortuosity(Enum):
    MILD = "خفیف"
    MODERATE = "متوسط"
    SEVERE = "شدید"


class ProcedureComplexity(Enum):
    SIMPLE = "ساده"
    MODERATE = "متوسط"
    COMPLEX = "پیچیده"
    VERY_COMPLEX = "بسیار پیچیده"


@dataclass
class PatientAnatomy:
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    colon_length: ColonLength
    splenic_flexure_angle: FlexureAngle
    hepatic_flexure_angle: FlexureAngle
    sigmoid_tortuosity: SigmoidTortuosity
    pelvic_size: PelvicSize
    has_sigmoid_loop: bool
    has_transverse_loop: bool
    prev_abdominal_surgery: bool
    chronic_constipation: bool


@dataclass
class BlindSpotWarning:
    location: str
    confidence: float
    reason: str
    estimated_encounter_time_min: float
    risk_level: str
    recommendations: List[str]
    anatomical_details: Dict[str, float]
    procedural_techniques: List[str]
    common_mistakes: List[str]


@dataclass
class SurgicalOptimization:
    optimal_path: List[str]
    time_allocation: Dict[str, float]
    instrument_selection: Dict[str, str]
    position_sequence: List[str]
    pressure_points: List[Tuple[str, str]]
    insufflation_strategy: Dict[str, float]


# ==================== ADVANCED ANATOMICAL MODEL ====================
class AdvancedAnatomicalModel:
    """
    مدل آناتومیک پیشرفته کولون
    این کلاس یک مدل چندپارامتری مبتنی بر بیومکانیک، آناتومی عملکردی و مدل‌سازی ریسک تجمعی است
    که برای پیش‌بینی سختی کولونوسکوپی استفاده می‌شود.
    """

    # ==================== COLON LENGTH PARAMETERS ====================
    # این مقادیر بر اساس اندازه‌گیری‌های رادیولوژیک کولون (CT Colonography)
    # مقدار اول: میانگین طول کولون (سانتی‌متر)
    # مقدار دوم: انحراف معیار طبیعی طول کولون
    COLON_LENGTH_VALUES = {
        ColonLength.SHORT: (80.0, 5.0),
        ColonLength.MEDIUM: (120.0, 10.0),
        ColonLength.LONG: (160.0, 15.0)
    }

    # ==================== FLEXURE ANGLE PARAMETERS ====================
    # زوایای خم‌های کولون (طحالی و کبدی)
    # این مقادیر از مطالعات آناتومیک و رادیولوژیک گرفته شده‌اند
    # زاویه کمتر → خم تیزتر → سختی عبور بیشتر
    ANGLE_VALUES = {
        FlexureAngle.ACUTE: (60.0, 10.0),
        FlexureAngle.RIGHT: (90.0, 5.0),
        FlexureAngle.OBTUSE: (120.0, 15.0)
    }

    # ==================== PELVIC SIZE FACTORS ====================
    # ضرایب اصلاح‌کننده آناتومیک لگن
    # مقدار اول: اثر بر آزادی حرکت کولون
    # مقدار دوم: محدودیت فضایی داخل لگن
    PELVIC_SIZE_FACTORS = {
        PelvicSize.SMALL: (0.8, 1.4),
        PelvicSize.MEDIUM: (1.0, 1.0),
        PelvicSize.LARGE: (1.2, 0.8)
    }

    # ==================== SIGMOID TORTUOSITY PARAMETERS ====================
    # پارامترهای پیچ‌خوردگی سیگموئید
    # curvature → میزان انحنای هندسی (Geometric Curvature)
    # torsion → پیچش فضایی سه‌بعدی (Torsion)
    # elasticity → خاصیت کشسانی دیواره کولون
    TORTUOSITY_PARAMS = {
        SigmoidTortuosity.MILD: {'curvature': 1.0, 'torsion': 0.5, 'elasticity': 0.8},
        SigmoidTortuosity.MODERATE: {'curvature': 1.8, 'torsion': 1.2, 'elasticity': 0.6},
        SigmoidTortuosity.SEVERE: {'curvature': 2.5, 'torsion': 2.0, 'elasticity': 0.4}
    }

    def __init__(self):
        self.patient_data = None
        self.anatomical_constants = self._initialize_constants()

    def _initialize_constants(self):
        """
        ثابت‌های بیومکانیکی کولون
        این ضرایب از مفاهیم فیزیولوژی بافت نرم، دینامیک احشایی و مقاومت جریان زیستی گرفته شده‌اند
        """
        return {
            # ضریب کشسانی دیواره کولون (Nonlinear Soft Tissue Elasticity)
            'colon_elasticity_coef': 0.0032,

            # تنش مزانتر (Mesenteric Tension) ناشی از تعلیق کولون
            'mesenteric_tension_factor': 0.0018,

            # مقاومت حرکات پریستالتیک (Viscoelastic Resistance)
            'peristaltic_resistance': 0.0045,

            # اصطکاک مخاط کولون (Mucosal Friction Coefficient)
            'mucosal_friction': 0.00085,

            # نوسانات عروقی (Vascular Pulsatility)
            'vascular_pulsatility': 0.0021,

            # مقاومت جریان لنفاوی مزانتر
            'lymphatic_flow_resistance': 0.0013
        }

    def calculate_bmi(self, patient_data: PatientAnatomy) -> float:
        """
        محاسبه BMI (Body Mass Index)
        فرمول استاندارد WHO:
        BMI = weight (kg) / height² (m²)

        تصحیح سنی:
        کاهش توده عضلانی و تغییر ترکیب بدن در سالمندان
        """
        bmi = patient_data.weight_kg / ((patient_data.height_cm / 100) ** 2)

        # Age-related body composition correction
        if patient_data.age > 60:
            age_correction = 1 - ((patient_data.age - 60) * 0.005)
            bmi *= age_correction

        return bmi

    def calculate_body_surface_area(self, patient_data: PatientAnatomy) -> float:
        """
        محاسبه سطح بدن (Body Surface Area)
        فرمول کلاسیک DuBois & DuBois (1916)
        یکی از معتبرترین روابط تجربی در پزشکی بالینی
        """
        bsa = 0.007184 * (patient_data.height_cm ** 0.725) * (patient_data.weight_kg ** 0.425)
        return bsa

    def sigmoid_difficulty_score(self, patient_data: PatientAnatomy) -> Dict[str, float]:
        """
        امتیاز سختی سیگموئید
        مدل ترکیبی چندعامله (Multiplicative Risk Model)
        الهام‌گرفته از:
        - Biomechanical stress models
        - Probabilistic anatomical variability
        """

        tort_params = self.TORTUOSITY_PARAMS[patient_data.sigmoid_tortuosity]

        # ==================== 1. CURVATURE FACTOR ====================
        # مدل‌سازی توزیع انحنای آناتومیک با توزیع Beta
        # Beta distribution برای پارامترهای محدودشده (0–1) بسیار رایج است
        curvature_base = tort_params['curvature']

        # وجود loop باعث افزایش غیرخطی انحنا می‌شود
        if patient_data.has_sigmoid_loop:
            curvature_base *= 1.7

        # طول زیاد کولون → انحنای بیشتر
        if patient_data.colon_length == ColonLength.LONG:
            curvature_base *= 1.3

        curvature_factor = beta.pdf(curvature_base / 3.0, 2, 5) * 2.5

        # ==================== 2. TORSION FACTOR ====================
        # پیچش فضایی کولون (3D torsion)
        # وابسته به اندازه لگن و BMI
        torsion_base = tort_params['torsion']
        pelvic_constraint = self.PELVIC_SIZE_FACTORS[patient_data.pelvic_size][1]

        torsion_factor = torsion_base * pelvic_constraint * (
            1 + 0.15 * self.calculate_bmi(patient_data) / 25
        )

        # ==================== 3. ELASTICITY FACTOR ====================
        # کاهش الاستیسیته با افزایش سن (Degenerative tissue changes)
        elasticity_factor = tort_params['elasticity']
        age_effect = max(0, 1 - (patient_data.age - 40) * 0.008)
        elasticity_factor *= age_effect

        # ==================== 4. CLINICAL FACTOR ====================
        # اثرات بالینی تجمعی (Chronic Load Model)
        clinical_factor = 1.0

        if patient_data.chronic_constipation:
            constipation_severity = 1 + (patient_data.age * 0.003)
            clinical_factor *= constipation_severity

        if patient_data.prev_abdominal_surgery:
            adhesion_effect = 1.25 + (self.calculate_bmi(patient_data) / 35 * 0.1)
            clinical_factor *= adhesion_effect

        # ==================== 5. MESENTERIC TENSION ====================
        # تنش تعلیقی کولون توسط مزانتر
        mesenteric_tension = self.anatomical_constants['mesenteric_tension_factor']

        if patient_data.gender == "female":
            mesenteric_tension *= 1.15 # تفاوت آناتومی لگن زنان

        # ==================== 6. FINAL NONLINEAR AGGREGATION ====================
        # مدل ضربی غیرخطی با توان‌های تنظیم‌شده
        # این ساختار مشابه مدل‌های Biomechanical Load Aggregation است
        difficulty_score = (
            curvature_factor ** 1.3 *
            torsion_factor ** 1.1 *
            (1 / elasticity_factor) ** 0.9 *
            clinical_factor ** 1.2 *
            (1 + mesenteric_tension * self.calculate_bmi(patient_data)) ** 1.15
        )

        # ==================== BIOLOGICAL NOISE ====================
        # نویز طبیعی زیستی (Inter-individual variability)
        biological_variation = 1 + np.random.normal(0, 0.08)
        difficulty_score *= biological_variation

        return {
            'total_score': min(difficulty_score, 10.0),
            'curvature_factor': curvature_factor,
            'torsion_factor': torsion_factor,
            'elasticity_score': elasticity_factor,
            'clinical_impact': clinical_factor,
            'mesenteric_tension': mesenteric_tension
        }

    def flexure_difficulty_analysis(self, angle: FlexureAngle, is_hepatic: bool = False) -> Dict[str, float]:
        """
        تحلیل سختی خم‌های کولون (Splenic / Hepatic Flexure)

        این مدل بر اساس:
        - تحلیل تنش زاویه‌ای (Angular Stress Analysis)
        - محدودیت دید آندوسکوپ (Visual Obstruction)
        - قیود آناتومیک اطراف (Ligaments & Organs)
        - کشش عروقی (Vascular Traction)
        طراحی شده است.
        """

        # استخراج میانگین و پراکندگی زاویه خم
        # angle_mean : زاویه غالب خم (درجه)
        # angle_std : تغییرپذیری آناتومیک زاویه
        angle_mean, angle_std = self.ANGLE_VALUES[angle]

        # ==================== 1. ANGULAR STRESS ====================
        # تحلیل تنش زاویه‌ای بر اساس انحراف از 90 درجه (Neutral Axis)
        # 90 درجه به‌عنوان حالت مرجع مکانیکی در نظر گرفته می‌شود
        if angle_mean <= 70:
            # خم‌های حاد → افزایش شدید تنش و کاهش دید
            angular_stress = (90 - angle_mean) * 0.035
            visual_obstruction = (90 - angle_mean) * 0.025
        elif angle_mean <= 110:
            # محدوده فیزیولوژیک
            angular_stress = abs(90 - angle_mean) * 0.02
            visual_obstruction = abs(90 - angle_mean) * 0.015
        else:
            # خم‌های باز → تنش کمتر ولی هدایت سخت‌تر
            angular_stress = (angle_mean - 90) * 0.015
            visual_obstruction = (angle_mean - 90) * 0.01

        # ==================== 2. POSITIONAL ANATOMICAL FACTOR ====================
        # اثر ساختارهای مجاور آناتومیک
        positional_factor = 1.0

        # ضرایب بر اساس آناتومی شناخته‌شده خم‌ها
        anatomical_variants = {
            'splenic': {
                'lienorenal_ligament': 1.2, # لیگامان طحالی–کلیوی
                'pancreatic_proximity': 1.1 # نزدیکی پانکراس
            },
            'hepatic': {
                'hepatoduodenal_ligament': 1.4, # لیگامان کبدی–دوازدهه
                'gallbladder_proximity': 1.15 # نزدیکی کیسه صفرا
            }
        }

        if is_hepatic:
            variant = anatomical_variants['hepatic']
            positional_factor *= (
                variant['hepatoduodenal_ligament'] *
                variant['gallbladder_proximity']
            )

            # اثر وزن کبد در بیماران با BMI بالا
            bmi_factor = 1 + (0.01 * max(0, 30)) # ثابت محافظه‌کارانه
            positional_factor *= bmi_factor
        else:
            variant = anatomical_variants['splenic']
            positional_factor *= (
                variant['lienorenal_ligament'] *
                variant['pancreatic_proximity']
            )

        # ==================== 3. VASCULAR TRACTION ====================
        # کشش واردشده توسط عروق اصلی
        vascular_traction = 1.0

        if is_hepatic:
            # شریان کبدی مشترک
            vascular_traction += 0.25
        else:
            # عروق طحالی
            vascular_traction += 0.15

        # ==================== 4. SPATIAL CONSTRAINT ====================
        # محدودیت فضایی ناشی از تغییرپذیری زاویه
        spatial_constraint = 1 + (angle_std * 0.03)

        # ==================== 5. FINAL AGGREGATION ====================
        # مدل ضربی غیرخطی (Nonlinear Stress Aggregation)
        difficulty_score = (
            (1 + angular_stress) ** 1.4 *
            (1 + visual_obstruction) ** 1.2 *
            positional_factor ** 1.3 *
            vascular_traction ** 1.15 *
            spatial_constraint ** 1.1
        )

        return {
            'total_score': min(difficulty_score, 8.0),
            'angular_stress': angular_stress,
            'visual_obstruction': visual_obstruction,
            'positional_factor': positional_factor,
            'vascular_traction': vascular_traction,
            'spatial_constraint': spatial_constraint
        }

    def estimate_procedure_dynamics(self, patient_data: PatientAnatomy) -> Dict[str, float]:
        """
        برآورد دینامیک زمانی پروسیجر کولونوسکوپی

        این مدل از نوع Time–Complexity Coupled Model است
        که زمان انجام هر فاز را به پیچیدگی آناتومیک مرتبط می‌کند.
        """

        # امتیازات پیچیدگی نواحی مختلف
        sigmoid_scores = self.sigmoid_difficulty_score(patient_data)
        splenic_scores = self.flexure_difficulty_analysis(
            patient_data.splenic_flexure_angle, False
        )
        hepatic_scores = self.flexure_difficulty_analysis(
            patient_data.hepatic_flexure_angle, True
        )

        # ==================== BASE TIMES ====================
        # زمان‌های پایه از داده‌های بالینی کولونوسکوپی
        base_times = {
            'insertion_base': 8.5, # ورود آندوسکوپ
            'withdrawal_base': 6.0, # خروج
            'examination_base': 10.0, # بررسی مخاط
            'therapeutic_base': 5.0 if any([
                patient_data.has_sigmoid_loop,
                patient_data.has_transverse_loop
            ]) else 2.0
        }

        # ==================== LENGTH FACTOR ====================
        # اثر طول کولون
        length_mean, length_std = self.COLON_LENGTH_VALUES[patient_data.colon_length]
        length_factor = 1 + (length_mean - 120) / 120 * 0.4

        # ==================== BMI FACTOR ====================
        bmi = self.calculate_bmi(patient_data)
        bmi_factor = 1 + max(0, (bmi - 25) * 0.015)

        # ==================== AGE FACTOR ====================
        age_factor = 1 + max(0, (patient_data.age - 50) * 0.01)

        # ==================== COMPLEXITY ACCUMULATION ====================
        # ترکیب وزن‌دار پیچیدگی نواحی
        complexity_accumulation = (
            sigmoid_scores['total_score'] * 0.35 +
            splenic_scores['total_score'] * 0.25 +
            hepatic_scores['total_score'] * 0.30 +
            (2.0 if patient_data.has_transverse_loop else 1.0) * 0.10
        )

        # ==================== FINAL TIME ESTIMATION ====================
        total_insertion = base_times['insertion_base'] * length_factor * bmi_factor * age_factor * (
            1 + complexity_accumulation * 0.3
        )

        total_withdrawal = base_times['withdrawal_base'] * length_factor * (
            1 + complexity_accumulation * 0.4
        )

        total_examination = base_times['examination_base'] * (
            1 + complexity_accumulation * 0.5
        )

        total_therapeutic = base_times['therapeutic_base'] * (
            1 + complexity_accumulation * 0.25
        )

        total_time = (
            total_insertion +
            total_withdrawal +
            total_examination +
            total_therapeutic
        )

        # ==================== SAFETY CAP ====================
        # محدودیت ایمنی بالینی
        total_time = min(total_time, 180)

        return {
            'total_time_min': total_time,
            'insertion_time': total_insertion,
            'withdrawal_time': total_withdrawal,
            'examination_time': total_examination,
            'therapeutic_time': total_therapeutic,
            'complexity_score': complexity_accumulation,
            'bmi_factor': bmi_factor,
            'age_factor': age_factor,
            'length_factor': length_factor
        }

    def predict_anatomical_risks(self, patient_data: PatientAnatomy) -> List[BlindSpotWarning]:
        """
        پیش‌بینی ریسک‌های آناتومیک کولونوسکوپی (Blind Spots & Difficult Segments)

        این تابع یک Risk-Stratification Model چندسطحی است
        که از خروجی مدل‌های سختی آناتومیک و دینامیک زمانی استفاده می‌کند
        و نواحی پرریسک را با سطح اطمینان (Confidence) گزارش می‌دهد.
        """

        warnings = []

        # ==================== PROCEDURE DYNAMICS ====================
        # استخراج زمان کل پروسیجر برای توزیع نسبی ریسک
        procedure_dynamics = self.estimate_procedure_dynamics(patient_data)
        total_time = procedure_dynamics['total_time_min']

        # ==================== 1. SIGMOID ANALYSIS ====================
        # تحلیل سیگموئید به‌عنوان شایع‌ترین ناحیه سخت
        sigmoid_scores = self.sigmoid_difficulty_score(patient_data)

        if sigmoid_scores['total_score'] >= 2.0:
            # Confidence بر اساس مدل احتمال غیرخطی (Power-law scaling)
            confidence = min(
                0.3 + 0.6 * (sigmoid_scores['total_score'] / 6.0) ** 1.4,
                0.97
            )

            # تخمین سهم زمانی سیگموئید از کل پروسیجر
            time_estimate = total_time * 0.15 * (
                1 + 0.35 * sigmoid_scores['total_score'] / 6.0
            )

            warnings.append(
                self._create_detailed_sigmoid_warning(
                    confidence,
                    time_estimate,
                    sigmoid_scores,
                    patient_data
                )
            )

        # ==================== 2. SPLENIC FLEXURE ====================
        # تحلیل خم طحالی (Splenic Flexure)
        splenic_scores = self.flexure_difficulty_analysis(
            patient_data.splenic_flexure_angle,
            False
        )

        if splenic_scores['total_score'] >= 1.8:
            confidence = min(
                0.35 + 0.5 * (splenic_scores['total_score'] / 5.0) ** 1.3,
                0.95
            )

            time_estimate = total_time * 0.18 * (
                1 + 0.3 * splenic_scores['total_score'] / 5.0
            )

            warnings.append(
                self._create_detailed_flexure_warning(
                    "خم طحالی",
                    confidence,
                    time_estimate,
                    splenic_scores,
                    False
                )
            )

        # ==================== 3. HEPATIC FLEXURE ====================
        # تحلیل خم کبدی (Hepatic Flexure)
        hepatic_scores = self.flexure_difficulty_analysis(
            patient_data.hepatic_flexure_angle,
            True
        )

        if hepatic_scores['total_score'] >= 2.0:
            confidence = min(
                0.4 + 0.55 * (hepatic_scores['total_score'] / 5.5) ** 1.35,
                0.96
            )

            time_estimate = total_time * 0.22 * (
                1 + 0.32 * hepatic_scores['total_score'] / 5.5
            )

            warnings.append(
                self._create_detailed_flexure_warning(
                    "خم کبدی",
                    confidence,
                    time_estimate,
                    hepatic_scores,
                    True
                )
            )

        # ==================== 4. TRANSVERSE COLON ====================
        # تحلیل کولون عرضی (Loop formation & Gravity effect)
        if patient_data.has_transverse_loop:
            transverse_score = self._calculate_transverse_complexity(patient_data)

            # Confidence خطی با شدت پیچیدگی
            confidence = 0.55 + 0.3 * transverse_score

            # سهم زمانی از کل پروسیجر
            time_estimate = total_time * 0.16 * (
                1 + 0.28 * transverse_score
            )

            warnings.append(
                self._create_detailed_transverse_warning(
                    confidence,
                    time_estimate,
                    transverse_score,
                    patient_data
                )
            )

        # ==================== 5. RECTOSIGMOID JUNCTION ====================
        # اتصال رکتوسیگموئید به‌عنوان ناحیه با محدودیت فضایی لگن
        if (
            patient_data.pelvic_size == PelvicSize.SMALL and
            patient_data.sigmoid_tortuosity != SigmoidTortuosity.MILD
        ):
            rectosigmoid_score = self._calculate_rectosigmoid_complexity(patient_data)

            confidence = 0.65 + 0.25 * rectosigmoid_score

            time_estimate = total_time * 0.08 * (
                1 + 0.22 * rectosigmoid_score
            )

            warnings.append(
                self._create_detailed_rectosigmoid_warning(
                    confidence,
                    time_estimate,
                    rectosigmoid_score,
                    patient_data
                )
            )

        # ==================== 6. BMI-RELATED RISKS ====================
        # افزایش ریسک به دلیل چاقی (Increased Visceral Fat & Reduced Maneuverability)
        bmi = self.calculate_bmi(patient_data)

        if bmi > 30:
            bmi_warning = self._create_bmi_related_warning(
                bmi,
                patient_data,
                total_time
            )
            warnings.append(bmi_warning)

        # ==================== 7. AGE-RELATED RISKS ====================
        # تغییرات وابسته به سن (Tissue fragility, reduced elasticity)
        if patient_data.age > 65:
            age_warning = self._create_age_related_warning(
                patient_data,
                total_time
            )
            warnings.append(age_warning)

        # ==================== SORTING ====================
        # مرتب‌سازی هشدارها بر اساس Confidence (Descending)
        warnings.sort(
            key=lambda x: x.confidence,
            reverse=True
        )

        return warnings

    def _calculate_transverse_complexity(self, patient_data: PatientAnatomy) -> float:
        """
        محاسبه پیچیدگی کولون عرضی

        این مدل اثرات زیر را ترکیب می‌کند:
        - طول کولون
        - BMI
        - سابقه جراحی
        - اثر گرانش بر حلقه‌های آزاد
        """

        complexity = 1.0

        # اثر طول زیاد کولون
        if patient_data.colon_length == ColonLength.LONG:
            complexity *= 1.4

        # اثر چاقی
        bmi = self.calculate_bmi(patient_data)
        if bmi > 30:
            complexity *= 1.3

        # اثر چسبندگی‌های جراحی
        if patient_data.prev_abdominal_surgery:
            complexity *= 1.25

        # اثر گرانش (Gravity-induced sagging)
        gravity_effect = 1.15

        return min(complexity * gravity_effect, 3.0)

    def _calculate_rectosigmoid_complexity(self, patient_data: PatientAnatomy) -> float:
        """
        محاسبه پیچیدگی اتصال رکتوسیگموئید

        این ناحیه به‌شدت تحت تأثیر:
        - اندازه لگن
        - پیچ‌خوردگی سیگموئید
        - تفاوت‌های جنسی آناتومیک
        """

        complexity = 1.0

        # ==================== PELVIC SIZE EFFECT ====================
        pelvic_factors = {
            PelvicSize.SMALL: 1.4, # محدودیت فضایی شدید
            PelvicSize.MEDIUM: 1.0,
            PelvicSize.LARGE: 0.8
        }
        complexity *= pelvic_factors[patient_data.pelvic_size]

        # ==================== TORTUOSITY EFFECT ====================
        tort_factors = {
            SigmoidTortuosity.MILD: 1.0,
            SigmoidTortuosity.MODERATE: 1.3,
            SigmoidTortuosity.SEVERE: 1.6
        }
        complexity *= tort_factors[patient_data.sigmoid_tortuosity]

        # ==================== GENDER EFFECT ====================
        # تفاوت‌های آناتومیک لگن زنان
        if patient_data.gender == "female":
            complexity *= 1.2

        return min(complexity, 2.5)

    def _create_detailed_sigmoid_warning(self, confidence: float, time_estimate: float,
                                         scores: Dict[str, float], patient_data: PatientAnatomy) -> BlindSpotWarning:
        """ایجاد هشدار دقیق برای سیگموئید"""

        risk_level = "high" if scores['total_score'] >= 4.0 else "medium" if scores['total_score'] >= 2.5 else "low"

        # توصیه‌های تخصصی پیشرفته
        recommendations = [
            "با چرخش‌های تدریجی ۹۰ درجه‌ای به جای چرخش‌های ناگهانی (Torque-steering) استفاده از تکنیک",
            "اعمال فشار شکمی کنترل شده در ربع چپ تحتانی با کمک دستیار برای کاهش حلقه‌های سیگموئید",
            "برای باز کردن پیچ‌های تند: قلاب کردن نوک اسکوپ در خم و کشیدن به عقب Hook-and-pull بکار گیری تکنیک",
            "برای تغییر مسیر آناتومیک Prone یا Left lateral چرخش بیمار به حالت",
            "Straightening: پس از عبور از سیگموئید، کشیدن ملایم برای صاف کردن حلقه‌ها",
            "برای بهبود دید و کاهش اسپاسم (Water immersion) بکارگیری آب به جای هوا",
            "در صورت پیچ خوردگی شدید و عدم پیشروی با اسکوپ استاندارد Pediatric colonoscope استفاده از",
            "در عبور اولیه برای جلوگیری از ایجاد حلقه زودرس Valves of Houston توجه ویژه به"
        ]

        if patient_data.has_sigmoid_loop:
            recommendations.extend([
                "استفاده از تکنیک Alpha maneuver: تشکیل حلقه آلفا برای عبور از سیگموئید پیچیده",
                "بکارگیری Abdominal pressure با دستیار در ربع چپ تحتانی و ساکرال مدیال",
                "محدود کردن Air insufflation به حداقل ضروری برای جلوگیری از اتساع بیش از حد"
            ])

        if patient_data.chronic_constipation:
            recommendations.extend([
                "زمان بیشتر برای بررسی دقیق Haustra ها به دلیل احتمال بالای fecal residue",
                "شستشوی کامل با آب قبل از بررسی مناطق مشکوک",
                "توجه ویژه به تغییرات مخاطی مرتبط با یبوست مزمن"
            ])

        # تکنیک‌های پروسیجری
        procedural_techniques = [
            "Single-handed torque technique",
            " Two-person technique در موارد پیچیده ( one person-torque،  other person-insertion)",
            "(Jiggling technique) برای عبور از مناطق اسپاسمودیک",
            "Pull-back and straighten هر ۱۰-۱۵ سانتی‌متر پیشروی"
        ]

        # خطاهای رایج
        common_mistakes = [
            ".یا آلفا می شود N بیش از حد که منجر به تشکیل حلقه torque اعمال",
            " استInsertion سریع بدون توجه به مقاومت که باعث perforation",
            "در مواجهه با مقاومت Position change عدم استفاده از",
            " که دید را بدتر می‌کندOver-insufflation تکنیک",
            "withdrawal عدم انجام (straighteningمناسب برای)"
        ]

        # جزئیات آناتومیک
        anatomical_details = {
            'curvature_score': scores['curvature_factor'],
            'torsion_index': scores['torsion_factor'],
            'elasticity_ratio': scores['elasticity_score'],
            'mesenteric_tension': scores['mesenteric_tension'],
            'predicted_loops': 2 if patient_data.has_sigmoid_loop else 1,
            'spasm_probability': 0.4 if patient_data.chronic_constipation else 0.2
        }

        return BlindSpotWarning(
            location="کولون سیگموئید",
            confidence=confidence,
            reason=f"پیچیدگی آناتومیک بالا با امتیاز {scores['total_score']:.2f} - ترکیب انحنای {scores['curvature_factor']:.2f} و پیچش {scores['torsion_factor']:.2f}",
            estimated_encounter_time_min=time_estimate,
            risk_level=risk_level,
            recommendations=recommendations,
            anatomical_details=anatomical_details,
            procedural_techniques=procedural_techniques,
            common_mistakes=common_mistakes
        )

    def _create_detailed_flexure_warning(self, location: str, confidence: float, time_estimate: float,
                                         scores: Dict[str, float], is_hepatic: bool) -> BlindSpotWarning:
        """ایجاد هشدار دقیق برای خم‌ها"""

        angle_type = "حاد" if scores['angular_stress'] > 0.3 else "تیز" if scores['angular_stress'] > 0.15 else "ملایم"
        risk_level = "high" if scores['total_score'] >= 4.5 else "medium" if scores['total_score'] >= 2.5 else "low"

        # توصیه‌های تخصصی بر اساس نوع خم
        if location == "خم طحالی":
            recommendations = [
                "برای بهره‌گیری از نیروی جاذبه در عبور از خم Right lateral position استفاده از",
                "تکنیک Slide-by: مستقیم push حرکت کناری اسکوپ به جای",
                "در ربع چپ فوقانی توسط دستیار Abdominal pressure اعمال",
                "قبل از ورود به خم Shaft shortening technique استفاده از",
                "بکارگیری External compression در ناحیه splenic flexure",
                "در حین عبور برای کاهش تنش ها Air insufflation محدود کردن",
                "در صورت زاویه بسیار حاد Pediatric colonoscope استفاده از",
                "تکنیک Two-stage approach: ابتدا به کولون نزولی، سپس عبور از خم"
            ]

            procedural_techniques = [
                "Splenic flexure straightening technique",
                "Right lateral position with clockwise torque",
                "Push-and-pull method with minimal air"
            ]

            common_mistakes = [
                "که باعث آسیب طحال می‌شود Forceful pushing حالت",
                "مناسب position change عدم استفاده از",
                "در حین عبور Over-torquing حالت",
                "که نشانه مقاومت است patient discomfort عدم توجه به"
            ]

        else:  # خم کبدی
            recommendations = [
                "برای عبور آسان‌تر Left lateral position یا Supine position استفاده از",
                "برای شل کردن خمیدگی کبد Jiggling روش",
                "در ربع راست فوقانی و زیر دنده‌ها Abdominal pressure اعمال",
                "برای کاهش اسپاسم و بهبود دید Water immersion استفاده از",
                "برای کاهش حلقه‌ها Suction-and-pull تکنیک",
                "در صورت موجود بودن Variable stiffness colonoscope استفاده از",
                "توجه به تنگی احتمالی ناشی از فشار کبد در بیماران چاق"
            ]

            procedural_techniques = [
                "Hepatic flexure negotiation with aspiration",
                "Left lateral position with anticlockwise rotation",
                "Stepwise advancement with frequent position changes"
            ]

            common_mistakes = [
                "که خطر سوراخ شدن داردPushing against resistance حالت",
                "percussionدر liver dullness عدم توجه به",
                "که عبور را سخت‌تر می‌کندOver-insufflationحالت",
                "عدم استفاده از ساکشنtension برای کاهش"
            ]

        # جزئیات آناتومیک
        anatomical_details = {
            'angular_stress': scores['angular_stress'],
            'visual_obstruction': scores['visual_obstruction'],
            'positional_factor': scores['positional_factor'],
            'vascular_traction': scores['vascular_traction'],
            'spatial_constraint': scores['spatial_constraint'],
            'is_hepatic': is_hepatic
        }

        return BlindSpotWarning(
            location=location,
            confidence=confidence,
            reason=f"پیکربندی خم {angle_type} با امتیاز پیچیدگی {scores['total_score']:.2f} - استرس زاویه‌ای {scores['angular_stress']:.2f} و محدودیت فضایی {scores['spatial_constraint']:.2f}",
            estimated_encounter_time_min=time_estimate,
            risk_level=risk_level,
            recommendations=recommendations,
            anatomical_details=anatomical_details,
            procedural_techniques=procedural_techniques,
            common_mistakes=common_mistakes
        )

    def _create_detailed_transverse_warning(self, confidence: float, time_estimate: float,
                                            score: float, patient_data: PatientAnatomy) -> BlindSpotWarning:
        """ایجاد هشدار دقیق برای کولون عرضی"""

        risk_level = "high" if score >= 2.0 else "medium"

        recommendations = [
            "استفاده از Abdominal pressure در ناحیه اپی‌گاستر و زیر دنده‌ها",
            "تکنیک Position change: از Supine به Prone یا Left lateral",
            "بکارگیری Hook-and-pull technique برای کاهش حلقه‌های عرضی",
            "استفاده از Variable stiffness colonoscope در صورت موجود بودن",
            "تکنیک Suction-and-pull برای کاهش اتساع و حلقه‌ها",
            "اعمال تکنیک Two-person برای کنترل بهتر در بیماران چاق",
            "محدود کردن Air insufflation و استفاده از CO2 برای کاهش discomfort",
            "توجه ویژه به ناحیه Mid-transverse colon که بیشترین احتمال loop formation دارد"
        ]

        if patient_data.prev_abdominal_surgery:
            recommendations.extend([
                "احتیاط بیشتر در بیماران با سابقه جراحی upper abdominal",
                "جستجوی چسبندگی‌های احتمالی در ناحیه transverse colon",
                "استفاده از Pediatric colonoscope در صورت شک به چسبندگی شدید"
            ])

        procedural_techniques = [
            "Transverse colon straightening with patient position change",
            "Epigastric pressure application technique",
            "Step-by-step advancement with frequent suction"
        ]

        common_mistakes = [
            "Pushing against gravity بدون position change",
            "عدم استفاده از abdominal pressure در بیماران با BMI بالا",
            "Over-insufflation که باعث افزایش حلقه‌ها می‌شود",
            "عدم توجه به patient position که کلید عبور از transverse است"
        ]

        anatomical_details = {
            'loop_severity': score,
            'gravity_sensitivity': 0.8,
            'position_dependency': 0.9,
            'bmi_impact': self.calculate_bmi(patient_data) / 30,
            'surgery_effect': 1.3 if patient_data.prev_abdominal_surgery else 1.0
        }

        return BlindSpotWarning(
            location="کولون عرضی",
            confidence=confidence,
            reason=f"وجود حلقه‌های عرضی با امتیاز پیچیدگی {score:.2f} - حساسیت بالا به پوزیشن و جاذبه",
            estimated_encounter_time_min=time_estimate,
            risk_level=risk_level,
            recommendations=recommendations,
            anatomical_details=anatomical_details,
            procedural_techniques=procedural_techniques,
            common_mistakes=common_mistakes
        )

    def _create_detailed_rectosigmoid_warning(self, confidence: float, time_estimate: float,
                                              score: float, patient_data: PatientAnatomy) -> BlindSpotWarning:
        """ایجاد هشدار دقیق برای اتصال رکتوسیگموئید"""

        risk_level = "medium" if score >= 1.5 else "low"

        recommendations = [
            "استفاده از Minimal air technique در طول ورود اولیه",
            "انجام Retroflexion دقیق در رکتوم برای بررسی کامل valves",
            "تکنیک Torque-and-lift برای عبور از rectosigmoid junction",
            "بکارگیری Water immersion برای بهبود دید و کاهش اسپاسم",
            "توجه ویژه به Valves of Houston در بررسی retroflexion",
            "استفاده از Pediatric colonoscope در صورت تنگی یا اسپاسم شدید",
            "تکنیک Jiggling برای شل کردن عضلات puborectalis",
            "Position change به Left lateral در صورت مشکل در عبور اولیه"
        ]

        if patient_data.gender == "female":
            recommendations.extend([
                "احتیاط بیشتر در زنان با سابقه هیسترکتومی",
                "توجه به چسبندگی‌های احتمالی ناحیه pouch of Douglas",
                "بررسی دقیق anterior wall در retroflexion"
            ])

        procedural_techniques = [
            "Rectosigmoid negotiation with minimal air",
            "Complete retroflexion examination technique",
            "Jiggling method for puborectalis relaxation"
        ]

        common_mistakes = [
            "Over-insufflation در ابتدای پروسیجر",
            "عدم انجام retroflexion کامل",
            "Forceful pushing در rectosigmoid junction",
            "عدم توجه به valves of Houston"
        ]

        anatomical_details = {
            'pelvic_constraint': self.PELVIC_SIZE_FACTORS[patient_data.pelvic_size][1],
            'tortuosity_effect': 1.5 if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE else 1.2,
            'gender_factor': 1.2 if patient_data.gender == "female" else 1.0,
            'spasm_tendency': 0.6 if patient_data.chronic_constipation else 0.3
        }

        return BlindSpotWarning(
            location="اتصال رکتوسیگموئید",
            confidence=confidence,
            reason=f"ترکیب اندازه لگن {patient_data.pelvic_size.value} و پیچ خوردگی {patient_data.sigmoid_tortuosity.value} با امتیاز پیچیدگی {score:.2f}",
            estimated_encounter_time_min=time_estimate,
            risk_level=risk_level,
            recommendations=recommendations,
            anatomical_details=anatomical_details,
            procedural_techniques=procedural_techniques,
            common_mistakes=common_mistakes
        )

    def _create_bmi_related_warning(self, bmi: float, patient_data: PatientAnatomy,
                                    total_time: float) -> BlindSpotWarning:
        """هشدارهای مرتبط با BMI بالا"""

        bmi_category = "چاقی درجه ۱" if 30 <= bmi < 35 else "چاقی درجه ۲" if 35 <= bmi < 40 else "چاقی درجه ۳"

        recommendations = [
            "استفاده از Longer colonoscope (160-170cm) برای دسترسی به سکوم",
            "بکارگیری Two-person technique برای کنترل بهتر",
            "استفاده از Abdominal pressure گسترده توسط دستیار",
            "Position change مکرر برای استفاده از جاذبه",
            "تخصیص زمان اضافی برای پروسیجر (حداقل ۲۰٪ بیشتر)",
            "استفاده از CO2 insufflation برای تحمل بهتر بیمار",
            "مانیتورینگ دقیق saturation و hemodynamics",
            "در نظر گرفتن Sedation depth مناسب برای بیماران چاق"
        ]

        if bmi >= 35:
            recommendations.extend([
                "در نظر گرفتن امکان استفاده از Endoscopic overtube",
                "استفاده از Pediatric colonoscope با قطر کمتر در صورت نیاز",
                "هماهنگی با متخصص بیهوشی برای مدیریت airway",
                "تهیه Equipment اضافی (scope طولانی‌تر، overtube)"
            ])

        procedural_techniques = [
            "Bariatric colonoscopy positioning protocol",
            "Two-person technique with coordinated movements",
            "Extended procedure time management"
        ]

        common_mistakes = [
            "عدم استفاده از colonoscope به اندازه کافی طولانی",
            "عدم درخواست کمک دستیار برای فشار شکمی",
            "تخصیص وقت ناکافی برای پروسیجر",
            "عدم توجه به positioning challenges"
        ]

        anatomical_details = {
            'bmi_value': bmi,
            'bmi_category': bmi_category,
            'mesenteric_fat_effect': min(1.0 + (bmi - 30) * 0.05, 1.3),
            'scope_length_requirement': 160 if bmi >= 35 else 150,
            'time_multiplier': 1 + (bmi - 25) * 0.02
        }

        return BlindSpotWarning(
            location="ملاحظات BMI بالا",
            confidence=0.7,
            reason=f"{bmi_category} (BMI: {bmi:.1f}) - افزایش چربی مزانتر و چالش‌های پوزیشن‌دهی",
            estimated_encounter_time_min=total_time * 0.1,
            risk_level="medium",
            recommendations=recommendations,
            anatomical_details=anatomical_details,
            procedural_techniques=procedural_techniques,
            common_mistakes=common_mistakes
        )

    def _create_age_related_warning(self, patient_data: PatientAnatomy, total_time: float) -> BlindSpotWarning:
        """هشدارهای مرتبط با سن بالا"""

        age_group = "سالمند" if 65 <= patient_data.age < 75 else "کهنسال" if patient_data.age >= 75 else "میانسال"

        recommendations = [
            "کاهش میزان Sedation برای جلوگیری از عوارض قلبی عروقی",
            "استفاده از Minimal air یا CO2 insufflation برای کاهش discomfort",
            "مانیتورینگ دقیق vital signs در طول پروسیجر",
            "توجه به داروهای ضد انعقاد و ضد پلاکت بیمار",
            "بررسی سابقه بیماری‌های قلبی و ریوی قبل از پروسیجر",
            "استفاده از Smaller diameter scope در صورت وجود دیورتیکولوز",
            "احتیاط بیشتر در بیماران با سابقه بیماری عروق مغزی",
            "زمان ریکاوری طولانی‌تر در نظر گرفته شود"
        ]

        if patient_data.age >= 75:
            recommendations.extend([
                "در نظر گرفتن امکان incomplete colonoscopy",
                "توجه ویژه به thin-walled colon در سالمندان",
                "کاهش فشار آب در irrigation",
                "احتیاط در biopsy به دلیل risk خونریزی بالاتر"
            ])

        procedural_techniques = [
            "Geriatric colonoscopy protocol",
            "Minimal sedation technique",
            "Careful insufflation management"
        ]

        common_mistakes = [
            "Over-sedation در بیماران سالمند",
            "عدم توجه به داروهای ضد انعقاد",
            "Over-insufflation در colon آتروفیک",
            "عدم در نظر گرفتن زمان ریکاوری طولانی"
        ]

        anatomical_details = {
            'age': patient_data.age,
            'age_group': age_group,
            'colon_atrophy_factor': 1 + (patient_data.age - 65) * 0.01,
            'diverticulosis_probability': min(0.3 + (patient_data.age - 65) * 0.015, 0.6),
            'sedation_sensitivity': 1.5 if patient_data.age >= 75 else 1.2
        }

        return BlindSpotWarning(
            location="ملاحظات سنی",
            confidence=0.65,
            reason=f"بیمار {age_group} ({patient_data.age} سال) - ملاحظات خاص سالمندی و احتمال دیورتیکولوز",
            estimated_encounter_time_min=total_time * 0.05,
            risk_level="medium",
            recommendations=recommendations,
            anatomical_details=anatomical_details,
            procedural_techniques=procedural_techniques,
            common_mistakes=common_mistakes
        )


# ==================== SURGICAL OPTIMIZATION ENGINE ====================
class SurgicalOptimizationEngine:
    """موتور بهینه‌سازی جراحی با الگوریتم‌های پیشرفته"""

    def __init__(self):
        self.optimization_methods = {
            'path_planning': self._optimize_navigation_path,
            'time_allocation': self._optimize_time_distribution,
            'instrument_selection': self._optimize_instrument_choice,
            'position_sequence': self._optimize_position_sequence,
            'pressure_strategy': self._optimize_pressure_points
        }

    def optimize_procedure(self, patient_data: PatientAnatomy,
                           warnings: List[BlindSpotWarning]) -> SurgicalOptimization:
        """بهینه‌سازی کامل پروسیجر با الگوریتم‌های چندهدفه"""

        # ۱. بهینه‌سازی مسیر ناوبری
        optimal_path = self._optimize_navigation_path(patient_data, warnings)

        # ۲. بهینه‌سازی تخصیص زمان
        time_allocation = self._optimize_time_distribution(warnings)

        # ۳. بهینه‌سازی انتخاب ابزار
        instrument_selection = self._optimize_instrument_choice(patient_data, warnings)

        # ۴. بهینه‌سازی توالی پوزیشن‌ها
        position_sequence = self._optimize_position_sequence(patient_data, warnings)

        # ۵. بهینه‌سازی نقاط فشار
        pressure_points = self._optimize_pressure_points(patient_data, warnings)

        # ۶. بهینه‌سازی استراتژی انسافلاسیون
        insufflation_strategy = self._optimize_insufflation_strategy(patient_data, warnings)

        return SurgicalOptimization(
            optimal_path=optimal_path,
            time_allocation=time_allocation,
            instrument_selection=instrument_selection,
            position_sequence=position_sequence,
            pressure_points=pressure_points,
            insufflation_strategy=insufflation_strategy
        )

    def _optimize_navigation_path(self, patient_data: PatientAnatomy, warnings: List[BlindSpotWarning]) -> List[str]:
        """بهینه‌سازی مسیر ناوبری با الگوریتم مسیریابی دینامیک"""

        # مناطق آناتومیک با هزینه‌های عبور
        anatomical_regions = {
            'رکتوم': {'base_cost': 1.0, 'risk_adjustment': 0.0},
            'اتصال رکتوسیگموئید': {'base_cost': 1.5, 'risk_adjustment': 0.0},
            'سیگموئید پروگزیمال': {'base_cost': 2.0, 'risk_adjustment': 0.0},
            'سیگموئید دیستال': {'base_cost': 2.5, 'risk_adjustment': 0.0},
            'کولون نزولی': {'base_cost': 1.8, 'risk_adjustment': 0.0},
            'خم طحالی': {'base_cost': 3.0, 'risk_adjustment': 0.0},
            'کولون عرضی پروگزیمال': {'base_cost': 2.2, 'risk_adjustment': 0.0},
            'کولون عرضی میانی': {'base_cost': 2.5, 'risk_adjustment': 0.0},
            'کولون عرضی دیستال': {'base_cost': 2.3, 'risk_adjustment': 0.0},
            'خم کبدی': {'base_cost': 3.2, 'risk_adjustment': 0.0},
            'کولون صعودی': {'base_cost': 2.0, 'risk_adjustment': 0.0},
            'سکوم': {'base_cost': 1.5, 'risk_adjustment': 0.0}
        }

        # اعمال تنظیمات خطر بر اساس هشدارها
        for warning in warnings:
            if warning.location in anatomical_regions:
                risk_multiplier = 2.0 if warning.risk_level == 'high' else 1.5 if warning.risk_level == 'medium' else 1.2
                anatomical_regions[warning.location]['risk_adjustment'] = risk_multiplier

        # محاسبه هزینه‌های نهایی
        final_costs = {}
        for region, params in anatomical_regions.items():
            final_costs[region] = params['base_cost'] * (1 + params['risk_adjustment'] * 0.3)

        # الگوریتم مسیریابی دینامیک (Dijkstra ساده شده)
        regions_order = list(anatomical_regions.keys())

        # اعمال بهینه‌سازی بر اساس هزینه‌ها
        optimized_path = []
        current_region = regions_order[0]
        optimized_path.append(current_region)

        remaining_regions = regions_order[1:]
        remaining_regions.sort(key=lambda x: final_costs[x])

        optimized_path.extend(remaining_regions)

        # اعمال اصلاحات خاص بر اساس آناتومی بیمار
        if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE:
            # در پیچ خوردگی شدید، عبور آرام‌تر از سیگموئید
            sigmoid_regions = ['سیگموئید پروگزیمال', 'سیگموئید دیستال']
            for region in sigmoid_regions:
                if region in optimized_path:
                    idx = optimized_path.index(region)
                    optimized_path.pop(idx)
                    optimized_path.insert(idx + 1, f"⚠️ {region}")

        return optimized_path

    def _optimize_time_distribution(self, warnings: List[BlindSpotWarning]) -> Dict[str, float]:
        """بهینه‌سازی توزیع زمان با الگوریتم تخصیص منابع"""

        # زمان پایه برای هر منطقه
        base_times = {
            'رکتوم': 2.0,
            'اتصال رکتوسیگموئید': 3.0,
            'سیگموئید': 8.0,
            'کولون نزولی': 4.0,
            'خم طحالی': 5.0,
            'کولون عرضی': 6.0,
            'خم کبدی': 5.0,
            'کولون صعودی': 4.0,
            'سکوم': 3.0
        }

        # محاسبه ضرایب خطر
        risk_coefficients = {}
        for warning in warnings:
            region_key = self._map_warning_to_region(warning.location)
            if region_key:
                if warning.risk_level == 'high':
                    risk_coefficients[region_key] = 1.8
                elif warning.risk_level == 'medium':
                    risk_coefficients[region_key] = 1.4
                else:
                    risk_coefficients[region_key] = 1.2

        # محاسبه زمان‌های بهینه
        optimized_times = {}
        total_allocated = 0

        for region, base_time in base_times.items():
            risk_coef = risk_coefficients.get(region, 1.0)
            # زمان بهینه با تابع لگاریتمی برای جلوگیری از افزایش بیش از حد
            optimized_time = base_time * (1 + np.log(risk_coef))
            optimized_times[region] = optimized_time
            total_allocated += optimized_time

        # نرمال‌سازی به ۴۵ دقیقه (زمان استاندارد)
        scale_factor = 45.0 / total_allocated
        for region in optimized_times:
            optimized_times[region] *= scale_factor

        # گرد کردن به یک رقم اعشار
        optimized_times = {k: round(v, 1) for k, v in optimized_times.items()}

        return optimized_times

    def _optimize_instrument_choice(self, patient_data: PatientAnatomy, warnings: List[BlindSpotWarning]) -> Dict[
        str, str]:
        """بهینه‌سازی انتخاب ابزار بر اساس پارامترهای بیمار"""

        instruments = {}

        # انتخاب نوع کولونوسکوپ
        bmi = patient_data.weight_kg / ((patient_data.height_cm / 100) ** 2)
        if bmi >= 35 or patient_data.colon_length == ColonLength.LONG:
            instruments['colonoscope_type'] = "کولونوسکوپ بلند (۱۶۰-۱۷۰ سانتی‌متر)"
        elif patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE:
            instruments['colonoscope_type'] = "کولونوسکوپ کودکان یا با قطر کوچک"
        else:
            instruments['colonoscope_type'] = "کولونوسکوپ استاندارد"

        # انتخاب ویژگی‌های اضافی
        features = []
        if any(w.risk_level == 'high' for w in warnings):
            features.append("قابلیت تغییر سختی (Variable stiffness)")

        if patient_data.age > 65:
            features.append("کنترل دقیق انسافلاسیون")

        if patient_data.has_sigmoid_loop or patient_data.has_transverse_loop:
            features.append("کنترل torque بالا")

        instruments['recommended_features'] = "، ".join(features) if features else "استاندارد"

        # تجهیزات جانبی پیشنهادی
        accessories = []
        if any('سیگموئید' in w.location for w in warnings):
            accessories.append("Overtube (در صورت نیاز)")

        if patient_data.prev_abdominal_surgery:
            accessories.append("Pediatric colonoscope (برای backup)")

        if bmi >= 30:
            accessories.append("Extension tubing برای انسافلاسیون")

        instruments['recommended_accessories'] = "، ".join(accessories) if accessories else "هیچ"

        # انتخاب نوک اسکوپ
        if patient_data.sigmoid_tortuosity in [SigmoidTortuosity.MODERATE, SigmoidTortuosity.SEVERE]:
            instruments['tip_selection'] = "نوک کوتاه با انعطاف‌پذیری بالا"
        else:
            instruments['tip_selection'] = "نوک استاندارد"

        return instruments

    def _optimize_position_sequence(self, patient_data: PatientAnatomy, warnings: List[BlindSpotWarning]) -> List[str]:
        """بهینه‌سازی توالی پوزیشن‌های بیمار"""

        base_positions = ["Left lateral", "Supine", "Right lateral", "Prone"]

        # تطبیق بر اساس هشدارها
        position_weights = {pos: 1.0 for pos in base_positions}

        for warning in warnings:
            if 'خم طحالی' in warning.location:
                position_weights['Right lateral'] *= 1.5
                position_weights['Prone'] *= 1.3

            if 'خم کبدی' in warning.location:
                position_weights['Left lateral'] *= 1.5
                position_weights['Supine'] *= 1.3

            if 'کولون عرضی' in warning.location:
                position_weights['Prone'] *= 1.4
                position_weights['Supine'] *= 1.2

        # تطبیق بر اساس BMI
        bmi = patient_data.weight_kg / ((patient_data.height_cm / 100) ** 2)
        if bmi >= 30:
            position_weights['Left lateral'] *= 1.2
            position_weights['Right lateral'] *= 1.2

        # مرتب‌سازی بر اساس وزن
        sorted_positions = sorted(position_weights.items(), key=lambda x: x[1], reverse=True)
        optimized_sequence = [pos for pos, _ in sorted_positions]

        # اضافه کردن توصیه‌های خاص
        if patient_data.has_transverse_loop:
            optimized_sequence.insert(1, "Prone (transverse colonبرای)")

        if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE:
            optimized_sequence.insert(0, "با بالش زیر لگن Left lateral")

        return optimized_sequence

    def _optimize_pressure_points(self, patient_data: PatientAnatomy, warnings: List[BlindSpotWarning]) -> List[
        Tuple[str, str]]:
        """بهینه‌سازی نقاط فشار شکمی"""

        pressure_points = []

        # نقاط فشار استاندارد
        standard_points = [
            ("Left lower quadrant", "فشار بر سیگموئید و descending colon"),
            ("Suprapubic", "فشار بر رکتوم و rectosigmoid junction"),
            ("Left upper quadrant", "فشار بر splenic flexure"),
            ("Right upper quadrant", "فشار بر hepatic flexure"),
            ("Epigastric", "فشار بر transverse colon")
        ]

        # انتخاب بر اساس هشدارها
        for warning in warnings:
            if 'سیگموئید' in warning.location:
                pressure_points.append(("Left lower quadrant",
                                        "فشار مستقیم بر حلقه‌های سیگموئید - حرکت دایره‌ای ملایم"))

            if 'خم طحالی' in warning.location:
                pressure_points.append(("Left upper quadrant under the ribs",
                                        "فشار به سمت پایین و راست برای کمک به عبور از خم"))

            if 'خم کبدی' in warning.location:
                pressure_points.append(("Right upper quadrant under the ribs)",
                                        "فشار به سمت پایین و چپ - اجتناب از فشار بر کبد"))

            if 'کولون عرضی' in warning.location:
                pressure_points.append(("Epigastric",
                                        "فشار عمیق به سمت ستون فقرات برای کاهش حلقه‌ها"))

        # حذف duplicates
        unique_points = []
        seen = set()
        for point in pressure_points:
            if point[0] not in seen:
                unique_points.append(point)
                seen.add(point[0])

        # اضافه کردن نقاط استاندارد اگر لازم
        if len(unique_points) < 2:
            unique_points.extend(standard_points[:2])

        return unique_points

    def _optimize_insufflation_strategy(self, patient_data: PatientAnatomy, warnings: List[BlindSpotWarning]) -> Dict[
        str, float]:
        """بهینه‌سازی استراتژی انسافلاسیون"""

        strategy = {
            'initial_insufflation': 20,  # mmHg
            'maintenance_pressure': 15,  # mmHg
            'max_pressure': 30,  # mmHg
            'suction_frequency': 0.3,  # suction/min
            'co2_preference': 0.8  # احتمال استفاده از CO2
        }

        # تنظیم بر اساس هشدارها
        for warning in warnings:
            if warning.risk_level == 'high':
                strategy['initial_insufflation'] *= 0.7
                strategy['suction_frequency'] *= 1.5
                strategy['co2_preference'] = 1.0  # قطعاً CO2

            if 'سیگموئید' in warning.location:
                strategy['initial_insufflation'] = min(strategy['initial_insufflation'], 15)

            if patient_data.age > 65:
                strategy['max_pressure'] = 25
                strategy['co2_preference'] = 1.0

        # تنظیم بر اساس آناتومی
        if patient_data.chronic_constipation:
            strategy['suction_frequency'] *= 1.3

        if patient_data.has_sigmoid_loop:
            strategy['initial_insufflation'] *= 0.6

        return strategy

    def _map_warning_to_region(self, warning_location: str) -> str:
        """نگاشت نام هشدار به منطقه آناتومیک"""
        mapping = {
            'کولون سیگموئید': 'سیگموئید',
            'خم طحالی': 'خم طحالی',
            'خم کبدی': 'خم کبدی',
            'کولون عرضی': 'کولون عرضی',
            'اتصال رکتوسیگموئید': 'اتصال رکتوسیگموئید'
        }
        return mapping.get(warning_location, '')


# ==================== ADVANCED VISUALIZATION ENGINE ====================
class AdvancedVisualizationEngine:
    """موتور بصری‌سازی پیشرفته با حساسیت بالا به پارامترها"""

    def __init__(self):
        self.colors = {
            'primary_yellow': '#FFD700',
            'primary_pink': '#FF69B4',
            'primary_green': '#32CD32',
            'dark_text': '#2C3E50',
            'warning_red': '#FF4444',
            'caution_orange': '#FF9966',
            'safe_green': '#44CC44'
        }

    def create_dynamic_colon_navigation_map(self, patient_data, warnings, risk_scores):
        """نقشه ناوبری دینامیک با حساسیت بالا به پارامترها"""

        # محاسبه پارامترهای دینامیک بر اساس آناتومی
        tortuosity_factor = {
            SigmoidTortuosity.MILD: 1.0,
            SigmoidTortuosity.MODERATE: 1.5,
            SigmoidTortuosity.SEVERE: 2.0
        }[patient_data.sigmoid_tortuosity]

        length_factor = {
            ColonLength.SHORT: 0.8,
            ColonLength.MEDIUM: 1.0,
            ColonLength.LONG: 1.3
        }[patient_data.colon_length]

        # تولید مختصات دینامیک بر اساس پارامترها
        t = np.linspace(0, 2 * np.pi, 100)

        # انحنای سیگموئید بر اساس پیچ خوردگی
        sigmoid_curvature = 2.0 * tortuosity_factor

        # طول مسیر بر اساس طول کولون
        path_length = 14 * length_factor

        # تولید مسیر دینامیک
        colon_x = []
        colon_y = []

        # بخش رکتوم (خطی)
        rect_length = 2
        for i in range(10):
            colon_x.append(i * rect_length / 10)
            colon_y.append(0)

        # بخش سیگموئید (منحنی)
        sigmoid_points = 30
        for i in range(sigmoid_points):
            t_val = i / sigmoid_points * np.pi
            x = rect_length + np.sin(t_val) * sigmoid_curvature
            y = np.cos(t_val) * sigmoid_curvature
            colon_x.append(x)
            colon_y.append(y)

        # بخش نزولی (خطی با شیب)
        descending_length = 3
        for i in range(15):
            x = colon_x[-1] + descending_length / 15
            y = colon_y[-1] - 0.1
            colon_x.append(x)
            colon_y.append(y)

        # خم طحالی (منحنی)
        splenic_curve = 20
        for i in range(splenic_curve):
            angle = i / splenic_curve * np.pi / 2
            x = colon_x[-1] + np.cos(angle) * 0.05
            y = colon_y[-1] + np.sin(angle) * 1.0
            colon_x.append(x)
            colon_y.append(y)

        # کولون عرضی (موجی)
        transverse_waves = 2
        for i in range(40):
            x = colon_x[-1] + 0.1
            y = colon_y[-1] + 0.05 * np.sin(i * 0.3)
            colon_x.append(x)
            colon_y.append(y)

        # خم کبدی (منحنی)
        hepatic_curve = 20
        for i in range(hepatic_curve):
            angle = i / hepatic_curve * np.pi / 2
            x = colon_x[-1] - np.sin(angle) * 0.05
            y = colon_y[-1] - np.cos(angle) * 1.0
            colon_x.append(x)
            colon_y.append(y)

        # کولون صعودی (خطی)
        ascending_length = 3
        for i in range(15):
            x = colon_x[-1] - 0.05
            y = colon_y[-1] - 0.15
            colon_x.append(x)
            colon_y.append(y)

        # سکوم (گشاد)
        cecum_points = 10
        for i in range(cecum_points):
            angle = i / cecum_points * np.pi
            x = colon_x[-1] + 0.2 * np.cos(angle)
            y = colon_y[-1] + 0.2 * np.sin(angle) - 0.1
            colon_x.append(x)
            colon_y.append(y)

        # نقاط آناتومیک کلیدی
        anatomical_points = {
            'رکتوم': (colon_x[5], colon_y[5]),
            'اتصال رکتوسیگموئید': (colon_x[15], colon_y[15]),
            'سیگموئید': (colon_x[25], colon_y[25]),
            'کولون نزولی': (colon_x[45], colon_y[45]),
            'خم طحالی': (colon_x[60], colon_y[60]),
            'کولون عرضی': (colon_x[80], colon_y[80]),
            'خم کبدی': (colon_x[100], colon_y[100]),
            'کولون صعودی': (colon_x[115], colon_y[115]),
            'سکوم': (colon_x[130], colon_y[130])
        }

        fig = go.Figure()

        # ترسیم مسیر با رنگ‌بندی بر اساس خطر
        for i in range(len(colon_x) - 1):
            segment_risk = 0

            # تعیین خطر بر اساس موقعیت
            if i < 15:  # رکتوم
                segment_risk = 0.2
            elif i < 40:  # سیگموئید
                segment_risk = 0.6 if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE else 0.3
            elif i < 65:  # نزولی و خم طحالی
                segment_risk = 0.5 if 'خم طحالی' in [w.location for w in warnings] else 0.3
            elif i < 105:  # عرضی
                segment_risk = 0.4 if patient_data.has_transverse_loop else 0.2
            elif i < 125:  # خم کبدی و صعودی
                segment_risk = 0.5 if 'خم کبدی' in [w.location for w in warnings] else 0.3
            else:  # سکوم
                segment_risk = 0.1

            # تعیین رنگ بر اساس خطر
            if segment_risk > 0.5:
                color = self.colors['warning_red']
            elif segment_risk > 0.3:
                color = self.colors['caution_orange']
            else:
                color = self.colors['safe_green']

            fig.add_trace(go.Scatter(
                x=[colon_x[i], colon_x[i + 1]],
                y=[colon_y[i], colon_y[i + 1]],
                mode='lines',
                line=dict(width=4, color=color),
                showlegend=False,
                hoverinfo='none'
            ))

        # اضافه کردن نقاط آناتومیک
        for name, (x, y) in anatomical_points.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=10, color=self.colors['dark_text'], symbol='circle',
                            line=dict(width=2, color='white')),
                text=[name],
                textposition="top center",
                name=name,
                hoverinfo='text',
                showlegend=False
            ))

        # اضافه کردن هشدارها
        warning_positions = {}
        for warning in warnings:
            if warning.location in anatomical_points:
                x, y = anatomical_points[warning.location]

                # اندازه بر اساس سطح اطمینان و ریسک
                size = 20 + (warning.confidence * 30)
                if warning.risk_level == 'high':
                    color = self.colors['warning_red']
                    symbol = 'x'
                elif warning.risk_level == 'medium':
                    color = self.colors['caution_orange']
                    symbol = 'triangle-up'
                else:
                    color = self.colors['primary_yellow']
                    symbol = 'circle'

                # افزودن انیمیشن‌های بصری برای هشدارهای مهم
                if warning.confidence > 0.7:
                    size *= 1.2

                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.7,
                        symbol=symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=f"⚠️ {warning.location}",
                    hovertemplate=(
                        f"<b>{warning.location}</b><br>"
                        f"سطح اطمینان: {warning.confidence * 100:.0f}%<br>"
                        f"سطح ریسک: {warning.risk_level}<br>"
                        f"زمان تخمینی: {warning.estimated_encounter_time_min:.1f} دقیقه<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=True
                ))

        # محاسبه محدوده دینامیک
        x_min, x_max = min(colon_x), max(colon_x)
        y_min, y_max = min(colon_y), max(colon_y)

        # اضافه کردن فضای حاشیه پویا
        x_margin = (x_max - x_min) * 0.2
        y_margin = (y_max - y_min) * 0.2

        fig.update_layout(
            title=dict(
                text=f"🎯 نقشه ناوبری دینامیک کولون<br><sub>پیچ خوردگی: {patient_data.sigmoid_tortuosity.value} | طول: {patient_data.colon_length.value}</sub>",
                font=dict(size=22, family='Arial Black')
            ),
            xaxis=dict(
                visible=False,
                range=[x_min - x_margin, x_max + x_margin]
            ),
            yaxis=dict(
                visible=False,
                range=[y_min - y_margin, y_max + y_margin]
            ),
            plot_bgcolor='rgba(250, 250, 250, 0.9)',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.85)',
                bordercolor='gray',
                borderwidth=1
            ),
            height=600,
            margin=dict(t=100, b=50, l=50, r=50)
        )

        return fig

    def create_dynamic_coverage_gauges(self, patient_data, warnings, procedure_dynamics):
        """گیج‌های دینامیک با حساسیت بالا"""

        if warnings:
            # محاسبات دینامیک بر اساس هشدارها
            high_risk_count = sum(1 for w in warnings if w.risk_level == 'high')
            medium_risk_count = sum(1 for w in warnings if w.risk_level == 'medium')

            total_confidence = sum(w.confidence for w in warnings) / len(warnings)

            # زمان تخمینی دینامیک
            estimated_time = sum(w.estimated_encounter_time_min for w in warnings)

            # امتیاز پیچیدگی دینامیک
            complexity_score = procedure_dynamics.get('complexity_score', 5.0)

        else:
            high_risk_count = 0
            medium_risk_count = 0
            total_confidence = 0
            estimated_time = 0
            complexity_score = 3.0

        # محاسبه امتیاز پوشش دینامیک
        coverage_base = 100 - (total_confidence * 40)

        # تنظیم بر اساس آناتومی بیمار
        if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE:
            coverage_base *= 0.9
        if patient_data.colon_length == ColonLength.LONG:
            coverage_base *= 0.95
        if patient_data.has_transverse_loop:
            coverage_base *= 0.92

        coverage_score = max(30, coverage_base)  # حداقل ۳۰٪

        # محاسبه امتیاز سختی دینامیک
        difficulty_base = complexity_score * 1.5

        # تنظیم بر اساس عوامل بالینی
        if patient_data.chronic_constipation:
            difficulty_base *= 1.2
        if patient_data.prev_abdominal_surgery:
            difficulty_base *= 1.15
        if patient_data.age > 65:
            difficulty_base *= 1.1

        difficulty_score = min(10, difficulty_base)

        # زمان دینامیک با در نظر گرفتن عوامل آناتومیک
        time_factor = 1.0
        if patient_data.colon_length == ColonLength.LONG:
            time_factor *= 1.3
        if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE:
            time_factor *= 1.25

        dynamic_estimated_time = estimated_time * time_factor

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=(
                f'پوشش تخمینی<br><span style="font-size:12px">بر اساس {len(warnings)} هشدار</span>',
                f'سختی پروسیجر<br><span style="font-size:12px">امتیاز: {complexity_score:.1f}/10</span>',
                f'زمان مورد نیاز<br><span style="font-size:12px">فاکتور: {time_factor:.2f}x</span>',
                f'مناطق پرریسک<br><span style="font-size:12px">{high_risk_count} بالا، {medium_risk_count} متوسط</span>'
            ),
            vertical_spacing=0.25,
            horizontal_spacing=0.2
        )

        # گیج ۱: پوشش تخمینی
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=coverage_score,
            domain={'row': 1, 'column': 1},
            title=dict(text="", font=dict(size=14)),
            delta=dict(
                reference=85,
                increasing=dict(color=self.colors['primary_green']),
                decreasing=dict(color=self.colors['warning_red'])
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor=self.colors['dark_text'],
                    tickfont=dict(size=10)
                ),
                bar=dict(color=self._get_coverage_color(coverage_score)),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 60], color="#FF6B6B", thickness=0.8),
                    dict(range=[60, 80], color="#FFD166", thickness=0.8),
                    dict(range=[80, 90], color="#06D6A0", thickness=0.8),
                    dict(range=[90, 100], color="#118AB2", thickness=0.8)
                ],
                threshold=dict(
                    line=dict(color="red", width=3),
                    thickness=0.8,
                    value=90
                )
            )
        ), row=1, col=1)

        # گیج ۲: سختی پروسیجر
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=difficulty_score,
            domain={'row': 1, 'column': 2},
            title=dict(text="", font=dict(size=14)),
            gauge=dict(
                axis=dict(
                    range=[0, 10],
                    tickwidth=1,
                    tickcolor=self.colors['dark_text'],
                    tickvals=[0, 2, 4, 6, 8, 10],
                    ticktext=["آسان", "نسبتا آسان", "متوسط", "سخت", "بسیار سخت", "پیچیده"],
                    tickfont=dict(size=9)
                ),
                bar=dict(color=self._get_difficulty_color(difficulty_score)),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 3], color=self.colors['primary_green'], thickness=0.8),
                    dict(range=[3, 6], color=self.colors['primary_yellow'], thickness=0.8),
                    dict(range=[6, 8], color=self.colors['caution_orange'], thickness=0.8),
                    dict(range=[8, 10], color=self.colors['warning_red'], thickness=0.8)
                ],
                threshold=dict(
                    line=dict(color="red", width=3),
                    thickness=0.8,
                    value=7
                )
            )
        ), row=1, col=2)

        # گیج ۳: زمان مورد نیاز
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=dynamic_estimated_time,
            domain={'row': 2, 'column': 1},
            title=dict(text="", font=dict(size=14)),
            gauge=dict(
                axis=dict(
                    range=[0, 60],
                    tickwidth=1,
                    tickcolor=self.colors['dark_text'],
                    tickvals=[0, 15, 30, 45, 60],
                    tickfont=dict(size=10)
                ),
                bar=dict(color="#9D4EDD"),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 20], color=self.colors['primary_green'], thickness=0.8),
                    dict(range=[20, 35], color=self.colors['primary_yellow'], thickness=0.8),
                    dict(range=[35, 50], color=self.colors['caution_orange'], thickness=0.8),
                    dict(range=[50, 60], color=self.colors['warning_red'], thickness=0.8)
                ],
                threshold=dict(
                    line=dict(color="red", width=3),
                    thickness=0.8,
                    value=40
                )
            )
        ), row=2, col=1)

        # گیج ۴: مناطق پرریسک
        total_risk_areas = high_risk_count + medium_risk_count
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=total_risk_areas,
            domain={'row': 2, 'column': 2},
            title=dict(text="مناطق پرریسک", font=dict(size=14)),
            gauge=dict(
                shape="bullet",
                axis=dict(
                    range=[0, 8],
                    tickwidth=1,
                    tickcolor=self.colors['dark_text'],
                    tickvals=[0, 2, 4, 6, 8],
                    ticktext=["کم", "متوسط", "زیاد", "خیلی زیاد", "بحرانی"],
                    tickfont=dict(size=9)
                ),
                bar=dict(color=self.colors['primary_pink']),
                bgcolor="white",
                steps=[
                    dict(range=[0, 2], color=self.colors['primary_green'], thickness=0.7),
                    dict(range=[2, 4], color=self.colors['primary_yellow'], thickness=0.7),
                    dict(range=[4, 6], color=self.colors['caution_orange'], thickness=0.7),
                    dict(range=[6, 8], color=self.colors['warning_red'], thickness=0.7)
                ],
                threshold=dict(
                    line=dict(color="red", width=2),
                    thickness=0.8,
                    value=3
                )
            )
        ), row=2, col=2)

        # به روزرسانی layout
        fig.update_layout(
            height=700,
            font=dict(family='Arial'),
            paper_bgcolor='white',
            margin=dict(t=120, b=50, l=50, r=50),
            showlegend=False,
            title=dict(
                text="📊 سنجش‌های دینامیک کیفیت پروسیجر",
                font=dict(size=20, family='Arial Black'),
                y=0.98
            )
        )

        # به روزرسانی عنوان‌های فرعی
        for i in range(4):
            fig.layout.annotations[i].font.size = 13

        return fig

    def _get_coverage_color(self, coverage_score):
        """رنگ‌بندی دینامیک بر اساس امتیاز پوشش"""
        if coverage_score >= 90:
            return self.colors['primary_green']
        elif coverage_score >= 80:
            return "#06D6A0"
        elif coverage_score >= 70:
            return self.colors['primary_yellow']
        elif coverage_score >= 60:
            return self.colors['caution_orange']
        else:
            return self.colors['warning_red']

    def _get_difficulty_color(self, difficulty_score):
        """رنگ‌بندی دینامیک بر اساس امتیاز سختی"""
        if difficulty_score <= 3:
            return self.colors['primary_green']
        elif difficulty_score <= 6:
            return self.colors['primary_yellow']
        elif difficulty_score <= 8:
            return self.colors['caution_orange']
        else:
            return self.colors['warning_red']

    def create_comprehensive_risk_heatmap(self, patient_data, warnings, risk_scores):
        """نقشه حرارتی جامع ریسک"""

        # مناطق آناتومیک با مقادیر ریسک پایه
        anatomical_regions = {
            'رکتوم': 0.2,
            'اتصال رکتوسیگموئید': 0.3,
            'سیگموئید پروگزیمال': 0.4,
            'سیگموئید دیستال': 0.5,
            'کولون نزولی': 0.3,
            'خم طحالی': 0.45,
            'کولون عرضی پروگزیمال': 0.35,
            'کولون عرضی میانی': 0.3,
            'کولون عرضی دیستال': 0.35,
            'خم کبدی': 0.5,
            'کولون صعودی': 0.3,
            'سکوم': 0.25
        }

        # اعمال تنظیمات بر اساس آناتومی بیمار
        if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE:
            anatomical_regions['سیگموئید پروگزیمال'] *= 1.6
            anatomical_regions['سیگموئید دیستال'] *= 1.8

        if patient_data.has_transverse_loop:
            anatomical_regions['کولون عرضی میانی'] *= 1.5

        if patient_data.pelvic_size == PelvicSize.SMALL:
            anatomical_regions['اتصال رکتوسیگموئید'] *= 1.4

        # اعمال هشدارها
        for warning in warnings:
            for region in anatomical_regions:
                if warning.location in region:
                    risk_multiplier = 1.8 if warning.risk_level == 'high' else 1.5 if warning.risk_level == 'medium' else 1.2
                    anatomical_regions[region] *= risk_multiplier

        # محدود کردن مقادیر به ۰-۱
        for region in anatomical_regions:
            anatomical_regions[region] = min(1.0, anatomical_regions[region])

        # ایجاد داده برای نقشه حرارتی
        regions = list(anatomical_regions.keys())
        risk_values = [anatomical_regions[region] * 100 for region in regions]  # تبدیل به درصد

        # مرتب‌سازی بر اساس ریسک
        sorted_indices = np.argsort(risk_values)
        sorted_regions = [regions[i] for i in sorted_indices]
        sorted_risks = [risk_values[i] for i in sorted_indices]

        # ایجاد رنگ‌ها
        colors = []
        for risk in sorted_risks:
            if risk >= 70:
                colors.append(self.colors['warning_red'])
            elif risk >= 50:
                colors.append(self.colors['caution_orange'])
            elif risk >= 30:
                colors.append(self.colors['primary_yellow'])
            else:
                colors.append(self.colors['primary_green'])

        fig = go.Figure()

        # اضافه کردن میله‌های افقی
        fig.add_trace(go.Bar(
            y=sorted_regions,
            x=sorted_risks,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{risk:.0f}%" for risk in sorted_risks],
            textposition='outside',
            textfont=dict(size=11, color='black')
        ))

        # خطوط راهنما
        fig.add_vline(x=30, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=70, line_dash="dash", line_color="gray", opacity=0.5)

        # برچسب‌های راهنما
        fig.add_annotation(x=15, y=len(regions) - 0.5, text="کم", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=40, y=len(regions) - 0.5, text="متوسط", showarrow=False, font=dict(color="orange"))
        fig.add_annotation(x=60, y=len(regions) - 0.5, text="زیاد", showarrow=False, font=dict(color="darkorange"))
        fig.add_annotation(x=85, y=len(regions) - 0.5, text="بسیار زیاد", showarrow=False, font=dict(color="red"))

        fig.update_layout(
            title=dict(
                text="🔥 نقشه حرارتی ریسک مناطق آناتومیک",
                font=dict(size=20, family='Arial Black')
            ),
            xaxis=dict(
                title="درصد ریسک نقاط کور",
                range=[0, 100],
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"]
            ),
            yaxis=dict(
                title="مناطق آناتومیک",
                tickfont=dict(size=11)
            ),
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(t=80, b=50, l=150, r=50)
        )

        return fig

    def create_advanced_action_plan(self, warnings, optimization_result):
        """نمودار اقدامات پیشرفته"""

        if not warnings:
            fig = go.Figure()
            fig.add_annotation(
                text="✅ هیچ اقدام خاصی مورد نیاز نیست",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=18, color="green")
            )
            fig.update_layout(
                title=dict(text="نمودار اقدامات", font=dict(size=16)),
                paper_bgcolor='white',
                height=400
            )
            return fig

        # دسته‌بندی اقدامات
        action_categories = {
            'تکنیک‌های ناوبری': ['Torque-steering', 'Hook-and-pull', 'Position change', 'Suction technique'],
            'مدیریت ابزار': ['Scope selection', 'Insufflation control', 'Accessory preparation'],
            'مدیریت بیمار': ['Positioning', 'Sedation strategy', 'Monitoring', 'Pressure application'],
            'اقدامات ایمنی': ['Risk assessment', 'Time allocation', 'Backup planning', 'Complication management']
        }

        # محاسبه امتیازات بر اساس هشدارها
        category_scores = {category: 0 for category in action_categories}

        for warning in warnings:
            # تخصیص امتیاز بر اساس نوع هشدار
            if 'سیگموئید' in warning.location:
                category_scores['تکنیک‌های ناوبری'] += 25
                category_scores['مدیریت ابزار'] += 15
            elif 'خم' in warning.location:
                category_scores['تکنیک‌های ناوبری'] += 20
                category_scores['مدیریت بیمار'] += 20
            elif 'عرضی' in warning.location:
                category_scores['مدیریت بیمار'] += 25
                category_scores['تکنیک‌های ناوبری'] += 15

            if warning.risk_level == 'high':
                category_scores['اقدامات ایمنی'] += 30

        # نرمال‌سازی امتیازات به ۰-۱۰۰
        max_score = max(category_scores.values()) if category_scores.values() else 1
        for category in category_scores:
            category_scores[category] = (category_scores[category] / max_score) * 100

        # ایجاد نمودار رادار
        categories = list(category_scores.keys())
        scores = list(category_scores.values())

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # بستن نمودار
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(255, 105, 180, 0.3)',
            line=dict(color=self.colors['primary_pink'], width=2),
            name='اولویت اقدامات',
            hoverinfo='text',
            text=[f'{cat}: {score:.0f}%' for cat, score in zip(categories, scores)]
        ))

        # اضافه کردن مناطق راهنما
        for threshold in [30, 60, 90]:
            fig.add_trace(go.Scatterpolar(
                r=[threshold] * (len(categories) + 1),
                theta=categories + [categories[0]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 30, 60, 90, 100],
                    ticktext=["کم", "متوسط", "زیاد", "خیلی زیاد", "بحرانی"],
                    tickfont=dict(size=10),
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                bgcolor='rgba(240, 240, 240, 0.3)'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.1
            ),
            title=dict(
                text="🎯 نمودار اولویت‌بندی اقدامات پروسیجری",
                font=dict(size=18, family='Arial Black')
            ),
            height=500,
            paper_bgcolor='white',
            margin=dict(l=80, r=150, t=80, b=50)
        )

        return fig


# ==================== STREAMLIT UI ====================
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color: #FF69B4;'>🏥 سیستم ناوبری هوشمند کولونوسکوپی پیشرفته</h1>
            <h3 style='color: #FFD700;'>نسخه ۶.۰ - هوش جراحی آناتومیک</h3>
            <p style='color: #2C3E50; font-size: 16px;'>
                تحلیل‌گر پیشرفته آناتومیک با الگوریتم‌های بهینه‌سازی چندهدفه
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar for patient input
    with st.sidebar:
        st.markdown("""
        <div class='success-box'>
            <h4>📋 ورودی‌های آناتومیک بیمار</h4>
            <p>لطفاً اطلاعات دقیق بیمار را وارد کنید</p>
        </div>
        """, unsafe_allow_html=True)

        # Patient information
        st.subheader("👤 اطلاعات دموگرافیک")
        age = st.slider("سن بیمار (سال)", 18, 120, 50, help="سن دقیق بیمار برای محاسبات مرتبط با سن")
        gender = st.radio("جنسیت", ["مرد", "زن"], help="برای محاسبات آناتومیک خاص جنسیت")
        height_cm = st.slider("قد (سانتی‌متر)", 100, 250, 170, help="قد برای محاسبه BMI و تناسب آناتومیک")
        weight_kg = st.slider("وزن (کیلوگرم)", 30, 300, 75, help="وزن برای محاسبه BMI و ملاحظات پروسیجری")

        st.markdown("---")

        # Anatomical features
        st.subheader("🔍 پارامترهای آناتومیک")
        colon_length = st.selectbox(
            "طول کولون",
            ["کوتاه (کمتر از ۱۰۰ سانتی‌متر)", "متوسط (۱۴۰-۱۰۰ سانتی‌متر)", "طویل (بیشتر از ۱۴۰ سانتی‌متر)"],
            help="طول تخمینی کولون بر اساس مطالعات آناتومیک"
        )

        splenic_angle = st.selectbox(
            "زاویه خم طحالی",
            ["حاد (کمتر از ۹۰ درجه)", "قائمه (نزدیک ۹۰ درجه)", "منفرجه (بیشتر از ۹۰ درجه)"],
            help="زاویه خم طحالی بر اساس تصویربرداری یا بررسی قبلی"
        )

        hepatic_angle = st.selectbox(
            "زاویه خم کبدی",
            ["حاد (کمتر از ۹۰ درجه)", "قائمه (نزدیک ۹۰ درجه)", "منفرجه (بیشتر از ۹۰ درجه)"],
            help="زاویه خم کبدی برای محاسبه سختی عبور"
        )

        tortuosity = st.selectbox(
            "درجه پیچ خوردگی سیگموئید",
            ["خفیف (کمتر از ۲ حلقه)", "متوسط (۳-۲ حلقه)", "شدید (بیشتر از ۳ حلقه)"],
            help="درجه پیچ خوردگی بر اساس کولونوسکوپی یا تصویربرداری قبلی"
        )

        pelvic_size = st.selectbox(
            "وسعت حفره لگن",
            ["کوچک", "متوسط", "بزرگ"],
            help="تخمین اندازه حفره لگن بر اساس معاینه یا تصویربرداری"
        )

        st.markdown("---")

        # Clinical information
        st.subheader("📝 فاکتورهای بالینی")
        has_sigmoid_loop = st.checkbox(
            "حلقه در سیگموئید وجود دارد",
            help="وجود حلقه‌های سیگموئید در بررسی‌های قبلی"
        )

        has_transverse_loop = st.checkbox(
            "حلقه در کولون عرضی وجود دارد",
            help="تشکیل حلقه در کولون عرضی در کولونوسکوپی‌های قبلی"
        )

        prev_abdominal_surgery = st.checkbox(
            "سابقه جراحی شکمی",
            help="هرگونه جراحی قبلی در حفره شکم"
        )

        chronic_constipation = st.checkbox(
            "یبوست مزمن",
            help="سابقه یبوست مزمن بیش از ۳ ماه"
        )

        # Calculate button
        st.markdown("---")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🧠 تحلیل آناتومیک پیشرفته", use_container_width=True, type="primary"):
                st.session_state['analyze'] = True
                st.session_state['advanced_mode'] = True

        with col2:
            if st.button("🗑️ پاکسازی نتایج", use_container_width=True):
                st.session_state['analyze'] = False
                st.session_state['advanced_mode'] = False

        if 'analyze' not in st.session_state:
            st.session_state['analyze'] = False
        if 'advanced_mode' not in st.session_state:
            st.session_state['advanced_mode'] = False

        # اطلاعات اضافی
        st.markdown("---")
        st.info("""
        **💡 نکته تخصصی:**  
        این سیستم از الگوریتم‌های پیشرفته آناتومیک و بهینه‌سازی چندهدفه برای تحلیل دقیق آناتومی کولون استفاده می‌کند.
        """)

    # Main content area
    if st.session_state['analyze']:
        try:
            # Convert inputs to model format
            colon_length_map = {
                "کوتاه (کمتر از ۱۰۰ سانتی‌متر)": ColonLength.SHORT,
                "متوسط (۱۴۰-۱۰۰ سانتی‌متر)": ColonLength.MEDIUM,
                "طویل (بیشتر از ۱۴۰ سانتی‌متر)": ColonLength.LONG
            }

            angle_map = {
                "حاد (کمتر از ۹۰ درجه)": FlexureAngle.ACUTE,
                "قائمه (نزدیک ۹۰ درجه)": FlexureAngle.RIGHT,
                "منفرجه (بیشتر از ۹۰ درجه)": FlexureAngle.OBTUSE
            }

            tortuosity_map = {
                "خفیف (کمتر از ۲ حلقه)": SigmoidTortuosity.MILD,
                "متوسط (۳-۲ حلقه)": SigmoidTortuosity.MODERATE,
                "شدید (بیشتر از ۳ حلقه)": SigmoidTortuosity.SEVERE
            }

            pelvic_size_map = {
                "کوچک": PelvicSize.SMALL,
                "متوسط": PelvicSize.MEDIUM,
                "بزرگ": PelvicSize.LARGE
            }

            # Create patient data object
            patient_data = PatientAnatomy(
                age=age,
                gender="male" if gender == "مرد" else "female",
                height_cm=height_cm,
                weight_kg=weight_kg,
                colon_length=colon_length_map[colon_length],
                splenic_flexure_angle=angle_map[splenic_angle],
                hepatic_flexure_angle=angle_map[hepatic_angle],
                sigmoid_tortuosity=tortuosity_map[tortuosity],
                pelvic_size=pelvic_size_map[pelvic_size],
                has_sigmoid_loop=has_sigmoid_loop,
                has_transverse_loop=has_transverse_loop,
                prev_abdominal_surgery=prev_abdominal_surgery,
                chronic_constipation=chronic_constipation
            )

            # Initialize models
            model = AdvancedAnatomicalModel()
            optimizer = SurgicalOptimizationEngine()
            visualizer = AdvancedVisualizationEngine()

            # Analyze
            with st.spinner("🧬 در حال تحلیل پیشرفته آناتومی با الگوریتم‌های چندمتغیره..."):
                warnings = model.predict_anatomical_risks(patient_data)
                procedure_dynamics = model.estimate_procedure_dynamics(patient_data)
                optimization_result = optimizer.optimize_procedure(patient_data, warnings)

                # Calculate additional metrics
                bmi = model.calculate_bmi(patient_data)
                bsa = model.calculate_body_surface_area(patient_data)
                total_time = procedure_dynamics['total_time_min']

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 خلاصه فنی", "🎯 ناوبری دینامیک", "📈 تحلیل ریسک",
                "⚠️ هشدارهای تخصصی", "⚙️ بهینه‌سازی پروسیجر", "🏥 پروتکل کامل"
            ])

            with tab1:
                st.markdown("""
                <div class='success-box'>
                    <h3>📊 خلاصه فنی تحلیل آناتومیک</h3>
                </div>
                """, unsafe_allow_html=True)

                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    bmi_status = "نرمال" if 18.5 <= bmi <= 24.9 else "اضافه وزن" if 25 <= bmi <= 29.9 else "چاق"
                    st.metric("شاخص توده بدنی", f"{bmi:.1f}", delta=bmi_status)

                with col2:
                    complexity = procedure_dynamics['complexity_score']
                    complexity_level = "ساده" if complexity < 4 else "متوسط" if complexity < 7 else "پیچیده"
                    st.metric("امتیاز پیچیدگی", f"{complexity:.1f}/10", delta=complexity_level)

                with col3:
                    st.metric("زمان تخمینی پروسیجر", f"{total_time:.0f} دقیقه")

                with col4:
                    high_risk_count = sum(1 for w in warnings if w.risk_level == 'high')
                    st.metric("مناطق پرریسک بالا", high_risk_count)

                st.markdown("---")

                # Detailed patient summary
                st.subheader("📋 جزئیات پارامترهای آناتومیک")

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**👤 اطلاعات پایه:**")
                    st.write(f"• سن: {patient_data.age} سال")
                    st.write(f"• جنسیت: {'مرد' if patient_data.gender == 'male' else 'زن'}")
                    st.write(f"• قد: {patient_data.height_cm} سانتی‌متر")
                    st.write(f"• وزن: {patient_data.weight_kg} کیلوگرم")
                    st.write(f"• سطح بدن: {bsa:.2f} m²")

                    st.info(f"**🔬 فاکتورهای آناتومیک:**")
                    st.write(f"• طول کولون: {patient_data.colon_length.value}")
                    st.write(f"• زاویه خم طحالی: {patient_data.splenic_flexure_angle.value}")
                    st.write(f"• زاویه خم کبدی: {patient_data.hepatic_flexure_angle.value}")

                with col2:
                    st.info(f"**📝 فاکتورهای بالینی:**")
                    st.write(f"• درجه پیچ خوردگی سیگموئید: {patient_data.sigmoid_tortuosity.value}")
                    st.write(f"• وسعت حفره لگن: {patient_data.pelvic_size.value}")
                    st.write(f"• حلقه سیگموئید: {'دارد' if patient_data.has_sigmoid_loop else 'ندارد'}")
                    st.write(f"• حلقه عرضی: {'دارد' if patient_data.has_transverse_loop else 'ندارد'}")
                    st.write(f"• سابقه جراحی شکمی: {'دارد' if patient_data.prev_abdominal_surgery else 'ندارد'}")
                    st.write(f"• یبوست مزمن: {'دارد' if patient_data.chronic_constipation else 'ندارد'}")

                    st.info(f"**📊 آمارهای پروسیجر:**")
                    st.write(f"• insertion time: {procedure_dynamics['insertion_time']:.1f} min")
                    st.write(f"• withdrawal time: {procedure_dynamics['withdrawal_time']:.1f} min")
                    st.write(f"• examination time: {procedure_dynamics['examination_time']:.1f} min")
                    st.write(f"• BMI factor: {procedure_dynamics['bmi_factor']:.2f}x")
                    st.write(f"• Age factor: {procedure_dynamics['age_factor']:.2f}x")

                # Complexity assessment
                st.markdown("---")
                st.subheader("🎯 ارزیابی سطح پیچیدگی پروسیجر")

                complexity_score = procedure_dynamics['complexity_score']
                if complexity_score < 4:
                    complexity_level = "ساده"
                    color = "green"
                    icon = "✅"
                elif complexity_score < 7:
                    complexity_level = "متوسط"
                    color = "orange"
                    icon = "⚠️"
                else:
                    complexity_level = "پیچیده"
                    color = "red"
                    icon = "🚨"

                st.markdown(f"""
                <div style='background: {color}20; padding: 20px; border-radius: 15px; border-left: 5px solid {color};'>
                    <h4 style='color: {color};'>{icon} سطح پیچیدگی: {complexity_level} (امتیاز: {complexity_score:.1f}/10)</h4>
                    <p>.توصیه‌ها بر اساس این سطح پیچیدگی ارائه شده‌اند</p>
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                st.markdown("""
                <div class='warning-box'>
                    <h3>🎯 نقشه ناوبری دینامیک کولون</h3>
                </div>
                """, unsafe_allow_html=True)

                # Dynamic navigation map
                try:
                    fig1 = visualizer.create_dynamic_colon_navigation_map(
                        patient_data, warnings, procedure_dynamics
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception as e:
                    st.error(f"خطا در ایجاد نقشه ناوبری: {str(e)}")

                # Coverage gauges
                st.markdown("---")
                st.subheader("📊 سنجش‌های دینامیک کیفیت")
                try:
                    fig2 = visualizer.create_dynamic_coverage_gauges(
                        patient_data, warnings, procedure_dynamics
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"خطا در ایجاد سنجش‌های کیفیت: {str(e)}")

                # Optimization highlights
                st.markdown("---")
                st.subheader("⚡ نکات کلیدی بهینه‌سازی")

                col1, col2 = st.columns(2)
                with col1:
                    st.success("**🎯 مسیر بهینه ناوبری:**")
                    for i, region in enumerate(optimization_result.optimal_path[:6], 1):
                        st.write(f"{i}. {region}")

                with col2:
                    st.success("**🔄 توالی پوزیشن‌های بهینه:**")
                    for i, position in enumerate(optimization_result.position_sequence[:4], 1):
                        st.write(f"{i}. {position}")

            with tab3:
                st.markdown("""
                <div class='danger-box'>
                    <h3>📈 تحلیل جامع ریسک آناتومیک</h3>
                </div>
                """, unsafe_allow_html=True)

                # Risk heatmap
                try:
                    fig3 = visualizer.create_comprehensive_risk_heatmap(
                        patient_data, warnings, procedure_dynamics
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.error(f"خطا در ایجاد نقشه حرارتی ریسک: {str(e)}")

                # Advanced action plan
                st.markdown("---")
                st.subheader("🎯 نمودار اولویت‌بندی اقدامات")
                try:
                    fig4 = visualizer.create_advanced_action_plan(warnings, optimization_result)
                    st.plotly_chart(fig4, use_container_width=True)
                except Exception as e:
                    st.error(f"خطا در ایجاد نمودار اقدامات: {str(e)}")

                # Time allocation
                st.markdown("---")
                st.subheader("⏱️ تخصیص زمانی بهینه")

                time_df = pd.DataFrame(
                    list(optimization_result.time_allocation.items()),
                    columns=['منطقه آناتومیک', 'زمان اختصاصی (دقیقه)']
                )
                time_df['درصد کل'] = (
                            time_df['زمان اختصاصی (دقیقه)'] / time_df['زمان اختصاصی (دقیقه)'].sum() * 100).round(1)

                st.dataframe(
                    time_df.sort_values('زمان اختصاصی (دقیقه)', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

            with tab4:
                st.markdown("""
                <div class='danger-box'>
                    <h3>⚠️ هشدارهای تخصصی آناتومیک</h3>
                </div>
                """, unsafe_allow_html=True)

                if warnings:
                    for i, warning in enumerate(warnings, 1):
                        # تعیین رنگ و آیکون بر اساس سطح ریسک
                        if warning.risk_level == 'high':
                            color = "#FF4444"
                            icon = "🚨"
                            container_class = "stAlert st-emotion-cache-1c7yyw e1f1d6gn3"
                        elif warning.risk_level == 'medium':
                            color = "#FF9966"
                            icon = "⚠️"
                            container_class = "stAlert st-emotion-cache-1c7yyw e1f1d6gn2"
                        else:
                            color = "#FFD700"
                            icon = "ℹ️"
                            container_class = "stAlert st-emotion-cache-1c7yyw e1f1d6gn1"

                        # ایجاد کانتینر هشدار
                        with st.container():
                            st.markdown(f"""
                            <div style='
                                border-radius: 15px;
                                border-left: 5px solid {color};
                                background: {color}15;
                                padding: 20px;
                                margin: 15px 0;
                            '>
                                <h4 style='color: {color}; margin-top: 0;'>
                                    {icon} {i}. {warning.location}
                                </h4>
                            </div>
                            """, unsafe_allow_html=True)

                        # متریک‌های سریع
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("سطح اطمینان", f"{warning.confidence * 100:.0f}%")

                        with col2:
                            st.metric("زمان تخمینی", f"{warning.estimated_encounter_time_min:.1f} دقیقه")

                        with col3:
                            risk_badge = {
                                'high': '🔴 بالا',
                                'medium': '🟡 متوسط',
                                'low': '🟢 پایین'
                            }.get(warning.risk_level, warning.risk_level)
                            st.metric("سطح ریسک", risk_badge)

                        with col4:
                            if warning.anatomical_details:
                                detail_key = list(warning.anatomical_details.keys())[0]
                                detail_val = warning.anatomical_details[detail_key]
                                st.metric("پارامتر کلیدی", f"{detail_val:.2f}")

                        # دلیل هشدار
                        st.write(f"**🔍 دلیل شناسایی:** {warning.reason}")

                        # جزئیات آناتومیک
                        with st.expander("📐 جزئیات آناتومیک و پارامترها", expanded=False):
                            if warning.anatomical_details:
                                details_df = pd.DataFrame(
                                    list(warning.anatomical_details.items()),
                                    columns=['پارامتر', 'مقدار']
                                )
                                st.dataframe(details_df, use_container_width=True, hide_index=True)

                        # توصیه‌های تخصصی
                        with st.expander("🎯 توصیه‌های تخصصی پروسیجری", expanded=True):
                            for rec in warning.recommendations:
                                st.write(f"• {rec}")

                        # تکنیک‌های پروسیجری
                        with st.expander("⚙️ تکنیک‌های پروسیجری پیشنهادی", expanded=False):
                            for tech in warning.procedural_techniques:
                                st.write(f"• {tech}")

                        # خطاهای رایج
                        with st.expander("❌ خطاهای رایج در این منطقه", expanded=False):
                            for mistake in warning.common_mistakes:
                                st.write(f"• ⚠️ {mistake}")

                        st.markdown("---")

                else:
                    st.success("""
                    ✅ **هیچ هشدار آناتومیک خاصی شناسایی نشد!**

                    شرایط آناتومیک بیمار در محدوده نرمال قرار دارد و پروسیجر با پروتکل استاندارد قابل انجام است.
                    """)

            with tab5:
                st.markdown("""
                <div class='success-box'>
                    <h3>⚙️ بهینه‌سازی هوشمند پروسیجر</h3>
                </div>
                """, unsafe_allow_html=True)

                # Instrument optimization
                st.subheader("🔧 بهینه‌سازی تجهیزات و ابزار")

                col1, col2 = st.columns(2)
                with col1:
                    st.info("**انتخاب کولونوسکوپ:**")
                    st.write(
                        f"• **نوع:** {optimization_result.instrument_selection.get('colonoscope_type', 'استاندارد')}")
                    st.write(
                        f"• **ویژگی‌ها:** {optimization_result.instrument_selection.get('recommended_features', 'استاندارد')}")
                    st.write(
                        f"• **نوک اسکوپ:** {optimization_result.instrument_selection.get('tip_selection', 'استاندارد')}")

                with col2:
                    st.info("**تجهیزات جانبی:**")
                    st.write(
                        f"• **اکسسوری‌ها:** {optimization_result.instrument_selection.get('recommended_accessories', 'هیچ')}")

                    if patient_data.age > 65:
                        st.write("• **ملاحظات سالمندی:** تجهیزات مانیتورینگ پیشرفته")

                    if procedure_dynamics['complexity_score'] > 7:
                        st.write("• **پروسیجر پیچیده:** آماده‌سازی تجهیزات پشتیبان")

                # Position optimization
                st.markdown("---")
                st.subheader("🔄 بهینه‌سازی توالی پوزیشن‌ها")

                positions_df = pd.DataFrame(
                    [(i + 1, pos) for i, pos in enumerate(optimization_result.position_sequence)],
                    columns=['اولویت', 'پوزیشن']
                )
                st.dataframe(positions_df, use_container_width=True, hide_index=True)

                # Pressure points optimization
                st.markdown("---")
                st.subheader("🖐️ بهینه‌سازی نقاط فشار شکمی")

                if optimization_result.pressure_points:
                    for location, description in optimization_result.pressure_points:
                        st.write(f"• **{location}:** {description}")
                else:
                    st.info("نیاز به فشار شکمی خاصی شناسایی نشد.")

                # Insufflation strategy
                st.markdown("---")
                st.subheader("💨 استراتژی بهینه انسافلاسیون")

                insufflation_df = pd.DataFrame(
                    list(optimization_result.insufflation_strategy.items()),
                    columns=['پارامتر', 'مقدار بهینه']
                )
                st.dataframe(insufflation_df, use_container_width=True, hide_index=True)

                # Special considerations
                st.markdown("---")
                st.subheader("🎯 ملاحظات خاص پروسیجر")

                considerations = []

                if patient_data.sigmoid_tortuosity == SigmoidTortuosity.SEVERE:
                    considerations.append("زمان اضافی برای عبور از سیگموئید پیچیده (حداقل ۱۰-۱۵ دقیقه)")

                if patient_data.has_transverse_loop:
                    considerations.append("استفاده مکرر از تغییر پوزیشن برای کنترل حلقه‌های عرضی")

                if patient_data.prev_abdominal_surgery:
                    considerations.append("احتیاط در مناطق احتمالی چسبندگی")

                if patient_data.chronic_constipation:
                    considerations.append("شستشوی کامل و بررسی دقیق مناطق احتمال باقی مانده مدفوع")

                if considerations:
                    for consideration in considerations:
                        st.write(f"• ⚠️ {consideration}")
                else:
                    st.info(".ملاحظات خاصی برای این پروسیجر شناسایی نشد")

            with tab6:
                st.markdown("""
                <div class='success-box'>
                    <h3>🏥 پروتکل کامل کولونوسکوپی بهینه</h3>
                </div>
                """, unsafe_allow_html=True)

                # Complete procedure protocol
                st.subheader("📋 پروتکل مرحله‌به‌مرحله")

                protocol_steps = [
                    ("۱. آماده‌سازی اولیه", [
                        "بررسی داروهای بیمار (ضد انعقاد، ضد پلاکت)",
                        "نیلاسیون وریدی و شروع مایع درمانی",
                        "مونیتورینگ پایه (فشار خون، نبض، اکسیژن خون)",
                        "تزریق پیش‌داروی سدیشن بر اساس پروتکل"
                    ]),

                    ("۲. پوزیشن‌دهی اولیه", [
                        f"پوزیشن: {optimization_result.position_sequence[0]}",
                        "قرارگیری بالش‌های مناسب برای حمایت آناتومیک",
                        "تنظیم ارتفاع تخت برای ارگونومی اندوسکوپیست"
                    ]),

                    ("۳. شروع پروسیجر", [
                        "بررسی دیجیتال رکتوم (در صورت نیاز)",
                        "روانکاری اسکوپ و بررسی عملکرد",
                        f"تنظیم فشار انسافلاسیون اولیه: {optimization_result.insufflation_strategy['initial_insufflation']} میلی متر جیوه",
                        "ورود آرام اسکوپ تحت دید مستقیم"
                    ]),

                    ("۴. ناوبری در مناطق پرریسک", [
                        "توجه ویژه به مناطق هشداردهنده طبق نقشه ناوبری",
                        "استفاده از تکنیک‌های توصیه‌شده برای هر منطقه",
                        "اعمال فشار شکمی در نقاط مشخص‌شده",
                        "تغییر پوزیشن در صورت مواجهه با مقاومت"
                    ]),

                    ("۵. بررسی سیستماتیک", [
                        "بررسی کامل تمام مناطق آناتومیک",
                        "عکس‌برداری از مناطق پرریسک و یافته‌ها",
                        "برداشتن نمونه‌برداری در صورت نیاز",
                        "بررسی مجدد مناطق مشکوک"
                    ]),

                    ("۶. پایان پروسیجر", [
                        "خروج کنترل شده اسکوپ",
                        "تخلیه هوای باقیمانده",
                        "بررسی نشانه‌های حیاتی بیمار",
                        "ثبت یافته‌ها و اقدامات انجام‌شده"
                    ]),

                    ("۷. مراقبت پس از پروسیجر", [
                        "مشاهده بیمار در ریکاوری",
                        "بررسی عوارض احتمالی",
                        "آموزش ترخیص به بیمار",
                        "تنظیم وقت پیگیری در صورت نیاز"
                    ])
                ]

                for step_title, step_details in protocol_steps:
                    with st.expander(f"{step_title}", expanded=False):
                        for detail in step_details:
                            st.write(f"• {detail}")

                # Team recommendations
                st.markdown("---")
                st.subheader("👥 توصیه‌های تیم اندوسکوپی")

                team_recommendations = [
                    "اندوسکوپیست اصلی: متخصص گوارش با تجربه کافی",
                    "دستیار اندوسکوپی: پرستار آموزش‌دیده اندوسکوپی",
                    "تعداد پرسنل: حداقل ۲ نفر برای پروسیجر‌های پیچیده"
                ]

                if procedure_dynamics['complexity_score'] > 7:
                    team_recommendations.append("**پیشنهاد: حضور اندوسکوپیست ارشد به عنوان سوپروایزر**")

                if any(w.risk_level == 'high' for w in warnings):
                    team_recommendations.append("آماده‌سازی تجهیزات اورژانسی در دسترس")

                for recommendation in team_recommendations:
                    st.write(f"• {recommendation}")

                # Download comprehensive report
                st.markdown("---")
                st.subheader("💾 گزارش جامع پروسیجر")

                report_text = f"""
                ====================================================
                گزارش جامع تحلیل آناتومیک و بهینه‌سازی کولونوسکوپی
                ====================================================

                تاریخ و زمان تولید: {datetime.now().strftime('%Y/%m/%d - %H:%M')}

                بخش ۱: اطلاعات بیمار
                ---------------------
                سن: {patient_data.age} سال
                جنسیت: {gender}
                قد: {patient_data.height_cm} سانتی‌متر
                وزن: {patient_data.weight_kg} کیلوگرم
                BMI: {bmi:.1f} ({bmi_status})
                سطح بدن: {bsa:.2f} متر مربع

                بخش ۲: پارامترهای آناتومیک
                --------------------------
                طول کولون: {patient_data.colon_length.value}
                زاویه خم طحالی: {patient_data.splenic_flexure_angle.value}
                زاویه خم کبدی: {patient_data.hepatic_flexure_angle.value}
                درجه پیچ خوردگی سیگموئید: {patient_data.sigmoid_tortuosity.value}
                وسعت حفره لگن: {patient_data.pelvic_size.value}
                حلقه سیگموئید: {'دارد' if patient_data.has_sigmoid_loop else 'ندارد'}
                حلقه عرضی: {'دارد' if patient_data.has_transverse_loop else 'ندارد'}
                سابقه جراحی شکمی: {'دارد' if patient_data.prev_abdominal_surgery else 'ندارد'}
                یبوست مزمن: {'دارد' if patient_data.chronic_constipation else 'ندارد'}

                بخش ۳: تحلیل ریسک
                -----------------
                تعداد هشدارها: {len(warnings)}
                هشدارهای پرریسک: {sum(1 for w in warnings if w.risk_level == 'high')}
                امتیاز پیچیدگی: {procedure_dynamics['complexity_score']:.1f}/10
                سطح پیچیدگی: {'ساده' if procedure_dynamics['complexity_score'] < 4 else 'متوسط' if procedure_dynamics['complexity_score'] < 7 else 'پیچیده'}
                زمان تخمینی پروسیجر: {total_time:.0f} دقیقه

                بخش ۴: بهینه‌سازی پروسیجر
                -------------------------
                کولونوسکوپ پیشنهادی: {optimization_result.instrument_selection.get('colonoscope_type', 'استاندارد')}
                توالی پوزیشن‌ها: {', '.join(optimization_result.position_sequence[:3])}
                مناطق پرریسک نیازمند توجه ویژه: {', '.join([w.location for w in warnings if w.risk_level == 'high'][:3])}

                بخش ۵: ملاحظات ویژه
                --------------------
                {chr(10).join(['• ' + item for item in considerations]) if considerations else '• هیچ ملاحظه ویژه‌ای شناسایی نشد.'}

                ====================================================
                *** این گزارش توسط سیستم هوشمند تحلیل آناتومیک نسخه ۶ تولید شده است ***
                """

                st.download_button(
                    label="📥 دانلود گزارش جامع (TXT)",
                    data=report_text,
                    file_name=f"colonoscopy_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    type="primary"
                )

        except Exception as e:
            st.error(f"""
            ❌ خطا در تحلیل آناتومیک پیشرفته

            **خطا:** {str(e)}

            لطفاً:
            ۱. اطلاعات ورودی را بررسی کنید
            ۲. دوباره تلاش کنید
            ۳. در صورت تداوم مشکل، با پشتیبانی تماس بگیرید
            """)

    else:
        # Initial screen - SIMPLIFIED as requested
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
        <p style="font-size:20px; font-weight:bold; text-align:center;">
        لطفا اطلاعات بیمار را در فرم سمت چپ وارد کنید 🩺📋
        </p>
        </div>
        """, unsafe_allow_html=True)


# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
    #فارسی انگلیسی هارم درستشون بکن
    #تحلیل خط به خط کد و رفرنس دهی
    #رفرنس مونده بدم و انگلیسیا درست کنم تو پاورم ک اعتبار و تحلیل میارم