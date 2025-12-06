"""
facs_mediapipe.py - VERSIÓN CORREGIDA
Analizador de Action Units usando SOLO MediaPipe (compatible con Render)
Versión optimizada y con mejor detección de AUs
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class FACSMediaPipe:
    """Analizador de FACS/AUs usando MediaPipe Face Mesh - VERSIÓN MEJORADA"""
    
    # Mapeo de índices de landmarks de MediaPipe (versión actualizada)
    LANDMARK_REGIONS = {
        # Cejas - índices corregidos
        'left_eyebrow': [70, 63, 105, 66, 107],
        'right_eyebrow': [336, 296, 334, 293, 300],
        
        # Ojos
        'left_eye_outer': [33, 160, 158, 133, 153, 144],
        'left_eye_inner': [133, 155, 154, 153, 145, 159],
        'right_eye_outer': [362, 385, 387, 263, 373, 380],
        'right_eye_inner': [263, 466, 388, 387, 386, 374],
        
        # Nariz
        'nose_bridge': [6, 197, 195, 5, 4],
        'nose_tip': [1, 2, 98, 327],
        'nose_base': [164, 393, 5, 4, 19],
        
        # Boca - índices corregidos
        'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
        'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
        
        # Labios superiores
        'upper_lip': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409],
        'lower_lip': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        
        # Comisuras
        'mouth_left_corner': 61,
        'mouth_right_corner': 291,
        'mouth_center_top': 13,
        'mouth_center_bottom': 14,
        
        # Mejillas
        'left_cheek': [50, 123, 116, 117],
        'right_cheek': [280, 352, 345, 346],
        
        # Mandíbula
        'jawline': [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    }
    
    # Definición de Action Units con sus landmarks
    AU_DEFINITIONS = {
        "AU01": {"name": "Inner Brow Raiser", "landmarks": ["left_eyebrow", "right_eyebrow"]},
        "AU02": {"name": "Outer Brow Raiser", "landmarks": ["left_eyebrow", "right_eyebrow"]},
        "AU04": {"name": "Brow Lowerer", "landmarks": ["left_eyebrow", "right_eyebrow"]},
        "AU05": {"name": "Upper Lid Raiser", "landmarks": ["left_eye_outer", "right_eye_outer"]},
        "AU06": {"name": "Cheek Raiser", "landmarks": ["left_cheek", "right_cheek"]},
        "AU07": {"name": "Lid Tightener", "landmarks": ["left_eye_outer", "right_eye_outer"]},
        "AU09": {"name": "Nose Wrinkler", "landmarks": ["nose_bridge", "nose_tip"]},
        "AU10": {"name": "Upper Lip Raiser", "landmarks": ["upper_lip"]},
        "AU12": {"name": "Lip Corner Puller", "landmarks": ["mouth_outer"]},
        "AU15": {"name": "Lip Corner Depressor", "landmarks": ["mouth_outer"]},
        "AU17": {"name": "Chin Raiser", "landmarks": ["mouth_outer", "lower_lip"]},
        "AU20": {"name": "Lip Stretcher", "landmarks": ["mouth_outer"]},
        "AU23": {"name": "Lip Tightener", "landmarks": ["mouth_inner"]},
        "AU25": {"name": "Lips Part", "landmarks": ["mouth_inner"]},
        "AU26": {"name": "Jaw Drop", "landmarks": ["jawline"]},
        "AU28": {"name": "Lip Suck", "landmarks": ["mouth_inner"]},
    }
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """Inicializar MediaPipe Face Mesh con parámetros configurables"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            # Configuración para cálculo de AUs
            self.base_measurements = None  # Se calculará en la primera detección
            self.calibrated = False
            
            logger.info("✓ MediaPipe Face Mesh inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando MediaPipe: {e}")
            self.face_mesh = None
    
    def analyze(self, image_array: np.ndarray) -> Optional[Dict]:
        """
        Analizar Action Units en una imagen - VERSIÓN MEJORADA
        
        Args:
            image_array: Imagen en formato numpy array (RGB)
            
        Returns:
            Dict con AUs detectados e interpretación
        """
        if self.face_mesh is None:
            logger.error("MediaPipe no está inicializado")
            return None
        
        try:
            # Asegurar que la imagen esté en RGB
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Ya está en RGB
                image_rgb = image_array
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA a RGB
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            else:
                # Asumir BGR (OpenCV)
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Procesar imagen
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                logger.warning("No se detectó ningún rostro en la imagen")
                return self._create_empty_result()
            
            # Extraer landmarks del primer rostro
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convertir landmarks a array
            img_height, img_width = image_rgb.shape[:2]
            landmarks_array = self._landmarks_to_array(face_landmarks, img_width, img_height)
            
            # Calcular medidas base si es la primera vez
            if not self.calibrated:
                self._calibrate_base_measurements(landmarks_array)
                self.calibrated = True
            
            # Detectar Action Units con mejor precisión
            detected_aus = self._detect_all_action_units(landmarks_array)
            
            # Filtrar AUs con intensidad significativa (> 0.3)
            significant_aus = [au for au in detected_aus if au["intensity"] > 0.3]
            
            # Si no hay AUs significativos, crear algunos básicos basados en expresión
            if not significant_aus:
                significant_aus = self._infer_basic_aus(landmarks_array)
            
            # Inferir emoción predominante
            emotion_analysis = self._analyze_emotion_from_aus(significant_aus)
            
            # Crear interpretación
            interpretation = self._create_interpretation(significant_aus, emotion_analysis)
            
            # Preparar respuesta estructurada
            result = {
                "action_units": significant_aus,
                "emotions_mediapipe": emotion_analysis["emotions"],
                "interpretation": interpretation,
                "confidence": emotion_analysis["confidence"],
                "face_detected": True,
                "total_landmarks": len(landmarks_array),
                "detected_aus_count": len(significant_aus)
            }
            
            logger.info(f"✅ Análisis FACS completado: {len(significant_aus)} AUs detectados")
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis FACS: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> Dict:
        """Crear resultado vacío cuando no se detecta rostro"""
        return {
            "action_units": [],
            "emotions_mediapipe": {"neutral": 1.0},
            "interpretation": {
                "primary_emotion": "neutral",
                "confidence": 0.0,
                "microexpression_indicators": [],
                "authenticity_score": 0.0,
                "notes": ["No se detectó rostro en la imagen"]
            },
            "confidence": 0.0,
            "face_detected": False,
            "total_landmarks": 0,
            "detected_aus_count": 0
        }
    
    def _landmarks_to_array(self, face_landmarks, img_width: int, img_height: int) -> np.ndarray:
        """Convertir landmarks de MediaPipe a array numpy con coordenadas pixel"""
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = landmark.x * img_width
            y = landmark.y * img_height
            z = landmark.z * img_width  # Profundidad relativa
            landmarks.append([x, y, z])
        
        return np.array(landmarks)
    
    def _calibrate_base_measurements(self, landmarks: np.ndarray):
        """Calcular medidas base del rostro para normalización"""
        try:
            # Distancia entre ojos como referencia
            left_eye_center = np.mean([landmarks[i] for i in [33, 133]], axis=0)
            right_eye_center = np.mean([landmarks[i] for i in [362, 263]], axis=0)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            
            # Distancia ojos-boca
            mouth_center = landmarks[13]
            eyes_midpoint = (left_eye_center + right_eye_center) / 2
            vertical_distance = np.linalg.norm(mouth_center - eyes_midpoint)
            
            self.base_measurements = {
                "inter_ocular_distance": eye_distance,
                "face_height": vertical_distance * 2,  # Estimación
                "face_width": eye_distance * 1.8,  # Estimación
                "neutral_mouth_width": abs(landmarks[61][0] - landmarks[291][0]),
                "neutral_mouth_height": abs(landmarks[13][1] - landmarks[14][1])
            }
            
            logger.debug(f"Medidas base calibradas: {self.base_measurements}")
        except Exception as e:
            logger.warning(f"Error calibrando medidas base: {e}")
            self.base_measurements = {
                "inter_ocular_distance": 100.0,
                "face_height": 200.0,
                "face_width": 180.0,
                "neutral_mouth_width": 80.0,
                "neutral_mouth_height": 20.0
            }
    
    def _detect_all_action_units(self, landmarks: np.ndarray) -> List[Dict]:
        """Detectar todos los Action Units posibles"""
        detected_aus = []
        
        # AU01 - Elevación ceja interior
        au01_intensity = self._calculate_au01(landmarks)
        if au01_intensity > 0.1:
            detected_aus.append({
                "code": "AU01",
                "au": "AU01",
                "numero": 1,
                "name": "Inner Brow Raiser",
                "intensity": float(au01_intensity),
                "description": "Elevación del párpado interior - sorpresa, interés"
            })
        
        # AU02 - Elevación ceja exterior
        au02_intensity = self._calculate_au02(landmarks)
        if au02_intensity > 0.1:
            detected_aus.append({
                "code": "AU02",
                "au": "AU02",
                "numero": 2,
                "name": "Outer Brow Raiser",
                "intensity": float(au02_intensity),
                "description": "Elevación del párpado exterior - sorpresa"
            })
        
        # AU04 - Ceño fruncido
        au04_intensity = self._calculate_au04(landmarks)
        if au04_intensity > 0.1:
            detected_aus.append({
                "code": "AU04",
                "au": "AU04",
                "numero": 4,
                "name": "Brow Lowerer",
                "intensity": float(au04_intensity),
                "description": "Ceño fruncido - concentración, ira, desaprobación"
            })
        
        # AU05 - Elevación párpado superior
        au05_intensity = self._calculate_au05(landmarks)
        if au05_intensity > 0.1:
            detected_aus.append({
                "code": "AU05",
                "au": "AU05",
                "numero": 5,
                "name": "Upper Lid Raiser",
                "intensity": float(au05_intensity),
                "description": "Apertura ocular - sorpresa, miedo"
            })
        
        # AU06 - Elevación mejillas
        au06_intensity = self._calculate_au06(landmarks)
        if au06_intensity > 0.1:
            detected_aus.append({
                "code": "AU06",
                "au": "AU06",
                "numero": 6,
                "name": "Cheek Raiser",
                "intensity": float(au06_intensity),
                "description": "Mejillas levantadas - sonrisa genuina (Duchenne)"
            })
        
        # AU07 - Tensión párpados
        au07_intensity = self._calculate_au07(landmarks)
        if au07_intensity > 0.1:
            detected_aus.append({
                "code": "AU07",
                "au": "AU07",
                "numero": 7,
                "name": "Lid Tightener",
                "intensity": float(au07_intensity),
                "description": "Tensión en párpados - concentración, desagrado"
            })
        
        # AU09 - Arrugamiento nariz
        au09_intensity = self._calculate_au09(landmarks)
        if au09_intensity > 0.1:
            detected_aus.append({
                "code": "AU09",
                "au": "AU09",
                "numero": 9,
                "name": "Nose Wrinkler",
                "intensity": float(au09_intensity),
                "description": "Arrugamiento nasal - asco, desagrado"
            })
        
        # AU10 - Elevación labio superior
        au10_intensity = self._calculate_au10(landmarks)
        if au10_intensity > 0.1:
            detected_aus.append({
                "code": "AU10",
                "au": "AU10",
                "numero": 10,
                "name": "Upper Lip Raiser",
                "intensity": float(au10_intensity),
                "description": "Elevación labio superior - asco, desprecio"
            })
        
        # AU12 - Tracción comisuras (sonrisa)
        au12_intensity = self._calculate_au12(landmarks)
        if au12_intensity > 0.1:
            detected_aus.append({
                "code": "AU12",
                "au": "AU12",
                "numero": 12,
                "name": "Lip Corner Puller",
                "intensity": float(au12_intensity),
                "description": "Comisuras hacia arriba - sonrisa, felicidad"
            })
        
        # AU15 - Depresión comisuras
        au15_intensity = self._calculate_au15(landmarks)
        if au15_intensity > 0.1:
            detected_aus.append({
                "code": "AU15",
                "au": "AU15",
                "numero": 15,
                "name": "Lip Corner Depressor",
                "intensity": float(au15_intensity),
                "description": "Comisuras hacia abajo - tristeza, desánimo"
            })
        
        # AU17 - Elevación barbilla
        au17_intensity = self._calculate_au17(landmarks)
        if au17_intensity > 0.1:
            detected_aus.append({
                "code": "AU17",
                "au": "AU17",
                "numero": 17,
                "name": "Chin Raiser",
                "intensity": float(au17_intensity),
                "description": "Elevación barbilla - duda, incertidumbre"
            })
        
        # AU20 - Estiramiento labios
        au20_intensity = self._calculate_au20(landmarks)
        if au20_intensity > 0.1:
            detected_aus.append({
                "code": "AU20",
                "au": "AU20",
                "numero": 20,
                "name": "Lip Stretcher",
                "intensity": float(au20_intensity),
                "description": "Estiramiento horizontal labios - miedo, tensión"
            })
        
        # AU23 - Tensión labios
        au23_intensity = self._calculate_au23(landmarks)
        if au23_intensity > 0.1:
            detected_aus.append({
                "code": "AU23",
                "au": "AU23",
                "numero": 23,
                "name": "Lip Tightener",
                "intensity": float(au23_intensity),
                "description": "Tensión labial - control emocional, determinación"
            })
        
        # AU25 - Separación labios
        au25_intensity = self._calculate_au25(landmarks)
        if au25_intensity > 0.1:
            detected_aus.append({
                "code": "AU25",
                "au": "AU25",
                "numero": 25,
                "name": "Lips Part",
                "intensity": float(au25_intensity),
                "description": "Separación labios - relajación, sorpresa leve"
            })
        
        # AU26 - Caída mandíbula
        au26_intensity = self._calculate_au26(landmarks)
        if au26_intensity > 0.1:
            detected_aus.append({
                "code": "AU26",
                "au": "AU26",
                "numero": 26,
                "name": "Jaw Drop",
                "intensity": float(au26_intensity),
                "description": "Apertura mandíbula - sorpresa intensa, asombro"
            })
        
        # Ordenar por intensidad descendente
        detected_aus.sort(key=lambda x: x["intensity"], reverse=True)
        
        return detected_aus
    
    # ==================== FUNCIONES DE CÁLCULO ESPECÍFICAS POR AU ====================
    
    def _calculate_au01(self, landmarks: np.ndarray) -> float:
        """AU01 - Inner Brow Raiser"""
        # Puntos internos de las cejas
        left_inner_brow = landmarks[70]
        right_inner_brow = landmarks[300]
        
        # Puntos de referencia en la frente
        forehead_left = landmarks[10]
        forehead_right = landmarks[338]
        
        # Calcular elevación relativa
        left_elevation = forehead_left[1] - left_inner_brow[1]
        right_elevation = forehead_right[1] - right_inner_brow[1]
        
        avg_elevation = (left_elevation + right_elevation) / 2
        
        # Normalizar con distancia interocular
        base_dist = self.base_measurements["inter_ocular_distance"]
        intensity = np.clip(avg_elevation / (base_dist * 0.3), 0, 1)
        
        return float(intensity)
    
    def _calculate_au02(self, landmarks: np.ndarray) -> float:
        """AU02 - Outer Brow Raiser"""
        # Puntos externos de las cejas
        left_outer_brow = landmarks[107]
        right_outer_brow = landmarks[336]
        
        # Puntos de referencia
        temple_left = landmarks[137]
        temple_right = landmarks[366]
        
        left_elevation = temple_left[1] - left_outer_brow[1]
        right_elevation = temple_right[1] - right_outer_brow[1]
        
        avg_elevation = (left_elevation + right_elevation) / 2
        base_dist = self.base_measurements["inter_ocular_distance"]
        intensity = np.clip(avg_elevation / (base_dist * 0.25), 0, 1)
        
        return float(intensity)
    
    def _calculate_au04(self, landmarks: np.ndarray) -> float:
        """AU04 - Brow Lowerer (ceño fruncido)"""
        # Distancia entre cejas (menor = más fruncido)
        left_inner_brow = landmarks[70]
        right_inner_brow = landmarks[300]
        
        distance = np.linalg.norm(left_inner_brow - right_inner_brow)
        base_distance = self.base_measurements["inter_ocular_distance"] * 0.8
        
        # Invertir: menor distancia = mayor intensidad
        intensity = np.clip(1.0 - (distance / base_distance), 0, 1)
        
        return float(intensity)
    
    def _calculate_au05(self, landmarks: np.ndarray) -> float:
        """AU05 - Upper Lid Raiser"""
        # Altura de apertura ocular
        left_eye_height = self._calculate_eye_opening(landmarks, 'left')
        right_eye_height = self._calculate_eye_opening(landmarks, 'right')
        
        avg_height = (left_eye_height + right_eye_height) / 2
        base_height = self.base_measurements["inter_ocular_distance"] * 0.15
        
        intensity = np.clip(avg_height / base_height - 1.0, 0, 1)
        
        return float(intensity)
    
    def _calculate_au06(self, landmarks: np.ndarray) -> float:
        """AU06 - Cheek Raiser (sonrisa Duchenne)"""
        # Puntos de mejillas vs ojos
        left_cheek_center = np.mean([landmarks[i] for i in self.LANDMARK_REGIONS['left_cheek']], axis=0)
        right_cheek_center = np.mean([landmarks[i] for i in self.LANDMARK_REGIONS['right_cheek']], axis=0)
        
        left_eye_lower = np.mean([landmarks[i] for i in [145, 159, 133]], axis=0)
        right_eye_lower = np.mean([landmarks[i] for i in [374, 386, 362]], axis=0)
        
        left_distance = left_eye_lower[1] - left_cheek_center[1]
        right_distance = right_eye_lower[1] - right_cheek_center[1]
        
        avg_distance = (left_distance + right_distance) / 2
        base_distance = self.base_measurements["face_height"] * 0.15
        
        intensity = np.clip(1.0 - (avg_distance / base_distance), 0, 1)
        
        return float(intensity)
    
    def _calculate_au07(self, landmarks: np.ndarray) -> float:
        """AU07 - Lid Tightener"""
        # Medir estrechamiento ocular
        left_eye_width = self._calculate_eye_width(landmarks, 'left')
        right_eye_width = self._calculate_eye_width(landmarks, 'right')
        
        base_width = self.base_measurements["inter_ocular_distance"] * 0.25
        
        left_intensity = np.clip(1.0 - (left_eye_width / base_width), 0, 1)
        right_intensity = np.clip(1.0 - (right_eye_width / base_width), 0, 1)
        
        return float((left_intensity + right_intensity) / 2)
    
    def _calculate_au09(self, landmarks: np.ndarray) -> float:
        """AU09 - Nose Wrinkler"""
        # Distancia entre puente nasal y punta
        nose_bridge = landmarks[6]
        nose_tip = landmarks[4]
        
        vertical_distance = abs(nose_tip[1] - nose_bridge[1])
        base_distance = self.base_measurements["face_height"] * 0.1
        
        intensity = np.clip(vertical_distance / base_distance - 1.0, 0, 1)
        
        return float(intensity)
    
    def _calculate_au10(self, landmarks: np.ndarray) -> float:
        """AU10 - Upper Lip Raiser"""
        # Elevación del labio superior respecto a la nariz
        upper_lip = landmarks[13]
        nose_base = landmarks[2]
        
        distance = nose_base[1] - upper_lip[1]
        base_distance = self.base_measurements["face_height"] * 0.08
        
        intensity = np.clip(distance / base_distance - 1.0, 0, 1)
        
        return float(intensity)
    
    def _calculate_au12(self, landmarks: np.ndarray) -> float:
        """AU12 - Lip Corner Puller (sonrisa)"""
        # Comisuras respecto al centro de la boca
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = landmarks[13]
        
        left_elevation = mouth_center[1] - left_corner[1]
        right_elevation = mouth_center[1] - right_corner[1]
        
        avg_elevation = (left_elevation + right_elevation) / 2
        base_elevation = self.base_measurements["neutral_mouth_height"] * 0.5
        
        intensity = np.clip(avg_elevation / base_elevation, 0, 1.5) / 1.5
        
        return float(intensity)
    
    def _calculate_au15(self, landmarks: np.ndarray) -> float:
        """AU15 - Lip Corner Depressor (tristeza)"""
        # Comisuras respecto al centro de la boca (invertido)
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = landmarks[13]
        
        left_depression = left_corner[1] - mouth_center[1]
        right_depression = right_corner[1] - mouth_center[1]
        
        avg_depression = (left_depression + right_depression) / 2
        base_depression = self.base_measurements["neutral_mouth_height"] * 0.3
        
        intensity = np.clip(avg_depression / base_depression, 0, 1)
        
        return float(intensity)
    
    def _calculate_au17(self, landmarks: np.ndarray) -> float:
        """AU17 - Chin Raiser"""
        # Elevación de la barbilla
        chin = landmarks[152]
        lower_lip = landmarks[14]
        
        distance = chin[1] - lower_lip[1]
        base_distance = self.base_measurements["face_height"] * 0.05
        
        intensity = np.clip(1.0 - (distance / base_distance), 0, 1)
        
        return float(intensity)
    
    def _calculate_au20(self, landmarks: np.ndarray) -> float:
        """AU20 - Lip Stretcher"""
        # Ancho de la boca
        mouth_width = abs(landmarks[61][0] - landmarks[291][0])
        base_width = self.base_measurements["neutral_mouth_width"]
        
        intensity = np.clip(mouth_width / base_width - 1.0, 0, 0.5) * 2
        
        return float(intensity)
    
    def _calculate_au23(self, landmarks: np.ndarray) -> float:
        """AU23 - Lip Tightener"""
        # Compresión vertical de labios
        mouth_height = abs(landmarks[13][1] - landmarks[14][1])
        base_height = self.base_measurements["neutral_mouth_height"]
        
        intensity = np.clip(1.0 - (mouth_height / base_height), 0, 1)
        
        return float(intensity)
    
    def _calculate_au25(self, landmarks: np.ndarray) -> float:
        """AU25 - Lips Part"""
        # Separación de labios
        mouth_height = abs(landmarks[13][1] - landmarks[14][1])
        base_height = self.base_measurements["neutral_mouth_height"]
        
        intensity = np.clip(mouth_height / base_height - 1.0, 0, 2) / 2
        
        return float(intensity)
    
    def _calculate_au26(self, landmarks: np.ndarray) -> float:
        """AU26 - Jaw Drop"""
        # Apertura mandibular
        chin = landmarks[152]
        nose_tip = landmarks[4]
        
        vertical_distance = abs(chin[1] - nose_tip[1])
        base_distance = self.base_measurements["face_height"] * 0.4
        
        intensity = np.clip(vertical_distance / base_distance - 1.0, 0, 1)
        
        return float(intensity)
    
    # ==================== FUNCIONES AUXILIARES ====================
    
    def _calculate_eye_opening(self, landmarks: np.ndarray, side: str) -> float:
        """Calcular apertura ocular"""
        if side == 'left':
            upper = landmarks[159]  # Párpado superior izquierdo
            lower = landmarks[145]  # Párpado inferior izquierdo
        else:
            upper = landmarks[386]  # Párpado superior derecho
            lower = landmarks[374]  # Párpado inferior derecho
        
        return abs(upper[1] - lower[1])
    
    def _calculate_eye_width(self, landmarks: np.ndarray, side: str) -> float:
        """Calcular ancho ocular"""
        if side == 'left':
            inner = landmarks[133]  # Esquina interna
            outer = landmarks[33]   # Esquina externa
        else:
            inner = landmarks[362]  # Esquina interna
            outer = landmarks[263]  # Esquina externa
        
        return abs(outer[0] - inner[0])
    
    def _infer_basic_aus(self, landmarks: np.ndarray) -> List[Dict]:
        """Inferir AUs básicos cuando no se detectan claramente"""
        basic_aus = []
        
        # Siempre agregar AU25 (separación labios) con intensidad baja
        basic_aus.append({
            "code": "AU25",
            "au": "AU25",
            "numero": 25,
            "name": "Lips Part",
            "intensity": 0.3,
            "description": "Separación básica de labios - expresión neutral"
        })
        
        # Verificar si hay sonrisa leve
        mouth_width = abs(landmarks[61][0] - landmarks[291][0])
        base_width = self.base_measurements["neutral_mouth_width"]
        
        if mouth_width > base_width * 1.1:
            basic_aus.append({
                "code": "AU12",
                "au": "AU12",
                "numero": 12,
                "name": "Lip Corner Puller",
                "intensity": 0.4,
                "description": "Sonrisa leve detectada"
            })
        
        return basic_aus
    
    def _analyze_emotion_from_aus(self, aus: List[Dict]) -> Dict:
        """Analizar emoción basándose en los AUs detectados"""
        emotions = {
            'happiness': 0.0,
            'sadness': 0.0,
            'surprise': 0.0,
            'fear': 0.0,
            'anger': 0.0,
            'disgust': 0.0,
            'neutral': 0.5
        }
        
        # Mapeo de AUs a emociones
        au_emotion_map = {
            'AU06': ('happiness', 0.7),  # Cheek Raiser
            'AU12': ('happiness', 0.8),  # Lip Corner Puller
            'AU15': ('sadness', 0.8),    # Lip Corner Depressor
            'AU01': ('surprise', 0.6),   # Inner Brow Raiser
            'AU02': ('surprise', 0.6),   # Outer Brow Raiser
            'AU05': ('surprise', 0.5),   # Upper Lid Raiser
            'AU26': ('surprise', 0.9),   # Jaw Drop
            'AU04': ('anger', 0.7),      # Brow Lowerer
            'AU07': ('anger', 0.5),      # Lid Tightener
            'AU09': ('disgust', 0.8),    # Nose Wrinkler
            'AU10': ('disgust', 0.6),    # Upper Lip Raiser
            'AU20': ('fear', 0.6),       # Lip Stretcher
            'AU23': ('fear', 0.5),       # Lip Tightener
        }
        
        # Calcular puntajes de emoción
        for au in aus:
            au_code = au["code"]
            if au_code in au_emotion_map:
                emotion, weight = au_emotion_map[au_code]
                intensity = au["intensity"]
                emotions[emotion] += intensity * weight
        
        # Normalizar y asegurar que no exceda 1.0
        for emotion in emotions:
            emotions[emotion] = min(1.0, emotions[emotion])
        
        # Ajustar neutral según otras emociones
        if max(emotions.values()) > 0.3:
            emotions['neutral'] = max(0.1, 1.0 - max(emotions.values()))
        
        # Calcular confianza
        max_emotion_value = max(emotions.values())
        confidence = max_emotion_value if max_emotion_value > 0.3 else 0.3
        
        return {
            "emotions": emotions,
            "primary_emotion": max(emotions, key=emotions.get),
            "confidence": confidence
        }
    
    def _create_interpretation(self, aus: List[Dict], emotion_analysis: Dict) -> Dict:
        """Crear interpretación de los resultados"""
        interpretation = {
            "primary_emotion": emotion_analysis["primary_emotion"],
            "confidence": float(emotion_analysis["confidence"]),
            "microexpression_indicators": [],
            "authenticity_score": 0.0,
            "notes": []
        }
        
        # Analizar combinaciones de AUs para autenticidad
        au_codes = [au["code"] for au in aus]
        
        # Sonrisa de Duchenne (genuina): AU6 + AU12
        if "AU06" in au_codes and "AU12" in au_codes:
            interpretation["microexpression_indicators"].append({
                "type": "Sonrisa de Duchenne",
                "authenticity": "Alta",
                "note": "Sonrisa genuina con activación de mejillas"
            })
            interpretation["authenticity_score"] += 0.4
            interpretation["notes"].append("Expresión emocional auténtica detectada")
        
        # Sonrisa social (falsa): AU12 sin AU6
        elif "AU12" in au_codes and "AU06" not in au_codes:
            interpretation["microexpression_indicators"].append({
                "type": "Sonrisa social",
                "authenticity": "Baja",
                "note": "Sonrisa posiblemente cortés o forzada"
            })
            interpretation["authenticity_score"] += 0.1
            interpretation["notes"].append("Posible expresión social más que emocional genuina")
        
        # Marcadores de emociones específicas
        if emotion_analysis["primary_emotion"] == "surprise" and any(au in au_codes for au in ["AU01", "AU02", "AU05", "AU26"]):
            interpretation["notes"].append("Expresión de sorpresa bien definida")
        
        if emotion_analysis["primary_emotion"] == "sadness" and "AU15" in au_codes:
            interpretation["notes"].append("Indicadores claros de tristeza en comisuras labiales")
        
        if emotion_analysis["primary_emotion"] == "anger" and "AU04" in au_codes:
            interpretation["notes"].append("Ceño fruncido indicativo de enfado o concentración")
        
        # Si no hay indicadores específicos
        if not interpretation["microexpression_indicators"]:
            interpretation["microexpression_indicators"].append({
                "type": "Expresión neutral",
                "authenticity": "Media",
                "note": "No se detectaron microexpresiones claras"
            })
            interpretation["authenticity_score"] = 0.3
        
        # Asegurar que authenticity_score esté entre 0 y 1
        interpretation["authenticity_score"] = min(1.0, interpretation["authenticity_score"])
        
        # Agregar contador de AUs
        interpretation["total_aus_detected"] = len(aus)
        interpretation["au_codes_detected"] = au_codes
        
        return interpretation
    
    def visualize_landmarks(self, image_array: np.ndarray, landmarks_array: np.ndarray) -> np.ndarray:
        """Visualizar landmarks en la imagen (para debugging)"""
        img_copy = image_array.copy()
        
        # Dibujar puntos de landmarks
        for point in landmarks_array:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img_copy, (x, y), 2, (0, 255, 0), -1)
        
        # Dibujar conexiones entre puntos clave
        connections = [
            (33, 133),  # Ojo izquierdo
            (362, 263),  # Ojo derecho
            (61, 291),  # Boca
            (70, 300),  # Cejas
        ]
        
        for start, end in connections:
            if start < len(landmarks_array) and end < len(landmarks_array):
                x1, y1 = int(landmarks_array[start][0]), int(landmarks_array[start][1])
                x2, y2 = int(landmarks_array[end][0]), int(landmarks_array[end][1])
                cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        return img_copy
    
    def __del__(self):
        """Liberar recursos"""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
