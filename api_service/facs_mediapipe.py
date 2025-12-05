"""
facs_mediapipe.py
Analizador de Action Units usando SOLO MediaPipe (compatible con Render)
Sin PyFeat - versión ligera y optimizada
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FACSMediaPipe:
    """Analizador de FACS/AUs usando MediaPipe Face Mesh"""
    
    # Mapeo de índices de landmarks de MediaPipe a regiones faciales
    LANDMARK_REGIONS = {
        # Cejas
        'left_eyebrow': [70, 63, 105, 66, 107],
        'right_eyebrow': [336, 296, 334, 293, 300],
        
        # Ojos
        'left_eye_upper': [159, 145, 158, 157, 173],
        'left_eye_lower': [133, 155, 154, 153, 145],
        'right_eye_upper': [386, 374, 385, 384, 398],
        'right_eye_lower': [362, 382, 381, 380, 374],
        
        # Nariz
        'nose_bridge': [6, 197, 195, 5],
        'nose_tip': [4, 19, 1, 2],
        
        # Boca
        'mouth_outer_upper': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        'mouth_outer_lower': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        'mouth_inner_upper': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        'mouth_inner_lower': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
        
        # Mejillas
        'left_cheek': [205, 216, 207, 187],
        'right_cheek': [425, 436, 427, 411],
        
        # Mandíbula
        'jaw_left': [172, 136, 150, 149],
        'jaw_right': [397, 365, 379, 378],
    }
    
    def __init__(self):
        """Inicializar MediaPipe Face Mesh"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✓ MediaPipe Face Mesh inicializado")
        except Exception as e:
            logger.error(f"Error inicializando MediaPipe: {e}")
            self.face_mesh = None
    
    def analyze(self, image_array: np.ndarray) -> Optional[Dict]:
        """
        Analizar Action Units en una imagen
        
        Args:
            image_array: Imagen en formato numpy array (RGB)
            
        Returns:
            Dict con AUs detectados e interpretación, o None si falla
        """
        if self.face_mesh is None:
            logger.error("MediaPipe no está inicializado")
            return None
        
        try:
            # Convertir BGR a RGB si es necesario
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Asumimos que viene en RGB de PIL
                image_rgb = image_array
            else:
                logger.error("Formato de imagen no válido")
                return None
            
            # Procesar imagen
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                logger.warning("No se detectó ningún rostro")
                return None
            
            # Extraer landmarks del primer rostro
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calcular características geométricas
            landmarks_array = self._landmarks_to_array(face_landmarks, image_rgb.shape)
            
            # Detectar AUs basándose en geometría facial
            detected_aus = self._detect_action_units(landmarks_array)
            
            # Inferir emoción predominante
            emotion = self._infer_emotion(detected_aus)
            
            # Interpretar resultados
            interpretation = self._interpret_aus(detected_aus, emotion)
            
            return {
                "action_units": detected_aus,
                "emotions_mediapipe": emotion,
                "interpretation": interpretation,
                "confidence": 0.75  # Confianza estimada de MediaPipe
            }
            
        except Exception as e:
            logger.error(f"Error en análisis FACS: {e}")
            return None
    
    def _landmarks_to_array(self, face_landmarks, image_shape) -> np.ndarray:
        """Convertir landmarks de MediaPipe a array numpy normalizado"""
        h, w = image_shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            landmarks.append([
                landmark.x * w,
                landmark.y * h,
                landmark.z * w  # Profundidad relativa
            ])
        
        return np.array(landmarks)
    
    def _detect_action_units(self, landmarks: np.ndarray) -> List[Dict]:
        """
        Detectar Action Units basándose en geometría de landmarks
        """
        detected_aus = []
        
        # AU01 - Elevación del párpado interior (Inner Brow Raiser)
        au01_intensity = self._calculate_eyebrow_raise(landmarks, 'left')
        if au01_intensity > 0.5:
            detected_aus.append({
                "code": "AU01",
                "intensity": float(au01_intensity),
                "description": "Elevación párpado interior (sorpresa, miedo)"
            })
        
        # AU02 - Elevación del párpado exterior (Outer Brow Raiser)
        au02_intensity = self._calculate_eyebrow_raise(landmarks, 'right')
        if au02_intensity > 0.5:
            detected_aus.append({
                "code": "AU02",
                "intensity": float(au02_intensity),
                "description": "Elevación párpado exterior (sorpresa)"
            })
        
        # AU04 - Ceño fruncido (Brow Lowerer)
        au04_intensity = self._calculate_brow_furrow(landmarks)
        if au04_intensity > 0.5:
            detected_aus.append({
                "code": "AU04",
                "intensity": float(au04_intensity),
                "description": "Ceño fruncido (concentración, ira)"
            })
        
        # AU06 - Mejillas levantadas (Cheek Raiser)
        au06_intensity = self._calculate_cheek_raise(landmarks)
        if au06_intensity > 0.5:
            detected_aus.append({
                "code": "AU06",
                "intensity": float(au06_intensity),
                "description": "Mejillas levantadas (sonrisa genuina)"
            })
        
        # AU12 - Comisuras labios arriba (Lip Corner Puller)
        au12_intensity = self._calculate_mouth_corners(landmarks)
        if au12_intensity > 0.5:
            detected_aus.append({
                "code": "AU12",
                "intensity": float(au12_intensity),
                "description": "Comisuras labios arriba (sonrisa)"
            })
        
        # AU15 - Comisuras labios abajo (Lip Corner Depressor)
        au15_intensity = self._calculate_mouth_corners_down(landmarks)
        if au15_intensity > 0.5:
            detected_aus.append({
                "code": "AU15",
                "intensity": float(au15_intensity),
                "description": "Comisuras labios abajo (tristeza)"
            })
        
        # AU25 - Labios separados (Lips Part)
        au25_intensity = self._calculate_lips_apart(landmarks)
        if au25_intensity > 0.5:
            detected_aus.append({
                "code": "AU25",
                "intensity": float(au25_intensity),
                "description": "Labios separados (relajación)"
            })
        
        # AU26 - Mandíbula caída (Jaw Drop)
        au26_intensity = self._calculate_jaw_drop(landmarks)
        if au26_intensity > 0.5:
            detected_aus.append({
                "code": "AU26",
                "intensity": float(au26_intensity),
                "description": "Mandíbula caída (sorpresa)"
            })
        
        return sorted(detected_aus, key=lambda x: x["intensity"], reverse=True)
    
    # ==================== FUNCIONES DE CÁLCULO GEOMÉTRICO ====================
    
    def _calculate_eyebrow_raise(self, landmarks: np.ndarray, side: str) -> float:
        """Calcular elevación de cejas"""
        if side == 'left':
            brow_points = self.LANDMARK_REGIONS['left_eyebrow']
            eye_points = self.LANDMARK_REGIONS['left_eye_upper']
        else:
            brow_points = self.LANDMARK_REGIONS['right_eyebrow']
            eye_points = self.LANDMARK_REGIONS['right_eye_upper']
        
        brow_y = np.mean([landmarks[i][1] for i in brow_points])
        eye_y = np.mean([landmarks[i][1] for i in eye_points])
        
        # Distancia normalizada (mayor distancia = ceja más elevada)
        distance = eye_y - brow_y
        # Normalizar a 0-1 (asumiendo rango típico de 20-60 píxeles)
        intensity = np.clip(distance / 40.0, 0, 1)
        
        return intensity
    
    def _calculate_brow_furrow(self, landmarks: np.ndarray) -> float:
        """Calcular ceño fruncido"""
        left_brow = self.LANDMARK_REGIONS['left_eyebrow']
        right_brow = self.LANDMARK_REGIONS['right_eyebrow']
        
        # Distancia entre cejas (menor = más fruncido)
        left_inner = landmarks[left_brow[-1]]
        right_inner = landmarks[right_brow[0]]
        
        distance = np.linalg.norm(left_inner - right_inner)
        
        # Invertir: menor distancia = mayor intensidad
        # Normalizar (asumiendo rango de 30-80 píxeles)
        intensity = np.clip(1 - (distance / 60.0), 0, 1)
        
        return intensity
    
    def _calculate_cheek_raise(self, landmarks: np.ndarray) -> float:
        """Calcular elevación de mejillas"""
        left_cheek = self.LANDMARK_REGIONS['left_cheek']
        right_cheek = self.LANDMARK_REGIONS['right_cheek']
        
        # Altura promedio de mejillas
        left_y = np.mean([landmarks[i][1] for i in left_cheek])
        right_y = np.mean([landmarks[i][1] for i in right_cheek])
        
        # Comparar con posición de ojos (mejillas altas = cerca de ojos)
        left_eye = self.LANDMARK_REGIONS['left_eye_lower']
        right_eye = self.LANDMARK_REGIONS['right_eye_lower']
        
        left_eye_y = np.mean([landmarks[i][1] for i in left_eye])
        right_eye_y = np.mean([landmarks[i][1] for i in right_eye])
        
        left_dist = left_eye_y - left_y
        right_dist = right_eye_y - right_y
        
        avg_dist = (left_dist + right_dist) / 2
        
        # Normalizar
        intensity = np.clip(1 - (avg_dist / 50.0), 0, 1)
        
        return intensity
    
    def _calculate_mouth_corners(self, landmarks: np.ndarray) -> float:
        """Calcular elevación de comisuras (sonrisa)"""
        # Comisuras de la boca
        left_corner = landmarks[61]  # Comisura izquierda
        right_corner = landmarks[291]  # Comisura derecha
        
        # Línea media de la boca
        mouth_center = landmarks[13]
        
        # Calcular si las comisuras están por encima del centro
        left_diff = mouth_center[1] - left_corner[1]
        right_diff = mouth_center[1] - right_corner[1]
        
        avg_diff = (left_diff + right_diff) / 2
        
        # Normalizar
        intensity = np.clip(avg_diff / 15.0, 0, 1)
        
        return intensity
    
    def _calculate_mouth_corners_down(self, landmarks: np.ndarray) -> float:
        """Calcular descenso de comisuras (tristeza)"""
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = landmarks[13]
        
        # Invertir: comisuras abajo del centro
        left_diff = left_corner[1] - mouth_center[1]
        right_diff = right_corner[1] - mouth_center[1]
        
        avg_diff = (left_diff + right_diff) / 2
        
        intensity = np.clip(avg_diff / 15.0, 0, 1)
        
        return intensity
    
    def _calculate_lips_apart(self, landmarks: np.ndarray) -> float:
        """Calcular separación de labios"""
        # Puntos superior e inferior de la boca
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        
        distance = np.linalg.norm(upper_lip - lower_lip)
        
        # Normalizar
        intensity = np.clip(distance / 20.0, 0, 1)
        
        return intensity
    
    def _calculate_jaw_drop(self, landmarks: np.ndarray) -> float:
        """Calcular apertura de mandíbula"""
        # Similar a lips_apart pero más extremo
        upper = landmarks[13]
        lower = landmarks[14]
        
        distance = np.linalg.norm(upper - lower)
        
        # Umbral más alto que lips_apart
        intensity = np.clip((distance - 10) / 30.0, 0, 1)
        
        return intensity
    
    def _infer_emotion(self, aus: List[Dict]) -> Dict[str, float]:
        """Inferir emoción basándose en AUs detectados"""
        emotions = {
            'happiness': 0.0,
            'sadness': 0.0,
            'surprise': 0.0,
            'fear': 0.0,
            'anger': 0.0,
            'disgust': 0.0,
            'neutral': 0.5
        }
        
        au_codes = {au['code']: au['intensity'] for au in aus}
        
        # Felicidad: AU06 + AU12
        if 'AU06' in au_codes and 'AU12' in au_codes:
            emotions['happiness'] = (au_codes['AU06'] + au_codes['AU12']) / 2
        elif 'AU12' in au_codes:
            emotions['happiness'] = au_codes['AU12'] * 0.6
        
        # Tristeza: AU15
        if 'AU15' in au_codes:
            emotions['sadness'] = au_codes['AU15']
        
        # Sorpresa: AU01 + AU02 + AU26
        surprise_aus = [au_codes.get(au, 0) for au in ['AU01', 'AU02', 'AU26']]
        if sum(surprise_aus) > 0:
            emotions['surprise'] = np.mean([x for x in surprise_aus if x > 0])
        
        # Ira: AU04
        if 'AU04' in au_codes:
            emotions['anger'] = au_codes['AU04']
        
        # Neutral por defecto si no hay emociones claras
        if max(emotions.values()) < 0.3:
            emotions['neutral'] = 0.7
        else:
            emotions['neutral'] = 0.2
        
        return emotions
    
    def _interpret_aus(self, aus: List[Dict], emotions: Dict) -> Dict:
        """Interpretar combinaciones de AUs"""
        au_codes = [au["code"] for au in aus]
        indicators = []
        authenticity = 0.0
        
        # Sonrisa de Duchenne (genuina): AU6 + AU12
        if "AU06" in au_codes and "AU12" in au_codes:
            indicators.append({
                "type": "Sonrisa de Duchenne",
                "authenticity": "Alta",
                "note": "Sonrisa genuina con activación de mejillas"
            })
            authenticity += 0.4
        
        # Sonrisa social (falsa): AU12 sin AU6
        elif "AU12" in au_codes and "AU06" not in au_codes:
            indicators.append({
                "type": "Sonrisa social",
                "authenticity": "Baja",
                "note": "Posiblemente cortés o forzada"
            })
            authenticity += 0.1
        
        # Tristeza: AU15
        if "AU15" in au_codes:
            indicators.append({
                "type": "Tristeza",
                "authenticity": "Media",
                "note": "Indicadores de abatimiento"
            })
            authenticity += 0.3
        
        # Sorpresa: AU01 + AU02 + AU26
        surprise_count = sum(1 for au in ["AU01", "AU02", "AU26"] if au in au_codes)
        if surprise_count >= 2:
            indicators.append({
                "type": "Sorpresa",
                "authenticity": "Media-Alta",
                "note": "Expresión de sorpresa o alarma"
            })
            authenticity += 0.3
        
        # Ira: AU04
        if "AU04" in au_codes:
            indicators.append({
                "type": "Concentración/Ira",
                "authenticity": "Media",
                "note": "Ceño fruncido detectado"
            })
            authenticity += 0.2
        
        primary_emotion = max(emotions, key=emotions.get) if emotions else "neutral"
        
        return {
            "primary_emotion": primary_emotion,
            "confidence": float(max(emotions.values())) if emotions else 0.0,
            "microexpression_indicators": indicators,
            "authenticity_score": min(1.0, authenticity)
        }
    
    def __del__(self):
        """Liberar recursos"""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
