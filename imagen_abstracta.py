import numpy as np
import cv2
from PIL import Image, ImageTk
import random
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime
import time

class GeneradorImagenes:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador de Imágenes Abstractas")
        
        # Variables adicionales para la captura en tiempo real
        self.captura = None
        self.capturando = False
        self.contador_capturas = 0
        
        # Variables existentes
        self.imagen_input = ""
        self.carpeta_output = "out"  # Carpeta por defecto
        
        # Asegurar que existe la carpeta por defecto
        if not os.path.exists(self.carpeta_output):
            os.makedirs(self.carpeta_output)
        
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Frame principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Botones existentes
        tk.Button(main_frame, text="Cargar Imagen", command=self.cargar_imagen,
                 width=20, height=2).pack(pady=10)
        
        tk.Button(main_frame, text="Seleccionar Carpeta Destino", 
                 command=self.seleccionar_carpeta, width=20, height=2).pack(pady=10)
        
        tk.Button(main_frame, text="Generar Imagen", command=self.generar,
                 width=20, height=2).pack(pady=10)
        
        # Nuevo botón para captura en tiempo real
        tk.Button(main_frame, text="Captura en Tiempo Real", 
                 command=self.toggle_captura_tiempo_real,
                 width=20, height=2).pack(pady=10)
        
        # Labels para mostrar rutas seleccionadas
        self.label_imagen = tk.Label(main_frame, text="Imagen no seleccionada",
                                   wraplength=300)
        self.label_imagen.pack(pady=5)
        
        self.label_carpeta = tk.Label(main_frame, text="Carpeta no seleccionada",
                                    wraplength=300)
        self.label_carpeta.pack(pady=5)
        
        # Preview de la imagen
        self.preview_label = tk.Label(main_frame)
        self.preview_label.pack(pady=10)
        
        # Frame para visualización en tiempo real
        self.frame_video = tk.Frame(main_frame)
        self.frame_video.pack(pady=10)
        
        # Labels para mostrar video y resultado
        self.label_video = tk.Label(self.frame_video)
        self.label_resultado = tk.Label(self.frame_video)
        
    def toggle_captura_tiempo_real(self):
        if not self.capturando:
            self.iniciar_captura()
        else:
            self.detener_captura()
            
    def iniciar_captura(self):
        if self.captura is None:
            self.captura = cv2.VideoCapture(0)
            if not self.captura.isOpened():
                messagebox.showerror("Error", "No se puede acceder a la cámara")
                return
        
        self.capturando = True
        self.root.state('zoomed')  # Maximizar ventana
        
        # Configurar layout para pantalla completa
        self.label_video.pack(side=tk.LEFT, padx=5)
        self.label_resultado.pack(side=tk.RIGHT, padx=5)
        
        self.ultima_captura = time.time()
        self.actualizar_frame()
        
    def detener_captura(self):
        self.capturando = False
        if self.captura is not None:
            self.captura.release()
            self.captura = None
        self.label_video.pack_forget()
        self.label_resultado.pack_forget()
        self.root.state('normal')  # Restaurar tamaño de ventana
        
    def actualizar_frame(self):
        if not self.capturando:
            return
            
        ret, frame = self.captura.read()
        if ret:
            # Mostrar frame de la cámara
            frame = cv2.flip(frame, 1)  # Espejo
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 480))
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.label_video.config(image=frame_tk)
            self.label_video.image = frame_tk
            
            # Verificar si han pasado 20 segundos
            tiempo_actual = time.time()
            if tiempo_actual - self.ultima_captura >= 20:
                self.ultima_captura = tiempo_actual
                self.procesar_frame_actual(frame)
                
        self.root.after(10, self.actualizar_frame)
        
    def procesar_frame_actual(self, frame):
        # Generar nombre del archivo
        self.contador_capturas += 1
        nombre_archivo = f"captura_{self.contador_capturas}.jpg"
        ruta_temp = os.path.join(self.carpeta_output, "temp.jpg")
        ruta_salida = os.path.join(self.carpeta_output, nombre_archivo)
        
        # Guardar frame actual
        cv2.imwrite(ruta_temp, frame)
        
        # Generar imagen abstracta
        self.generar_imagen_abstracta(ruta_temp, ruta_salida)
        
        # Mostrar resultado
        resultado = Image.open(ruta_salida)
        resultado = resultado.resize((640, 480))
        resultado_tk = ImageTk.PhotoImage(resultado)
        self.label_resultado.config(image=resultado_tk)
        self.label_resultado.image = resultado_tk
        
        # Eliminar archivo temporal
        os.remove(ruta_temp)
        
    def cargar_imagen(self):
        self.imagen_input = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if self.imagen_input:
            self.label_imagen.config(text=f"Imagen: {self.imagen_input}")
            self.mostrar_preview()
            
    def seleccionar_carpeta(self):
        self.carpeta_output = filedialog.askdirectory()
        if self.carpeta_output:
            self.label_carpeta.config(text=f"Carpeta: {self.carpeta_output}")
            
    def mostrar_preview(self):
        # Mostrar preview de la imagen seleccionada
        imagen = Image.open(self.imagen_input)
        imagen = imagen.resize((200, 200), Image.Resampling.LANCZOS)
        imagen = ImageTk.PhotoImage(imagen)
        self.preview_label.config(image=imagen)
        self.preview_label.image = imagen
        
    def generar_imagen_abstracta(self, imagen_input, output_path):
        # Cargar y redimensionar la imagen
        img = cv2.imread(imagen_input)
        img = cv2.resize(img, (800, 800))
        
        # Obtener más colores dominantes (aumentado a 15)
        pixels = img.reshape(-1, 3)
        colores_dominantes = pixels[np.random.choice(pixels.shape[0], 15, replace=False)]
        
        # Añadir variaciones de color
        colores_adicionales = []
        for color in colores_dominantes:
            color_claro = np.clip(color * 1.5, 0, 255)
            color_oscuro = color * 0.5
            color_saturado = np.clip(color * 2.0, 0, 255)
            colores_adicionales.extend([color_claro, color_oscuro, color_saturado])
        
        colores_totales = np.vstack([colores_dominantes, np.array(colores_adicionales)])
        
        # Crear lienzo con fondo aleatorio
        canvas = np.ones((800, 800, 3), dtype=np.uint8) * random.randint(200, 255)
        
        # Funciones para generar diferentes formas
        def generar_curva_bezier():
            pts = np.random.randint(0, 800, (4, 2))
            t = np.linspace(0, 1, 100)
            curva = np.zeros((100, 2))
            for i in range(100):
                t_i = t[i]
                curva[i] = (1-t_i)**3 * pts[0] + 3*(1-t_i)**2*t_i * pts[1] + \
                           3*(1-t_i)*t_i**2 * pts[2] + t_i**3 * pts[3]
            return curva.astype(np.int32)

        def deformar_puntos(puntos, intensidad=1.0):
            """Aplica deformaciones aleatorias a los puntos"""
            puntos = np.array(puntos, dtype=np.float32)
            
            # Diferentes tipos de deformaciones
            deformaciones = [
                # Ondulación
                lambda p: p + intensidad * np.sin(p * random.uniform(0.01, 0.05)) * random.uniform(10, 30),
                
                # Torsión
                lambda p: np.array([
                    p[:, 0] * np.cos(p[:, 1] * 0.01 * intensidad) + p[:, 1] * np.sin(p[:, 0] * 0.01 * intensidad),
                    p[:, 1] * np.cos(p[:, 0] * 0.01 * intensidad) - p[:, 0] * np.sin(p[:, 1] * 0.01 * intensidad)
                ]).T,
                
                # Explosión/Implosión (corregida)
                lambda p: p * np.column_stack([
                    1 + random.uniform(-0.3, 0.3) * intensidad * np.sin(np.arctan2(p[:, 1], p[:, 0])),
                    1 + random.uniform(-0.3, 0.3) * intensidad * np.sin(np.arctan2(p[:, 1], p[:, 0]))
                ]),
                
                # Ruido Perlin-like
                lambda p: p + np.array([
                    [np.sin(x*0.05 + y*0.03) * random.uniform(10, 20) * intensidad 
                     for x, y in zip(p[:, 0], p[:, 1])]
                    for _ in range(2)
                ]).T
            ]
            
            # Aplicar múltiples deformaciones aleatorias
            num_deformaciones = random.randint(1, 3)
            for _ in range(num_deformaciones):
                deformacion = random.choice(deformaciones)
                puntos = deformacion(puntos)
                
            return puntos.astype(np.int32)

        def generar_forma_organica():
            centro = np.random.randint(100, 700, 2)
            puntos = []
            num_puntos = random.randint(12, 25)  # Aumentado el número de puntos
            
            # Generar forma base
            for i in range(num_puntos):
                angulo = 2 * np.pi * i / num_puntos
                radio = random.randint(30, 100)
                # Añadir variación al radio
                radio *= (1 + random.uniform(-0.3, 0.3))
                x = centro[0] + int(radio * np.cos(angulo))
                y = centro[1] + int(radio * np.sin(angulo))
                puntos.append([x, y])
            
            # Deformar la forma
            puntos = deformar_puntos(puntos, random.uniform(0.5, 2.0))
            return puntos

        def generar_espiral():
            centro = np.random.randint(100, 700, 2)
            puntos = []
            vueltas = random.uniform(2, 5)
            pasos = 200
            
            for i in range(pasos):
                t = i / pasos * 2 * np.pi * vueltas
                radio = t * 2
                # Añadir variación al radio
                radio *= (1 + random.uniform(-0.2, 0.2))
                x = centro[0] + int(radio * np.cos(t))
                y = centro[1] + int(radio * np.sin(t))
                puntos.append([x, y])
            
            puntos = deformar_puntos(np.array(puntos), random.uniform(0.3, 1.5))
            return puntos

        def generar_patron_recursivo():
            puntos = []
            nivel = random.randint(3, 5)
            tam_inicial = random.randint(100, 200)
            
            def recursion(x, y, tam, nivel):
                if nivel <= 0:
                    return
                puntos.extend([[x, y], [x+tam, y], [x+tam, y+tam], [x, y+tam]])
                nuevo_tam = tam // 2
                recursion(x, y, nuevo_tam, nivel-1)
                recursion(x+tam//2, y, nuevo_tam, nivel-1)
                
            recursion(200, 200, tam_inicial, nivel)
            return np.array(puntos)

        def generar_onda_sinusoidal():
            puntos = []
            amplitud = random.randint(50, 150)
            frecuencia = random.uniform(0.01, 0.03)
            for x in range(0, 800, 2):
                y = 400 + amplitud * np.sin(x * frecuencia)
                puntos.append([x, int(y)])
            return np.array(puntos)

        def generar_explosion():
            centro = np.random.randint(100, 700, 2)
            puntos = []
            num_lineas = random.randint(15, 30)
            
            for i in range(num_lineas):
                angulo = 2 * np.pi * i / num_lineas
                # Variar la longitud de cada línea
                longitud = random.randint(50, 200) * (1 + random.uniform(-0.4, 0.4))
                # Añadir puntos intermedios para deformación
                num_puntos = random.randint(3, 6)
                for j in range(num_puntos):
                    t = j / (num_puntos - 1)
                    actual_longitud = longitud * t
                    x = centro[0] + int(actual_longitud * np.cos(angulo))
                    y = centro[1] + int(actual_longitud * np.sin(angulo))
                    puntos.append([x, y])
                
            puntos = deformar_puntos(np.array(puntos), random.uniform(0.5, 1.8))
            return puntos

        def generar_patron_voronoi():
            puntos_centro = np.random.randint(0, 800, (random.randint(5, 10), 2))
            puntos = []
            for centro in puntos_centro:
                for _ in range(20):
                    offset = np.random.normal(0, 30, 2)
                    punto = centro + offset
                    puntos.append(punto)
            return np.array(puntos, dtype=np.int32)

        # Generar capas de formas
        for _ in range(100):
            color = colores_totales[random.randint(0, len(colores_totales)-1)]
            
            # Elegir tipo de forma aleatoriamente (añadidos nuevos tipos)
            forma_tipo = random.choice([
                'poligono', 'curva', 'organica', 'circulos',
                'espiral', 'recursivo', 'onda', 'explosion', 'voronoi'
            ])
            
            # Aplicar deformaciones a todas las formas
            if forma_tipo == 'poligono':
                puntos = np.random.randint(0, 800, (random.randint(3, 8), 2))
                puntos = deformar_puntos(puntos, random.uniform(0.3, 1.5))
                cv2.fillPoly(canvas, [puntos], color.tolist())
                
            elif forma_tipo == 'curva':
                puntos = generar_curva_bezier()
                puntos = deformar_puntos(puntos, random.uniform(0.2, 1.0))
                cv2.polylines(canvas, [puntos], False, color.tolist(), 
                             random.randint(2, 8), cv2.LINE_AA)
                
            elif forma_tipo == 'organica':
                puntos = generar_forma_organica()
                cv2.fillPoly(canvas, [puntos], color.tolist())
                
            elif forma_tipo == 'circulos':
                centro = (random.randint(0, 800), random.randint(0, 800))
                radio = random.randint(20, 150)
                cv2.circle(canvas, centro, radio, color.tolist(), 
                          random.choice([-1, random.randint(5, 20)]))
            
            elif forma_tipo == 'espiral':
                puntos = generar_espiral()
                cv2.polylines(canvas, [puntos], False, color.tolist(), 
                             random.randint(2, 8), cv2.LINE_AA)
                
            elif forma_tipo == 'recursivo':
                puntos = generar_patron_recursivo()
                cv2.polylines(canvas, [puntos], True, color.tolist(), 
                             random.randint(1, 4), cv2.LINE_AA)
                
            elif forma_tipo == 'onda':
                puntos = generar_onda_sinusoidal()
                cv2.polylines(canvas, [puntos], False, color.tolist(), 
                             random.randint(3, 10), cv2.LINE_AA)
                
            elif forma_tipo == 'explosion':
                puntos = generar_explosion()
                cv2.polylines(canvas, [puntos], False, color.tolist(), 
                             random.randint(1, 5), cv2.LINE_AA)
                
            elif forma_tipo == 'voronoi':
                puntos = generar_patron_voronoi()
                hull = cv2.convexHull(puntos)
                cv2.fillPoly(canvas, [hull], color.tolist())
            
        # Aplicar efectos finales
        canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
        canvas = cv2.convertScaleAbs(canvas, alpha=1.1, beta=10)
        
        # Guardar la imagen
        cv2.imwrite(output_path, canvas)
        
    def generar(self):
        if not self.imagen_input or not self.carpeta_output:
            messagebox.showerror("Error", "Por favor selecciona una imagen y una carpeta de destino")
            return
            
        # Generar nombre único para el archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"imagen_abstracta_{timestamp}.jpg"
        ruta_completa = os.path.join(self.carpeta_output, nombre_archivo)
        
        try:
            self.generar_imagen_abstracta(self.imagen_input, ruta_completa)
            messagebox.showinfo("Éxito", f"Imagen generada exitosamente:\n{ruta_completa}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar la imagen: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneradorImagenes(root)
    root.mainloop()
