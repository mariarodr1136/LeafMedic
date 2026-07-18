#!/usr/bin/env python3
"""Build docs/data/treatments.es.json from the English knowledge base.

Severity values are copied verbatim (they are an enum, not prose). Every other
field is translated. Usage: python3 tools/build_translations.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CARE = "Continúa con los cuidados y la vigilancia habituales"
NO_DISEASE = "La planta no muestra signos de enfermedad."

ES = {
"Apple___Apple_scab": {
 "common_name": "Roña del manzano", "plant": "Manzano", "disease": "Roña del manzano",
 "description": "Enfermedad fúngica causada por Venturia inaequalis, que produce lesiones oscuras y costrosas en hojas y frutos.",
 "symptoms": ["Lesiones de verde oliva oscuro a marrón en las hojas", "Manchas costrosas en el fruto", "Caída prematura de las hojas", "Frutos agrietados o deformados"],
 "treatments": ["Retira y destruye las hojas caídas", "Aplica fungicidas (captan, miclobutanil) en periodos húmedos", "Poda los árboles para mejorar la circulación del aire", "Utiliza variedades de manzano resistentes"],
 "prevention": ["Elige cultivares resistentes a la roña", "Mantén un espaciado adecuado entre árboles", "Evita el riego por aspersión", "Retira los restos vegetales infectados"]},
"Apple___Black_rot": {
 "common_name": "Podredumbre negra del manzano", "plant": "Manzano", "disease": "Podredumbre negra",
 "description": "Enfermedad fúngica que afecta a hojas, frutos y corteza, causada por Botryosphaeria obtusa.",
 "symptoms": ["Manchas moradas en las hojas que se extienden en grandes zonas marrones", "Frutos podridos con anillos concéntricos", "Chancros en las ramas", "Frutos negros y arrugados (momias)"],
 "treatments": ["Retira todo el fruto y la madera infectados", "Aplica fungicidas durante la floración y el cuajado", "Poda los chancros de las ramas", "Destruye los frutos momificados"],
 "prevention": ["Higiene y poda adecuadas", "Elimina la madera muerta", "Evita el estrés del árbol con una fertilización correcta", "Aplica fungicidas preventivos"]},
"Apple___Cedar_apple_rust": {
 "common_name": "Roya del manzano y el cedro", "plant": "Manzano", "disease": "Roya del manzano y el cedro",
 "description": "Enfermedad fúngica que necesita tanto manzanos como cedros para completar su ciclo de vida.",
 "symptoms": ["Manchas amarillo-anaranjadas en el haz de la hoja", "Proyecciones tubulares en el envés", "Lesiones en el fruto", "Defoliación prematura"],
 "treatments": ["Aplica fungicidas (miclobutanil, mancozeb) desde la brotación", "Retira los cedros cercanos si es posible", "Elimina las agallas de los cedros en invierno", "Utiliza variedades de manzano resistentes a la roya"],
 "prevention": ["Planta variedades de manzano resistentes", "Separa los manzanos de los cedros y enebros", "Vigila y trata al principio de la temporada"]},
"Apple___healthy": {
 "common_name": "Manzano sano", "plant": "Manzano", "disease": "Ninguna — sana",
 "description": "La planta no muestra signos de enfermedad. Las hojas están verdes y vigorosas.",
 "treatments": [CARE],
 "prevention": ["Mantén buenas prácticas de cultivo", "Revisa con regularidad para detectar enfermedades a tiempo", "Asegura una nutrición y un riego adecuados"]},
"Blueberry___healthy": {
 "common_name": "Arándano sano", "plant": "Arándano", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Mantén el suelo ácido (pH 4,5-5,5)", "Poda adecuada", "Drenaje suficiente"]},
"Cherry_(including_sour)___Powdery_mildew": {
 "common_name": "Oídio del cerezo", "plant": "Cerezo", "disease": "Oídio",
 "description": "Enfermedad fúngica que produce un polvillo blanco sobre hojas y brotes.",
 "symptoms": ["Capa blanca polvorienta sobre las hojas", "Enrollamiento y deformación de las hojas", "Crecimiento atrofiado de los brotes", "Menor calidad del fruto"],
 "treatments": ["Aplica azufre o bicarbonato potásico en aspersión", "Utiliza fungicidas sistémicos (miclobutanil)", "Retira los brotes muy infectados", "Mejora la circulación del aire"],
 "prevention": ["Elige variedades resistentes", "Evita el exceso de fertilización nitrogenada", "Espaciado adecuado para que circule el aire", "Retira los restos infectados"]},
"Cherry_(including_sour)___healthy": {
 "common_name": "Cerezo sano", "plant": "Cerezo", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Técnicas de poda adecuadas", "Buena circulación del aire", "Vigilancia periódica"]},
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
 "common_name": "Mancha gris del maíz", "plant": "Maíz", "disease": "Mancha gris de la hoja",
 "description": "Enfermedad fúngica que produce lesiones grises rectangulares en las hojas.",
 "symptoms": ["Lesiones grises largas y estrechas en las hojas", "Lesiones paralelas a los nervios", "Muerte prematura de la hoja", "Menor rendimiento"],
 "treatments": ["Aplica fungicidas foliares (azoxistrobina, propiconazol)", "Practica la rotación de cultivos", "Utiliza híbridos resistentes", "Labra para enterrar los restos del cultivo"],
 "prevention": ["Planta híbridos resistentes", "Rota los cultivos (evita maíz continuo)", "Gestiona los restos del cultivo", "Evita el riego por aspersión"]},
"Corn_(maize)___Common_rust_": {
 "common_name": "Roya común del maíz", "plant": "Maíz", "disease": "Roya común",
 "description": "Enfermedad fúngica que crea pústulas de color óxido en las hojas.",
 "symptoms": ["Pequeñas pústulas marrón rojizas en ambas caras de la hoja", "Las pústulas liberan esporas de color óxido", "Amarilleo y muerte de la hoja", "Menor fotosíntesis"],
 "treatments": ["Aplica fungicidas si la enfermedad es grave", "Planta híbridos resistentes", "Retira los restos vegetales infectados", "En climas secos no suele requerir tratamiento"],
 "prevention": ["Utiliza híbridos resistentes", "Siembra temprano para evitar los picos de la enfermedad", "Buen drenaje del terreno"]},
"Corn_(maize)___Northern_Leaf_Blight": {
 "common_name": "Tizón foliar norteño", "plant": "Maíz", "disease": "Tizón foliar norteño",
 "description": "Enfermedad fúngica que produce lesiones alargadas con forma de puro en las hojas del maíz.",
 "symptoms": ["Lesiones largas de gris verdoso a canela", "Manchas con forma de puro", "Las lesiones pueden unirse y matar hojas enteras", "Posible pérdida grave de rendimiento"],
 "treatments": ["Aplica fungicidas foliares (estrobilurinas, triazoles)", "Utiliza híbridos resistentes", "Practica la rotación de cultivos", "Entierra los restos del cultivo con laboreo"],
 "prevention": ["Planta híbridos resistentes", "Rota con cultivos no hospedantes", "Gestiona los restos del cultivo", "Evita la siembra tardía"]},
"Corn_(maize)___healthy": {
 "common_name": "Maíz sano", "plant": "Maíz", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Fertilización adecuada", "Riego suficiente", "Control de malas hierbas", "Vigila plagas y enfermedades"]},
"Grape___Black_rot": {
 "common_name": "Podredumbre negra de la vid", "plant": "Vid", "disease": "Podredumbre negra",
 "description": "Enfermedad fúngica grave de la vid que provoca la momificación del fruto.",
 "symptoms": ["Manchas canela o marrones con bordes oscuros en las hojas", "Bayas podridas de aspecto negro y arrugado", "Frutos momificados", "Menor rendimiento"],
 "treatments": ["Aplica fungicidas (mancozeb, captan) desde la brotación hasta el cuajado", "Retira y destruye los frutos momificados", "Poda para favorecer la circulación del aire", "Retira las hojas infectadas"],
 "prevention": ["Gestión adecuada del follaje", "Retira las momias y los restos infectados", "Programa preventivo de fungicidas", "Buena circulación del aire"]},
"Grape___Esca_(Black_Measles)": {
 "common_name": "Yesca de la vid (sarampión negro)", "plant": "Vid", "disease": "Yesca (sarampión negro)",
 "description": "Enfermedad compleja causada por varios hongos que afecta a la estructura de la cepa y al fruto.",
 "symptoms": ["Patrones de rayas atigradas en las hojas", "Manchas negras en las bayas", "Marchitez y decaimiento de la cepa", "Pudrición de la madera"],
 "treatments": ["No hay cura eficaz: gestiona los síntomas", "Poda y elimina la madera infectada", "Protege las heridas de poda", "Arranca las cepas muy afectadas", "Mantén el vigor de la cepa"],
 "prevention": ["Usa herramientas de poda limpias", "Sella las heridas de poda grandes", "Evita el estrés de la cepa", "Selecciona material de plantación sano"]},
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
 "common_name": "Tizón foliar de la vid", "plant": "Vid", "disease": "Tizón foliar (mancha de Isariopsis)",
 "description": "Enfermedad fúngica que provoca manchas foliares y defoliación prematura.",
 "symptoms": ["Manchas marrones angulosas en las hojas", "Caída prematura de las hojas", "Menor fotosíntesis", "Cepas debilitadas"],
 "treatments": ["Aplica fungicidas (a base de cobre, mancozeb)", "Retira las hojas infectadas", "Mejora la circulación del aire", "Mantén la salud de la cepa"],
 "prevention": ["Gestión adecuada del follaje", "Evita el riego por aspersión", "Buena higiene del cultivo", "Variedades resistentes"]},
"Grape___healthy": {
 "common_name": "Vid sana", "plant": "Vid", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Poda adecuada", "Buen flujo de aire", "Vigilancia periódica", "Nutrición equilibrada"]},
"Orange___Haunglongbing_(Citrus_greening)": {
 "common_name": "Huanglongbing (HLB)", "plant": "Naranjo", "disease": "Huanglongbing (enverdecimiento de los cítricos)",
 "description": "Enfermedad bacteriana devastadora transmitida por psílidos. Actualmente no tiene cura.",
 "symptoms": ["Brotes amarillos con moteado irregular", "Frutos asimétricos y amargos", "Caída prematura del fruto", "Decaimiento y muerte del árbol"],
 "treatments": ["SIN CURA: arranca y destruye los árboles infectados", "Controla el psílido vector con insecticidas", "Terapia nutricional para prolongar la vida del árbol", "Establece cuarentena en las zonas infectadas"],
 "prevention": ["Usa plantones certificados libres de enfermedad", "Controla el psílido asiático de los cítricos", "Retira los árboles infectados de inmediato", "Planta bajo malla protectora"]},
"Peach___Bacterial_spot": {
 "common_name": "Mancha bacteriana del melocotonero", "plant": "Melocotonero", "disease": "Mancha bacteriana",
 "description": "Enfermedad bacteriana que afecta a hojas y frutos de los frutales de hueso.",
 "symptoms": ["Pequeñas manchas oscuras con halo amarillo en las hojas", "Manchas en el fruto", "Caída prematura de las hojas", "Menor calidad del fruto"],
 "treatments": ["Aplica bactericidas a base de cobre", "Usa antibióticos (estreptomicina) donde esté permitido", "Poda las ramas infectadas", "Planta variedades resistentes"],
 "prevention": ["Elige cultivares resistentes", "Evita el riego por aspersión", "Espaciado adecuado", "Tratamientos de cobre en reposo invernal"]},
"Peach___healthy": {
 "common_name": "Melocotonero sano", "plant": "Melocotonero", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Poda adecuada", "Fertilización suficiente", "Vigilancia periódica", "Buena higiene del cultivo"]},
"Pepper,_bell___Bacterial_spot": {
 "common_name": "Mancha bacteriana del pimiento", "plant": "Pimiento", "disease": "Mancha bacteriana",
 "description": "Enfermedad bacteriana que provoca manchas foliares y lesiones en el fruto.",
 "symptoms": ["Pequeñas manchas marrones en las hojas", "Manchas abultadas en el fruto", "Amarilleo y caída de las hojas", "Menor rendimiento y calidad del fruto"],
 "treatments": ["Aplica bactericidas a base de cobre", "Retira y destruye las plantas infectadas", "Mejora la circulación del aire", "Evita el riego por encima del follaje"],
 "prevention": ["Usa semillas y plantones libres de enfermedad", "Rota los cultivos (rotación de 3 años)", "No manipules las plantas mojadas", "Aplica cobre de forma preventiva"]},
"Pepper,_bell___healthy": {
 "common_name": "Pimiento sano", "plant": "Pimiento", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Rotación de cultivos", "Riego adecuado", "Buena higiene del cultivo", "Vigilancia periódica"]},
"Potato___Early_blight": {
 "common_name": "Alternariosis de la patata", "plant": "Patata", "disease": "Tizón temprano",
 "description": "Enfermedad fúngica que provoca manchas foliares y reduce el rendimiento.",
 "symptoms": ["Manchas marrón oscuro con anillos concéntricos (tipo diana)", "Las hojas inferiores se afectan primero", "Amarilleo alrededor de las lesiones", "Defoliación prematura"],
 "treatments": ["Aplica fungicidas (clorotalonil, mancozeb)", "Retira las hojas inferiores infectadas", "Mejora la circulación del aire", "Acolcha para evitar salpicaduras de tierra"],
 "prevention": ["Usa patata de siembra certificada", "Rota los cultivos (3-4 años)", "Espaciado adecuado entre plantas", "Evita el riego por aspersión"]},
"Potato___Late_blight": {
 "common_name": "Mildiu de la patata", "plant": "Patata", "disease": "Tizón tardío",
 "description": "Enfermedad devastadora que causó la Gran Hambruna Irlandesa. Puede destruir cultivos con enorme rapidez.",
 "symptoms": ["Manchas de aspecto acuoso en las hojas", "Crecimiento fúngico blanco en el envés", "Lesiones marrón negruzcas en los tallos", "Podredumbre de los tubérculos"],
 "treatments": ["Aplica fungicidas de inmediato (clorotalonil, metalaxil)", "Retira y destruye las plantas infectadas", "Cosecha antes de que la enfermedad alcance los tubérculos", "Destruye todos los restos infectados"],
 "prevention": ["Usa variedades resistentes", "Vigila las condiciones meteorológicas favorables", "Aplicaciones preventivas de fungicida", "Rotación de cultivos adecuada"]},
"Potato___healthy": {
 "common_name": "Patata sana", "plant": "Patata", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Rotación de cultivos", "Patata de siembra certificada", "Buen drenaje", "Vigilancia periódica"]},
"Raspberry___healthy": {
 "common_name": "Frambueso sano", "plant": "Frambueso", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Poda adecuada", "Buena circulación del aire", "Vigilancia periódica", "Elimina las cañas viejas"]},
"Soybean___healthy": {
 "common_name": "Soja sana", "plant": "Soja", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Rotación de cultivos", "Densidad de siembra adecuada", "Control de malas hierbas", "Vigila la aparición de enfermedades"]},
"Squash___Powdery_mildew": {
 "common_name": "Oídio de la calabaza", "plant": "Calabaza", "disease": "Oídio",
 "description": "Enfermedad fúngica frecuente que forma un polvillo blanco en las hojas de las cucurbitáceas.",
 "symptoms": ["Capa blanca polvorienta sobre las hojas", "Manchas amarillas bajo el polvillo", "Muerte prematura de la hoja", "Menor rendimiento y calidad del fruto"],
 "treatments": ["Aplica fungicidas (azufre, bicarbonato potásico, aceite de nim)", "Retira las hojas muy infectadas", "Mejora la circulación del aire", "Usa una aspersión de leche (1:9 leche y agua)"],
 "prevention": ["Planta variedades resistentes", "Espaciado adecuado", "Evita el riego por encima del follaje", "Aplica fungicidas preventivos"]},
"Strawberry___Leaf_scorch": {
 "common_name": "Quemadura foliar del fresal", "plant": "Fresa", "disease": "Quemadura foliar",
 "description": "Enfermedad fúngica que provoca manchas foliares con borde morado.",
 "symptoms": ["Manchas moradas irregulares en las hojas", "Las manchas se agrandan y se vuelven marrones", "Aspecto quemado", "Menor vigor de la planta"],
 "treatments": ["Retira y destruye las hojas infectadas", "Aplica fungicidas (captan, tiram)", "Mejora la circulación del aire", "Renueva la plantación tras la cosecha"],
 "prevention": ["Usa material de plantación libre de enfermedad", "Evita el riego por aspersión", "Espaciado adecuado entre plantas", "Retira las hojas viejas"]},
"Strawberry___healthy": {
 "common_name": "Fresa sana", "plant": "Fresa", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Buena higiene del cultivo", "Espaciado adecuado", "Renovación periódica", "Vigilancia periódica"]},
"Tomato___Bacterial_spot": {
 "common_name": "Mancha bacteriana del tomate", "plant": "Tomate", "disease": "Mancha bacteriana",
 "description": "Enfermedad bacteriana que afecta a las hojas y los frutos del tomate.",
 "symptoms": ["Pequeñas manchas marrón oscuro en las hojas", "Halos amarillos alrededor de las manchas", "Lesiones marrones abultadas en el fruto", "Defoliación prematura"],
 "treatments": ["Aplica bactericidas a base de cobre", "Retira las plantas infectadas", "Mejora la circulación del aire", "Evita el riego por encima del follaje"],
 "prevention": ["Usa semillas certificadas libres de enfermedad", "Rota los cultivos (no con pimientos)", "No manipules las plantas mojadas", "Tratamientos con cobre"]},
"Tomato___Early_blight": {
 "common_name": "Alternariosis del tomate", "plant": "Tomate", "disease": "Tizón temprano",
 "description": "Enfermedad fúngica común del tomate que produce lesiones características en forma de diana.",
 "symptoms": ["Manchas marrón oscuro con anillos concéntricos", "Las hojas inferiores se afectan primero", "Amarilleo alrededor de las lesiones", "Podredumbre del cuello en plántulas"],
 "treatments": ["Aplica fungicidas (clorotalonil, mancozeb)", "Retira las hojas inferiores infectadas", "Entutora las plantas para mejorar el flujo de aire", "Acolcha para evitar salpicaduras de tierra"],
 "prevention": ["Rota los cultivos", "Espaciado adecuado entre plantas", "Acolcha las plantas", "Riega en la base de la planta"]},
"Tomato___Late_blight": {
 "common_name": "Mildiu del tomate", "plant": "Tomate", "disease": "Tizón tardío",
 "description": "Enfermedad devastadora de tipo fúngico que puede destruir un cultivo de tomate en pocos días.",
 "symptoms": ["Grandes lesiones marrón negruzcas en las hojas", "Moho blanco en el envés con humedad alta", "Lesiones en los tallos", "Podredumbre del fruto"],
 "treatments": ["Aplica fungicidas de inmediato (clorotalonil, cobre)", "Retira y destruye todas las plantas infectadas", "No compostes el material infectado", "Aplica fungicidas preventivos con tiempo fresco y húmedo"],
 "prevention": ["Usa variedades resistentes", "Espaciado adecuado", "Evita el riego por encima del follaje", "Vigila la meteorología"]},
"Tomato___Leaf_Mold": {
 "common_name": "Moho foliar del tomate", "plant": "Tomate", "disease": "Moho foliar",
 "description": "Enfermedad fúngica frecuente en tomate de invernadero con ventilación deficiente.",
 "symptoms": ["Manchas amarillas en el haz de la hoja", "Vello de verde oliva a marrón en el envés", "Enrollamiento y muerte de la hoja", "Menor rendimiento"],
 "treatments": ["Mejora la ventilación y reduce la humedad", "Aplica fungicidas (clorotalonil, cobre)", "Retira las hojas infectadas", "Aumenta el espaciado entre plantas"],
 "prevention": ["Usa variedades resistentes", "Asegura una buena circulación del aire", "Evita el riego por encima del follaje", "Controla la humedad en los invernaderos"]},
"Tomato___Septoria_leaf_spot": {
 "common_name": "Septoriosis del tomate", "plant": "Tomate", "disease": "Mancha foliar por Septoria",
 "description": "Enfermedad fúngica muy extendida que produce numerosas manchas pequeñas en las hojas.",
 "symptoms": ["Manchas circulares pequeñas con centro gris y borde oscuro", "Numerosas manchas primero en las hojas inferiores", "Puntos negros (picnidios) en el centro de las manchas", "Defoliación progresiva"],
 "treatments": ["Aplica fungicidas (clorotalonil, mancozeb)", "Retira las hojas inferiores infectadas", "Acolcha para evitar salpicaduras de tierra", "Mejora la circulación del aire"],
 "prevention": ["Rota los cultivos", "Riega a ras de suelo", "Acolcha las plantas", "Espacia bien las plantas"]},
"Tomato___Spider_mites Two-spotted_spider_mite": {
 "common_name": "Araña roja de dos manchas", "plant": "Tomate", "disease": "Infestación de araña roja",
 "description": "No es una enfermedad sino daño por plaga: diminutos arácnidos que succionan la savia de la planta.",
 "symptoms": ["Punteado amarillo en las hojas", "Telarañas finas en la planta", "Hojas bronceadas o plateadas", "Caída de hojas y decaimiento de la planta"],
 "treatments": ["Pulveriza con agua para desalojar los ácaros", "Aplica jabón insecticida o aceite de nim", "Usa acaricidas si la infestación es grave", "Introduce ácaros depredadores (control biológico)"],
 "prevention": ["Mantén una humedad del suelo adecuada", "Evita los ambientes polvorientos", "Favorece la presencia de insectos beneficiosos", "Vigila con frecuencia en tiempo caluroso y seco"]},
"Tomato___Target_Spot": {
 "common_name": "Mancha diana del tomate", "plant": "Tomate", "disease": "Mancha diana",
 "description": "Enfermedad fúngica similar al tizón temprano pero con anillos concéntricos de patrón distinto.",
 "symptoms": ["Manchas marrones con anillos concéntricos", "Lesiones en hojas, tallos y frutos", "Defoliación prematura", "Menor rendimiento"],
 "treatments": ["Aplica fungicidas (clorotalonil, azoxistrobina)", "Retira los restos vegetales infectados", "Mejora la circulación del aire", "Acolcha para reducir las salpicaduras de tierra"],
 "prevention": ["Rotación de cultivos", "Espaciado adecuado entre plantas", "Evita el riego por aspersión", "Usa tutores y jaulas limpios"]},
"Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
 "common_name": "Virus del rizado amarillo del tomate", "plant": "Tomate", "disease": "Virus del rizado amarillo (TYLCV)",
 "description": "Enfermedad viral transmitida por la mosca blanca que provoca pérdidas graves de rendimiento.",
 "symptoms": ["Rizado y amarilleo intensos de las hojas", "Crecimiento atrofiado de la planta", "Menor cuajado de frutos", "Hojas ahuecadas hacia arriba"],
 "treatments": ["SIN CURA: arranca y destruye las plantas infectadas", "Controla la mosca blanca vectora con insecticidas", "Usa acolchados reflectantes para repeler la mosca blanca", "Planta variedades resistentes al virus"],
 "prevention": ["Usa variedades resistentes", "Controla la mosca blanca", "Usa mallas antiinsectos", "Retira las plantas infectadas de inmediato"]},
"Tomato___Tomato_mosaic_virus": {
 "common_name": "Virus del mosaico del tomate", "plant": "Tomate", "disease": "Virus del mosaico del tomate (ToMV)",
 "description": "Enfermedad viral muy contagiosa que se transmite mecánicamente por contacto.",
 "symptoms": ["Patrón moteado de verde claro y oscuro en las hojas", "Deformación y enrollamiento de las hojas", "Crecimiento atrofiado", "Menor calidad y rendimiento del fruto"],
 "treatments": ["SIN CURA: arranca y destruye las plantas infectadas", "Desinfecta las herramientas con lejía al 10 %", "Lávate las manos antes de manipular las plantas", "No fumes cerca de las plantas (el virus puede venir del tabaco)"],
 "prevention": ["Usa variedades resistentes al virus", "Desinfecta herramientas y manos", "Retira las plantas infectadas de inmediato", "No plantes cerca del tabaco"]},
"Tomato___healthy": {
 "common_name": "Tomate sano", "plant": "Tomate", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Espaciado adecuado", "Buena circulación del aire", "Vigilancia periódica", "Fertilización equilibrada"]},
"Soybean___Frogeye_Leaf_Spot": {
 "common_name": "Mancha ojo de rana de la soja", "plant": "Soja", "disease": "Mancha ojo de rana",
 "description": "Enfermedad fúngica que produce lesiones circulares con centro gris y borde oscuro en las hojas de soja.",
 "symptoms": ["Manchas circulares con centro gris", "Bordes marrón rojizo oscuro", "Las lesiones pueden unirse", "Defoliación prematura"],
 "treatments": ["Aplica fungicidas (azoxistrobina, trifloxistrobina)", "Usa variedades resistentes", "Rota con cultivos no hospedantes", "Retira los restos vegetales infectados"],
 "prevention": ["Planta cultivares resistentes", "Rota con maíz o trigo", "Usa semilla libre de enfermedad", "Asegura una buena circulación del aire"]},
"Soybean___Downy_Mildew": {
 "common_name": "Mildiu de la soja", "plant": "Soja", "disease": "Mildiu velloso",
 "description": "Enfermedad de tipo fúngico que produce lesiones amarillas y un vello blanco o gris en el envés de la hoja.",
 "symptoms": ["Manchas amarillas angulosas en el haz", "Vello blanco o gris en el envés", "Crecimiento atrofiado", "Menor formación de vainas"],
 "treatments": ["Aplica fungicidas con metalaxil", "Retira y destruye las plantas infectadas", "Mejora la circulación del aire", "Reduce la humedad sobre la hoja"],
 "prevention": ["Usa variedades resistentes", "Evita el riego por aspersión", "Espaciado adecuado entre plantas", "Rotación de cultivos"]},
"Corn_(maize)___Lethal_Necrosis": {
 "common_name": "Necrosis letal del maíz", "plant": "Maíz", "disease": "Necrosis letal (MLN)",
 "description": "Enfermedad viral devastadora que provoca pérdidas graves de rendimiento. La transmiten insectos y virus del suelo.",
 "symptoms": ["Muerte del cogollo (las hojas jóvenes mueren)", "Moteado clorótico en las hojas", "Muerte prematura de la planta", "Crecimiento atrofiado", "Posible pérdida total del cultivo"],
 "treatments": ["SIN CURA: arranca y destruye las plantas infectadas de inmediato", "Controla los trips y pulgones vectores con insecticidas", "Planta híbridos resistentes al virus", "Usa semilla certificada libre de enfermedad"],
 "prevention": ["Usa variedades resistentes a la MLN", "Controla pronto los insectos vectores", "Elimina las plantas de maíz espontáneas", "Rotación de cultivos", "Evita sembrar cerca de campos infectados"]},
"Cabbage___healthy": {
 "common_name": "Col sana", "plant": "Col", "disease": "Ninguna — sana",
 "description": NO_DISEASE,
 "treatments": [CARE],
 "prevention": ["Rotación de cultivos", "Buena higiene del cultivo", "Espaciado adecuado", "Vigila las plagas"]},
"Cabbage___Black_Rot": {
 "common_name": "Podredumbre negra de la col", "plant": "Col", "disease": "Podredumbre negra",
 "description": "Enfermedad bacteriana grave que provoca lesiones en forma de V y decoloración vascular.",
 "symptoms": ["Lesiones amarillas en forma de V desde el borde de la hoja", "Nervios ennegrecidos", "Marchitez", "Podredumbre del cogollo en casos graves"],
 "treatments": ["Retira y destruye las plantas infectadas", "Aplica bactericidas a base de cobre", "Mejora el drenaje", "Trata la semilla con agua caliente (50 °C durante 25 min)"],
 "prevention": ["Usa plantones y semilla libres de enfermedad", "Rota los cultivos (evita brasicáceas 2 años o más)", "Controla los insectos (las alticas propagan la enfermedad)", "Evita el riego por aspersión"]},
"Tomato___Spider_mites": {
 "common_name": "Araña roja del tomate", "plant": "Tomate", "disease": "Araña roja (ácaro de dos manchas)",
 "description": "Diminutos arácnidos chupadores de savia que proliferan con calor y sequedad y se reproducen con rapidez en el envés de las hojas.",
 "symptoms": ["Punteado fino amarillo o blanco en el haz de la hoja", "Hojas de aspecto bronceado o polvoriento", "Finas telarañas en el envés y entre los tallos", "Enrollamiento, desecación y caída prematura de las hojas"],
 "treatments": ["Pulveriza las plantas con un chorro fuerte de agua para desprender los ácaros", "Aplica jabón insecticida o aceite hortícola en el envés", "Suelta ácaros depredadores (Phytoseiulus persimilis) como control biológico", "Retira y destruye las hojas muy infestadas"],
 "prevention": ["Mantén las plantas bien regadas: evita el estrés hídrico", "Aumenta la humedad ambiental alrededor de las plantas", "Inspecciona el envés con regularidad, sobre todo con calor", "Evita insecticidas de amplio espectro que matan a los depredadores naturales"]},
}


def main() -> None:
    en = json.loads((REPO / "data" / "treatments.json").read_text())
    missing = set(en) - set(ES)
    extra = set(ES) - set(en)
    if missing or extra:
        raise SystemExit(f"label mismatch\n  missing: {sorted(missing)}\n  extra: {sorted(extra)}")

    out = {}
    for label, entry in en.items():
        es = dict(ES[label])
        # severity is an enum shared across languages, never translated
        if "severity" in entry:
            es["severity"] = entry["severity"]
        # keep field presence identical to the English record; healthy entries
        # carry an empty symptoms list rather than omitting the key
        for field in ("symptoms", "treatments", "prevention"):
            if field in entry and field not in es:
                if not entry[field]:
                    es[field] = []
                else:
                    raise SystemExit(f"{label}: missing {field}")
            if field in es and field not in entry:
                raise SystemExit(f"{label}: unexpected {field}")
            if field in entry and len(es[field]) != len(entry[field]):
                raise SystemExit(
                    f"{label}: {field} length {len(es[field])} != English {len(entry[field])}"
                )
        out[label] = es

    dest = REPO / "docs" / "data" / "treatments.es.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {dest} ({len(out)} entries)")


if __name__ == "__main__":
    main()
