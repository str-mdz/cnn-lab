'use strict';

const image = [[0,0,0,0,0],[0,0,255,255,0],[0,0,255,255,0],[0,0,0,0,0],[0,0,0,0,0]];
const normalized = image.map(row => row.map(value => value / 255));
const kernel = [[-1,1],[-1,1]];
function convolve(input, filter, stride = 1, padding = 0) {
  const size = Math.floor((input.length + 2 * padding - filter.length) / stride) + 1;
  return Array.from({length:size}, (_,r) => Array.from({length:size}, (_,c) =>
    filter.reduce((sum,row,i) => sum + row.reduce((s,w,j) => s + w * (input[r*stride+i-padding]?.[c*stride+j-padding] ?? 0), 0), 0)
  ));
}
const featureMap = convolve(normalized, kernel);
const relu = featureMap.map(row => row.map(value => Math.max(0, value)));
const pooled = [0,2].map(r => [0,2].map(c => Math.max(...relu.slice(r,r+2).flatMap(row => row.slice(c,c+2)))));
const vector = pooled.flat();
const weights = [[1,0,2,0],[0,0,1,0]];
const logits = weights.map(row => row.reduce((sum,w,i) => sum + w * vector[i], 0));
function softmax(values) {
  const exp = values.map(value => Math.exp(value - Math.max(...values)));
  const total = exp.reduce((a,b) => a+b, 0);
  return exp.map(value => value / total);
}
const probabilities = softmax(logits);
const number = value => Number.isInteger(value) ? String(value) : String(Number(value.toFixed(3)));

const references = [
  ['IBM','¿Qué son las redes neuronales convolucionales?','IBM','https://www.ibm.com/es-es/think/topics/convolutional-neural-networks'],
  ['Google for Developers','Introducción a las redes neuronales convolucionales','Machine Learning','https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks?hl=es-419'],
  ['IBM Data and AI Team','IA vs. machine learning vs. deep learning vs. redes neuronales: ¿cuál es la diferencia?','IBM','https://www.ibm.com/es-es/think/topics/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks'],
  ['Google Codelabs','Redes neuronales convolucionales, con Keras y TPU','Google Codelabs','https://codelabs.developers.google.com/codelabs/keras-flowers-convnets?hl=es-419'],
  ['Google for Developers','Redes neuronales','Curso intensivo de aprendizaje automático','https://developers.google.com/machine-learning/crash-course/neural-networks?hl=es-419'],
  ['Google for Developers','Conjuntos de datos: división del conjunto de datos original','Machine Learning','https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets?hl=es-419'],
  ['Google for Developers','Evita el sobreajuste','Práctica de clasificación de imágenes','https://developers.google.com/machine-learning/practica/image-classification/preventing-overfitting?hl=es-419'],
  ['Google for Developers','Clasificación: Exactitud, recuperación, precisión y métricas relacionadas','Machine Learning','https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=es-419'],
  ['IBM','¿Qué es la visión artificial?','IBM','https://www.ibm.com/es-es/think/topics/computer-vision'],
  ['UNESCO','Recomendación sobre la Ética de la Inteligencia Artificial','UNESCO','https://www.unesco.org/es/legal-affairs/recommendation-ethics-artificial-intelligence'],
  ['PyTorch','vgg16','Torchvision Documentation','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html'],
  ['Google for Developers','Sobreajuste: Interpretación de curvas de pérdida','Machine Learning','https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves?hl=es-419']
];
function cite(id) { return `<a href="${references[id-1][3]}" target="_blank" rel="noopener noreferrer" aria-label="Fuente ${id}: ${references[id-1][1]}">[${id}]</a>`; }
function bibliography(first, last) {
  return `<ol class="bibliography">${references.slice(first-1,last).map(([author,title,site,url],i) => `<li><strong>[${first+i}]</strong> ${author}, «${title}», <em>${site}</em>. [En línea]. Disponible: <a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a> [Consultado: 17-nov-2026].</li>`).join('')}</ol>`;
}

const slides = [
  ["Redes neuronales convolucionales","",[1,2],"Una CNN trabaja con números organizados en matrices. Primero seguiremos una imagen pequeña con pesos elegidos para entender las cuentas. Después veremos el código y los resultados de VGG16 documentados en el PDF. No necesitas instalar nada: cada estudiante avanza a su ritmo."],
  ["Índice","",[1,2]],
  ["¿Cómo reconoce una computadora una imagen?","Una imagen digital es una cuadrícula de píxeles. Toca uno para conocer su posición y su intensidad.",[1,2],"En escala de grises de 8 bits, 0 representa negro y 255 blanco. Los valores intermedios representan intensidades intermedias. La ubicación también importa: reordenar los mismos números puede producir una imagen distinta. En color, cada posición tiene tres intensidades: rojo, verde y azul. Aquí las coordenadas empiezan en cero."],
  ["De la inteligencia artificial a las CNN","Las CNN pertenecen al aprendizaje profundo, que a su vez forma parte del aprendizaje automático y de la IA.",[3],"La IA incluye diversas maneras de resolver tareas, incluso reglas escritas por personas. Machine learning aprende a partir de ejemplos. Deep learning utiliza redes de varias capas. Las CNN son una familia de redes profundas que aprovecha las relaciones espaciales entre valores vecinos."],
  ["¿Por qué necesitamos CNN?","Las imágenes tienen muchos valores. Un filtro pequeño reutiliza sus pesos en distintas regiones.",[1,2],"En una capa Dense, cada entrada se conecta con cada neurona. Una convolución aplica un filtro local usando los mismos pesos a lo largo de la imagen. Esto permite buscar patrones en posiciones diferentes. Esta comparación ilustra dos capas, sin sesgos; no compara redes completas equivalentes."],
  ["¿Qué es una red neuronal convolucional?","Una arquitectura de aprendizaje profundo que utiliza filtros de pesos compartidos para extraer características y aprender patrones espaciales.",[1,2],"La arquitectura organiza las capas. Los pesos compartidos significan que un mismo filtro se aplica a varias regiones: no se aprende un filtro diferente por cada posición. Las primeras capas pueden responder a contrastes; otras combinan esas respuestas para representar patrones más complejos."],
  ["¿Cómo interpreta una CNN una imagen?","Usaremos esta imagen de 5 × 5 durante el ejemplo. Normalizarla significa dividir cada intensidad entre 255.",[2,4],"Dividir entre 255 lleva los valores de 8 bits al intervalo [0,1]. No convierte cualquier intensidad en 0 o 1: por ejemplo, 128/255 ≈ 0.502. Nuestra imagen solo contiene 0 y 255, por eso la salida usa ceros y unos. Las imágenes RGB tienen tres matrices, una por canal."],
  ["Arquitectura general de una CNN","",[2],"Cada etapa recibe la salida de la anterior. La convolución produce respuestas locales; ReLU introduce no linealidad; pooling resume regiones; Flatten organiza los valores en un vector; Dense combina características y Softmax transforma puntuaciones en probabilidades. Es un ejemplo simplificado, no una regla para todas las CNN."],
  ["La operación de convolución","Mueve el filtro sobre la imagen. Multiplica los cuatro pares y suma: así se obtiene una casilla de salida.",[2,4],"El filtro conserva sus pesos mientras cambia de posición. Aquí el stride es 1 y no hay padding: una entrada 5 × 5 y un filtro 2 × 2 producen 4 × 4 resultados. Se sigue la convención de las bibliotecas de aprendizaje profundo, sin invertir el kernel. Un resultado negativo indica una respuesta opuesta a la orientación del filtro."],
  ["Ejemplo de convolución","",[2],"Cada valor de la región se multiplica por el peso que ocupa la misma posición. Los productos no nulos son 1, −1 y −1: su suma es −1. Este ejercicio es independiente y no modifica el ejemplo de 5 × 5."],
  ["Mapas de características","",[1,2],"Las primeras capas pueden detectar bordes o contrastes. Las siguientes combinan respuestas y construyen representaciones más complejas. Para RGB, cada filtro cubre los tres canales: por ejemplo, un kernel 3 × 3 × 3 tiene 27 pesos. Sus contribuciones se suman con un sesgo para producir un mapa; ocho filtros producen ocho mapas."],
  ["Stride y padding","Stride indica cuánto avanza el filtro. Padding añade ceros alrededor de la imagen.",[2,4],"El tamaño de salida por dimensión es O = floor((N + 2P − K)/S) + 1. N es la entrada, K el filtro, S el stride y P el padding. Un stride mayor reduce el número de posiciones. El padding permite controlar las dimensiones y considerar los bordes. Estas variantes no cambian el mapa base de las siguientes pantallas."],
  ["Función de activación ReLU","ReLU(x) = max(0, x). Los negativos se convierten en cero; los demás valores se conservan.",[2],"ReLU se aplica a cada elemento sin cambiar las dimensiones de la matriz. En nuestro mapa modifica tres casillas negativas. Introduce una no linealidad: sin activaciones no lineales, varias transformaciones lineales seguidas seguirían siendo una sola transformación lineal."],
  ["Agrupamiento o pooling","Max Pooling conserva el máximo de cada región. Aquí usamos bloques 2 × 2 con stride 2.",[1,4],"Las cuatro regiones no se solapan. La región superior izquierda contiene [0,1,0,2], cuyo máximo es 2. Pooling reduce dimensiones y también descarta información. Average Pooling usaría el promedio: en esa misma región, (0+1+0+2)/4 = 0.75."],
  ["Flatten, Dense y Softmax","",[2,4],"Flatten enumera las filas: [2,0,1,0]. Dense calcula sumas de entradas por pesos más un sesgo; aquí los sesgos son cero. Softmax divide cada exponencial por la suma de todas. "],
  ["¿Cómo aprende una CNN?","",[4,5],"El proceso incluye inicialización, predicción, cálculo de pérdida, retropropagación y actualización. La retropropagación calcula derivadas; el optimizador usa esas derivadas para cambiar los pesos. El pequeño ejercicio usa un solo peso: no entrena una CNN. Inicializaciones como Glorot y He buscan escalas iniciales que faciliten el aprendizaje."],
  ["Preparación y división de datos","",[6],"Entrenamiento ajusta los pesos. Validación permite comparar configuraciones y detectar problemas. Prueba se reserva para la evaluación final. "],
  ["Evaluación y sobreajuste","Si entrenamiento mejora pero validación empeora, la red puede estar aprendiendo demasiado las particularidades de sus ejemplos.",[7,8,12],"Accuracy es la proporción de aciertos. La pérdida mide la discrepancia según la función elegida. Generalizar significa funcionar con datos nuevos. Más diversidad, aumento de datos, regularización o parada temprana pueden ayudar ante el sobreajuste."],
  ["Tareas que realizan las CNN","",[9],"Clasificación asigna una etiqueta a toda la imagen. Detección localiza objetos con recuadros. Segmentación asigna una categoría a cada píxel."],
  ["Aplicaciones y limitaciones","Las CNN pueden apoyar distintas tareas visuales, pero también pueden equivocarse y presentar sesgos.",[1,10],"Un modelo puede fallar cuando cambian la iluminación, la cámara o la población respecto a sus datos de entrenamiento. Un promedio de aciertos puede ocultar errores frecuentes en ciertos grupos. Se necesitan datos representativos, evaluación en el contexto de uso y supervisión adecuada. Una probabilidad alta no garantiza una respuesta correcta."],
  ["Implementación de una CNN","",[11]],
  ["Modelo y datos: VGG16 + ImageNet","",[11]],
  ["Preparación: modelo e imagen","Lee el código por partes: cargar pesos, preparar la imagen y construir el tensor.",[11]],
  ["Entrada: la imagen como números","",[2,11]],
  ["Filtros de la primera capa","",[11]],
  ["Mapas de características","",[11]],
  ["Softmax y Top 5","",[11]],
  ["Conclusiones","Una CNN transforma píxeles en características y combina esas características para generar una predicción.",[1,2]],
  ["Referencias","",[1,2,3,4,5,6]],
  ["Referencias","",[7,8,9,10,11,12]],
  ["Gracias por participar","",[1,2,11]]
];
const chapters = [
  ["Fundamentos",1,6],
  ["Arquitectura",7,15],
  ["Entrenamiento",16,18],
  ["Aplicación y demostración",19,27],
  ["Conclusiones",28,31]
];

const team = [
  ['Martinez Ojeda, Jonhatan Alexis', 'jonhalexismaro@gmail.com'],
  ['Méndez López, Leonardo Octavio', 'leonardo.mendez.fi@gmail.com'],
  ['Mendoza Camacho, Estrella de Maria', 'starmendozacamacho@gmail.com'],
  ['Sánchez Yañez, Valeria Ximena', 'valeriaximenasy07@gmail.com'],
  ['Solorio Garduño, Alexis Jair', 'salexisjair14@gmail.com']
];

function matrix(data, {title='', size=240, selected=[], current=-1, reveal=Infinity, picture=false, clickable=false, filter=false, heat=false, changed=[]} = {}) {
  const values = data.flat(), maximum = Math.max(1,...values.map(Math.abs));
  return `<div class="matrix-block"><div class="matrix-title">${title}</div><div class="matrix ${picture?'picture':''} ${heat?'heat':''}" style="--cols:${data[0].length};--size:${size}px" role="group" aria-label="${title || 'Matriz'}">${values.map((value,i) => {
    const r=Math.floor(i/data[0].length), c=i%data[0].length, tag=clickable?'button':'span';
    return `<${tag} class="cell ${value<0?'negative':value>0?'positive':''} ${selected.includes(i)?'selected':''} ${i===current?'current':''} ${filter?'kernel':''} ${changed.includes(i)?'changed':''}" style="--gray:${value};--heat:hsl(${value<0?335:156} 35% ${96-Math.abs(value/maximum)*24}%)" ${clickable?`data-action="pixel" data-value="${i}"`:''} aria-label="Fila ${r}, columna ${c}: ${i>reveal?'pendiente':number(value)}">${i>reveal?'·':number(value)}</${tag}>`;
  }).join('')}</div></div>`;
}
const arrow = '<span class="arrow" aria-hidden="true">→</span>';
const choices = (labels, selected) => `<div class="controls">${labels.map((label,i) => `<button class="choice" data-action="choice" data-value="${i}" aria-pressed="${i===selected}">${label}</button>`).join('')}</div>`;
const go = (label,n) => `<button class="choice" data-action="go" data-value="${n}">${label} →</button>`;
const equation = text => `<div class="equation">${text}</div>`;
const note = text => `<p class="note">${text}</p>`;
const heading = title => `<h2 class="example-head">${title}</h2>`;
const flow = items => `<div class="pipeline">${items.map(item=>`<span>${item}</span>`).join('')}</div>`;
function bars(labels, values, digits=2) {
  return `<div class="bars">${labels.map((label,i)=>`<div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill" style="width:${values[i]}%"></div></div><strong>${values[i].toFixed(digits)} %</strong></div>`).join('')}</div>`;
}
function pdfImage(name, caption, css='pdf-photo') {
  const source = document.querySelector('#pdf-images').content.querySelector(`[data-image="${name}"]`).getAttribute('src');
  return `<figure><img class="${css}" src="${source}" alt="${caption || (name==='dog'?'Fotografía del perro':name)}">${caption?`<figcaption>${caption}</figcaption>`:''}</figure>`;
}
function scene(task) {
  return `<svg class="scene" viewBox="0 0 480 310" role="img" aria-label="Casa y árbol: ${['clasificación','detección','segmentación'][task]}"><rect width="480" height="310" rx="10" fill="${task===2?'#307fc2':'#dceaf7'}"/><path d="M0 220H480V310H0Z" fill="${task===2?'#398d61':'#d6e5c5'}"/><circle cx="410" cy="55" r="23" fill="#f6cd64"/><path d="M83 151h142v118H83Z" fill="${task===2?'#e9a22f':'#efd9b3'}"/><path d="M63 154l91-88 94 88Z" fill="${task===2?'#e9a22f':'#b96369'}"/><path d="M129 214h36v55h-36" fill="${task===2?'#e9a22f':'#8c6746'}"/><path d="M356 168h24v100h-24Z" fill="${task===2?'#7e59a6':'#8c6746'}"/><circle cx="367" cy="145" r="57" fill="${task===2?'#7e59a6':'#64966b'}"/>${task===0?'<rect x="17" y="17" width="96" height="35" rx="5" fill="white"/><text x="36" y="41" font-size="18" fill="#18558d">Casa</text>':''}${task===1?'<g fill="none" stroke-width="3"><rect x="58" y="62" width="196" height="213" stroke="#ad641e"/><rect x="301" y="80" width="132" height="196" stroke="#18558d"/></g><text x="58" y="50" font-size="16" fill="#18558d">Casa</text><text x="301" y="68" font-size="16" fill="#18558d">Árbol</text>':''}${task===2?'<g fill="white" font-size="18"><text x="22" y="40">Cielo</text><text x="128" y="187">Casa</text><text x="345" y="149">Árbol</text><text x="22" y="290">Pasto</text></g>':''}</svg>`;
}

const states = slides.map(() => ({choice:0,pixel:7,step:0,stride:1,padding:0}));
function example(n) {
  const s=states[n-1], pick=s.choice, pixelValue=image[Math.floor(s.pixel/5)][s.pixel%5];
  switch(n) {
    case 1: return `<div class="hero"><div><h1 id="title-1">Redes neuronales<br>convolucionales<span>(CNN)</span></h1><p class="cover-subtitle">Inteligencia Artificial · Grupo 5</p><ul class="team-list">${team.map(([name])=>`<li>${name}</li>`).join('')}</ul><div class="controls">${go('Comenzar',2)}</div></div><div class="hero-art">${matrix(image,{size:210,picture:true})}</div></div>`;
    case 2: return `<ol class="index-list">${chapters.map(([title],i)=>`<li><span class="index-number">0${i+1}</span><span>${title}</span></li>`).join('')}</ol>`;
    case 3: return choices(['Ver imagen','Ver números'],pick)+`<div class="matrix-row">${matrix(image,{title:'Toca un píxel · imagen 5 × 5',size:250,picture:pick===0,clickable:true,current:s.pixel})}</div>`+equation(`Píxel (${Math.floor(s.pixel/5)}, ${s.pixel%5}) → intensidad <strong>${pixelValue}</strong> de 255`);
    case 4: {
      const concepts=[['Inteligencia artificial','Resolver tareas asociadas con la inteligencia. Incluye métodos de aprendizaje y sistemas de reglas.'],['Machine learning','Aprender patrones a partir de datos, por ejemplo para estimar un precio.'],['Deep learning','Aprender representaciones con redes de varias capas, por ejemplo para procesar voz.'],['CNN','Usar filtros compartidos para aprender patrones espaciales, por ejemplo en imágenes.']];
      return heading('De lo general a lo específico')+flow(['IA','Machine learning','Deep learning','CNN'])+choices(['IA','ML','DL','CNN'],pick)+`<h3>${concepts[pick][0]}</h3><p>${concepts[pick][1]}</p>`;
    }
    case 5: return heading('Una comparación de capas, sin sesgos')+`<div class="columns"><div class="fact"><h3>Conexión completa · Dense</h3><p>Imagen RGB de 100 × 100</p>${equation('100 × 100 × 3 = 30,000 entradas<br>30,000 × 100 neuronas')}<div class="stat">3,000,000 <small>pesos</small></div></div><div class="fact"><h3>Un filtro convolucional</h3><p>Kernel que cubre tres canales</p>${equation('3 × 3 × 3 = 27 pesos<br>Se reutilizan en cada posición')}<div class="stat">27 <small>pesos por filtro</small></div></div></div>`;
    case 6: return `<blockquote>Filtros de pesos compartidos que aprenden patrones espaciales de los datos.</blockquote>`+flow(['Entrada<b>Píxeles</b>','Procesamiento<b>Características</b>','Salida<b>Predicción</b>'])+note('Compartir pesos significa aplicar el mismo filtro en distintas posiciones.');
    case 7: return heading('Normalizar: x / 255')+choices(['Valores originales','Dividir entre 255'],pick)+`<div class="matrix-row">${matrix(pick?normalized:image,{title:pick?'Matriz normalizada':'Matriz original',size:250,clickable:true,current:s.pixel})}</div>`+equation(`${pixelValue} / 255 = <strong>${pixelValue/255}</strong>`)+flow(['R<small>Una matriz roja</small>','G<small>Una matriz verde</small>','B<small>Una matriz azul</small>'])+`<p class="hint">RGB: tres intensidades por píxel. En este ejemplo usamos un solo canal de grises.</p>`;
    case 8: {
      const names=['Imagen','Convolución','ReLU','Pooling','Flatten','Dense','Softmax'];
      const data=[normalized,featureMap,relu,pooled,[vector],[logits],[probabilities]];
      const descriptions=['Entrada normalizada · 5 × 5','Respuestas del filtro · 4 × 4','Sin valores negativos · 4 × 4','Máximos locales · 2 × 2','Vector de cuatro valores','Dos puntuaciones o logits','Dos probabilidades que suman 1'];
      return heading('Sigue los mismos valores')+choices(names,pick)+`<div class="matrix-row">${matrix(data[pick],{title:descriptions[pick],size:pick>=4?270:240})}</div>`;
    }
    case 9: {
      const r=Math.floor(s.step/4),c=s.step%4;
      const pairs=kernel.flatMap((row,i)=>row.map((weight,j)=>({pixel:normalized[r+i][c+j],weight,index:(r+i)*5+c+j})));
      const products=pairs.map(pair=>pair.pixel*pair.weight);
      return `<div class="matrix-row">${matrix(normalized,{title:'Entrada normalizada · 5 × 5',size:250,selected:pairs.map(p=>p.index)})}${arrow}${matrix(kernel,{title:'Filtro · 2 × 2',size:125,filter:true})}${arrow}${matrix(featureMap,{title:'Mapa de salida · 4 × 4',size:220,current:s.step,reveal:s.step})}</div><h3>¿De dónde sale el valor (${r}, ${c})?</h3><div class="products">${pairs.map(p=>`<span>${p.pixel} × (${p.weight}) = <strong>${p.pixel*p.weight}</strong></span>`).join('')}</div>`+equation(`${products.map(p=>`(${p})`).join(' + ')} = <strong>${products.reduce((a,b)=>a+b,0)}</strong>`)+`<p class="hint">O = (5 − 2) / 1 + 1 = 4. Hay 4 × 4 = 16 posiciones.<br>Stride 1 · padding 0 · sesgo 0.</p><div class="operation-controls"><span>Posición ${s.step+1} de 16</span><button data-action="back" ${s.step===0?'disabled':''}>← Posición anterior</button><button data-action="step" ${s.step===15?'disabled':''}>Siguiente posición →</button><button data-action="reset">Reiniciar</button></div>`;
    }
    case 10: {
      const region=[[1,2,0],[0,1,1],[0,0,1]],filter=[[1,0,-1],[1,0,-1],[1,0,-1]],products=region.flat().map((v,i)=>v*filter.flat()[i]);
      return `<div class="matrix-row">${matrix(region,{title:'Región',size:180})}${arrow}${matrix(filter,{title:'Filtro',size:180,filter:true})}</div><div class="products">${region.flat().map((v,i)=>`<span>${v} × (${filter.flat()[i]}) = ${products[i]}</span>`).join('')}</div>${choices(['Ver productos','Mostrar resultado'],pick)}`+equation(pick===1?'Suma = <strong>'+products.reduce((a,b)=>a+b,0)+'</strong>':'Suma los nueve productos.');
    }
    case 11: {
      const selectedKernel=pick===0?kernel:[[-1,-1],[1,1]];
      return choices(['Filtro vertical','Filtro horizontal'],pick)+`<div class="matrix-row">${matrix(selectedKernel,{title:'Filtro',size:125,filter:true})}${arrow}${matrix(convolve(normalized,selectedKernel),{title:'Mapa calculado',heat:true})}</div>`+note('Un filtro RGB 3 × 3 × 3 suma sus tres contribuciones, más el sesgo, para producir un mapa. Ocho filtros producen ocho mapas.');
    }
    case 12: {
      const out=convolve(normalized,kernel,s.stride,s.padding),size=5+2*s.padding;
      const padded=Array.from({length:size},(_,r)=>Array.from({length:size},(_,c)=>normalized[r-s.padding]?.[c-s.padding]??0));
      return heading('Prueba dos cambios pequeños')+`<div class="controls"><label>Stride<select data-setting="stride"><option value="1" ${s.stride===1?'selected':''}>1</option><option value="2" ${s.stride===2?'selected':''}>2</option></select></label><label>Padding<select data-setting="padding"><option value="0" ${s.padding===0?'selected':''}>0</option><option value="1" ${s.padding===1?'selected':''}>1</option></select></label></div><div class="matrix-row">${matrix(padded,{title:`Entrada con borde · ${size} × ${size}`,size:250})}${arrow}${matrix(out,{title:`Salida · ${out.length} × ${out.length}`,size:220})}</div>`+equation(`O = floor((5 + 2 × ${s.padding} − 2) / ${s.stride}) + 1 = <strong>${out.length}</strong>`);
    }
    case 13: return heading('De la convolución a ReLU')+choices(['Antes','Aplicar ReLU'],pick)+`<div class="matrix-row">${matrix(featureMap,{title:'Convolución original',size:230})}${arrow}${matrix(pick?relu:featureMap,{title:pick?'Después de ReLU':'Sin aplicar',size:230,changed:pick?featureMap.flat().flatMap((v,i)=>v<0?[i]:[]):[]})}</div>`+equation('max(0, −2) = <strong>0</strong> · max(0, 2) = <strong>2</strong>');
    case 14: {
      const r=Math.floor(pick/2)*2,c=(pick%2)*2,indices=[r*4+c,r*4+c+1,(r+1)*4+c,(r+1)*4+c+1],values=indices.map(i=>relu.flat()[i]);
      return heading('Elige una región 2 × 2')+choices(['Superior izquierda','Superior derecha','Inferior izquierda','Inferior derecha'],pick)+`<div class="matrix-row">${matrix(relu,{title:'Entrada después de ReLU',size:250,selected:indices})}${arrow}${matrix(pooled,{title:'Max Pooling · 2 × 2',size:135,current:pick})}</div>`+equation(`max(${values.join(', ')}) = <strong>${Math.max(...values)}</strong>`);
    }
    case 15: return heading('De una matriz a dos probabilidades')+choices(['1. Flatten','2. Dense','3. Softmax'],pick)+`<div class="matrix-row">${matrix(pooled,{title:'Pooling',size:125})}${arrow}${matrix([vector],{title:'Flatten · por filas',size:270})}</div>`+(pick===0?equation('[[2, 0], [1, 0]] → <strong>[2, 0, 1, 0]</strong>'):pick===1?weights.map((row,i)=>equation(`z${i+1} = ${row.map((w,j)=>`(${w} × ${vector[j]})`).join(' + ')} + 0 = <strong>${logits[i]}</strong>`)).join(''):equation(`Logits: [${logits.join(', ')}]<br>p(A) = e⁴ / (e⁴ + e¹)<br>p(B) = e¹ / (e⁴ + e¹)`)+bars(['Clase A','Clase B'],probabilities.map(p=>p*100)))+note('Pesos: [[1,0,2,0],[0,0,1,0]] · sesgos cero.');
    case 16: {
      const weight=pick === 0 ? 0.2 : 0.44,prediction=weight*2,loss=(prediction-1)**2;
      return flow(['Inicializar','Predecir','Medir pérdida','Retropropagar','Actualizar'])+equation(`x = 2 · objetivo = 1 · η = 0.1<br>w = ${weight} → predicción = ${number(prediction)}<br>L = (wx − 1)² = <strong>${Number(loss.toFixed(4))}</strong>`)+(pick?equation('Gradiente inicial = 2 × 2 × (0.4 − 1) = −2.4<br>w_nuevo = 0.2 − 0.1 × (−2.4) = <strong>0.44</strong><br>Predicción nueva = 0.44 × 2 = <strong>0.88</strong>'):'')+choices(['Reiniciar','Actualizar peso'],pick);
    }
    case 17: {
      const percentages=pick?[80,10,10]:[70,15,15];
      return choices(['70 / 15 / 15','80 / 10 / 10'],pick)+flow(['Entrenamiento','Validación','Prueba'].map((name,i)=>`${name}<b>${percentages[i]*10} imágenes</b><small>${percentages[i]} %</small>`))+equation(`${percentages.join(' % + ')} % = <strong>100 %</strong>`);
    }
    case 18: {
      const train=Array.from({length:30},(_,i)=>.95*Math.exp(-i/7)+.04);
      const validation=train.map((v,i)=>pick?.85*Math.exp(-i/5)+.13+Math.max(0,i-10)*.018:v+.07);
      const path=data=>data.map((v,i)=>`${i?'L':'M'}${50+i*18},${240-v*180}`).join(' ');
      return heading('Curvas de pérdida')+choices(['Buen aprendizaje','Sobreajuste'],pick)+`<svg class="chart" viewBox="0 0 630 290" role="img" aria-label="Pérdida de entrenamiento y validación en 30 épocas"><path d="M50 30V240H600" stroke="#aebfd1" fill="none"/><path d="${path(train)}" stroke="#18558d" stroke-width="3" fill="none"/><path d="${path(validation)}" stroke="#75529e" stroke-width="3" stroke-dasharray="7 4" fill="none"/><text x="10" y="30">Loss</text><text x="48" y="265">1</text><text x="560" y="265">30</text><text x="280" y="283">Épocas</text></svg><p class="hint">━━ Entrenamiento · azul &nbsp; ┄┄ Validación · morado</p>`+note(pick?'La validación empieza a empeorar mientras entrenamiento mejora: señal de sobreajuste.':'Ambas curvas mejoran: es necesario comprobar el desempeño con datos nuevos.');
    }
    case 19: return heading('Tres preguntas sobre la misma escena')+choices(['Clasificación','Detección','Segmentación'],pick)+scene(pick)+equation(['¿Qué hay? → Una etiqueta: «casa».','¿Qué hay y dónde? → Objetos y recuadros.','¿Qué es cada píxel? → Categorías por región.'][pick]);
    case 20: {
      const applications=[['Salud','Apoyar la revisión de imágenes médicas.','El desempeño puede cambiar con la población o el equipo de imagen.'],['Industria','Identificar defectos en piezas.','Un cambio de iluminación puede parecer un defecto.'],['Agricultura','Detectar signos de daño en hojas.','Una variedad de cultivo diferente puede confundir al modelo.'],['Transporte','Detectar peatones en imágenes.','La lluvia, la noche y las oclusiones dificultan la tarea.']];
      return heading('Una aplicación y una limitación')+choices(applications.map(a=>a[0]),pick)+`<h3>${applications[pick][0]}</h3><p><strong>Aplicación:</strong> ${applications[pick][1]}</p><p><strong>Limitación:</strong> ${applications[pick][2]}</p>`;
    }
    case 21: return `<div class="columns">${pdfImage('dog','')}<div>${flow(['Imagen','Preprocesamiento','VGG16','Filtros y mapas','Top 5'])}</div></div>`;
    case 22: return heading('Modelo ≠ conjunto de datos')+choices(['VGG16','ImageNet-1K'],pick)+(pick?`<h3>ImageNet-1K · datos</h3><p>Imágenes etiquetadas en 1,000 categorías. Los ejemplos sirvieron para aprender pesos y las etiquetas permiten interpretar las salidas.</p>`:`<h3>VGG16 · arquitectura</h3><p>13 capas convolucionales + 3 capas lineales con pesos. Sus bloques extraen características antes de la clasificación.</p>`)+flow(['2 Conv · 64','2 Conv · 128','3 Conv · 256','3 Conv · 512','3 Conv · 512']);
    case 23: {
      const parts = [
        ['import torch\nfrom PIL import Image\nfrom torchvision import models, transforms','Importa PyTorch, Pillow y las herramientas de Torchvision.'],
        ['weights = models.VGG16_Weights.DEFAULT\nvgg = models.vgg16(weights=weights)','Carga los pesos preentrenados de VGG16.'],
        ['vgg.eval()','Activa el modo evaluación: por ejemplo, desactiva Dropout.'],
        ['imagen = Image.open("dog.jpg").convert("RGB")','Lee la fotografía como imagen RGB.'],
        ['transform = transforms.Compose([\n    transforms.Resize((224, 224)),\n    transforms.ToTensor(),\n    transforms.Normalize(\n        [0.485, 0.456, 0.406],\n        [0.229, 0.224, 0.225])\n])','Redimensiona, convierte a tensor y normaliza cada canal.'],
        ['x = transform(imagen).unsqueeze(0)','Añade la dimensión de lote: (1, 3, 224, 224).']
      ];
      return choices(['Importar','Pesos','Evaluación','Imagen','Transformar','Tensor'],pick)+`<pre><code>${parts[pick][0]}</code></pre>`+note(parts[pick][1]);
    }
    case 24: return `<div class="columns">${pdfImage('dog','')}<div>${equation('(1, 3, 224, 224)')}${flow(['1<small>Imagen en el lote</small>','3<small>Canales RGB</small>','224<small>Alto</small>','224<small>Ancho</small>'])}</div></div>`+equation('Ejemplo de intensidad en el canal rojo:<br>(128 / 255 − 0.485) / 0.229 ≈ <strong>'+number((128/255-.485)/.229)+'</strong>');
    case 25: return `<div class="filters">${[1,2,3,4].map(i=>pdfImage('filter-'+i,'Filtro '+i,'pdf-filter')).join('')}</div>`+equation('Un filtro: 3 × 3 × 3 = 27 pesos<br>Primera capa: 64 filtros → 64 mapas');
    case 26: {
      const names=['Conv2d','ReLU','Pool 1','Pool 3','Pool 5'],shapes=['64, 224, 224','64, 224, 224','64, 112, 112','256, 28, 28','512, 7, 7'];
      return choices(names,pick)+pdfImage('activation-'+(pick+1),names[pick]+' · canal 0','pdf-map')+equation(`Forma (canales, alto, ancho) = <strong>(${shapes[pick]})</strong>`);
    }
    case 27: return bars(['Golden retriever','Labrador retriever','Kuvasz','Great Pyrenees','English setter'],[92.7,2.8,2.3,.6,.5],1);
    case 28: return heading('Si algo no quedó claro, regresa a ese tema')+`<div class="controls">${[['Píxeles',7],['Filtros',9],['Características',11],['Predicción',15],['Evaluación',18]].map(([label,id])=>go(label,id)).join('')}</div><ol><li>Los filtros aprenden características de los datos.</li><li>La arquitectura conecta las operaciones.</li><li>Evaluar con datos nuevos permite comprobar generalización.</li></ol>`;
    case 29: return bibliography(1,6);
    case 30: return bibliography(7,12);
    case 31: return `<div class="thanks"><h1 id="title-31">Gracias por<br>su atención.</h1><p class="closing-subtitle">Redes neuronales convolucionales<br>Facultad de Ingeniería, UNAM · Grupo 5</p><ul class="contact-list">${team.map(([name,email])=>`<li><span>${name}</span><a href="mailto:${email}">${email}</a></li>`).join('')}</ul>${go('Volver al inicio',1)}</div>`;
    default: return '';
  }
}

const container = document.getElementById('slides');
const jump = document.getElementById('jump');
const menu = document.getElementById('slide-menu');
let current = 0;
container.innerHTML = slides.map(([title,intro,refs,extra],i) => {
  const n=i+1, chapter=chapters.find(([,first,last])=>n>=first&&n<=last)[0];
  const badge=n>=7&&n<=18?'<span class="badge">Ejemplo A</span>':n>=21&&n<=27?'<span class="badge">Ejemplo B</span>':'';
  const more=n<21&&n!==2?`<details class="more"><summary aria-label="Información adicional"><span aria-hidden="true">+</span></summary><div class="more-body"><p>${extra} ${refs.map(cite).join(', ')}</p></div></details>`:'';
  return `<section id="slide-${n}" class="slide slide-${n}" aria-labelledby="title-${n}" hidden>${n===1||n===31?'':`<div><div class="section-meta">${chapter}${badge}</div><h1 id="title-${n}">${title}</h1>${intro?`<p class="intro">${intro} ${refs.map(cite).join(', ')}</p>`:''}</div>`}<div class="example"></div><footer class="source-footer"><span>Fuentes:</span>${refs.map(cite).join(', ')}</footer>${more}</section>`;
}).join('');
jump.innerHTML = slides.map((slide,i)=>`<button type="button" data-slide="${i}"><span>${String(i+1).padStart(2,'0')}</span>${slide[0]}</button>`).join('');
function refreshExample() { container.children[current].querySelector('.example').innerHTML=example(current+1); }
function showSlide(index, scroll=true) {
  current = Math.max(0,Math.min(slides.length-1,index));
  [...container.children].forEach((section,i)=>section.hidden=i!==current);
  refreshExample();
  jump.querySelectorAll('button').forEach((button,i)=>{button.classList.toggle('active',i===current);if(i===current)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');});
  document.getElementById('counter').textContent=`Diapositiva ${current+1} de 31`;
  document.getElementById('progress').style.width=`${(current+1)/31*100}%`;
  document.getElementById('previous').disabled=current===0;
  document.getElementById('next').disabled=current===slides.length-1;
  document.title=`${current+1} · ${slides[current][0]} | CNN Lab`;
  history.replaceState(null,'','#'+(current+1));
  if(scroll){container.scrollTop=0;container.querySelectorAll('.more').forEach(details=>details.open=false);}
  if(compactMenu.matches)menu.open=false;
}
document.getElementById('previous').addEventListener('click',()=>showSlide(current-1));
document.getElementById('next').addEventListener('click',()=>showSlide(current+1));
jump.addEventListener('click',event=>{const button=event.target.closest('button[data-slide]');if(button)showSlide(Number(button.dataset.slide));});
container.addEventListener('click',event=>{
  const button=event.target.closest('button[data-action]');
  if(!button||button.disabled)return;
  const s=states[current],value=Number(button.dataset.value),action=button.dataset.action;
  if(action==='go'){showSlide(value-1);return;}
  if(action==='choice')s.choice=value;
  if(action==='pixel')s.pixel=value;
  if(action==='step')s.step=Math.min(15,s.step+1);
  if(action==='back')s.step=Math.max(0,s.step-1);
  if(action==='reset')s.step=0;
  refreshExample();

  const replacement=[...container.children[current].querySelectorAll('button')].find(b=>b.dataset.action===action&&b.dataset.value===button.dataset.value);
  replacement?.focus({preventScroll:true});
});
container.addEventListener('change',event=>{
  if(!event.target.dataset.setting)return;
  states[current][event.target.dataset.setting]=Number(event.target.value);
  refreshExample();
});
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'){container.querySelectorAll('.more').forEach(details=>details.open=false);if(compactMenu.matches)menu.open=false;return;}
  if(event.target.closest('input,select,button,summary')||event.ctrlKey||event.metaKey||event.altKey)return;
  if(event.key==='ArrowRight'){event.preventDefault();showSlide(current+1);}
  if(event.key==='ArrowLeft'){event.preventDefault();showSlide(current-1);}
});
window.addEventListener('hashchange',()=>{const n=Number(location.hash.slice(1));if(Number.isInteger(n)&&n>=1&&n<=31)showSlide(n-1);});
const compactMenu=window.matchMedia('(max-width: 900px)');
compactMenu.addEventListener('change',()=>{menu.open=!compactMenu.matches;});
menu.open=!compactMenu.matches;

const requested=Number(location.hash.slice(1));
showSlide(Number.isInteger(requested)&&requested>=1&&requested<=31?requested-1:0,false);
