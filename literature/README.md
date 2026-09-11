# Literatura del curso

Esta carpeta reúne la bibliografía aplicada del curso *Big Data y Machine Learning para Economía Aplicada* (Maestría en Economía, Universidad Icesi).

## Los dos artículos que definen el estilo

| Archivo | Referencia | Papel en el curso |
|---|---|---|
| `Kim_Kilinsky_2024_Division_Does_Not_Imply_Predictability.pdf` | Kim, S.-y. S., & Zilinsky, J. (2024). Division Does Not Imply Predictability: Demographics Continue to Reveal Little About Voting and Partisanship. *Political Behavior*, 46, 67–87. | El tipo de artículo aplicado que los estudiantes leen y presentan: una pregunta predictiva, un modelo de referencia (logit) frente a un modelo flexible (random forest), evaluación fuera de muestra con intervalos y una lectura honesta de lo que el modelo no dice. |
| `democratization_AIM4D_Notre_Dame.pdf` | Gelvez, J. D., Cardiles, A., Martínez-González, E. F., & Muñoz, M. (2026). How Predictable Is an Election? A Machine-Learning Approach to Electoral Behavior in Colombia. Documento de trabajo. | El paper hilo conductor: sus datos (panel de mesas electorales con censo) y sus ejercicios (AUC y R² fuera de muestra, jerarquía de bloques de predictores, SHAP) son la aplicación guiada de las sesiones 1, 3, 6 y 7. |

## `revision/` — revisión de literatura (139 artículos)

Revisión construida el 9 de septiembre de 2026 para el rediseño del curso: qué se predice en economía y finanzas con machine learning, cómo se evalúa y cómo se interpreta. Cubre revistas top 5 y sus *Papers & Proceedings*, revistas top de campo (desarrollo, real estate, macro, econometría aplicada), finanzas (JF, JFE, RFS), ciencia general (*Science*, *Nature*, *PNAS*) y los artículos metodológicos de referencia.

| Archivo | Qué contiene |
|---|---|
| `revision-literatura-bdml.html` | La revisión navegable: los dos papers modelo, cómo se construyó la muestra, panorama de métodos e interpretación, ficha de cada artículo (qué predice, con qué datos y métodos, cómo evalúa, por qué importa) y una sugerencia de uso por sesión. Se abre en cualquier navegador. |
| `papers.csv` | Una fila por artículo: sesión sugerida, autores, año, título, revista, nivel, dominio, métodos, herramientas de interpretación, DOI, citas (OpenAlex), resumen y justificación en español. |
| `papers.json` | Los mismos registros con todos los campos, para filtrar o reutilizar. |
| `papers.bib` | BibTeX generado automáticamente a partir de `papers.json` (revisar autores completos antes de citar). |
| `metodos-por-paper.csv` | Clasificación de cada artículo por familias de ML usadas (regularización, árboles, random forest, boosting, redes, etc.), número de familias, presencia de un modelo de referencia simple y sesión a partir de la cual el estudiante conoce todos sus métodos. Incluye seis artículos añadidos el 10 de septiembre de 2026 (Kim & Zilinsky; Gelvez et al.; Chinco, Clark-Joseph & Ye 2019; Kozak, Nagel & Santosh 2020; Rapach, Strauss & Zhou 2013; Gorodnichenko, Pham & Talavera 2023). |

La columna `sesion_sugerida` sigue la secuencia del syllabus: 1 marco, 2 regresión y evaluación, 3 clasificación, 4 validación, 5 regularización, 6 árboles y ensambles, 7 interpretación, 8 redes e imágenes, 9 estructura sin *Y* y frontera causal.

Los PDF de los artículos no se distribuyen en este repositorio; cada fila trae el DOI o la URL de acceso.
