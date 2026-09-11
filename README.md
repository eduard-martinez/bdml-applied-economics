# Big Data y Machine Learning para Economía Aplicada

Material del curso de la Maestría en Economía de la Universidad Icesi, dictado por [Eduard F. Martínez-González](https://eduard-martinez.github.io).

- **Página del curso:** <https://eduard-martinez.github.io/teaching/bdml-applied-economics/>
- **Syllabus:** [`syllabus/syllabus.pdf`](syllabus/syllabus.pdf)
- **Proyecto final:** [`final-project/projecto_final.pdf`](final-project/projecto_final.pdf)

## Estructura del repositorio

| Carpeta | Contenido | Sección de la página del curso |
|---|---|---|
| `syllabus/` | Syllabus (PDF + fuente LaTeX) | Encabezado |
| `lectures/week-01/ … week-09/` | Diapositivas de cada sesión en PDF (`week-XX.pdf`); las fuentes LaTeX y las versiones anteriores se conservan fuera del repositorio | Schedule |
| `applications/` | Código R de la parte aplicada de cada sesión — una carpeta `week-XX/` por sesión (por publicar) | Schedule · *R application* |
| `problem-sets/pset-1/`, `problem-sets/pset-2/` | Problem Sets 1 (sesiones 1–5) y 2 (sesiones 6–9) (por publicar); estructura en [`problem-sets/README.md`](problem-sets/README.md) | Problem sets |
| `final-project/` | Lineamientos del proyecto final | Evaluation · Final project |
| `literature/` | Los dos papers modelo y la revisión de literatura del curso ([`literature/revision/`](literature/revision/), 139 artículos con métodos, métricas y sesión sugerida) | Reading library |
| `books/` | Copia del texto guía (ISL, 2.ª ed.) | Core bibliography |
| `data/` | Datos de las aplicaciones (por publicar) | — |

## Cómo agregar material nuevo

La página del curso enlaza los archivos de este repositorio por su ruta, así que basta con respetar las convenciones de nombres:

- **Slides nuevas o actualizadas:** reemplazar `lectures/week-XX/week-XX.pdf` (la página siempre enlaza ese nombre). Sólo se versiona el PDF: el `.gitignore` deja fuera las fuentes `.tex`, los auxiliares de LaTeX y las carpetas `archive/`.
- **Aplicación en R de la sesión XX:** crear `applications/week-XX/` con el script `.R` (o un `.zip` con script y datos pequeños).
- **Problem Set:** dejar el PDF y sus datos en `problem-sets/pset-1/` o `problem-sets/pset-2/`.
- **Paper nuevo para presentaciones:** dejar el PDF en `literature/`.

Después de subir material nuevo, la página del curso se actualiza para agregar los enlaces correspondientes.

> **Nota:** GitHub rechaza archivos de más de 100 MB (y advierte desde 50 MB). Los datos grandes van comprimidos o se publican por fuera y se enlazan desde la página.
