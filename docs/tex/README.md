# Code Guide (LaTeX)

Questa cartella contiene la guida tecnica del codice:

- `main.tex`: documento principale.
- `chapters/`: capitoli per area (Simulator, RBC, MPC, RL).
- `files/`: descrizione puntuale dei file Python core.

## Build (opzionale)

Con `tectonic`:

```powershell
tectonic docs/tex/main.tex --outdir outputs/docs
```

Con `pdflatex`:

```powershell
cd docs/tex
pdflatex main.tex
```
