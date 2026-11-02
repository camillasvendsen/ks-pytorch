# PyTorch 101

## Kom i gang

### GitHub Codespaces

1. Logg inn på GitHub og åpne [kursrepoet](https://github.com/namsmann1/ks-pytorch).
2. Velg **Code → Codespaces → Create codespace on main**
3. Vent til miljøet er ferdig satt opp.
4. Åpne `oppgaver/01_tensorer.ipynb`. Velg **Select Kernel → Python Environments**
   og prosjektets `.venv`-miljø hvis du blir bedt om å velge kjerne.

Du kan kjøre notatbøkene direkte i VS Code i nettleseren.
Se [GitHubs Codespaces-veiledning](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository)
ved behov.

### Arbeid med oppgavene

Kjør cellene ovenfra og ned. Les eksemplet, løs oppgaven i den tomme kodecellen,
og fyll ut Markdown-cellene merket **Mitt svar**. Dobbeltklikk på
en Markdown-celle for å redigere; bruk Shift+Enter for å vise svaret.

## Rekkefølge

| Notatbok | Tema |
|---|---|
| [01](oppgaver/01_tensorer.ipynb) | Tensorer, akser, broadcasting og matriseprodukt |
| [02](oppgaver/02_bilder.ipynb) | Bilder, klassefordeling og enkel referanse |
| [03](oppgaver/03_prediksjoner.ipynb) | Prediksjoner |
| [04](oppgaver/04_lineaer_klassifikator.ipynb) | Trening og evaluering av en lineær modell på Fashion-MNIST |
| [05](oppgaver/05_nevralt_nettverk.ipynb) | Utvid modellen med et skjult lag. Feilanalyse |
| [06 (bonus)](oppgaver/06_bonus.ipynb) |

## Forelesningsmateriell

Se [Slides](slides/pytorch-101.pdf) for pdf-versjon av presentasjonen

Eventuelt, bygg slidev-prosjektet selv (krever node)

```bash
cd slides
npm ci
npm run dev
```

## Data og løsninger

Se `src/pytorch_101/utils/data.py` for hjelpefunksjoner til nedlasting av data.

Datasett: [Zalando Research – Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

## Referanser

- [PyTorch introduksjon](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
