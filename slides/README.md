# Forelesning: PyTorch 101

En norsk Slidev-presentasjon som følger notatbok 01–05, med overgang til det
valgfrie prosjektet i 06. 36 lysbilder, med omtrent 50–60 minutter forelesning fordelt på fem
bolker, avhengig av spørsmål og tempo. Oppgavearbeidet kommer mellom bolkene.

## Kjøring

Installer Node.js 22.12 eller nyere. Fra prosjektroten:

```bash
cd slides
npm ci
npm run dev
```

Åpne http://localhost:3030. Bruk piltastene for å navigere og `p` for
foreleservisning med notater. På lysbildet «Én parameteroppdatering» fremheves
kodelinjene trinnvis. Nettverksfiguren har knapper for valg av nevron, og pikselillustrasjonen har
en skyveknapp for lysstyrke. De øvrige lysbildene vises samlet.

```bash
npm run build    # Nettversjon i dist/
npm run export   # PDF i pytorch-101.pdf
```

PDF-eksport bruker Chromium fra `playwright-chromium`. Hvis nettleseren ikke ble
installert sammen med avhengighetene, kjør `npx playwright install chromium`.
En eksportert PDF har ikke forelesernotater eller interaktive fremhevinger.
Interaktive figurer vises med sine startverdier og kan leses uten interaksjon.
Det ferdigbygde nettstedet bruker lokale fonter og bilder og trenger ingen CDN.

## Foreleserveiledning

| Bolk | Tid | Arbeid etterpå |
|---|---:|---|
| Introduksjon, tensorer og matriser | 15–17 min | 01, ca. 45–60 min |
| Bilder og datasett | 10–12 min | 02, ca. 30–45 min |
| Prediksjoner og lag | 5–7 min | 03, ca. 40–55 min |
| Trening og evaluering | 10–12 min | 04, ca. 60–90 min |
| Nettverk og feilanalyse | 8–10 min | 05, ca. 90–120 min |

Tidene er veiledende, uten pauser. Bruk resterende tid på hjelp, diskusjon og
bonusprosjekt. Et mulig skille er 01–04 på dag 1 og 05 på dag 2. Hvis gruppen
trenger mer tid, flytt 04 til starten av dag 2 og kort ned bonusarbeidet.
Start dag 2 med å la studentene forklare de fem stegene i treningsløkken.

De grønne oppgaveslidene er stoppunkter. La studentene jobbe før neste bolk.
Etter notatbok 01, 03 og 04 følger et kontrollspørsmål. Sett av 30–60 sekunder
til diskusjon i par, og gjennomgå svaret muntlig. Fasit står i forelesernotatene.
HTML-kommentarene i `slides.md` er korte norske forelesernotater. Praktiske tidsanslag og gjennomføringsråd står her i
veiledningen. Lysbildene introduserer begrepene før kodeeksemplene; de er ikke
en gjennomgang av hver oppgavecelle eller en sammenhengende Python-fil.
Korte PyTorch-eksempler knytter teorien til syntaksen. Fullstendige løkker,
utregninger og API-detaljer utforskes i notatbøkene, som også er stedet for
kjøring og demonstrasjoner. Nettverk introduseres som helhet før vektede summer. Beregningen i ett nevron
introduseres med en generell formel før det numeriske nettverkseksemplet.

Hold hovedinnstillingene lik notatbøkene: 20 000 treningsbilder, 1 000 testbilder,
batchstørrelse 64, Adam med læringsrate 0.001 og 25 epoker. Nettverket har 128
skjulte nevroner. Sammenligningen med 8 nevroner og prosjektet i 06 er valgfrie.
Backpropagation forklares bare som beregning av gradienter, uten utledning.

## Filer og kilder

- `slides.md`: innhold og forelesernotater.
- `styles/index.css`: farger, typografi og oppsett.
- `components/LearningCurves.vue`: skjematiske trenings- og valideringskurver.
- `components/PixelGrid.vue`: tallene i en 4 × 4-pikselmatrise.
- `public/fashion-mnist.png`: faktiske treningsbilder fra kursets faste utvalg.
- `scripts/make_dataset_figure.py`: gjenskaper figuren med prosjektets Python-miljø.

Bildene kommer fra [Zalando Research, Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
under [MIT-lisensen](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).
Figuren inneholder ett bilde per klasse; den viser ikke modellprediksjoner.
Læringskurvene er skjematiske og viser ikke målte resultater fra notatbøkene.

Rekkefølgen følger kursnotatbøkene. Begrepsinndelingen er også inspirert av
[CS231n PyTorch Tutorial, Guanzhi Wang (2021)](https://cs231n.stanford.edu/2021/slides/2021/discussion_4_pytorch.pdf).
Tekst, manus og illustrasjoner er laget for dette kurset. Supplerende dokumentasjon:

- [PyTorch: tensorer](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch: trening og evaluering](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Slidev: bruk og bygging](https://sli.dev/guide/)

## Visuelle forklaringer

- `components/SingleNeuron.vue`: innganger, vekter, bias og aktivering for ett nevron.
- `components/NetworkDiagram.vue`: fullkoblet 3–4–2-nettverk med beregnede
  verdier og valg av skjult nevron. Konstruerte vekter, ikke en trent modell.
- `components/PixelValues.vue`: sammenheng mellom gråtone, råverdi og skalering.
- `components/ImageChannels.vue`: et RGB-bilde dekomponert i tre kanaler.

Nettverksillustrasjonen er inspirert av neuron- og vektvisualiseringene i
[3Blue1Brown: But what is a Neural Network?](https://www.3blue1brown.com/lessons/neural-networks/).
Figurene er laget for dette kurset. Nettverket bruker ReLU og rå utgangsskårer,
og viser ikke sigmoid-aktiveringer eller sannsynligheter.
