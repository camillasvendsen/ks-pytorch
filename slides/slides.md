---
theme: default
title: PyTorch 101
info: 36 lysbilder med forelesernotater. Omtrent 50–60 minutter, fordelt over kurset.
class: cover
colorSchema: light
aspectRatio: 16/9
canvasWidth: 1100
fonts:
  sans: DejaVu Sans
  mono: DejaVu Sans Mono
  provider: none
htmlAttrs:
  lang: nb
transition: none
mdc: true
drawings:
  enabled: true
---

# PyTorch 101

<div class="lead">Tensorer, nevrale nettverk og bildeklassifikasjon</div>

<!--
PyTorch - et rammeverk i python for å jobbe med nevrale nett og dyp læring.
Vi skal gå gjennom:

* De grunnleggende byggesteinene i pytorch
* En liten recap på matriser, med dimensjoner og hvordan de ganges sammen
* Bittelitt bildebehandling
* Bygge og trene en modell på et datasett

Hvorfor?

* Anthropic og OAI bruker Pytorch til å lage claude og gpt!
* Veldig relevant for analyse av satelittbilder, dronebilder osv
* Veldig kult

Quiz

* Hvem har hørt om dyp læring før?
* Er vi komfortable med å gange sammen 2 matriser?
-->

---

# Bildeklassifikasjon

<img src="/fashion-mnist.png" class="h-55 mx-auto" alt="Ett bilde fra hver av de ti Fashion-MNIST-klassene" />

<div class="flow"><div class="box">Bilde</div><span>→</span><div class="box">Modell</div><span>→</span><div class="box">Ti klasseskårer</div><span>→</span><div class="box">Predikert klasse</div></div>


<div class="small">Fashion-MNIST · Zalando Research</div>

<!--
Casen vi skal jobbe med i dag og i morgen

* skal skille klær, sko og vesker i ti klasser.
* For å gjøre det: trene en modell som bruker treningsdata (altså eksempler med (bilde, fasit)-par)
* Fashio-MNIST fra zalando research
-->

---

# Nevrale nettverk

<div class="flow"><div class="box">Innganger<br>pikselverdier</div><span>→</span><div class="box">Skjult lag<br>mellomverdier</div><span>→</span><div class="box">Utganger<br>klasseskårer</div></div>

Et nevralt nettverk består av lag med lærbare parametere: **vekter og bias**.

Ulineære aktiveringsfunksjoner mellom lagene gjør at nettverket kan representere ulineære sammenhenger.

<div class="note">PyTorch gir oss tensorer, modellag og automatisk beregning av gradienter.</div>

<!--
Hva er et nevralt nett?
Kort sagt:

* En funksjonsapproksimator
* Her: å tilpasse en funksjon som mapper bilder til en klasse (tall fra 0-9)
* Består av lærbare parametre: vekter og bias, organisert i et valgfritt antall lag
* Ulineære aktiveringsfunksjoner som lar nettverket representere ulineære sammenhenger
* Pytorch gir oss en del primitiver som er greie å jobbe med: tensorer, forhåndsdefinerte lag, tapsfunksjoner og viktigst: automatisk beregning av gradienter

Dette kommer vi tilbake til

-->

---

# Ett nevron

<div class="grid grid-cols-[1.15fr_1fr] gap-8 items-center mt-10">
<div>

<SingleNeuron />

</div>
<div>

**Vektet sum + bias**

$$
z = \sum_{i=1}^{D} w_i x_i + b
  = \mathbf{w}^{\mathsf T}\mathbf{x} + b
$$

**Aktivering**

$$
h = \operatorname{ReLU}(z) = \max(0, z)
$$

</div>
</div>

<!--
Eksempel med ett nevron fra et nettverk

* Vi gjør en vektet sum av innganger og legger til en bias for å produsere z
* Kjører z gjennom en ulineær aktiveringsfunksjon og får h, som er utgangen
* Altså: x_1*w_1 + x_2*w_2 + ... + x_D*w_D + b, også max av det
-->

---

# Eksempel

<NetworkDiagram />


<!--
Eksempel:

* 3 innganger, 4 utganger

For første nevron i det skjulte laget:

* følg eksempelet
-->

---

# Tensorer og former

| Tensor | Eksempel på form | Betydning |
|---|---|---|
| Skalar | `[]` | Ett tall |
| Vektor | `[3]` | Tre tall langs én akse |
| Matrise | `[2, 3]` | To rader, tre kolonner |
| Bildebatch | `[B, C, H, W]` | Bilder, kanaler, høyde, bredde |

```python
import torch
values = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
values.shape  # torch.Size([2, 3])
```

<!--
Vi hopper tilbake til den grunnleggende enheten i Pytorch: Tensorer.

Tensor er fellesnavnet for tall organisert langs null eller flere akser. Formen/dimensjonen forteller hvor mange verdier hver akse har.
-->

---

# Indeksering og akser

```python
values = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
values[0]                  # Første rad: [1., 2., 3.]
values[:, 1]               # Andre kolonne: [2., 5.]
values.mean(dim=1)         # Ett gjennomsnitt per rad: [2., 5.]
values + 2                 # Legg 2 til hvert tall
```

**`dim` velger aksen vi regner over.**

`shape`: organisering · `dtype`: talltype · `device`: CPU eller GPU

<!--
Tensorer indekseres omtrent som vanlige python-lister:

* index 0 gir første rad
* bruk slicing for å hente ut kolonne 2 med index 1

Tensorer har en rekke metoder, feks:

* .mean() regner ut gjennomsnitt
* ved å oppgi dim=1, summerer vi over kolonnene, og ender opp med ett tall per rad
-->

---

# Matrisemultiplikasjon

$$
C=AB,\qquad C_{ik}=\sum_{j=1}^{D} A_{ij}B_{jk}
$$

$$
[B,D]\; @\; [D,K]\;\longrightarrow\;[B,K]
$$

Elementet $C_{ik}$ er prikkproduktet av rad $i$ i $A$ og kolonne $k$ i $B$.

```python
result = inputs @ weights
```

<!--
Dyp læring handler i stor grad om å gange sammen matriser, så det må vi kunne.
Her er formelen. La oss si vi har 2 matriser, A og B, og lar C=AB.
Da er hvert element i C, C_ik prikkproduktet av rad _i_ i A med kolonne k i B.

Noe av det viktigste rent praktisk er å forstå dimensjonene til matriser som skal ganges sammen.
Altså, hvorfor noe funker, og noe gir feilmeldinger om upassende dimesjoner. Det skal dere få jobbe med etterpå.
Men greia er:

* B,D-matrise, ganger D,K-matrise, gir en B,K-matrise.

I PyTorch bruker vi @, eller .dot()
-->

---

# Matrisemultiplikasjon: eksempel

$$
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
\begin{bmatrix}2&0\\1&1\\0&2\end{bmatrix}
=\begin{bmatrix}4&8\\13&17\end{bmatrix}
$$

Første tall: $1\cdot2+2\cdot1+3\cdot0=4$.

```python
inputs = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
weights = torch.tensor([[2., 0.], [1., 1.], [0., 2.]])
result = inputs @ weights  # Form [2, 2]
```

**`@`** beregner matriseprodukt. **`*`** ganger elementvis.

<!--
-->

---
class: exercise
---

# Oppgaver 01 · Tensorer

`oppgaver/01_tensorer.ipynb`

Indeksering, reduksjoner, broadcasting og matrisemultiplikasjon.

<!--
-->

---

# Quiz

`scores` har formen `[32, 10]`: 32 bilder og 10 klasseskårer per bilde.

```python
predictions = scores.argmax(dim=1)
```

Hva blir dimensjonene til `predictions`, og hva betyr tallene?

<!--
Diskuter med sidemannen.
Svar: Formen er [32]. Hver verdi er indeksen til klassen med høyest skår for ett bilde, altså et heltall fra 0 til 9. Vi velger langs klasseaksen, dim=1.
Valgfritt oppfølgingsspørsmål: Hva gir argmax(dim=0)? Formen blir [10], med indeksen til bildet med høyest skår for hver klasse, ikke én prediksjon per bilde.
-->

---

# Bilder som tensorer

<div class="cols">
<div>
<PixelGrid />
</div>
<div>

Én piksel = én lysstyrke.

**0 er svart. 1 er hvitt.**

```python
image[0, row, column]
```

Et gråtonebilde: `[1, H, W]`

Et RGB-bilde: `[3, H, W]`

</div></div>

<!--
Et digitalt bilde er bare en matrise med lysintensiteter. Her ser vi på en kanal av et bilde. 
-->

---

# Pikselverdier og skalering

<PixelValues />

```python
raw_pixels = torch.tensor([0, 128, 255], dtype=torch.uint8)
pixels = raw_pixels.float() / 255
# tensor([0.0000, 0.5020, 1.0000])
```

Skalering endrer tallområdet, men bevarer relative lysstyrker.

<!--
Bilder er på 0-255 fordi de lagres som 8-bit heltall

Vi skal jobbe med skalaen 0-1, derfor deler vi på 255.
0=svart, 1=hvitt
-->

---

# Fargekanaler

<ImageChannels />

```python
red = image[0]    # [H, W]
green = image[1]  # [H, W]
blue = image[2]   # [H, W]
```

Fashion-MNIST har én gråtonekanal: `[1, 28, 28]`.

<!--
Her er et RGB bilde med 4 piksler, og 3 kanaler (RGB).
Hver kanal er en matrise.

Dette er altså en matrise med 3 dimensjoner: dybde, høyde og bredde.
-->

---

# Bildeform og batchakse

<div class="flow"><div class="box">Pikselmatrise<br><code>[28, 28]</code></div><span>→</span><div class="box">Én kanal<br><code>[1, 28, 28]</code></div><span>→</span><div class="box">64 bilder<br><code>[64, 1, 28, 28]</code></div></div>

```python
image.shape             # [1, 28, 28]
image.squeeze(0).shape  # [28, 28], for visning med imshow
```

| Akse | Betydning |
|---|---|
| `B` | Antall bilder i batchen |
| `C` | Antall kanaler per bilde |
| `H, W` | Høyde og bredde i piksler |

<!--
Vi skal jobbe med 4 dimensjoner, bare for å gjøre det enda vanskeligere :)

Så vi skal jobbe med mange bilder samtidig, feks 64.
Da får vi 64 bilder, som hver har 1 fargekanal, og 28x28 piksler.

Det første tallet kalles for batch-size.
-->

---

# Fashion-MNIST

```python
from pytorch_101.utils import CLASS_NAMES, download_fashion_mnist

train_data, test_data = download_fashion_mnist()
image, label = train_data[0]
image.shape          # [1, 28, 28]
CLASS_NAMES[label]   # Navnet på riktig klasse
```

**10 klasser · 20 000 treningsbilder · 1 000 testbilder**


<!--
Som sagt jobbe med et kles-datasett. Vi har 10 klasser, 20k treningsbilder og 1k testbilder.

Vi har laget ferdig kode som laster ned og pakker datasettet for dere.
-->

---

# Klassefordeling og referansenøyaktighet

Anta 100 bilder:

| Klasse | Antall |
|---|---:|
| T-skjorte | 80 |
| Bukse | 10 |
| Sko | 10 |

En modell som alltid velger T-skjorte får **80 % nøyaktighet**, men finner ingen bukser eller sko.

```python
baseline_accuracy = max(class_counts) / sum(class_counts)
```

<!--
Et par ord om nøyaktighet:

Anta vi har 100 bilder, hvor 80 av dem tilhører en klasse. 

Da vil en modell som alltid gjetter tskjorte få 80% nøyaktighet.

Samlet nøyaktighet må tolkes i lys av klassefordelingen.
-->

---
class: exercise
---

# Oppgaver 02 · Bilder

`oppgaver/02_bilder.ipynb`

Visualisering av bilder, klassefordeling og nøyaktighet ved flertallsgjetting.

<!--
Stopp forelesningen her og la studentene arbeide. Bruk eksemplene og hintene i notatboken ved behov.
-->

---

# Batching og DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))
images.shape  # [64, 1, 28, 28]
labels.shape  # [64]
```

**Dataset** henter eksempler. **DataLoader** samler dem i grupper.

`shuffle=True` stokker rekkefølgen for hver gjennomgang.

<!--
Vi skal som nevnt jobbe med batches (altså mange bilder i parallell). 

Bruker dataloaders til dette.
Datasett henter enkelt-eksempler; Dataloader samler mange eksempler i batches.
-->

---

# Lineær klassifikasjon

$$
Z=XW+b
$$

$$
[B,784]\;@\;[784,10]+[10]\;\longrightarrow\;[B,10]
$$

```python
flat_images = images.flatten(start_dim=1)
weights = torch.randn(784, 10) * 0.01
bias = torch.zeros(10)
scores = flat_images @ weights + bias
```

**Vekter** bestemmer bidragene. **Bias** gir ett konstantledd per klasse.

<!--
Vi skal jo trene en ML-modell. 
Modellene vi skal lage, kan ikke regne på romlige bilder direkte, så vi må flate de ut. 
Fra 28x28 til 784.
Da bruker vi flatten, og ganger sammen på vanlig måte
-->

---

# nn.Linear og nn.Sequential

```python
from torch import nn

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
scores = model(images)             # [B, 10]
predictions = scores.argmax(dim=1) # [B]
```

**`nn.Linear`** eier vekter og bias. **`nn.Sequential`** setter lag i rekkefølge.

Skårene kalles **logits**. 

Predikert klasse er indeksen til den høyeste skåren.

Nøyaktighet = antall riktige / antall eksempler.

<!--
Pytorch gir oss slike lineære lag gratis, med lærbare parametre.
Linear gjør den samme beregningen som vi nettopp skrev for hånd. 
Argmax velger klassekolonnen med størst skår for hvert bilde.

Vi kan så regne ut nøyaktigheten ved å ta antall riktige delt på antall eksempler.
-->

---
class: exercise
---

# Oppgaver 03 · Prediksjoner

`oppgaver/03_prediksjoner.ipynb`

Vektede summer, klasseskårer og en lineær modell med `nn.Linear`.

<!--
-->

---

# Quiz

En modell gir ti klasseskårer for hvert bilde. 

**Er disse sannsynligheter?**

**Hvorfor kan vi likevel bruke argmax til å velge klasse?**

<!--
The scores are logits—they can be negative and don’t necessarily sum to one. argmax selects the highest score; converting to probabilities with softmax would preserve that ordering
-->

---

# Trening av modellparametere

<div class="flow"><div class="box">Bilder</div><span>→</span><div class="box">Modell</div><span>→</span><div class="box">Skårer</div><span>→</span><div class="box">Tap mot fasiten</div></div>

1. Beregn prediksjoner og tap (**forward**).
2. Beregn hvordan parameterne påvirker tapet (**backward**).
3. Juster vekter og bias (**optimizer**).

Treningen justerer parameterne for å redusere tapet på treningsdataene.

<!--
Nå skal vi begynne å trene modellen.
En PyTorch treningsløkke ser omtrent sånn her ut:

Forward pass:

* Kjør input gjennom modellen og regn ut tap mot faisten.

Backward pass
* Beregn gradienten til tapet med hensyn på alle parameterne.

Til slutt: Juster vekter og bias med en optimaliseringsteknikk
-->

---

# Kryssentropi og nøyaktighet

Fasit: klasse 1.

| Sannsynligheter | Predikert klasse | Riktig? | Tap $-\log(p_y)$ |
|---|---|---|---:|
| `[0.4, 0.6]` | 1 | Ja | 0.51 |
| `[0.1, 0.9]` | 1 | Ja | 0.11 |

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(scores, labels)
```

<!--
Vi skal bruke kryssentropi som tap i dag.
Så anta at vi har 2 korrekte prediksjoner. 
Den ene er 60% sikker, den andre 90%. 

Selv om begge har rett, vil tapet være vesentlig større i det første tilfellet.
-->

---

# Gradienter og læringsrate

Gradienten beskriver hvordan tapet endres ved små endringer i parameterne.

Enkel gradient descent:

$$
w_{\mathrm{ny}}=w-\eta\frac{\partial L}{\partial w}
$$

**Læringsraten** $\eta$ styrer størrelsen på steget.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Vi bruker **Adam**, en populær optimizer.

<!--
Gradienter brukes for å justere parameterne i nettverket (vekter og bias).

Formelen for å justere en parameter gitt en gradient er som følger, mest for at dere skal ha sett det.
-->

---

# Én parameteroppdatering

```python {all|1|2-3|4|5|all}
optimizer.zero_grad()           # Nullstill gamle gradienter
scores = model(images)         # Beregn skårer
loss = loss_fn(scores, labels)  # Sammenlign med fasiten
loss.backward()                # Beregn gradienter
optimizer.step()               # Oppdater vekter og bias
```

**Batch:** én gruppe bilder per oppdatering.

**Epoke:** én gjennomgang av treningsutvalget.

<!--
I kode ser det ca sånn ut:

* Nullstill gamle gradienter
* Kjør input gjennom modell
* Regn ut avviket mellom prediksjon og fasit
* Regn nye gradienter
* Oppdater parametre
-->


---
class: exercise
---

# Oppgaver 04 · Trening

`oppgaver/04_lineaer_klassifikator.ipynb`

Treningsløkke, tapskurve og testnøyaktighet for den lineære modellen.

<!--
-->

---

# Quiz

```python
loss.backward()
optimizer.step()
```

**Hva er forskjellen mellom disse to kallene?**

Hva beregnes, og når endres modellens parametere?

<!--
Svar: loss.backward() beregner gradientene til tapet med hensyn på modellens parametere og akkumulerer dem i parameter.grad. Kallet oppdaterer ikke vekter eller bias. optimizer.step() bruker gradientene til å oppdatere parameterne etter optimizerens regel, her Adam.
-->

---

# Nettverk med ett skjult lag

<div class="flow"><div class="box">Piksler<br><strong>784</strong></div><span>→</span><div class="box">Skjult lag + ReLU<br><strong>128</strong></div><span>→</span><div class="box">Klasseskårer<br><strong>10</strong></div></div>

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

Det skjulte laget lærer **mellomliggende representasjoner**.

<!--
Til nå: lineær modell.
Nå skal vi legge til et skjult lag, og ulineære aktiveringsfunksjoner.
-->

---

# Aktiveringsfunksjoner: ReLU

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

```python
values = torch.tensor([-2., -0.5, 0., 1., 3.])
torch.relu(values)  # tensor([0., 0., 0., 1., 3.])
```

ReLU lager et **knekkpunkt**: negative verdier blir null.

Flere lineære lag uten aktivering kan slås sammen til ett lineært lag.

<!--
ReLU endrer ikke formen og har ingen parametere som læres. Flere knekkpunkter kan kombineres til mer sammensatte funksjoner. La studentene gjøre den algebraiske begrunnelsen og utforske kurvefiguren i notatboken.
-->

---


# Overtilpasning og generalisering

<LearningCurves />

**Trening:** lær parametere · **Validering:** velg modell · **Test:** sluttvurdering

<!--
Når treningstapet synker mens valideringstapet øker, kan det være overtilpasning. En treningskurve alene kan ikke avgjøre dette. Vi har ikke et eget valideringssett i hovedoppgavene. Før systematisk modellvalg bør vi sette av valideringsdata, i stedet for å velge etter testresultatene.
-->

---
class: exercise
---

# Oppgaver 05 · Nevrale nettverk

`oppgaver/05_nevralt_nettverk.ipynb`

Trening nettverk med flere lag, sammenligning modeller og feilanalyse

<!--
-->

---

# Oppsummering

| Del | PyTorch | Rolle |
|---|---|---|
| Data | `torch.Tensor`, `DataLoader` | Tall, former og batcher |
| Modell | `nn.Linear`, `nn.ReLU` | Inngang → skårer |
| Tap | `nn.CrossEntropyLoss` | Skårer og fasit → tap |
| Gradienter | `loss.backward()` | Deriver tapet med hensyn på parameterne |
| Oppdatering | `optimizer.step()` | Juster parameterne |
| Evaluering | `eval()`, `no_grad()` | Undersøk usette data |

<!--
Be studentene forklare beregningen fra ett bilde til en prediksjon, og hva som skjer i tillegg under trening. Neste steg er det valgfrie prosjektet: bruk den samme arbeidsflyten på nye data.
-->
