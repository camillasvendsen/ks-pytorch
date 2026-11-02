<template>
  <svg viewBox="0 0 510 350" class="w-full" role="img" aria-labelledby="single-neuron-title single-neuron-desc">
    <title id="single-neuron-title">Beregningen i ett nevron</title>
    <desc id="single-neuron-desc">Inngangene x1 til xD multipliseres med vektene w1 til wD. Summen får et bias b og gir z. ReLU gir aktiveringen h.</desc>
    <defs>
      <marker id="single-neuron-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#53616a" /></marker>
    </defs>
    <g fill="none" stroke="#53616a" stroke-width="2" marker-end="url(#single-neuron-arrow)">
      <path d="M80 70 L235 167" />
      <path d="M80 155 L230 178" />
      <path d="M80 290 L235 195" />
      <path d="M265 75 V150" />
      <path d="M300 180 H365" />
      <path d="M435 180 H490" />
    </g>
    <g fill="#eef3f0" stroke="#176b68" stroke-width="2">
      <circle cx="55" cy="70" r="25" />
      <circle cx="55" cy="155" r="25" />
      <circle cx="55" cy="290" r="25" />
      <circle cx="265" cy="180" r="35" />
      <rect x="365" y="155" width="70" height="50" rx="5" />
    </g>
    <g text-anchor="middle" fill="#202d37" style="font-size: 22px !important">
      <text x="55" y="77">x₁</text><text x="55" y="162">x₂</text><text x="55" y="297">x<tspan baseline-shift="sub" style="font-size: 15px !important">D</tspan></text>
      <text x="55" y="230">⋮</text>
      <text x="265" y="188">Σ</text>
      <text x="265" y="62">b</text>
      <text x="330" y="165">z</text>
      <text x="480" y="165">h</text>
      <text x="400" y="186" style="font-size: 18px !important">ReLU</text>
    </g>
    <g fill="#176b68" text-anchor="middle" style="font-size: 22px !important">
      <text x="150" y="96">w₁</text><text x="150" y="151">w₂</text><text x="150" y="268">w<tspan baseline-shift="sub" style="font-size: 15px !important">D</tspan></text>
    </g>
    <g fill="#53616a" text-anchor="middle" style="font-size: 17px !important">
      <text x="55" y="340">Innganger</text><text x="160" y="340">Vekter</text><text x="265" y="35">Bias</text>
    </g>
  </svg>
</template>
