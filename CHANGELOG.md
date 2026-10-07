# Changelog

## [0.1.0](https://github.com/JoeHelbing/mam-bench/compare/v0.0.1...v0.1.0) (2026-10-07)


### Features

* **agents:** clarify model choices and align agent structure ([#17](https://github.com/JoeHelbing/mam-bench/issues/17)) ([2313158](https://github.com/JoeHelbing/mam-bench/commit/2313158b7078967ba32d5f66d4b3cc412e0fbfd5))
* **civil:** reserve one-call actions by target square ([#21](https://github.com/JoeHelbing/mam-bench/issues/21)) ([224359a](https://github.com/JoeHelbing/mam-bench/commit/224359aca8eab598b8bd935a1245e30d069a5c02))
* **config:** configure strict parameterless tools per model ([#22](https://github.com/JoeHelbing/mam-bench/issues/22)) ([2c23e52](https://github.com/JoeHelbing/mam-bench/commit/2c23e5273bdbdc9b2d678d5e2f1e3abd7380ffb7))
* **config:** ship six matched-seed default cases ([#26](https://github.com/JoeHelbing/mam-bench/issues/26)) ([605f35d](https://github.com/JoeHelbing/mam-bench/commit/605f35d4246414bdad1a5fdae883aebd23c49ad2))
* **schelling:** land one-call movement on dev ([#24](https://github.com/JoeHelbing/mam-bench/issues/24)) ([abbab38](https://github.com/JoeHelbing/mam-bench/commit/abbab38afd8b410c7c7ffcc5b6599a0865205d29))
* **schelling:** render dual-view agent observations ([#19](https://github.com/JoeHelbing/mam-bench/issues/19)) ([5172518](https://github.com/JoeHelbing/mam-bench/commit/5172518e77977f4cd94b456962846f2b895d14f1))
* **sessions:** return explicit turn outcomes ([03b4630](https://github.com/JoeHelbing/mam-bench/commit/03b4630ecd252533f784fb5dd45c17e2c8398168))
* **sessions:** return explicit turn outcomes ([b476872](https://github.com/JoeHelbing/mam-bench/commit/b476872ab168c8f38a95bc2183b8442ef24a0d64))


### Bug Fixes

* **runtime:** normalize SGLang weight-version telemetry ([#23](https://github.com/JoeHelbing/mam-bench/issues/23)) ([62355ae](https://github.com/JoeHelbing/mam-bench/commit/62355ae9775b68330c0562fad6a31d134eaf9490))
* **simulations:** match paired scheduling and random slots ([56971cb](https://github.com/JoeHelbing/mam-bench/commit/56971cbd372a91d69381ac2191f7d2d9b6c3b10a))
* **simulations:** match paired scheduling and random slots ([30e491f](https://github.com/JoeHelbing/mam-bench/commit/30e491f9bd28765016ee0a59d42ebe7766e16897))

## 0.0.1 (2026-09-29)


### ⚠ BREAKING CHANGES

* **schelling:** use floats for case proportions
* implement paired single-model benchmark suites
* simplify benchmark runtime and persist conversations

### Features

* add persistent agent session runtime ([29fc385](https://github.com/JoeHelbing/mam-bench/commit/29fc3854cc2c0a466b361de254244c6e665e5c54))
* add shared communication and rolling sessions ([91c4b63](https://github.com/JoeHelbing/mam-bench/commit/91c4b63ec5ef7729fe74499bf2031a07bbea7c96))
* establish MAM-Bench benchmark foundation ([7c63229](https://github.com/JoeHelbing/mam-bench/commit/7c63229f1afbc09a14568fb23c4ec3c9ced21c88))
* implement paired single-model benchmark suites ([f310794](https://github.com/JoeHelbing/mam-bench/commit/f31079475fd71237371b0bac5b4cf6ac9fb4aecd))
* **schelling:** add locally embodied actor turns ([0f5e779](https://github.com/JoeHelbing/mam-bench/commit/0f5e779c6f3115a0790fe385b81a49e68c7a3027))
* **schelling:** enforce actor turn policy ([e9ef263](https://github.com/JoeHelbing/mam-bench/commit/e9ef263b171c71479e78d609c84ddffe99b6a484))
* **schelling:** finalize v2 with simpler agent sessions ([fc65d5e](https://github.com/JoeHelbing/mam-bench/commit/fc65d5eedd7228d6ec8825dee1ad1a580d274e81))
* **simulations:** add tactical Civil Violence evaluations ([17d102f](https://github.com/JoeHelbing/mam-bench/commit/17d102f62477c6a8a407a1a77b0da4a34696a557))
* updated vision movement schelling ([1d408b1](https://github.com/JoeHelbing/mam-bench/commit/1d408b1d41394a0a9cfb254aa9839cb0a2c82a7e))
* updated vision movement schelling ([2810a09](https://github.com/JoeHelbing/mam-bench/commit/2810a09e4001757229b36b6630513e230c7b49c4))


### Bug Fixes

* **ci:** align Dependabot titles with release convention ([f299b87](https://github.com/JoeHelbing/mam-bench/commit/f299b87d7bc0428b30669d4b116f20d37c7b068f))
* **release:** align alpha version ([4569846](https://github.com/JoeHelbing/mam-bench/commit/45698465da6a14c2ef80f0b3c299d97a7f26972d))
* remove readme cruft that agents love to shove in there ([7234fa1](https://github.com/JoeHelbing/mam-bench/commit/7234fa1c7728d3e4663e0d59e2c35af367874508))
* remove readme cruft that agents love to shove in there ([a14203e](https://github.com/JoeHelbing/mam-bench/commit/a14203e8ba0c1c99bce59ce27c3f18076606f607))


### Documentation

* link MAM-Bench project post ([eccab31](https://github.com/JoeHelbing/mam-bench/commit/eccab31c4814de3f5d73b7ad30a666a0917ce9c6))
* mark MAM-Bench as work in progress ([c77bbf6](https://github.com/JoeHelbing/mam-bench/commit/c77bbf6db6baf44cb0acf2cf01a706852c683105))


### Code Refactoring

* **schelling:** use floats for case proportions ([0face31](https://github.com/JoeHelbing/mam-bench/commit/0face31b0ac44fb570daf65c53be5a2a882101c9))
* simplify benchmark runtime and persist conversations ([e94b424](https://github.com/JoeHelbing/mam-bench/commit/e94b42446ac36ad29d4866f6a35175935e5c5f45))
