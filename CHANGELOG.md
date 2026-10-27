# Changelog

## [2.0.0](https://github.com/JoeHelbing/mam-bench/compare/v1.0.2...v2.0.0) (2026-10-27)


### ⚠ BREAKING CHANGES

* **runs:** case results require a `null` outcome and add `null_score` and `adjusted_score`; BenchmarkResult version is 0.2.0 and total_score sums adjusted scores. Cases write null.jsonl and null-outcome.json.
* **scoring:** case results require a `null` outcome and add `null_score` and `adjusted_score`; BenchmarkResult version is 0.2.0 and total_score sums adjusted scores. Cases write null.jsonl and null-outcome.json.
* **settings:** Anthropic efforts now go out as output_config.effort instead of PydanticAI's budget mapping; budget-only models may refuse.
* **agents:** add private Modal workspaces ([#63](https://github.com/JoeHelbing/mam-bench/issues/63))

### Features

* **agents:** add private Modal workspaces ([#63](https://github.com/JoeHelbing/mam-bench/issues/63)) ([0a9d5bb](https://github.com/JoeHelbing/mam-bench/commit/0a9d5bb1253c890744cae2235c72018a563bbb49))
* **anthropic:** support native API benchmark runs ([#61](https://github.com/JoeHelbing/mam-bench/issues/61)) ([ae2c582](https://github.com/JoeHelbing/mam-bench/commit/ae2c582f74b81d8767a57e72d642a2d9eed69490))
* **openai:** support native API benchmark runs ([#60](https://github.com/JoeHelbing/mam-bench/issues/60)) ([8550df9](https://github.com/JoeHelbing/mam-bench/commit/8550df930323f2abb0ca194e1e5c67292663eafb))
* **runs:** fail runs whose model cannot play; retry provider errors instead of falling back ([#69](https://github.com/JoeHelbing/mam-bench/issues/69)) ([2a11516](https://github.com/JoeHelbing/mam-bench/commit/2a11516b1332c9eaf0c7d432988e42df61f006a4))
* **scoring:** score each case against a null-policy world ([#68](https://github.com/JoeHelbing/mam-bench/issues/68)) ([9be6032](https://github.com/JoeHelbing/mam-bench/commit/9be6032ef50d744a2d326bc4d6ac7ed0d3d5decd))
* **settings:** let providers judge reasoning_effort and sampling values ([#65](https://github.com/JoeHelbing/mam-bench/issues/65)) ([abf060b](https://github.com/JoeHelbing/mam-bench/commit/abf060ba0dd58dc2ff322690aa4be17369de7eee))


### Bug Fixes

* **deps:** update pydantic-ai ([#80](https://github.com/JoeHelbing/mam-bench/issues/80)) ([a041fa6](https://github.com/JoeHelbing/mam-bench/commit/a041fa64cbd2d08ec22f8f95ec4266483db6842c))
* **models:** isolate client workarounds and document serving fixes ([#53](https://github.com/JoeHelbing/mam-bench/issues/53)) ([0cccf3d](https://github.com/JoeHelbing/mam-bench/commit/0cccf3d1a586c722b62a58b0bcd82362b1520221))
* recover transient API call failures within turns ([#57](https://github.com/JoeHelbing/mam-bench/issues/57)) ([cac856b](https://github.com/JoeHelbing/mam-bench/commit/cac856b8d237336db234718ca1c09ead7721b097))
* **runtime:** bring streaming capture changes into dev ([#59](https://github.com/JoeHelbing/mam-bench/issues/59)) ([25baebf](https://github.com/JoeHelbing/mam-bench/commit/25baebf387806388ac80e9f1d48985e6b8f45fa0))

## [1.0.2](https://github.com/JoeHelbing/mam-bench/compare/v1.0.1...v1.0.2) (2026-10-15)


### Bug Fixes

* **runtime:** cap default output at a quarter of context ([3bfab73](https://github.com/JoeHelbing/mam-bench/commit/3bfab73de53d3c55981e9e2f531fed481dbf51b4))
* **runtime:** cap default output at a quarter of context ([bc2171d](https://github.com/JoeHelbing/mam-bench/commit/bc2171dd5500acef460787bc43a2a0732d5b3a14))
* **telemetry:** capture conversation content by default ([b52f7b1](https://github.com/JoeHelbing/mam-bench/commit/b52f7b1f691d78250e769d4ee9c3bb7b7af51812))
* **telemetry:** capture conversation content by default ([234b5fe](https://github.com/JoeHelbing/mam-bench/commit/234b5fe4b14e618302c6b3b9d87592e8f7842444))

## [1.0.1](https://github.com/JoeHelbing/mam-bench/compare/v1.0.0...v1.0.1) (2026-10-14)


### Bug Fixes

* **telemetry:** load Logfire settings from .env ([847b8f7](https://github.com/JoeHelbing/mam-bench/commit/847b8f78ccb53fc9ab658d7a356baccf3b3b9f89))
* **telemetry:** load Logfire settings from .env ([4803ec5](https://github.com/JoeHelbing/mam-bench/commit/4803ec54ccb254e3faa567793931b98b5de56bba))

## [1.0.0](https://github.com/JoeHelbing/mam-bench/compare/v0.2.0...v1.0.0) (2026-10-14)


### ⚠ BREAKING CHANGES

* **config:** discover limits and preserve endpoint defaults
* **runtime:** use native vLLM provider safeguards
* **agents:** accept ready actions without claim priority

### Features

* **observability:** collect native OpenTelemetry signals ([505d2f3](https://github.com/JoeHelbing/mam-bench/commit/505d2f394962bfd4722d78c07b76c72491fdeeaf))
* **observability:** collect native OpenTelemetry signals ([ac4f821](https://github.com/JoeHelbing/mam-bench/commit/ac4f821ab144606562b38d032a98c1794663efa8))
* **openrouter:** resolve endpoint-aware model settings ([26c24a6](https://github.com/JoeHelbing/mam-bench/commit/26c24a6020a7ced5adb2a44ed7da0d2af1b6fd39))


### Bug Fixes

* **agents:** accept ready actions without claim priority ([14a97f9](https://github.com/JoeHelbing/mam-bench/commit/14a97f91649912ff376892211652980b5467e363))
* **runtime:** handle vLLM repetition detection stops ([e66d225](https://github.com/JoeHelbing/mam-bench/commit/e66d22574c7ed1f8dd160a7eab500dc81aff562a))
* **runtime:** handle vLLM repetition detection stops ([1716471](https://github.com/JoeHelbing/mam-bench/commit/1716471b5c8740621f99842ca1807e726ab8e2ce))


### Code Refactoring

* **config:** discover limits and preserve endpoint defaults ([3614eaa](https://github.com/JoeHelbing/mam-bench/commit/3614eaa41e89db8ab1c1e854fb3badf870e5df92))
* **runtime:** use native vLLM provider safeguards ([dc32d22](https://github.com/JoeHelbing/mam-bench/commit/dc32d22367f8541be619539d3fcf19319d9b0309))

## [0.2.0](https://github.com/JoeHelbing/mam-bench/compare/v0.1.0...v0.2.0) (2026-10-07)


### Features

* **civil:** allow configurable step limits ([ff99d8f](https://github.com/JoeHelbing/mam-bench/commit/ff99d8f63fabd83c38f5f22d534026955fe18d9f))
* **civil:** allow configurable step limits ([9ddce5f](https://github.com/JoeHelbing/mam-bench/commit/9ddce5f5b93eb35956a1cfd244a0c57f7edbf787))


### Documentation

* refresh public GitHub landing page ([d29f845](https://github.com/JoeHelbing/mam-bench/commit/d29f845521927a7e620dabf4ee1d3242f4ac51b4))
* refresh public GitHub landing page ([e4a5698](https://github.com/JoeHelbing/mam-bench/commit/e4a56983e659311876945419e8ee23b6d1248071))

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
